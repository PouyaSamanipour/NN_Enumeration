"""
verify_certificate_face.py
==========================
Alternative to verify_certificate_new.py that refines the (n-1)-dimensional
Zero Level Set (ZLS) face directly instead of slicing the full n-D polytope.

Why this is better in 12D
--------------------------
The current approach (verify_certificate_new) calls Enumerator_rapid on the
full n-D polytope (V vertices, V can be 500–5000 in 12D) — O(V²) per cut.
This quickly hits the max_slicer_verts guard and returns INCONCLUSIVE.

Here we cut the ZLS face (x_stars, K vertices, typically K ≪ V) using
slice_face_wide — O(K²) per cut with the correct (n-2)-shared-bit adjacency
condition for an (n-1)-D face.  No Enumerator_rapid, no sub-cell vertex
enumeration, no re-evaluation of B at sub-cell vertices.

Refinement strategy
-------------------
When _two_step_label is INCONCLUSIVE:

  1. Find farthest pair of ZLS points; define n_refs-1 equally-spaced
     parallel cutting hyperplanes along that direction.

  2. Apply cuts sequentially, peeling off n_refs slabs from the ZLS face via
     slice_face_wide (adjacency: n-2 shared bitmask bits).

  3. Run _two_step_label on each slab.  The Taylor radius r_i shrinks with
     each slab, tightening the bound.

  4. Queue INCONCLUSIVE slabs and repeat up to max_depth levels.

Public API (identical to verify_certificate_new)
------------------------------------------------
    summary = verify_barrier(
        BC, sv, layer_W, layer_b, W_out,
        boundary_H, boundary_b,
        barrier_model, dynamics_name="quadrotor"
    )
"""

from __future__ import annotations

import time
import warnings
import numpy as np
import torch
from collections import deque
from typing import List, Tuple, Optional
from .bitwise_utils import generate_mask, slice_polytope_with_hyperplane

try:
    from .verify_certificate_new import (
        Label, CellResult, VerificationSummary, Counterexample,
        DynamicsEvaluator, _two_step_label, _find_farthest_pair_nb,
    )
    from .hessian_bound import HessianBounder, compute_local_gradient
    from .Dynamics import load_dynamics
    from .bitwise_utils import (
        get_cell_hyperplanes_input_space,
        generate_mask_wide,
        slice_polytope_wide,
        slice_face_wide,
        Enumerator_rapid_face,
    )
except ImportError:
    from verify_certificate_new import (
        Label, CellResult, VerificationSummary, Counterexample,
        DynamicsEvaluator, _two_step_label, _find_farthest_pair_nb,
    )
    from hessian_bound import HessianBounder, compute_local_gradient
    from Dynamics import load_dynamics
    from bitwise_utils import (
        get_cell_hyperplanes_input_space,
        generate_mask_wide,
        slice_polytope_wide,
        slice_face_wide,
        Enumerator_rapid_face,
    )
def _get_zero_level_set_crossings(
    vertices      : np.ndarray,
    sv_i          : np.ndarray,
    B_vals        : np.ndarray,
    layer_W       : list,
    layer_b       : list,
    boundary_H    : np.ndarray,
    boundary_b    : np.ndarray,
    use_wide      : bool,
    barrier_model = None,
    model_dtype   = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find exact zero level set crossings using bitmask slicing.

    Returns
    -------
    x_stars      : (K, n) crossing points with B(x*)≈0
    x_masks      : (K,)   bitmasks for x_stars
    verts_B_neg  : vertices of the B≤0 sub-polytope (x_stars included)
    verts_B_pos  : vertices of the B≥0 sub-polytope (x_stars included)

    verts_B_neg / verts_B_pos are the two halves produced by slicing the
    cell at B=0; the caller can use these to construct a ZLS-facet sub-cell
    for adaptive refinement without recomputing crossings.
    """
    import warnings

    n                = vertices.shape[1]
    zls_residual_tol = 1e-3
    empty_masks      = np.zeros(0, dtype=np.uint64)
    empty_verts      = np.zeros((0, n), dtype=np.float64)

    if B_vals.min() >= -1e-9 or B_vals.max() <= 1e-9:
        return empty_verts, empty_masks, empty_verts, empty_verts

    H_all, b_all = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b
    )

    verts_f64 = vertices.astype(np.float64)
    H_f64     = H_all.astype(np.float64)
    b_f64     = b_all.astype(np.float64)
    B_f64     = B_vals.astype(np.float64)

    h_idx = len(H_all)

    if use_wide:
        masks = generate_mask_wide(verts_f64, H_f64, b_f64)
        polytopes, mask_lists, created_verts = slice_polytope_wide(
            verts_f64, B_f64, masks, h_idx, n
        )
    else:
        masks = generate_mask(verts_f64, H_f64, b_f64)
        polytopes, mask_lists, created_verts = slice_polytope_with_hyperplane(
            verts_f64, B_f64, masks, h_idx, n
        )

    if len(created_verts) == 0:
        return empty_verts, empty_masks, empty_verts, empty_verts

    created_verts = np.array(created_verts)

    # Keep the two sub-polytope vertex arrays before any filtering
    verts_B_neg = np.asarray(polytopes[0], dtype=np.float64)  # B ≤ 0 side
    verts_B_pos = np.asarray(polytopes[1], dtype=np.float64)  # B ≥ 0 side

    n_orig_in     = len(polytopes[0]) - len(created_verts)
    created_masks = np.array(mask_lists[0][n_orig_in:])

    if barrier_model is not None and model_dtype is not None:
        np_dtype = np.float64 if model_dtype == torch.float64 else np.float32
        with torch.no_grad():
            B_cross = barrier_model(
                torch.tensor(created_verts.astype(np_dtype), dtype=model_dtype)
            ).detach().numpy().ravel().astype(np.float64)

        residuals    = np.abs(B_cross)
        max_residual = float(residuals.max())

        if max_residual > zls_residual_tol:
            n_spurious = int((residuals > zls_residual_tol).sum())
            warnings.warn(
                f"_get_zero_level_set_crossings: {n_spurious}/{len(created_verts)} "
                f"crossing point(s) have |B(x*)| > {zls_residual_tol:.0e} "
                f"(max residual = {max_residual:.2e}). "
                f"Filtering spurious crossings."
            )

        valid         = residuals <= zls_residual_tol
        created_verts = created_verts[valid]
        created_masks = created_masks[valid]

        if len(created_verts) == 0:
            return empty_verts, empty_masks, empty_verts, empty_verts

    return created_verts, created_masks, verts_B_neg, verts_B_pos


# ═══════════════════════════════════════════════════════════════════════════
# ZLS face extraction — always uses wide masks
# ═══════════════════════════════════════════════════════════════════════════

def _get_zls_face(
    vertices      : np.ndarray,
    B_vals        : np.ndarray,
    H_all         : np.ndarray,
    b_all         : np.ndarray,
    barrier_model = None,
    model_dtype   = None,
) -> Tuple[np.ndarray, int]:
    """
    Find ZLS crossing points for the face slicer.

    Returns
    -------
    x_stars : (K, n) crossing points with B(x*) ≈ 0
    h_idx   : bit index assigned to the B=0 cutting hyperplane (= len(H_all))
              Refinement cuts should start at h_idx+1.
    """
    n = vertices.shape[1]
    zls_tol = 1e-3
    empty_v = np.zeros((0, n), dtype=np.float64)

    if B_vals.min() >= -1e-9 or B_vals.max() <= 1e-9:
        return empty_v, -1

    h_idx = len(H_all)

    verts_f64 = vertices.astype(np.float64)
    H_f64     = H_all.astype(np.float64)
    b_f64     = b_all.astype(np.float64)
    B_f64     = B_vals.astype(np.float64)

    masks     = generate_mask_wide(verts_f64, H_f64, b_f64)

    # Ensure masks are wide enough to hold bit h_idx
    min_words = (h_idx + 64) // 64
    if masks.shape[1] < min_words:
        pad   = np.zeros((len(masks), min_words - masks.shape[1]), dtype=np.uint64)
        masks = np.hstack([masks, pad])

    polytopes, _, created_verts = slice_polytope_wide(
        verts_f64, B_f64, masks, h_idx, n
    )

    if len(created_verts) == 0:
        return empty_v, h_idx

    created_verts = np.array(created_verts, dtype=np.float64)

    if barrier_model is not None and model_dtype is not None:
        np_dtype = np.float64 if model_dtype == torch.float64 else np.float32
        with torch.no_grad():
            B_cross = barrier_model(
                torch.tensor(created_verts.astype(np_dtype), dtype=model_dtype)
            ).detach().numpy().ravel().astype(np.float64)
        residuals = np.abs(B_cross)
        max_res   = float(residuals.max())
        if max_res > zls_tol:
            n_spur = int((residuals > zls_tol).sum())
            warnings.warn(
                f"_get_zls_face: {n_spur}/{len(created_verts)} crossing(s) "
                f"have |B|>{zls_tol:.0e} (max={max_res:.2e}) — filtering."
            )
        valid         = residuals <= zls_tol
        created_verts = created_verts[valid]

    if len(created_verts) == 0:
        return empty_v, h_idx

    return created_verts, h_idx


# ═══════════════════════════════════════════════════════════════════════════
# ZLS face refinement
# ═══════════════════════════════════════════════════════════════════════════

def _refine_zls_face(
    x_stars              : np.ndarray,
    H_all                : np.ndarray,
    b_all                : np.ndarray,
    worst_cond           : float,
    remainder            : float,
    M_i_parent           : float,
    p_i                  : np.ndarray,
    barrier_model,
    dyn                  : DynamicsEvaluator,
    hb,
    cell_idx             : int,
    continuous_time      : bool,
    n                    : int,
    h_idx                : int,
    model_dtype,
    max_depth            : int   = 20,
    n_refs               : int   = 10,
    refinement_timeout_s : float = 120.0,
    use_wide             : bool  = False,
) -> Tuple[Label, float, float, float, Optional[Counterexample]]:
    """
    Iterative ZLS face refinement using slice_face_wide.

    Delegates all cutting to Enumerator_rapid_face (same design as
    Enumerator_rapid) which recomputes masks fresh per face per cut.
    Queue items accumulate the boundary hyperplanes of all cuts so far,
    mirroring the bH_child pattern in verify_certificate_new.

    Queue items: (x_stars, worst_cond, remainder, depth, bH, bb)
      bH/bb : accumulated boundary = H_all + all cuts that carved this sub-face
    """
    from .bitwise_utils import finding_side
    H_all,b_all=finding_side(H_all, x_stars, b_all)
    queue = deque()
    queue.append((x_stars, worst_cond, remainder, max_depth,
                  H_all.astype(np.float64), b_all.astype(np.float64)))

    worst_M            = 0.0
    worst_r            = 0.0
    worst_rem          = 0.0
    any_inconclusive   = False
    n_safe_sub         = 0
    n_inconclusive_sub = 0
    n_queued           = 0
    leaf_report : List[Tuple[int, float, float, float]] = []
    deadline           = time.perf_counter() + refinement_timeout_s

    while queue:
        if time.perf_counter() > deadline:
            print(f"  [cell {cell_idx}] face-refine timeout "
                  f"({refinement_timeout_s:.1f}s), {len(queue)} items remain, "
                  f"safe_sub={n_safe_sub}, queued_total={n_queued}")
            any_inconclusive = True
            break

        xs, wc, rem, depth, bH, bb = queue.popleft()

        if depth == 0 or len(xs) < 2:
            any_inconclusive   = True
            n_inconclusive_sub += 1
            if len(xs) >= 2:
                c_leaf = xs.mean(axis=0)
                r_leaf = float(np.linalg.norm(xs - c_leaf, axis=1).max())
                leaf_report.append((len(xs), r_leaf, rem, wc, 'depth_limit'))
            continue

        # ── find farthest pair ────────────────────────────────────────────────
        idx_a, idx_b, max_dist = _find_farthest_pair_nb(
            np.ascontiguousarray(xs, dtype=np.float64)
        )
        if max_dist < 1e-12:
            any_inconclusive   = True
            n_inconclusive_sub += 1
            leaf_report.append((len(xs), 0.0, rem, wc, 'degenerate'))
            continue

        normal    = (xs[idx_b] - xs[idx_a]) / max_dist
        base_proj = float(xs[idx_a] @ normal)
        n_refs    = 5 # cap n_refs for tiny faces
        step_proj = float((xs[idx_b] - xs[idx_a]) @ normal) / n_refs
        cut_vals  = [base_proj + k * step_proj for k in range(1, n_refs)]

        H_refine = np.tile(normal, (n_refs - 1, 1)).astype(np.float64)
        b_refine = np.array([-c for c in cut_vals],  dtype=np.float64)

        # ── delegate to Enumerator_rapid_face ────────────────────────────────
        # Each input face carries its own (verts, bh, bb); each output face
        # gets a dedicated (bh, bb) from finding_side — no shared accumulator.
        sub_faces = Enumerator_rapid_face(
            H_refine, b_refine, [(xs, bH, bb)], n,
            mask_tolerance=1e-5,
            use_wide=use_wide,
        )
        # sub_faces: list of (face_verts, face_bh, face_bb)

        if len(sub_faces) <= 1:
            any_inconclusive   = True
            n_inconclusive_sub += 1
            _c = xs.mean(axis=0); _r = float(np.linalg.norm(xs - _c, axis=1).max())
            leaf_report.append((len(xs), _r, rem, wc, 'no_split'))
            continue

        # ── two-step check on each sub-face ──────────────────────────────────
        for face_v, face_bh, face_bb in sub_faces:
            if len(face_v) < 2:
                any_inconclusive   = True
                n_inconclusive_sub += 1
                leaf_report.append((len(face_v), 0.0, rem, wc, 'tiny_slab'))
                continue

            label, M_i, r_i, rem_i, wc_i, ce = _two_step_label(
                face_v, barrier_model, dyn, hb, p_i, cell_idx, continuous_time,
                M_i_override=None,
            )
            worst_M   = max(worst_M,   M_i)
            worst_r   = max(worst_r,   r_i)
            worst_rem = max(worst_rem, rem_i)

            if label == Label.UNSAFE:
                return Label.UNSAFE, M_i, r_i, rem_i, ce

            if label == Label.INCONCLUSIVE:
                queue.append((face_v, wc_i, rem_i, depth - 1, face_bh, face_bb))
                n_queued += 1
            else:
                n_safe_sub += 1

    final = Label.INCONCLUSIVE if any_inconclusive else Label.SAFE_REFINEMENT
    print(f"  [cell {cell_idx}] face-refine done: "
          f"safe_sub={n_safe_sub}  inconclusive_sub={n_inconclusive_sub}  "
          f"total_queued={n_queued}  "
          f"worst_rem={worst_rem:.3e}  worst_r={worst_r:.3e}  -> {final.value}")

    if leaf_report:
        leaf_report.sort(key=lambda t: t[2], reverse=True)  # sort by remainder desc
        print(f"  [cell {cell_idx}] remaining inconclusive sub-faces ({len(leaf_report)}):")
        print(f"    {'#':>4}  {'k_pts':>6}  {'r_i':>8}  {'remainder':>10}  {'worst_cond':>11}  reason")
        for idx, (k, r, rem_l, cond_l, reason) in enumerate(leaf_report, 1):
            print(f"    {idx:>4}  {k:>6}  {r:>8.4f}  {rem_l:>10.4e}  {cond_l:>11.4e}  {reason}")

    if any_inconclusive:
        return Label.INCONCLUSIVE, worst_M, worst_r, worst_rem, None
    return Label.SAFE_REFINEMENT, worst_M, worst_r, worst_rem, None


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def verify_barrier(
    BC                  : list,
    sv                  : np.ndarray,
    layer_W             : list,
    layer_b             : list,
    W_out               : np.ndarray,
    boundary_H          : np.ndarray,
    boundary_b          : np.ndarray,
    barrier_model,
    dynamics_name       : str,
    continuous_time     : bool  = True,
    early_exit          : bool  = True,
    refinement_max_depth: int   = 20,
    refinement_timeout_s: float = float('inf'),
    TH                  : list  = None,
    n_refs              : int   = 10,
    cell_timeout_s      : Optional[float] = None,
) -> VerificationSummary:
    """
    Formally verify a barrier certificate using ZLS face refinement.

    Identical interface to verify_certificate_new.verify_barrier.
    The only behavioural difference: INCONCLUSIVE cells are refined by
    cutting the (n-1)-D ZLS face (slice_face_wide, O(K²)) instead of the
    full n-D polytope (Enumerator_rapid, O(V²)).

    Parameters
    ----------
    n_refs : int
        Number of slabs per refinement level (n_refs-1 cuts per iteration).
    """
    import sympy as sp

    symbols, f_sym = load_dynamics(dynamics_name)
    dyn            = DynamicsEvaluator(symbols, f_sym)

    _is_linear = all(
        sp.diff(fi, xj, xk) == 0
        for fi in f_sym for xj in symbols for xk in symbols
    )
    if _is_linear:
        print("  Linear dynamics detected — skipping HessianBounder (M_i=0).")
        hb = None
    else:
        hb = HessianBounder(symbols, f_sym)

    model_dtype   = next(barrier_model.parameters()).dtype
    n_dim         = layer_W[0].shape[1]
    total_neurons = sum(W.shape[0] for W in layer_W)
    use_wide      = (total_neurons + len(boundary_H) > 64)

    if TH is None:
        TH = list(boundary_b[:n_dim])

    summary = VerificationSummary(mode="barrier_face", dynamics_name=dynamics_name)
    _eps    = 1e-6

    print(f"\nBarrier verification (ZLS face refinement): {len(BC)} cells, "
          f"dynamics='{dynamics_name}', "
          f"mode={'continuous' if continuous_time else 'discrete'}-time, "
          f"use_wide={use_wide}, max_depth={refinement_max_depth}, n_refs={n_refs}")
    t0 = time.perf_counter()

    for i in range(len(BC)):
        t_cell = time.perf_counter()

        vertices = np.asarray(BC[i], dtype=float)
        BC[i]    = None
        sv_i     = sv[i].ravel()
        p_i      = compute_local_gradient(sv_i, layer_W, W_out)

        np_dtype = np.float64 if model_dtype == torch.float64 else np.float32
        with torch.no_grad():
            B_vals = barrier_model(
                torch.tensor(vertices.astype(np_dtype), dtype=model_dtype)
            ).numpy().ravel()

        # Fast path: affine p_i·f(x) for linear dynamics
        if _is_linear and continuous_time:
            F_verts    = dyn.eval_batch(vertices)
            cond_verts = F_verts @ p_i
            if float(cond_verts.max()) <= _eps:
                summary.results.append(CellResult(
                    label=Label.SAFE_TAYLOR, cell_idx=i,
                    M_i=0.0, r_i=0.0, remainder=0.0,
                    max_condition=float(cond_verts.max()),
                    time_s=time.perf_counter() - t_cell,
                ))
                summary.n_safe_taylor += 1
                continue

        # ── ZLS face extraction ───────────────────────────────────────────────
        H_all, b_all = get_cell_hyperplanes_input_space(
            sv_i, layer_W, layer_b, boundary_H, boundary_b
        )
        x_stars, h_idx = _get_zls_face(
            vertices, B_vals, H_all, b_all,
            barrier_model=barrier_model, model_dtype=model_dtype,
        )
        if len(x_stars) == 0:
            summary.results.append(CellResult(
                label=Label.SAFE_TAYLOR, cell_idx=i,
                M_i=0.0, r_i=0.0, remainder=0.0, max_condition=-1.0,
                time_s=time.perf_counter() - t_cell,
            ))
            summary.n_safe_taylor += 1
            continue

        if cell_timeout_s is not None and (time.perf_counter() - t_cell) > cell_timeout_s:
            print(f"  [cell {i}] per-cell timeout after ZLS — INCONCLUSIVE")
            summary.n_inconclusive += 1
            summary.results.append(CellResult(
                label=Label.INCONCLUSIVE, cell_idx=i,
                M_i=0.0, r_i=0.0, remainder=0.0, max_condition=0.0,
                time_s=time.perf_counter() - t_cell,
            ))
            continue

        # ── two-step label on full ZLS face ───────────────────────────────────
        label, M_i, r_i, remainder, worst_cond, ce = _two_step_label(
            x_stars, barrier_model, dyn, hb, p_i, i, continuous_time
        )

        # ── ZLS face refinement if inconclusive ───────────────────────────────
        if label == Label.INCONCLUSIVE:
            _budget = refinement_timeout_s
            if cell_timeout_s is not None:
                _budget = min(_budget,
                              cell_timeout_s - (time.perf_counter() - t_cell))

            label, M_i, r_i, remainder, ce = _refine_zls_face(
                x_stars, H_all, b_all,
                worst_cond, remainder, M_i,
                p_i, barrier_model, dyn, hb,
                i, continuous_time,
                n_dim, h_idx, model_dtype,
                max_depth=refinement_max_depth,
                n_refs=n_refs,
                refinement_timeout_s=_budget,
                use_wide=use_wide,
            )

        if ce is not None and summary.counterexample is None:
            summary.counterexample = ce

        summary.results.append(CellResult(
            label=label, cell_idx=i,
            M_i=M_i, r_i=r_i, remainder=remainder, max_condition=0.0,
            time_s=time.perf_counter() - t_cell,
        ))

        if label == Label.SAFE_TAYLOR:
            summary.n_safe_taylor += 1
        elif label == Label.SAFE_REFINEMENT:
            summary.n_safe_refinement += 1
        elif label == Label.UNSAFE:
            summary.n_unsafe += 1
            if early_exit:
                break
        else:
            summary.n_inconclusive += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(BC):
            print(f"  [{i+1}/{len(BC)}] "
                  f"safe_taylor={summary.n_safe_taylor} "
                  f"safe_ref={summary.n_safe_refinement} "
                  f"unsafe={summary.n_unsafe} "
                  f"inconclusive={summary.n_inconclusive}")

    summary.runtime_s = time.perf_counter() - t0
    summary.print_summary()
    return summary
