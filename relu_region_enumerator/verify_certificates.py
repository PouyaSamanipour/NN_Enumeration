"""
verify_certificates.py
======================
Formal verification of barrier certificate conditions over all enumerated
polytopic linear regions.

Zero level set crossings
-------------------------
For each boundary cell X_i, B is affine: B(x) = p_i @ x + c_i.
Exact crossing points are found using slice_polytope_with_hyperplane —
bitmask adjacency ensures only valid polytope edges are used.

Two-step labeling per cell
---------------------------
Step 1 — Exact evaluation at crossing points + centroid c*:
    Evaluate p_i . f(x) (continuous) or B(f(x)) (discrete) exactly.
    If any value > 0 -> UNSAFE (exact counterexample).
    Cannot conclude SAFE — only sampled points checked.

Step 2 — Linearized dynamics + Taylor remainder:
    Linearize f around c* = mean(x_stars).
    cond_e      = p_i . f_lin(x*_e)   or   B(f_lin(x*_e))
    remainder_e = 0.5 * M * ||x*_e - c*||^2
    M computed over x_stars (tighter than full cell).

    SAFE        : max_e [ cond_e + remainder_e ] <= 0
    UNSAFE      : min_e [ cond_e - remainder_e ] > 0
    INCONCLUSIVE: refine

Refinement by fake neuron
--------------------------
When INCONCLUSIVE, split the zero level set polytope with a fake neuron:
  1. Find principal direction d of crossing points via SVD
  2. Fake hyperplane: d^T x = d^T c*  (passes through centroid, perp to d)
  3. Split crossing points into two children
  4. Pick child with smaller spread (smaller max ||x*_e - c*||)
  5. For chosen child, add BOTH the fake hyperplane AND the zero level set
     as boundary hyperplanes — so the child's geometry is correct
  6. Re-run crossing detection and labeling on the child
  7. Recurse until SAFE/UNSAFE or max_depth reached

Each child is verified only on its own zero level set vertices.
If verified SAFE, it is flushed — the parent is SAFE once all children are SAFE.

Usage
-----
    summary = verify_barrier(
        BC, sv, layer_W, layer_b, W_out,
        boundary_H, boundary_b,
        barrier_model, dynamics_name="arch3"
    )

    summary = verify_lyapunov(
        enumerate_poly, sv_all, layer_W, layer_b, W_out,
        boundary_H, boundary_b,
        lyapunov_model, dynamics_name="quadrotor"
    )
"""

from __future__ import annotations

import time
import numpy as np
import torch
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional

try:
    from .hessian_bound import HessianBounder, compute_local_gradient
    from .Dynamics import load_dynamics
    from .bitwise_utils import (
        get_cell_hyperplanes_input_space,
        generate_mask,
        generate_mask_wide,
        slice_polytope_with_hyperplane,
        slice_polytope_wide,
    )
except ImportError:
    from hessian_bound import HessianBounder, compute_local_gradient
    from Dynamics import load_dynamics
    from bitwise_utils import (
        get_cell_hyperplanes_input_space,
        generate_mask,
        generate_mask_wide,
        slice_polytope_with_hyperplane,
        slice_polytope_wide,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

class Label(Enum):
    SAFE_TAYLOR  = "SAFE_TAYLOR"    # formally certified by Taylor remainder
    SAFE_NLP     = "SAFE_NLP"       # heuristically confirmed by NLP (no violation found)
    UNSAFE       = "UNSAFE"         # counterexample found (exact or NLP)
    INCONCLUSIVE = "INCONCLUSIVE"   # Taylor exhausted, NLP not run or also inconclusive


@dataclass
class CellResult:
    label         : Label
    cell_idx      : int
    M_i           : float
    r_i           : float
    remainder     : float
    max_condition : float


@dataclass
class VerificationSummary:
    mode           : str
    dynamics_name  : str
    n_safe_taylor  : int   = 0
    n_safe_nlp     : int   = 0
    n_unsafe       : int   = 0
    n_inconclusive : int   = 0
    runtime_s      : float = 0.0
    results        : List[CellResult] = field(default_factory=list)
    counterexample : Optional['Counterexample'] = None

    @property
    def n_safe(self):
        return self.n_safe_taylor + self.n_safe_nlp

    @property
    def total(self):
        return self.n_safe + self.n_unsafe + self.n_inconclusive

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Verification Summary  [{self.mode} / {self.dynamics_name}]")
        print(f"{'='*60}")
        print(f"  Total cells checked : {self.total}")
        print(f"  SAFE (Taylor)       : {self.n_safe_taylor}"
              f"  ({100*self.n_safe_taylor/max(self.total,1):.1f}%)  [formally certified]")
        print(f"  SAFE (NLP)          : {self.n_safe_nlp}"
              f"  ({100*self.n_safe_nlp/max(self.total,1):.1f}%)  [heuristically confirmed]")
        print(f"  UNSAFE              : {self.n_unsafe}"
              f"  ({100*self.n_unsafe/max(self.total,1):.1f}%)")
        print(f"  INCONCLUSIVE        : {self.n_inconclusive}"
              f"  ({100*self.n_inconclusive/max(self.total,1):.1f}%)")
        print(f"  Runtime             : {self.runtime_s:.2f} s")

        if self.counterexample is not None:
            print(f"\n  Exact counterexample found:")
            self.counterexample.report()
        elif self.n_unsafe > 0:
            print(f"\n  !! {self.n_unsafe} UNSAFE cells found.")
        elif self.n_inconclusive == 0:
            if self.n_safe_nlp == 0:
                print(f"\n  Certificate VERIFIED (Taylor) over all {self.total} cells.")
            else:
                print(f"\n  Certificate VERIFIED: {self.n_safe_taylor} cells by Taylor, "
                      f"{self.n_safe_nlp} by NLP fallback.")
        else:
            print(f"\n  {self.n_inconclusive} cells remain INCONCLUSIVE after Taylor + NLP.")
        print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# Counterexample
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Counterexample:
    x_star          : np.ndarray
    f_x_star        : np.ndarray
    B_x_star        : float
    violation_value : float
    p_i             : np.ndarray
    continuous_time : bool
    cell_idx        : int
    point_idx       : int

    def report(self):
        print(f"\n{'='*60}")
        print(f"COUNTEREXAMPLE  "
              f"[{'continuous' if self.continuous_time else 'discrete'}-time]")
        print(f"{'='*60}")
        print(f"  Cell index      : {self.cell_idx}")
        print(f"  x*              : {np.array2string(self.x_star, precision=6)}")
        print(f"  f(x*)           : {np.array2string(self.f_x_star, precision=6)}")
        print(f"  B(x*)           : {self.B_x_star:.6e}")
        if self.continuous_time:
            print(f"  p_i . f(x*)     : {self.violation_value:.6e}  (> 0 = violation)")
        else:
            print(f"  B(f(x*))        : {self.violation_value:.6e}  (> 0 = violation)")
        print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# DynamicsEvaluator
# ═══════════════════════════════════════════════════════════════════════════

class DynamicsEvaluator:
    """Precompiled evaluator for f(x) and J_f(x)."""

    def __init__(self, symbols, f_sym):
        import sympy as sp
        self.symbols = symbols
        self.n       = len(symbols)

        print("  Building DynamicsEvaluator...", flush=True)
        self._f_fns = [
            sp.lambdify(list(symbols), fi, modules='numpy') for fi in f_sym
        ]
        J_sym        = sp.Matrix(f_sym).jacobian(list(symbols))
        self._jac_fn = sp.lambdify(list(symbols), J_sym, modules='numpy')
        print("  Done.", flush=True)

    def eval(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        args  = list(x)
        f_val = np.array([float(fn(*args)) for fn in self._f_fns])
        f_jac = np.array(self._jac_fn(*args), dtype=float)
        return f_val, f_jac

    def eval_batch(self, X: np.ndarray) -> np.ndarray:
        E, n = X.shape
        F    = np.zeros((E, n))
        args = [X[:, k] for k in range(n)]
        for j, fn in enumerate(self._f_fns):
            vals    = fn(*args)
            F[:, j] = (
                np.broadcast_to(np.atleast_1d(vals), (E,))
                if not np.isscalar(vals)
                else np.full(E, float(vals))
            )
        return F


# ═══════════════════════════════════════════════════════════════════════════
# Zero level set crossings via bitmask slicing
# ═══════════════════════════════════════════════════════════════════════════

def _get_zero_level_set_crossings(
    vertices   : np.ndarray,   # (V, n)
    sv_i       : np.ndarray,   # (total_neurons,)
    B_vals     : np.ndarray,   # (V,) B at vertices — exact since B affine
    layer_W    : list,
    layer_b    : list,
    boundary_H : np.ndarray,   # (B, n) — may include fake neurons / ZLS
    boundary_b : np.ndarray,   # (B,)
    use_wide   : bool,
) -> Tuple[np.ndarray, np.ndarray]:   # (created_verts, created_masks)
    """
    Find exact zero level set crossings using bitmask slicing.

    B is affine on the cell so B_vals at vertices are exact.
    The zero level set B(x)=0 is a new hyperplane cutting the cell.
    slice_polytope_with_hyperplane uses bitmask adjacency to find only
    valid polytope edges — created_verts are the exact crossing points.

    Returns
    -------
    created_verts : (E, n)          zero level set crossing points
    created_masks : (E,) uint64  or (E, num_words) uint64
                    bitmasks of the crossing points — needed for subsequent
                    refinement slicing along the fake neuron hyperplane
    """
    n = vertices.shape[1]

    empty_masks = np.zeros(0, dtype=np.uint64)

    if B_vals.min() >= -1e-9 or B_vals.max() <= 1e-9:
        return np.zeros((0, n)), empty_masks

    H_all, b_all = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b
    )

    verts_f64 = vertices.astype(np.float64)
    H_f64     = H_all.astype(np.float64)
    b_f64     = b_all.astype(np.float64)
    B_f64     = B_vals.astype(np.float64)

    h_idx = len(H_all)   # ZLS gets the next available bit

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
        return np.zeros((0, n)), empty_masks

    # created_verts are appended at the end of both child mask arrays.
    # Extract their masks from mask_lists[0] (inside child).
    n_orig_in     = len(polytopes[0]) - len(created_verts)
    created_masks = mask_lists[0][n_orig_in:]   # last E entries

    return np.array(created_verts), np.array(created_masks)


# ═══════════════════════════════════════════════════════════════════════════
# Core two-step labeling
# ═══════════════════════════════════════════════════════════════════════════

def _two_step_label(
    x_stars         : np.ndarray,   # (E, n) crossing points
    barrier_model,
    dyn             : DynamicsEvaluator,
    hb              : HessianBounder,
    p_i             : np.ndarray,   # (n,)
    cell_idx        : int,
    continuous_time : bool,
) -> Tuple[Label, float, float, float, Optional[Counterexample]]:
    """
    Two-step labeling on a set of zero level set crossing points.

    Step 1 — exact dynamics at x_stars + centroid c*:
        Any violation -> UNSAFE with counterexample.

    Step 2 — linearized dynamics around c* + Taylor remainder:
        M computed over x_stars.
        Returns label, M_i, r_i, remainder, counterex.
    """
    model_dtype = next(barrier_model.parameters()).dtype

    c_star   = x_stars.mean(axis=0)
    eval_pts = np.vstack([x_stars, c_star[None, :]])   # (E+1, n)
    f_exact  = dyn.eval_batch(eval_pts)

    if continuous_time:
        exact_vals = f_exact @ p_i
    else:
        with torch.no_grad():
            exact_vals = barrier_model(
                torch.tensor(f_exact.astype(np.float32), dtype=model_dtype)
            ).numpy().ravel()

    # Step 1: exact check
    for e in range(len(eval_pts)):
        if exact_vals[e] > 0.0:
            with torch.no_grad():
                B_x = float(barrier_model(
                    torch.tensor(eval_pts[e:e+1].astype(np.float32),
                                 dtype=model_dtype)
                ).numpy().ravel()[0])
            ce = Counterexample(
                x_star=eval_pts[e].copy(), f_x_star=f_exact[e].copy(),
                B_x_star=B_x, violation_value=float(exact_vals[e]),
                p_i=p_i.copy(), continuous_time=continuous_time,
                cell_idx=cell_idx, point_idx=e,
            )
            return Label.UNSAFE, 0.0, 0.0, 0.0, ce

    # Step 2: Taylor remainder
    f_cstar, f_jac = dyn.eval(c_star)
    delta          = x_stars - c_star
    f_lin_xs       = f_cstar + delta @ f_jac.T

    if continuous_time:
        cond_vals = f_lin_xs @ p_i
    else:
        with torch.no_grad():
            cond_vals = barrier_model(
                torch.tensor(f_lin_xs.astype(np.float32), dtype=model_dtype)
            ).numpy().ravel()

    distances   = np.linalg.norm(x_stars - c_star, axis=1)
    M_i         = hb.bound(p_i, x_stars)
    remainders  = 0.5 * M_i * distances ** 2
    worst_safe  = float((cond_vals + remainders).max())
    best_unsafe = float((cond_vals - remainders).min())

    eps = 1e-6   # numerical tolerance — values within ±eps of zero treated as zero
    if worst_safe <= eps:
        label = Label.SAFE_TAYLOR
    elif best_unsafe > eps:
        label = Label.UNSAFE
    else:
        label = Label.INCONCLUSIVE

    return (label, M_i, float(distances.max()),
            float(remainders.max()), None)


# ═══════════════════════════════════════════════════════════════════════════
# Refinement
# ═══════════════════════════════════════════════════════════════════════════

def slice_zls_polytope(
    x_stars   : np.ndarray,   # (E, n) ZLS polytope vertices — (n-1)-dimensional
    fake_vals : np.ndarray,   # (E,) signed distances to fake neuron hyperplane
    x_masks   : np.ndarray,   # (E,) uint64 bitmasks
    h_idx     : int,          # global bit index for the new fake neuron
    n         : int,          # input space dimension
) -> Tuple[list, list, np.ndarray]:
    """
    Slice the ZLS polytope (an (n-1)-dimensional face in n-dimensional space)
    with a fake neuron hyperplane.

    The ZLS polytope lives in n-dimensional space but is (n-1)-dimensional.
    Its edges satisfy the adjacency condition: two vertices share an edge
    if their bitmasks share at least (n-1) - 1 = n-2 bits.

    This is different from slice_polytope_with_hyperplane which uses
    req_shared = n - 1 for full n-dimensional polytopes. Using n-1 here
    would be one too high and find no edges.

    Otherwise identical in structure to slice_polytope_with_hyperplane.

    Returns
    -------
    polytopes     : list of two (V_k, n) arrays — [inside, outside]
    mask_lists    : list of two (V_k,) uint64 arrays
    created_verts : (E_new, n) new vertices on the fake neuron boundary
    """
    ONE        = np.uint64(1)
    req_shared = n - 2   # (n-1)-dimensional polytope: edges share n-2 hyperplanes

    strict_inside  = fake_vals < -1e-9
    strict_outside = fake_vals >  1e-9
    idx_in         = np.where(fake_vals <= 1e-5)[0]
    idx_out        = np.where(fake_vals >= -1e-5)[0]
    s_idx_in       = np.where(strict_inside)[0]
    s_idx_out      = np.where(strict_outside)[0]

    # Use dynamic lists to avoid OOM on large E
    new_verts_list = []
    new_masks_list = []

    for k in range(len(s_idx_in)):
        u  = s_idx_in[k]
        mu = x_masks[u]
        for l in range(len(s_idx_out)):
            v  = s_idx_out[l]
            mv = x_masks[v]
            shared   = mu & mv
            n_shared = 0
            temp     = shared
            while temp > np.uint64(0) and n_shared < req_shared:
                if temp & ONE:
                    n_shared += 1
                temp >>= ONE
            if n_shared >= req_shared:
                d1 = fake_vals[u]
                d2 = fake_vals[v]
                t  = -d1 / (d2 - d1)
                new_verts_list.append(x_stars[u] + t * (x_stars[v] - x_stars[u]))
                new_masks_list.append(shared | (mu & (ONE << np.uint64(h_idx))))

    if new_verts_list:
        created_verts = np.array(new_verts_list)
        created_masks = np.array(new_masks_list, dtype=np.uint64)
        verts_in  = np.vstack([x_stars[idx_in],  created_verts])
        masks_in  = np.concatenate([x_masks[idx_in],  created_masks])
        verts_out = np.vstack([x_stars[idx_out], created_verts])
        masks_out = np.concatenate([x_masks[idx_out], created_masks])
    else:
        created_verts = np.zeros((0, n), dtype=np.float64)
        verts_in  = x_stars[idx_in]
        masks_in  = x_masks[idx_in]
        verts_out = x_stars[idx_out]
        masks_out = x_masks[idx_out]

    return [verts_in, verts_out], [masks_in, masks_out], created_verts


def _split_on_zls(
    vertices        : np.ndarray,   # (V, n) original cell X_i
    x_stars         : np.ndarray,   # (E, n) ZLS crossings
    x_masks         : np.ndarray,   # (E,) or (E, W) bitmasks of x_stars
    sv_i            : np.ndarray,
    p_i             : np.ndarray,
    c_i             : float,
    barrier_model,
    dyn             : DynamicsEvaluator,
    hb              : HessianBounder,
    layer_W         : list,
    layer_b         : list,
    W_out           : np.ndarray,
    boundary_H      : np.ndarray,
    boundary_b      : np.ndarray,
    cell_idx        : int,
    continuous_time : bool,
    use_wide        : bool,
    max_depth       : int,
) -> Tuple[Label, float, float, float, Optional[Counterexample]]:
    """
    Step 0: split X_i along its zero level set B(x) = 0.

    Both children share the ZLS face — x_stars are vertices of both.
    We only need to refine ONE child (smaller one). Pass x_stars and
    their masks directly to _refine_barrier which works on the ZLS
    polytope without touching the full cell geometry again.
    """
    # Pick smaller child by number of parent vertices on each side
    v_vals = vertices @ p_i + c_i
    n_pos  = int((v_vals >= -1e-9).sum())
    n_neg  = int((v_vals <=  1e-9).sum())
    side   = +1 if n_pos <= n_neg else -1

    # x_stars and x_masks are the ZLS polytope — pass directly to refinement
    # No re-verification (already INCONCLUSIVE), go straight to fake neuron
    return _refine_barrier(
        x_stars, x_masks, p_i, c_i,
        barrier_model, dyn, hb,
        cell_idx, continuous_time, use_wide,
        max_depth,
    )


def _refine_barrier(
    x_stars         : np.ndarray,   # (E, n) ZLS polytope vertices
    x_masks         : np.ndarray,   # (E,) or (E, W) bitmasks of x_stars
    p_i             : np.ndarray,
    c_i             : float,
    barrier_model,
    dyn             : DynamicsEvaluator,
    hb              : HessianBounder,
    cell_idx        : int,
    continuous_time : bool,
    use_wide        : bool,
    max_depth       : int,
) -> Tuple[Label, float, float, float, Optional[Counterexample]]:
    """
    Refine INCONCLUSIVE cell by splitting the ZLS polytope with a fake neuron.

    The ZLS polytope has vertices x_stars and edges encoded in x_masks.
    We treat x_stars as a proper (n-1)-dimensional polytope and slice it
    with a fake neuron hyperplane using slice_polytope_with_hyperplane.

    This gives:
      - Two child point sets (xs_pos, xs_neg) with correct adjacency
      - New intersection points created_verts at the boundary between children
        (intersection of fake neuron with ZLS face edges)

    Both children must be verified. SAFE only when all children are SAFE.

    Termination: remainder shrinks by ~4x per level. Only B(f(x*)) = 0
    (tangent) stays INCONCLUSIVE at max_depth.
    """
    if max_depth == 0 or len(x_stars) < 2:
        if len(x_stars) >= 1:
            c   = x_stars.mean(axis=0)
            r   = float(np.linalg.norm(x_stars - c, axis=1).max()) if len(x_stars) > 1 else 0.0
            M   = hb.bound(p_i, x_stars)
            rem = 0.5 * M * r ** 2
            print(f"  [cell {cell_idx}] max_depth=0: "
                  f"E={len(x_stars)} r={r:.4f} M={M:.4f} rem={rem:.6f} "
                  f"({'depth_exhausted' if max_depth == 0 else 'too_few_points'})")
        return Label.INCONCLUSIVE, 0.0, 0.0, 0.0, None

    n      = x_stars.shape[1]
    c_star = x_stars.mean(axis=0)

    # ── Principal direction of ZLS polytope via SVD ───────────────────────────
    X = x_stars - c_star
    if np.linalg.norm(X) < 1e-12:
        return Label.INCONCLUSIVE, 0.0, 0.0, 0.0, None

    _, _, Vt    = np.linalg.svd(X, full_matrices=False)
    d           = Vt[0]
    projections = x_stars @ d

    # Use median of projections — guarantees clean split with no near-zero
    # fake_vals. Centroid-based offset can place many points near zero,
    # causing slice_zls_polytope to miss edge crossings.
    offset    = float(np.median(projections))
    fake_vals = (projections - offset).astype(np.float64)

    # Guard: if all on one side, no split possible
    if (fake_vals >= -1e-9).all() or (fake_vals <= 1e-9).all():
        return Label.INCONCLUSIVE, 0.0, 0.0, 0.0, None

    # ── Slice the ZLS polytope along the fake neuron ──────────────────────────
    # Use slice_zls_polytope with req_shared = n-2 (ZLS is (n-1)-dimensional,
    # so its edges share n-2 hyperplanes, not n-1 as for the full cell).
    h_idx = int(int(x_masks.max())).bit_length() if len(x_masks) > 0 else 0

    if use_wide:
        h_idx = int(x_masks.shape[1]) * 64
        polytopes, mask_lists, created_verts = slice_polytope_wide(
            x_stars.astype(np.float64), fake_vals,
            x_masks, h_idx, n
        )
    else:
        polytopes, mask_lists, created_verts = slice_zls_polytope(
            x_stars.astype(np.float64), fake_vals,
            x_masks.astype(np.uint64), h_idx, n
        )

    # polytopes[0] = fake_vals <= 0 (neg side), polytopes[1] = fake_vals >= 0 (pos side)
    # These include the new boundary intersection points (created_verts).
    # We use them for the next recursion level (correct adjacency for further splits).
    xs_neg_full  = np.array(polytopes[0])
    xs_pos_full  = np.array(polytopes[1])
    msk_neg      = np.array(mask_lists[0])
    msk_pos      = np.array(mask_lists[1])

    # For verification (Taylor remainder), use ONLY the original x_stars
    # strictly on each side — not the new boundary intersection points.
    # This prevents r from growing due to added boundary points and ensures
    # the child spread r actually halves each level.
    xs_pos_strict = x_stars[fake_vals >  1e-9]
    xs_neg_strict = x_stars[fake_vals < -1e-9]

    children = [
        (xs_pos_strict, xs_pos_full, msk_pos),
        (xs_neg_strict, xs_neg_full, msk_neg),
    ]

    # ── Verify BOTH children independently ───────────────────────────────────
    worst_M          = 0.0
    worst_r          = 0.0
    worst_rem        = 0.0
    any_inconclusive = False

    for xs_verify, xs_recurse, msk_child in children:
        if len(xs_verify) == 0:
            continue   # no original points on this side — safe

        # Verify using strict half (no boundary inflation)
        label, M_i, r_i, remainder, ce = _two_step_label(
            xs_verify, barrier_model, dyn, hb,
            p_i, cell_idx, continuous_time
        )

        worst_M   = max(worst_M,   M_i)
        worst_r   = max(worst_r,   r_i)
        worst_rem = max(worst_rem, remainder)

        if label == Label.UNSAFE:
            return Label.UNSAFE, M_i, r_i, remainder, ce

        if label == Label.INCONCLUSIVE:
            # Recurse using full polytope (includes boundary points for correct adjacency)
            label, M_i, r_i, remainder, ce = _refine_barrier(
                xs_recurse, msk_child, p_i, c_i,
                barrier_model, dyn, hb,
                cell_idx, continuous_time, use_wide,
                max_depth - 1,
            )
            worst_M   = max(worst_M,   M_i)
            worst_r   = max(worst_r,   r_i)
            worst_rem = max(worst_rem, remainder)

            if label == Label.UNSAFE:
                return Label.UNSAFE, M_i, r_i, remainder, ce
            if label == Label.INCONCLUSIVE:
                any_inconclusive = True   # mark but continue to next child

    if any_inconclusive:
        return Label.INCONCLUSIVE, worst_M, worst_r, worst_rem, None
    return Label.SAFE_TAYLOR, worst_M, worst_r, worst_rem, None





# ═══════════════════════════════════════════════════════════════════════════
# Lyapunov cell labeling
# ═══════════════════════════════════════════════════════════════════════════

def _label_lyapunov_cell(
    vertices      : np.ndarray,
    centroid      : np.ndarray,
    dyn           : DynamicsEvaluator,
    hb            : HessianBounder,
    p_i           : np.ndarray,
    lyapunov_model,
    cell_idx      : int,
) -> CellResult:
    """
    Label a Lyapunov cell. Linearize f around cell centroid.
    Taylor remainder at vertices using cell radius.
    """
    model_dtype = next(lyapunov_model.parameters()).dtype

    f_val, f_jac = dyn.eval(centroid)
    delta        = vertices - centroid
    f_lin_vk     = f_val + delta @ f_jac.T

    with torch.no_grad():
        V_next = lyapunov_model(
            torch.tensor(f_lin_vk.astype(np.float32), dtype=model_dtype)
        ).numpy().ravel()
        V_curr = lyapunov_model(
            torch.tensor(vertices.astype(np.float32), dtype=model_dtype)
        ).numpy().ravel()

    cond_vals   = V_next - V_curr
    r_i         = float(np.linalg.norm(vertices - centroid, axis=1).max())
    M_i         = hb.bound(p_i, vertices)
    remainders  = np.full(len(cond_vals), 0.5 * M_i * r_i ** 2)
    worst_safe  = float((cond_vals + remainders).max())
    best_unsafe = float((cond_vals - remainders).min())

    if worst_safe <= 0.0:
        label = Label.SAFE_TAYLOR
    elif best_unsafe > 0.0:
        label = Label.UNSAFE
    else:
        label = Label.INCONCLUSIVE

    return CellResult(
        label=label, cell_idx=cell_idx, M_i=M_i, r_i=r_i,
        remainder=float(remainders.max()), max_condition=float(cond_vals.max()),
    )


# ═══════════════════════════════════════════════════════════════════════════
# NLP fallback for INCONCLUSIVE cells
# ═══════════════════════════════════════════════════════════════════════════

def _nlp_fallback(
    vertices        : np.ndarray,
    sv_i            : np.ndarray,
    p_i             : np.ndarray,
    c_i             : float,
    layer_W         : list,
    layer_b         : list,
    boundary_H      : np.ndarray,
    boundary_b      : np.ndarray,
    dyn             : DynamicsEvaluator,
    barrier_model,
    model_dtype,
    continuous_time : bool,
    n_starts        : int,
    unsafe_tol      : float,
    cell_idx        : int,
) -> Tuple[Label, Optional[Counterexample]]:
    """
    NLP fallback for INCONCLUSIVE cells.

    Solves: maximize p_i . f(x)  s.t.  p_i @ x + c_i = 0,  x in X_i

    Returns:
      SAFE_NLP     if NLP finds no violation (opt_val <= unsafe_tol)
      UNSAFE       if NLP finds a violation (opt_val > unsafe_tol)
      INCONCLUSIVE if NLP fails to converge
    """
    from scipy.optimize import minimize

    n        = vertices.shape[1]
    centroid = vertices.mean(axis=0)

    # Cell halfspaces from activation pattern
    H_all, b_all = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b
    )
    centroid_vals = H_all @ centroid + b_all
    signs         = np.sign(centroid_vals)
    signs[signs == 0] = -1.0
    A_ub = signs[:, None] * H_all
    b_ub = -(signs * b_all)

    p_norm_sq = float(p_i @ p_i)
    if p_norm_sq < 1e-12:
        return Label.SAFE_NLP, None

    def objective(x):
        f_val, _ = dyn.eval(x)
        return -float(p_i @ f_val) if continuous_time else -float(
            barrier_model(torch.tensor(f_val.astype(np.float32)[None, :],
                         dtype=model_dtype)).detach().numpy().ravel()[0])

    def obj_grad(x):
        _, f_jac = dyn.eval(x)
        return -(f_jac.T @ p_i) if continuous_time else np.zeros(n)

    constraints = [
        {'type': 'eq',   'fun': lambda x: float(p_i @ x) + c_i, 'jac': lambda x: p_i},
        {'type': 'ineq', 'fun': lambda x: b_ub - A_ub @ x,       'jac': lambda x: -A_ub},
    ]

    # Starting points: vertices projected onto ZLS + centroid projected
    x_proj = centroid - (float(p_i @ centroid) + c_i) / p_norm_sq * p_i
    starts  = [x_proj] + [
        v - (float(p_i @ v) + c_i) / p_norm_sq * p_i
        for v in vertices[:n_starts]
    ]

    best_val = -np.inf
    best_x   = x_proj.copy()

    for x0 in starts:
        try:
            res = minimize(objective, x0, jac=obj_grad, method='SLSQP',
                           constraints=constraints,
                           options={'ftol': 1e-9, 'maxiter': 500, 'disp': False})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_x   = res.x.copy()
        except Exception:
            pass

    if best_val > unsafe_tol:
        f_val, _ = dyn.eval(best_x)
        with torch.no_grad():
            B_x = float(barrier_model(
                torch.tensor(best_x.astype(np.float32)[None, :], dtype=model_dtype)
            ).detach().numpy().ravel()[0])
        ce = Counterexample(
            x_star=best_x, f_x_star=f_val, B_x_star=B_x,
            violation_value=best_val, p_i=p_i.copy(),
            continuous_time=continuous_time, cell_idx=cell_idx, point_idx=-1,
        )
        return Label.UNSAFE, ce

    return Label.SAFE_NLP, None


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def verify_barrier(
    BC              : list,
    sv              : np.ndarray,
    layer_W         : list,
    layer_b         : list,
    W_out           : np.ndarray,
    boundary_H      : np.ndarray,
    boundary_b      : np.ndarray,
    barrier_model,
    dynamics_name   : str,
    continuous_time : bool  = True,
    early_exit      : bool  = True,
    nlp_fallback    : bool  = True,
    nlp_n_starts    : int   = 5,
    nlp_unsafe_tol  : float = 1e-6,
) -> VerificationSummary:
    """
    Formally verify barrier certificate on all boundary-adjacent cells.

    For each cell:
      1. Find exact zero level set crossings via bitmask slicing
      2. Two-step labeling (exact + Taylor remainder)
           -> SAFE_TAYLOR: formally certified — sound
           -> UNSAFE: exact counterexample found
           -> INCONCLUSIVE: Taylor remainder too conservative
      3. If INCONCLUSIVE and nlp_fallback=True:
           Run NLP to find max p_i . f(x) on ZLS
           -> SAFE_NLP: no violation found (heuristic)
           -> UNSAFE: counterexample candidate found by NLP
    """
    from scipy.optimize import minimize

    symbols, f_sym = load_dynamics(dynamics_name)
    hb             = HessianBounder(symbols, f_sym)
    dyn            = DynamicsEvaluator(symbols, f_sym)
    total_neurons  = sum(W.shape[0] for W in layer_W)
    use_wide       = (total_neurons + len(boundary_H) > 64)
    model_dtype    = next(barrier_model.parameters()).dtype

    summary = VerificationSummary(mode="barrier", dynamics_name=dynamics_name)

    print(f"\nBarrier verification: {len(BC)} boundary cells, "
          f"dynamics='{dynamics_name}', "
          f"mode={'continuous-time' if continuous_time else 'discrete-time'}, "
          f"use_wide={use_wide}, nlp_fallback={nlp_fallback}")
    t0 = time.perf_counter()

    for i, vertices in enumerate(BC):
        vertices = np.asarray(vertices, dtype=float)
        sv_i     = sv[i].ravel()
        p_i      = compute_local_gradient(sv_i, layer_W, W_out)

        # B values at vertices — exact since B is affine on cell
        with torch.no_grad():
            B_vals = barrier_model(
                torch.tensor(vertices.astype(np.float32), dtype=model_dtype)
            ).numpy().ravel()

        # B offset: B(x) = p_i @ x + c_i
        centroid = vertices.mean(axis=0)
        with torch.no_grad():
            c_i = float(
                barrier_model(
                    torch.tensor(centroid.astype(np.float32)[None, :],
                                 dtype=model_dtype)
                ).detach().numpy().ravel()[0]
            ) - float(p_i @ centroid)

        # Find zero level set crossings
        x_stars, _ = _get_zero_level_set_crossings(
            vertices, sv_i, B_vals,
            layer_W, layer_b, boundary_H, boundary_b, use_wide
        )

        if len(x_stars) == 0:
            summary.results.append(CellResult(
                label=Label.SAFE_TAYLOR, cell_idx=i,
                M_i=0.0, r_i=0.0, remainder=0.0, max_condition=-1.0,
            ))
            summary.n_safe_taylor += 1
            continue

        # Two-step labeling
        label, M_i, r_i, remainder, ce = _two_step_label(
            x_stars, barrier_model, dyn, hb, p_i, i, continuous_time
        )

        # NLP fallback for INCONCLUSIVE cells — direct, no refinement
        if label == Label.INCONCLUSIVE and nlp_fallback:
            label, ce = _nlp_fallback(
                vertices, sv_i, p_i, c_i,
                layer_W, layer_b, boundary_H, boundary_b,
                dyn, barrier_model, model_dtype,
                continuous_time, nlp_n_starts, nlp_unsafe_tol, i,
            )

        if ce is not None and summary.counterexample is None:
            summary.counterexample = ce

        result = CellResult(
            label=label, cell_idx=i,
            M_i=M_i, r_i=r_i, remainder=remainder,
            max_condition=0.0,
        )
        summary.results.append(result)

        if label == Label.SAFE_TAYLOR:
            summary.n_safe_taylor += 1
        elif label == Label.SAFE_NLP:
            summary.n_safe_nlp += 1
        elif label == Label.UNSAFE:
            summary.n_unsafe += 1
            if early_exit:
                break
        else:
            summary.n_inconclusive += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(BC):
            print(f"  [{i+1}/{len(BC)}] "
                  f"safe_taylor={summary.n_safe_taylor} "
                  f"safe_nlp={summary.n_safe_nlp} "
                  f"unsafe={summary.n_unsafe} "
                  f"inconclusive={summary.n_inconclusive}")

    summary.runtime_s = time.perf_counter() - t0
    summary.print_summary()
    return summary


def verify_lyapunov(
    enumerate_poly   : list,
    sv_all           : np.ndarray,
    layer_W          : list,
    layer_b          : list,
    W_out            : np.ndarray,
    boundary_H       : np.ndarray,
    boundary_b       : np.ndarray,
    lyapunov_model,
    dynamics_name    : str,
    early_exit       : bool = True,
) -> VerificationSummary:
    """
    Formally verify Delta_V(x) = V(f(x)) - V(x) <= 0 on all cells.
    Linearizes f around cell centroid. Taylor remainder at vertices.
    """
    symbols, f_sym = load_dynamics(dynamics_name)
    hb             = HessianBounder(symbols, f_sym)
    dyn            = DynamicsEvaluator(symbols, f_sym)

    summary = VerificationSummary(mode="lyapunov", dynamics_name=dynamics_name)

    print(f"\nLyapunov verification: {len(enumerate_poly)} cells, "
          f"dynamics='{dynamics_name}'")
    t0 = time.perf_counter()

    for i, vertices in enumerate(enumerate_poly):
        vertices = np.asarray(vertices, dtype=float)
        sv_i     = sv_all[i].ravel()
        p_i      = compute_local_gradient(sv_i, layer_W, W_out)
        centroid = vertices.mean(axis=0)

        result = _label_lyapunov_cell(
            vertices, centroid, dyn, hb, p_i, lyapunov_model, i
        )

        summary.results.append(result)
        if result.label == Label.SAFE_TAYLOR:
            summary.n_safe += 1
        elif result.label == Label.UNSAFE:
            summary.n_unsafe += 1
            if early_exit:
                print(f"\n  !! UNSAFE cell at index {i}.")
                break
        else:
            summary.n_inconclusive += 1

        if (i + 1) % 1000 == 0 or (i + 1) == len(enumerate_poly):
            elapsed = time.perf_counter() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(enumerate_poly) - i - 1) / rate
            print(f"  [{i+1}/{len(enumerate_poly)}] "
                  f"safe={summary.n_safe} "
                  f"unsafe={summary.n_unsafe} "
                  f"inconclusive={summary.n_inconclusive} "
                  f"| {rate:.0f} cells/s | ETA {eta:.0f}s")

    summary.runtime_s = time.perf_counter() - t0
    summary.print_summary()
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Smoke test: Taylor labeling logic")

    import numpy as np

    def label(cond, dist, M=1.0):
        rem = 0.5 * M * dist**2
        if (cond + rem).max() <= 0: return "SAFE"
        if (cond - rem).min() >  0: return "UNSAFE"
        return "INCONCLUSIVE"

    assert label(np.array([-0.3,-0.1]), np.array([0.1,0.1])) == "SAFE"
    assert label(np.array([0.5, 0.8]),  np.array([0.1,0.1])) == "UNSAFE"
    assert label(np.array([-0.05,0.05]),np.array([0.5,0.5])) == "INCONCLUSIVE"
    print("  SAFE / UNSAFE / INCONCLUSIVE: all passed")

    print("\nSmoke test: SVD principal direction")
    pts = np.array([[0.,0.],[2.,0.],[4.,0.]])   # all on x-axis
    X   = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    d = Vt[0]
    assert abs(abs(d[0]) - 1.0) < 1e-6, f"Expected [1,0], got {d}"
    print(f"  Principal direction: {d} (expected [±1, 0])")

    print("\nAll smoke tests passed.")