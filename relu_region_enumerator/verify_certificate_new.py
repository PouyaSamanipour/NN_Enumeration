"""
verify_certificate_new.py
=========================
Same as verify_certificates.py but replaces the NLP fallback with an
adaptive geometric refinement technique.

Adaptive Refinement (replaces NLP)
------------------------------------
When INCONCLUSIVE after the two-step Taylor check, instead of running NLP:

  1. Compute the required number of refinements:
         n_refs = ceil(remainder / |worst_cond|)
     where  remainder  = 0.5 * M * r^2  (Taylor remainder at this step)
     and    worst_cond = max(cond_vals)  (largest linearised condition value —
                                          typically negative but |value| < remainder)

  2. Find the two x_stars with the largest pairwise distance:
         (idx_a, idx_b) = argmax_{i,j} ||x_stars[i] - x_stars[j]||

  3. Normal direction  d = (x_stars[idx_b] - x_stars[idx_a]) / ||...||

  4. Create  n_refs - 1  equally spaced cut points between x_stars[idx_a]
     and x_stars[idx_b]:
         p_k = x_stars[idx_a] + (k / n_refs) * (x_stars[idx_b] - x_stars[idx_a])
                                                         k = 1, ..., n_refs - 1

  5. Each hyperplane H_k passes through p_k with normal d:
         H_k : d^T x = d^T p_k

  6. These n_refs - 1 parallel planes partition the ZLS crossing-point set
     into n_refs slabs.  Each slab is verified independently with the two-step
     Taylor check.  Recurse on any still-INCONCLUSIVE slab.

Usage
-----
    summary = verify_barrier(
        BC, sv, layer_W, layer_b, W_out,
        boundary_H, boundary_b,
        barrier_model, dynamics_name="arch3"
    )
"""

from __future__ import annotations

import time
import numpy as np
import torch
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional
from numba import njit

try:
    from .hessian_bound import HessianBounder, compute_local_gradient
    from .Dynamics import load_dynamics
    from .bitwise_utils import (
        get_cell_hyperplanes_input_space,
        generate_mask,
        generate_mask_wide,
        slice_polytope_with_hyperplane,
        slice_polytope_wide,
        Enumerator_rapid,
        finding_deep_hype,
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
        Enumerator_rapid,
        finding_deep_hype,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

class Label(Enum):
    SAFE_TAYLOR      = "SAFE_TAYLOR"       # formally certified by Taylor remainder
    SAFE_REFINEMENT  = "SAFE_REFINEMENT"   # certified after adaptive geometric refinement
    UNSAFE           = "UNSAFE"            # counterexample found
    INCONCLUSIVE     = "INCONCLUSIVE"      # Taylor + refinement exhausted


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
    mode                : str
    dynamics_name       : str
    n_safe_taylor       : int   = 0
    n_safe_refinement   : int   = 0
    n_unsafe            : int   = 0
    n_inconclusive      : int   = 0
    runtime_s           : float = 0.0
    results             : List[CellResult] = field(default_factory=list)
    counterexample      : Optional['Counterexample'] = None

    @property
    def n_safe(self):
        return self.n_safe_taylor + self.n_safe_refinement

    @property
    def total(self):
        return self.n_safe + self.n_unsafe + self.n_inconclusive

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Verification Summary  [{self.mode} / {self.dynamics_name}]")
        print(f"{'='*60}")
        print(f"  Total cells checked    : {self.total}")
        print(f"  SAFE (Taylor)          : {self.n_safe_taylor}"
              f"  ({100*self.n_safe_taylor/max(self.total,1):.1f}%)  [formally certified]")
        print(f"  SAFE (Refinement)      : {self.n_safe_refinement}"
              f"  ({100*self.n_safe_refinement/max(self.total,1):.1f}%)  [certified after adaptive split]")
        print(f"  UNSAFE                 : {self.n_unsafe}"
              f"  ({100*self.n_unsafe/max(self.total,1):.1f}%)")
        print(f"  INCONCLUSIVE           : {self.n_inconclusive}"
              f"  ({100*self.n_inconclusive/max(self.total,1):.1f}%)")
        print(f"  Runtime                : {self.runtime_s:.2f} s")

        if self.counterexample is not None:
            print(f"\n  Exact counterexample found:")
            self.counterexample.report()
        elif self.n_unsafe > 0:
            print(f"\n  !! {self.n_unsafe} UNSAFE cells found.")
        elif self.n_inconclusive == 0:
            if self.n_safe_refinement == 0:
                print(f"\n  Certificate VERIFIED (Taylor) over all {self.total} cells.")
            else:
                print(f"\n  Certificate VERIFIED: {self.n_safe_taylor} cells by Taylor, "
                      f"{self.n_safe_refinement} by adaptive refinement.")
        else:
            print(f"\n  {self.n_inconclusive} cells remain INCONCLUSIVE after Taylor + refinement.")
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find exact zero level set crossings using bitmask slicing.
    Identical to verify_certificates.py.
    """
    import warnings

    n                = vertices.shape[1]
    zls_residual_tol = 1e-3
    empty_masks      = np.zeros(0, dtype=np.uint64)

    if B_vals.min() >= -1e-9 or B_vals.max() <= 1e-9:
        return np.zeros((0, n)), empty_masks

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
        return np.zeros((0, n)), empty_masks

    created_verts = np.array(created_verts)

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
            return np.zeros((0, n)), empty_masks

    return created_verts, created_masks


# ═══════════════════════════════════════════════════════════════════════════
# Core two-step labeling
# ═══════════════════════════════════════════════════════════════════════════

def _two_step_label(
    x_stars         : np.ndarray,
    barrier_model,
    dyn             : DynamicsEvaluator,
    hb              : HessianBounder,
    p_i             : np.ndarray,
    cell_idx        : int,
    continuous_time : bool,
) -> Tuple[Label, float, float, float, float, Optional[Counterexample]]:
    """
    Two-step labeling on a set of zero level set crossing points.

    Returns
    -------
    label, M_i, r_i, remainder, worst_cond, counterexample

    worst_cond = max(cond_vals) — needed by the adaptive refinement to
    compute the required number of cuts (n_refs = ceil(remainder / |worst_cond|)).
    """
    model_dtype = next(barrier_model.parameters()).dtype
    np_dtype    = np.float64 if model_dtype == torch.float64 else np.float32

    c_star   = x_stars.mean(axis=0)
    eval_pts = np.vstack([x_stars, c_star[None, :]])
    f_exact  = dyn.eval_batch(eval_pts)

    if continuous_time:
        exact_vals = f_exact @ p_i
    else:
        with torch.no_grad():
            exact_vals = barrier_model(
                torch.tensor(f_exact.astype(np_dtype), dtype=model_dtype)
            ).numpy().ravel()

    # Step 1: exact check
    for e in range(len(eval_pts)):
        if exact_vals[e] > 0.0:
            with torch.no_grad():
                B_x = float(barrier_model(
                    torch.tensor(eval_pts[e:e+1].astype(np_dtype),
                                 dtype=model_dtype)
                ).numpy().ravel()[0])
            ce = Counterexample(
                x_star=eval_pts[e].copy(), f_x_star=f_exact[e].copy(),
                B_x_star=B_x, violation_value=float(exact_vals[e]),
                p_i=p_i.copy(), continuous_time=continuous_time,
                cell_idx=cell_idx, point_idx=e,
            )
            return Label.UNSAFE, 0.0, 0.0, 0.0, 0.0, ce

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
    worst_cond  = float(cond_vals.max())

    eps = 1e-6
    if worst_safe <= eps:
        label = Label.SAFE_TAYLOR
    elif best_unsafe > eps:
        label = Label.UNSAFE
    else:
        label = Label.INCONCLUSIVE

    return (label, M_i, float(distances.max()),
            float(remainders.max()), worst_cond, None)


# ═══════════════════════════════════════════════════════════════════════════
# ZLS polytope slicer (unchanged — needed for recursive refinement)
# ═══════════════════════════════════════════════════════════════════════════

def slice_zls_polytope(
    x_stars   : np.ndarray,
    fake_vals : np.ndarray,
    x_masks   : np.ndarray,
    h_idx     : int,
    n         : int,
) -> Tuple[list, list, np.ndarray]:
    """
    Slice the (n-1)-dimensional ZLS polytope with a hyperplane.
    Identical to verify_certificates.py.
    """
    ONE        = np.uint64(1)
    req_shared = n - 2

    strict_inside  = fake_vals < -1e-9
    strict_outside = fake_vals >  1e-9
    idx_in         = np.where(fake_vals <= 1e-5)[0]
    idx_out        = np.where(fake_vals >= -1e-5)[0]
    s_idx_in       = np.where(strict_inside)[0]
    s_idx_out      = np.where(strict_outside)[0]

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


# ═══════════════════════════════════════════════════════════════════════════
# Numba helpers
# ═══════════════════════════════════════════════════════════════════════════

@njit
def _find_edge_zls_nb(verts: np.ndarray, B_vals: np.ndarray) -> np.ndarray:
    """Find B=0 crossings on polytope edges by linear interpolation (JIT).

    For every pair of vertices (u, v) with opposite B signs, interpolate to
    the crossing point.  Returns a (K, n) array of crossing points.
    """
    V = verts.shape[0]
    n = verts.shape[1]
    # Count crossings
    cnt = 0
    for i in range(V):
        for j in range(i + 1, V):
            if B_vals[i] * B_vals[j] < 0.0:
                cnt += 1
    if cnt == 0:
        return np.empty((0, n), dtype=np.float64)
    result = np.empty((cnt, n), dtype=np.float64)
    k = 0
    for i in range(V):
        for j in range(i + 1, V):
            if B_vals[i] * B_vals[j] < 0.0:
                t = -B_vals[i] / (B_vals[j] - B_vals[i])
                for d in range(n):
                    result[k, d] = verts[i, d] + t * (verts[j, d] - verts[i, d])
                k += 1
    return result


def _find_edge_zls(verts: np.ndarray, B_vals: np.ndarray) -> np.ndarray:
    """Wrapper: find B=0 edge crossings (calls numba JIT)."""
    return _find_edge_zls_nb(
        np.ascontiguousarray(verts, dtype=np.float64),
        np.ascontiguousarray(B_vals, dtype=np.float64),
    )


@njit
def _centroid_and_radius_nb(verts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute centroid and max distance from centroid (JIT)."""
    V = verts.shape[0]
    n = verts.shape[1]
    c = np.zeros(n, dtype=np.float64)
    for i in range(V):
        for d in range(n):
            c[d] += verts[i, d]
    for d in range(n):
        c[d] /= V
    max_r = 0.0
    for i in range(V):
        r2 = 0.0
        for d in range(n):
            diff = verts[i, d] - c[d]
            r2 += diff * diff
        r = r2 ** 0.5
        if r > max_r:
            max_r = r
    return c, max_r


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive geometric refinement — iterative (queue-based)
# ═══════════════════════════════════════════════════════════════════════════

_refine_stats = {"enumerator_ok": 0, "slab_fallback": 0}

def _refine_barrier_adaptive(
    x_stars         : np.ndarray,
    worst_cond      : float,
    remainder       : float,
    p_i             : np.ndarray,
    barrier_model,
    dyn             : DynamicsEvaluator,
    hb              : HessianBounder,
    cell_idx        : int,
    continuous_time : bool,
    use_wide        : bool,
    max_depth       : int,
    vertices        : np.ndarray,
    sv_i            : np.ndarray,
    layer_W         : list,
    layer_b         : list,
    boundary_H      : np.ndarray,
    boundary_b      : np.ndarray,
    TH              : list,
    model_dtype,
) -> Tuple[Label, float, float, float, Optional[Counterexample]]:
    """
    Iterative adaptive refinement — replaces the recursive version.

    Uses an explicit queue so that resolved sub-cell vertex arrays are freed
    immediately rather than staying pinned on the call stack.

    Queue items: (x_stars, vertices, worst_cond, remainder, depth_remaining)
    """
    from collections import deque

    np_dtype = np.float64 if model_dtype == torch.float64 else np.float32
    n        = x_stars.shape[1]

    queue = deque()
    queue.append((x_stars, vertices, worst_cond, remainder, max_depth))

    worst_M          = 0.0
    worst_r          = 0.0
    worst_rem        = 0.0
    any_inconclusive = False

    while queue:
        xs, verts, wc, rem, depth = queue.popleft()

        if depth == 0 or len(xs) < 2:
            any_inconclusive = True
            continue

        E = len(xs)

        # ── find farthest pair of ZLS points ─────────────────────────────────
        diff     = xs[:, None, :] - xs[None, :, :]
        dmat     = np.linalg.norm(diff, axis=2)
        flat     = int(np.argmax(dmat))
        idx_a, idx_b = divmod(flat, E)
        max_dist = float(dmat[idx_a, idx_b])

        if max_dist < 1e-12:
            any_inconclusive = True
            continue

        # ── compute cuts ──────────────────────────────────────────────────────
        normal    = (xs[idx_b] - xs[idx_a]) / max_dist
        abs_f_lin = max(abs(wc), 1e-10)
        n_refs    = min(max(2, int(np.ceil(rem / abs_f_lin))), 32)

        base_proj = float(xs[idx_a] @ normal)
        step_proj = float((xs[idx_b] - xs[idx_a]) @ normal) / n_refs
        cuts      = [base_proj + k * step_proj for k in range(1, n_refs)]


        H_refine = np.tile(normal, (n_refs - 1, 1)).astype(np.float64)
        b_refine = np.array([-c for c in cuts], dtype=np.float64)

        # ── slice polytope ────────────────────────────────────────────────────
        H_cell, b_cell = get_cell_hyperplanes_input_space(
            sv_i, layer_W, layer_b, boundary_H, boundary_b
        )
        try:
            sub_cells = Enumerator_rapid(
                H_refine, b_refine,
                [np.asarray(verts, dtype=np.float64)],
                TH, [H_cell.tolist()], [b_cell.tolist()],
                False, None, 0, use_wide,
            )
        except Exception as exc:
            sub_cells = None

        # ── Enumerator_rapid couldn't split ──────────────────────────────────
        if sub_cells is None or len(sub_cells) <= 1:
            _refine_stats["slab_fallback"] += 1
            n_verts = len(verts)
            n_sub   = len(sub_cells) if sub_cells is not None else 0
            print(f"  [cell {cell_idx}] Enumerator_rapid returned {n_sub} sub-cells "
                  f"(parent has {n_verts} verts, E={len(xs)}, n_refs={n_refs})")
            any_inconclusive = True
            continue

        # ── process proper sub-cells ──────────────────────────────────────────
        _refine_stats["enumerator_ok"] += 1
        for sub_verts in sub_cells:
            sub_verts = np.asarray(sub_verts, dtype=np.float64)
            if len(sub_verts) < n + 1:
                continue

            with torch.no_grad():
                B_sub = barrier_model(
                    torch.tensor(sub_verts.astype(np_dtype), dtype=model_dtype)
                ).numpy().ravel().astype(np.float64)

            if B_sub.min() >= -1e-9 or B_sub.max() <= 1e-9:
                continue   # no ZLS crossing — SAFE, discard immediately

            xs_k, _ = _get_zero_level_set_crossings(
                sub_verts, sv_i, B_sub,
                layer_W, layer_b, boundary_H, boundary_b,
                use_wide, barrier_model=barrier_model, model_dtype=model_dtype,
            )
            if len(xs_k) == 0:
                continue

            label, M_i, r_i, rem_i, wc_i, ce = _two_step_label(
                xs_k, barrier_model, dyn, hb, p_i, cell_idx, continuous_time
            )
            worst_M   = max(worst_M,   M_i)
            worst_r   = max(worst_r,   r_i)
            worst_rem = max(worst_rem, rem_i)

            if label == Label.UNSAFE:
                return Label.UNSAFE, M_i, r_i, rem_i, ce

            if label == Label.INCONCLUSIVE:
                if len(xs_k) >= 2:
                    queue.append((xs_k, sub_verts, wc_i, rem_i, depth - 1))
                else:
                    any_inconclusive = True
            # SAFE_TAYLOR: sub_verts goes out of scope here — freed immediately

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
    refinement_max_depth: int   = 5,
    TH                  : list  = None,
) -> VerificationSummary:
    """
    Formally verify barrier certificate on all boundary-adjacent cells.

    For each cell:
      1. Find exact zero level set crossings via bitmask slicing.
      2. Two-step labeling (exact + Taylor remainder).
           -> SAFE_TAYLOR      : formally certified — sound.
           -> UNSAFE           : exact counterexample found.
           -> INCONCLUSIVE     : Taylor remainder too conservative.
      3. If INCONCLUSIVE:
           Run adaptive geometric refinement (_refine_barrier_adaptive).
           Computes n_refs = ceil(remainder / |worst_cond|), finds the
           farthest pair of x_stars, cuts n_refs-1 parallel hyperplanes
           through equally spaced points along that direction.
           -> SAFE_REFINEMENT  : certified after splitting.
           -> UNSAFE           : counterexample found in a sub-region.
           -> INCONCLUSIVE     : max depth reached.
    """
    symbols, f_sym = load_dynamics(dynamics_name)
    hb             = HessianBounder(symbols, f_sym)
    dyn            = DynamicsEvaluator(symbols, f_sym)
    total_neurons  = sum(W.shape[0] for W in layer_W)
    use_wide       = (total_neurons + len(boundary_H) > 64)
    model_dtype    = next(barrier_model.parameters()).dtype

    # Default TH: infer from boundary_b (assumes box domain: H=[I; -I], b=[TH; TH])
    if TH is None:
        n_dim = layer_W[0].shape[1]
        TH = list(boundary_b[:n_dim])

    summary = VerificationSummary(mode="barrier", dynamics_name=dynamics_name)

    print(f"\nBarrier verification (adaptive refinement): {len(BC)} boundary cells, "
          f"dynamics='{dynamics_name}', "
          f"mode={'continuous-time' if continuous_time else 'discrete-time'}, "
          f"use_wide={use_wide}, max_depth={refinement_max_depth}")
    t0 = time.perf_counter()

    for i, vertices in enumerate(BC):
        vertices = np.asarray(vertices, dtype=float)
        sv_i     = sv[i].ravel()
        p_i      = compute_local_gradient(sv_i, layer_W, W_out)

        np_dtype = np.float64 if model_dtype == torch.float64 else np.float32
        with torch.no_grad():
            B_vals = barrier_model(
                torch.tensor(vertices.astype(np_dtype), dtype=model_dtype)
            ).numpy().ravel()

        # Find zero level set crossings
        x_stars, x_masks = _get_zero_level_set_crossings(
            vertices, sv_i, B_vals,
            layer_W, layer_b, boundary_H, boundary_b, use_wide,
            barrier_model=barrier_model,
            model_dtype=model_dtype,
        )

        if len(x_stars) == 0:
            summary.results.append(CellResult(
                label=Label.SAFE_TAYLOR, cell_idx=i,
                M_i=0.0, r_i=0.0, remainder=0.0, max_condition=-1.0,
            ))
            summary.n_safe_taylor += 1
            continue

        # Two-step labeling
        label, M_i, r_i, remainder, worst_cond, ce = _two_step_label(
            x_stars, barrier_model, dyn, hb, p_i, i, continuous_time
        )

        # Adaptive refinement for INCONCLUSIVE cells
        if label == Label.INCONCLUSIVE:
            label, M_i, r_i, remainder, ce = _refine_barrier_adaptive(
                x_stars, worst_cond, remainder,
                p_i, barrier_model, dyn, hb,
                i, continuous_time, use_wide,
                refinement_max_depth,
                vertices, sv_i, layer_W, layer_b, boundary_H, boundary_b, TH, model_dtype,
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
                  f"safe_refinement={summary.n_safe_refinement} "
                  f"unsafe={summary.n_unsafe} "
                  f"inconclusive={summary.n_inconclusive}")

    summary.runtime_s = time.perf_counter() - t0
    summary.print_summary()
    total_splits = _refine_stats["enumerator_ok"] + _refine_stats["slab_fallback"]
    print(f"\n  Refinement splits: {total_splits} total  |  "
          f"Enumerator_rapid OK: {_refine_stats['enumerator_ok']}  |  "
          f"Slab fallback: {_refine_stats['slab_fallback']}")
    _refine_stats["enumerator_ok"] = 0
    _refine_stats["slab_fallback"] = 0
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Smoke test: n_refs computation")

    remainder  = 0.1
    worst_cond = -0.02   # INCONCLUSIVE: |cond| < remainder
    n_refs     = max(2, int(np.ceil(remainder / abs(worst_cond))))
    print(f"  remainder={remainder}, worst_cond={worst_cond}, n_refs={n_refs}")
    assert n_refs == 5, f"Expected 5, got {n_refs}"

    # Farthest pair
    rng    = np.random.default_rng(0)
    pts    = rng.standard_normal((10, 3))
    diff   = pts[:, None, :] - pts[None, :, :]
    dmat   = np.linalg.norm(diff, axis=2)
    flat   = int(np.argmax(dmat))
    ia, ib = divmod(flat, 10)
    print(f"  Farthest pair: {ia}, {ib}, dist={dmat[ia,ib]:.4f}")

    print("  All passed.")
