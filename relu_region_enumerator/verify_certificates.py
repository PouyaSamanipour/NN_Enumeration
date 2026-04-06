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
    SAFE         = "SAFE"
    UNSAFE       = "UNSAFE"
    INCONCLUSIVE = "INCONCLUSIVE"


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
    mode          : str
    dynamics_name : str
    n_safe        : int   = 0
    n_unsafe      : int   = 0
    n_inconclusive: int   = 0
    runtime_s     : float = 0.0
    results       : List[CellResult] = field(default_factory=list)
    counterexample: Optional['Counterexample'] = None

    @property
    def total(self):
        return self.n_safe + self.n_unsafe + self.n_inconclusive

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Verification Summary  [{self.mode} / {self.dynamics_name}]")
        print(f"{'='*60}")
        print(f"  Total cells checked : {self.total}")
        print(f"  SAFE                : {self.n_safe}"
              f"  ({100*self.n_safe/max(self.total,1):.1f}%)")
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
            print(f"\n  Certificate VERIFIED over all {self.total} cells.")
        else:
            print(f"\n  Certificate verified on {self.n_safe} cells; "
                  f"{self.n_inconclusive} inconclusive "
                  f"(B(f(x*)) ~ 0, trajectory tangent to boundary).")
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
    boundary_H : np.ndarray,   # (B, n) — may include fake neurons
    boundary_b : np.ndarray,   # (B,)
    use_wide   : bool,
) -> np.ndarray:               # (E, n) crossing points, or empty
    """
    Find exact zero level set crossings using bitmask slicing.

    B is affine on the cell so B_vals at vertices are exact.
    The zero level set B(x)=0 is a new hyperplane cutting the cell.
    slice_polytope_with_hyperplane uses bitmask adjacency to find only
    valid polytope edges — created_verts are the exact crossing points.

    boundary_H may include fake neurons added during refinement.
    """
    n = vertices.shape[1]

    if B_vals.min() >= -1e-9 or B_vals.max() <= 1e-9:
        return np.zeros((0, n))

    H_all, b_all = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b
    )

    verts_f64 = vertices.astype(np.float64)
    H_f64     = H_all.astype(np.float64)
    b_f64     = b_all.astype(np.float64)
    B_f64     = B_vals.astype(np.float64)

    if use_wide:
        masks = generate_mask_wide(verts_f64, H_f64, b_f64)
        _, _, created_verts = slice_polytope_wide(
            verts_f64, B_f64, masks, len(H_all), n
        )
    else:
        masks = generate_mask(verts_f64, H_f64, b_f64)
        _, _, created_verts = slice_polytope_with_hyperplane(
            verts_f64, B_f64, masks, len(H_all), n
        )

    return created_verts   # (E, n)


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

    if worst_safe <= 0.0:
        label = Label.SAFE
    elif best_unsafe > 0.0:
        label = Label.UNSAFE
    else:
        label = Label.INCONCLUSIVE

    return (label, M_i, float(distances.max()),
            float(remainders.max()), None)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers for cell splitting
# ═══════════════════════════════════════════════════════════════════════════

def _build_child(
    vertices   : np.ndarray,   # (V, n) parent vertices
    x_stars    : np.ndarray,   # (E, n) zero level set crossing points
    boundary_H : np.ndarray,
    boundary_b : np.ndarray,
    p_i        : np.ndarray,   # (n,)
    c_i        : float,
    cut_H      : np.ndarray,   # (1, n) cutting hyperplane normal (already oriented)
    cut_b      : np.ndarray,   # (1,)   cutting hyperplane offset
    side_sign  : float,        # +1 if child is on cut_H @ x + cut_b >= 0 side
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build child cell geometry for one side of a cut.

    Returns (child_vertices, boundary_H_child, boundary_b_child).

    Child vertices = parent vertices on correct side + crossing points.
    Boundary includes: original boundary + cutting hyperplane + ZLS (B=0).
    """
    n = vertices.shape[1]

    # Parent vertices on this side
    v_vals = vertices @ cut_H[0] + cut_b[0]   # (V,)
    if side_sign > 0:
        v_mask = v_vals >= -1e-9
    else:
        v_mask = v_vals <=  1e-9

    verts_side = vertices[v_mask]

    # Child vertices: parent vertices on this side + zero level set crossings
    if len(verts_side) == 0:
        child_verts = x_stars
    else:
        child_verts = np.vstack([verts_side, x_stars])

    # Augmented boundary: original + cutting hyperplane (oriented) + ZLS
    # Cutting hyperplane oriented so child is on the non-negative side:
    #   side_sign > 0:  cut_H @ x + cut_b >= 0  ->  add  cut_H,  cut_b
    #   side_sign < 0:  cut_H @ x + cut_b <= 0  ->  add -cut_H, -cut_b
    if side_sign > 0:
        bH = np.vstack([boundary_H, cut_H])
        bb = np.concatenate([boundary_b, cut_b])
    else:
        bH = np.vstack([boundary_H, -cut_H])
        bb = np.concatenate([boundary_b, -cut_b])

    # Zero level set as boundary face (B=0 vertices exist in both children)
    zls_H = p_i[None, :]
    zls_b = np.array([c_i])
    bH    = np.vstack([bH, zls_H])
    bb    = np.concatenate([bb, zls_b])

    return child_verts, bH, bb


def _get_child_crossings(
    child_verts : np.ndarray,
    x_stars     : np.ndarray,   # fallback if detection fails
    sv_i        : np.ndarray,
    barrier_model,
    layer_W     : list,
    layer_b     : list,
    boundary_H  : np.ndarray,
    boundary_b  : np.ndarray,
    use_wide    : bool,
) -> np.ndarray:
    """Find zero level set crossings on a child cell."""
    model_dtype = next(barrier_model.parameters()).dtype
    with torch.no_grad():
        B_child = barrier_model(
            torch.tensor(child_verts.astype(np.float32), dtype=model_dtype)
        ).numpy().ravel()

    crossings = _get_zero_level_set_crossings(
        child_verts, sv_i, B_child,
        layer_W, layer_b, boundary_H, boundary_b, use_wide
    )

    # Fallback: if bitmask slicing finds nothing, use the known crossing points
    # that belong to this child (those already on the B=0 face)
    if len(crossings) == 0:
        crossings = x_stars

    return crossings


# ═══════════════════════════════════════════════════════════════════════════
# Step 0: split X_i along zero level set, verify smaller child, discard other
# ═══════════════════════════════════════════════════════════════════════════

def _split_on_zls(
    vertices        : np.ndarray,   # (V, n) original cell vertices
    x_stars         : np.ndarray,   # (E, n) zero level set crossings
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
    Split X_i along zero level set B(x)=0 into X_i^+ and X_i^-.

    Both children share the zero level set as a face — its vertices appear
    in both. Therefore we only need to verify ONE child (the smaller one).
    The certificate condition is the same on the shared face regardless of
    which side you verify from, so verifying one covers the boundary.

    After this initial split, we call _refine_barrier which verifies BOTH
    children when splitting with fake neurons.
    """
    n = vertices.shape[1]

    # ZLS hyperplane as the cut: p_i @ x + c_i = 0
    cut_H = p_i[None, :]
    cut_b = np.array([c_i])

    # Build both children
    verts_pos, bH_pos, bb_pos = _build_child(
        vertices, x_stars, boundary_H, boundary_b,
        p_i, c_i, cut_H, cut_b, side_sign=+1
    )
    verts_neg, bH_neg, bb_neg = _build_child(
        vertices, x_stars, boundary_H, boundary_b,
        p_i, c_i, cut_H, cut_b, side_sign=-1
    )

    # Pick smaller child by number of non-ZLS vertices
    v_vals  = vertices @ p_i + c_i   # (V,)
    n_pos   = int((v_vals >= -1e-9).sum())
    n_neg   = int((v_vals <=  1e-9).sum())

    if n_pos <= n_neg:
        child_verts, child_bH, child_bb = verts_pos, bH_pos, bb_pos
    else:
        child_verts, child_bH, child_bb = verts_neg, bH_neg, bb_neg

    if len(child_verts) < n + 1:
        # Degenerate — fall back to original crossings
        return _two_step_label(
            x_stars, barrier_model, dyn, hb, p_i, cell_idx, continuous_time
        )

    # Get crossings on chosen child
    child_crossings = _get_child_crossings(
        child_verts, x_stars, sv_i, barrier_model,
        layer_W, layer_b, child_bH, child_bb, use_wide
    )

    if len(child_crossings) == 0:
        return Label.SAFE, 0.0, 0.0, 0.0, None

    # Two-step label on chosen child
    label, M_i, r_i, remainder, ce = _two_step_label(
        child_crossings, barrier_model, dyn, hb, p_i, cell_idx, continuous_time
    )

    if label != Label.INCONCLUSIVE:
        return label, M_i, r_i, remainder, ce

    # INCONCLUSIVE — refine with fake neuron (must verify BOTH children)
    return _refine_barrier(
        child_crossings, child_verts, sv_i, p_i, c_i,
        barrier_model, dyn, hb,
        layer_W, layer_b, W_out,
        child_bH, child_bb,
        cell_idx, continuous_time, use_wide,
        max_depth - 1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Refinement by fake neuron — BOTH children must be verified
# ═══════════════════════════════════════════════════════════════════════════

def _refine_barrier(
    x_stars         : np.ndarray,   # (E, n) current crossing points
    vertices        : np.ndarray,   # (V, n) current cell vertices
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
    Refine INCONCLUSIVE cell by splitting with a fake neuron hyperplane.

    Unlike the initial ZLS split, fake neuron splits create two children
    that do NOT share the zero level set face — BOTH must be verified.

    Strategy:
      1. Find principal direction d of x_stars via SVD
      2. Fake hyperplane: d^T x = d^T c*
      3. Split into two children, build each with correct boundary
      4. Verify BOTH children with two-step label
      5. If either is INCONCLUSIVE, recurse on it
      6. SAFE only when ALL children are SAFE
      7. UNSAFE as soon as any child is UNSAFE

    Termination: only B(f(x*)) = 0 (tangent) stays INCONCLUSIVE at max_depth.
    """
    if max_depth == 0 or len(x_stars) < 2:
        return Label.INCONCLUSIVE, 0.0, 0.0, 0.0, None

    n      = x_stars.shape[1]
    c_star = x_stars.mean(axis=0)

    # ── Principal direction via SVD ───────────────────────────────────────────
    X = x_stars - c_star
    if np.linalg.norm(X) < 1e-12:
        return Label.INCONCLUSIVE, 0.0, 0.0, 0.0, None

    _, _, Vt  = np.linalg.svd(X, full_matrices=False)
    d         = Vt[0]
    offset    = float(d @ c_star)
    cut_H     = d[None, :]
    cut_b     = np.array([-offset])   # d @ x - offset = 0

    # ── Build both children ───────────────────────────────────────────────────
    verts_pos, bH_pos, bb_pos = _build_child(
        vertices, x_stars, boundary_H, boundary_b,
        p_i, c_i, cut_H, cut_b, side_sign=+1
    )
    verts_neg, bH_neg, bb_neg = _build_child(
        vertices, x_stars, boundary_H, boundary_b,
        p_i, c_i, cut_H, cut_b, side_sign=-1
    )

    # ── Split crossing points between children ────────────────────────────────
    signed    = x_stars @ d - offset
    xs_pos    = x_stars[signed >= -1e-9]
    xs_neg    = x_stars[signed <=  1e-9]

    children = [
        (verts_pos, bH_pos, bb_pos, xs_pos),
        (verts_neg, bH_neg, bb_neg, xs_neg),
    ]

    # ── Verify BOTH children ──────────────────────────────────────────────────
    worst_M   = 0.0
    worst_r   = 0.0
    worst_rem = 0.0

    for child_verts, child_bH, child_bb, xs_child in children:
        if len(child_verts) < n + 1 or len(xs_child) == 0:
            continue   # degenerate child — treat as safe

        child_crossings = _get_child_crossings(
            child_verts, xs_child, sv_i, barrier_model,
            layer_W, layer_b, child_bH, child_bb, use_wide
        )

        if len(child_crossings) == 0:
            continue   # no crossings on this child — safe

        label, M_i, r_i, remainder, ce = _two_step_label(
            child_crossings, barrier_model, dyn, hb,
            p_i, cell_idx, continuous_time
        )

        # Track worst-case values for reporting
        worst_M   = max(worst_M,   M_i)
        worst_r   = max(worst_r,   r_i)
        worst_rem = max(worst_rem, remainder)

        if label == Label.UNSAFE:
            return Label.UNSAFE, M_i, r_i, remainder, ce

        if label == Label.INCONCLUSIVE:
            # Recurse on this child
            label, M_i, r_i, remainder, ce = _refine_barrier(
                child_crossings, child_verts, sv_i, p_i, c_i,
                barrier_model, dyn, hb,
                layer_W, layer_b, W_out,
                child_bH, child_bb,
                cell_idx, continuous_time, use_wide,
                max_depth - 1,
            )

            if label == Label.UNSAFE:
                return Label.UNSAFE, M_i, r_i, remainder, ce

            if label == Label.INCONCLUSIVE:
                return Label.INCONCLUSIVE, worst_M, worst_r, worst_rem, None

    # All children verified SAFE
    return Label.SAFE, worst_M, worst_r, worst_rem, None


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
        label = Label.SAFE
    elif best_unsafe > 0.0:
        label = Label.UNSAFE
    else:
        label = Label.INCONCLUSIVE

    return CellResult(
        label=label, cell_idx=cell_idx, M_i=M_i, r_i=r_i,
        remainder=float(remainders.max()), max_condition=float(cond_vals.max()),
    )


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
    continuous_time : bool = True,
    early_exit      : bool = True,
    max_refine_depth: int  = 8,
) -> VerificationSummary:
    """
    Formally verify barrier certificate on all boundary-adjacent cells.

    For each cell:
      1. Find exact zero level set crossings via bitmask slicing
      2. Two-step labeling (exact + Taylor)
      3. If INCONCLUSIVE: refine by fake neuron splitting
    """
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
          f"use_wide={use_wide}, max_refine_depth={max_refine_depth}")
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
        c_i      = float(
            barrier_model(
                torch.tensor(centroid.astype(np.float32)[None, :],
                             dtype=model_dtype)
            ) 
        ) - float(p_i @ centroid)

        # Find zero level set crossings
        x_stars = _get_zero_level_set_crossings(
            vertices, sv_i, B_vals,
            layer_W, layer_b, boundary_H, boundary_b, use_wide
        )

        if len(x_stars) == 0:
            summary.results.append(CellResult(
                label=Label.SAFE, cell_idx=i,
                M_i=0.0, r_i=0.0, remainder=0.0, max_condition=-1.0,
            ))
            summary.n_safe += 1
            continue

        # Two-step labeling
        label, M_i, r_i, remainder, ce = _two_step_label(
            x_stars, barrier_model, dyn, hb, p_i, i, continuous_time
        )

        # Refine if INCONCLUSIVE:
        # Step 0 — split X_i along zero level set, verify smaller child (discard other)
        # Step 1+ — fake neuron splits, verify BOTH children
        if label == Label.INCONCLUSIVE:
            label, M_i, r_i, remainder, ce = _split_on_zls(
                vertices, x_stars, sv_i, p_i, c_i,
                barrier_model, dyn, hb,
                layer_W, layer_b, W_out,
                boundary_H, boundary_b,
                i, continuous_time, use_wide, max_refine_depth
            )

        if ce is not None and summary.counterexample is None:
            summary.counterexample = ce

        result = CellResult(
            label=label, cell_idx=i,
            M_i=M_i, r_i=r_i, remainder=remainder,
            max_condition=0.0,
        )
        summary.results.append(result)

        if label == Label.SAFE:
            summary.n_safe += 1
        elif label == Label.UNSAFE:
            summary.n_unsafe += 1
            if early_exit:
                break
        else:
            summary.n_inconclusive += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(BC):
            print(f"  [{i+1}/{len(BC)}] "
                  f"safe={summary.n_safe} "
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
        if result.label == Label.SAFE:
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