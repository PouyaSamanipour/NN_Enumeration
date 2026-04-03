"""
verify_certificates.py
======================
Formal verification of Lyapunov decrease and barrier certificate conditions
over all enumerated polytopic linear regions, using the Taylor remainder bound.

For each cell X_i the network is affine, so the certificate value B(x) or V(x)
is affine on X_i and its extremum is attained at a vertex. The nonlinear
dynamics f(x) introduces a correction bounded by the Taylor remainder:

    |B(f(x)) - B(f_lin(x))| <= 0.5 * M_i * r_i^2

where:
    f_lin(x) = linearization of f at centroid of X_i
    M_i      = guaranteed upper bound on ||H_cell(x)||_2 over X_i
    r_i      = max_k ||v_k - centroid_i||  (cell radius)

Verification labels per cell
-----------------------------
    SAFE        : condition holds with formal guarantee
    UNSAFE      : condition is violated
    INCONCLUSIVE: remainder bound too loose to decide; cell needs refinement

Barrier certificate condition (boundary cells only)
----------------------------------------------------
    B(f(x)) <= 0  for all x in X_i with B(x) near 0

    Verified as:
        max_k B(f_lin(v_k)) + 0.5 * M_i * r_i^2 <= 0  -> SAFE
        min_k B(f_lin(v_k)) - 0.5 * M_i * r_i^2 >  0  -> UNSAFE
        otherwise                                        -> INCONCLUSIVE

Lyapunov decrease condition (all cells)
----------------------------------------
    Delta_V(x) = V(f(x)) - V(x) <= 0  for all x in X_i

    Verified as:
        max_k Delta_V_lin(v_k) + 0.5 * M_i * r_i^2 <= 0  -> SAFE
        min_k Delta_V_lin(v_k) - 0.5 * M_i * r_i^2 >  0  -> UNSAFE
        otherwise                                           -> INCONCLUSIVE

Usage
-----
    from verify_certificates import verify_barrier, verify_lyapunov

    # Barrier (boundary cells only — fast, competes with Ren et al.)
    summary = verify_barrier(
        BC, sv, hyperplanes, W, b,
        barrier_model, dynamics_name="arch3"
    )

    # Lyapunov (all cells)
    summary = verify_lyapunov(
        enumerate_poly, sv_all, hyperplanes, W, b,
        lyapunov_model, A_d, dynamics_name="quadrotor"
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
except ImportError:
    from hessian_bound import HessianBounder, compute_local_gradient
    from Dynamics import load_dynamics


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

class Label(Enum):
    SAFE         = "SAFE"
    UNSAFE       = "UNSAFE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class CellResult:
    label      : Label
    cell_idx   : int
    M_i        : float
    r_i        : float
    remainder  : float
    max_condition : float   # max value of the condition at vertices
    # positive = potentially violated, negative = satisfied


@dataclass
class VerificationSummary:
    mode          : str    # "barrier" or "lyapunov"
    dynamics_name : str
    n_safe        : int = 0
    n_unsafe      : int = 0
    n_inconclusive: int = 0
    runtime_s     : float = 0.0
    results       : List[CellResult] = field(default_factory=list)

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
        if self.n_unsafe > 0:
            print(f"\n  !! {self.n_unsafe} UNSAFE cells found.")
        elif self.n_inconclusive == 0:
            print(f"\n  Certificate VERIFIED over all {self.total} cells.")
        else:
            print(f"\n  Certificate verified on {self.n_safe} cells; "
                  f"{self.n_inconclusive} inconclusive (refine to resolve).")
        print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# Core: evaluate condition at vertices under linearized dynamics
# ═══════════════════════════════════════════════════════════════════════════

def _find_zero_crossings(
    vertices     : np.ndarray,   # (V, n)
    B_curr       : np.ndarray,   # (V,) B evaluated at each vertex
    centroid     : np.ndarray,   # (n,)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find all zero level set crossing points of B on cell edges.

    For each edge (v_u, v_v) where B changes sign, the exact crossing is:
        x* = v_u + t * (v_v - v_u)   where t = B(v_u) / (B(v_u) - B(v_v))

    This is equation (10) from the paper with B playing the role of P^q.

    Returns
    -------
    x_stars   : (E, n) array of zero crossing points
    distances : (E,)   array of ||x* - centroid|| for each crossing
    """
    from itertools import combinations
    x_stars   = []
    distances = []

    V = len(vertices)
    for u, v in combinations(range(V), 2):
        bu, bv = B_curr[u], B_curr[v]
        if bu * bv < 0:   # sign change on this edge
            t      = bu / (bu - bv)
            x_star = vertices[u] + t * (vertices[v] - vertices[u])
            x_stars.append(x_star)
            distances.append(float(np.linalg.norm(x_star - centroid)))

    if len(x_stars) == 0:
        return np.zeros((0, vertices.shape[1])), np.zeros(0)

    return np.array(x_stars), np.array(distances)


def _eval_barrier_at_vertices(
    vertices   : np.ndarray,      # (V, n)
    centroid   : np.ndarray,      # (n,)
    f_jac      : np.ndarray,      # (n, n) Jacobian of f at centroid
    f_val      : np.ndarray,      # (n,)   f(centroid)
    barrier_model,                # TorchScript network
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate B(f_lin(x*)) at zero level set crossing points.

    The barrier condition B(x) = 0 => B(f(x)) <= 0 only needs to hold
    on the zero level set {x : B(x) = 0} intersected with X_i.

    Since B is affine on X_i, the zero level set is a face whose vertices
    are the edge crossing points x* where B changes sign. Since B(f_lin(x))
    is affine in x, its maximum over this face is attained at these points.

    The remainder at each crossing point uses ||x* - centroid|| rather than
    the full cell radius r_i, giving a tighter bound.

    Parameters
    ----------
    vertices      : (V, n) all cell vertices
    centroid      : (n,)
    f_jac         : (n, n) Jacobian of f at centroid
    f_val         : (n,)   f(centroid)
    barrier_model : TorchScript network

    Returns
    -------
    B_next    : (E,) B(f_lin(x*)) at each zero crossing point
    distances : (E,) ||x* - centroid|| for per-point remainder
                Returns ([-1.0], [0.0]) if no crossings found (trivially safe).
    """
    model_dtype = next(barrier_model.parameters()).dtype

    # Step 1: evaluate B at all cell vertices
    with torch.no_grad():
        B_curr = barrier_model(
            torch.tensor(vertices.astype(np.float32), dtype=model_dtype)
        ).numpy().ravel()   # (V,)

    # Step 2: find zero crossing points on edges
    x_stars, distances = _find_zero_crossings(vertices, B_curr, centroid)

    if len(x_stars) == 0:
        # No zero crossings — cell does not straddle B=0
        # Should not happen for boundary cells, but handle gracefully
        return np.array([-1.0]), np.array([0.0])

    # Step 3: evaluate B(f_lin(x*)) at each crossing point
    delta    = x_stars - centroid              # (E, n)
    f_lin_xs = f_val + delta @ f_jac.T        # (E, n)

    with torch.no_grad():
        B_next = barrier_model(
            torch.tensor(f_lin_xs.astype(np.float32), dtype=model_dtype)
        ).numpy().ravel()   # (E,)

    return B_next, distances


def _eval_delta_V_at_vertices(
    vertices      : np.ndarray,   # (V, n)
    centroid      : np.ndarray,   # (n,)
    f_jac         : np.ndarray,   # (n, n)
    f_val         : np.ndarray,   # (n,)
    lyapunov_model,               # TorchScript network
) -> np.ndarray:
    """
    Evaluate Delta_V_lin(v_k) = V(f_lin(v_k)) - V(v_k) at all vertices.

    Returns array of shape (V,).
    """
    delta    = vertices - centroid
    f_lin_vk = f_val + delta @ f_jac.T      # (V, n)

    model_dtype = next(lyapunov_model.parameters()).dtype
    with torch.no_grad():
        V_next = lyapunov_model(
            torch.tensor(f_lin_vk.astype(np.float32), dtype=model_dtype)
        ).numpy().ravel()
        V_curr = lyapunov_model(
            torch.tensor(vertices.astype(np.float32),  dtype=model_dtype)
        ).numpy().ravel()

    return V_next - V_curr                   # (V,)


def _compute_jacobian(f_sym, symbols, centroid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute f(centroid) and Jacobian J_f at centroid numerically from
    precompiled lambdified functions.

    Parameters
    ----------
    f_sym    : list of sympy expressions (length n)
    symbols  : tuple of sympy symbols
    centroid : (n,) array

    Returns
    -------
    f_val  : (n,) array
    f_jac  : (n, n) array
    """
    import sympy as sp
    n = len(symbols)

    # Evaluate f at centroid
    subs = dict(zip(symbols, centroid))
    f_val = np.array([float(fi.subs(subs)) for fi in f_sym])

    # Jacobian J[i,j] = df_i/dx_j at centroid
    f_jac = np.zeros((n, n))
    for i, fi in enumerate(f_sym):
        for j, xj in enumerate(symbols):
            dfi_dxj = sp.diff(fi, xj)
            f_jac[i, j] = float(dfi_dxj.subs(subs))

    return f_val, f_jac


# ═══════════════════════════════════════════════════════════════════════════
# Per-cell verify
# ═══════════════════════════════════════════════════════════════════════════

def _verify_cell(
    condition_vals : np.ndarray,            # (E,) condition at zero crossings
    M_i            : float,                 # Hessian bound over full cell
    cell_idx       : int,
    distances      : Optional[np.ndarray] = None,  # (E,) ||x* - centroid||
    r_i            : float = 0.0,           # fallback if distances not provided
) -> CellResult:
    """
    Label one cell using per-point remainder bounds.

    For each zero crossing point x*_e:
        remainder_e = 0.5 * M_i * ||x*_e - centroid||^2

    The cell is SAFE if for all crossing points:
        B(f_lin(x*_e)) + remainder_e <= 0

    The cell is UNSAFE if for any crossing point:
        B(f_lin(x*_e)) - remainder_e > 0

    Note: M_i is computed over the full cell (not just crossing points)
    because the Taylor bound must hold everywhere, but the distances
    ||x* - centroid|| are specific to each crossing point — tighter
    than the full cell radius r_i.

    Parameters
    ----------
    condition_vals : B(f_lin(x*)) at each zero crossing point
    M_i            : Hessian bound over full cell X_i
    cell_idx       : index for reporting
    distances      : ||x* - centroid|| per crossing point (tighter remainder)
    r_i            : fallback radius if distances not provided
    """
    if distances is None or len(distances) == 0:
        distances = np.full(len(condition_vals), r_i)

    # Per-point remainders — tighter than using global r_i
    remainders = 0.5 * M_i * distances ** 2   # (E,)

    max_cond = float(condition_vals.max())
    min_cond = float(condition_vals.min())

    # Worst-case check: is max(B_next + remainder) <= 0?
    worst_safe   = float((condition_vals + remainders).max())
    # Best-case check: is min(B_next - remainder) > 0?
    best_unsafe  = float((condition_vals - remainders).min())

    if worst_safe <= 0.0:
        label = Label.SAFE
    elif best_unsafe > 0.0:
        label = Label.UNSAFE
    else:
        label = Label.INCONCLUSIVE

    # Report the tightest remainder for summary
    remainder = float(remainders.max())

    return CellResult(
        label         = label,
        cell_idx      = cell_idx,
        M_i           = M_i,
        r_i           = float(distances.max()),
        remainder     = remainder,
        max_condition = max_cond,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Public API: barrier certificate verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_barrier(
    BC             : list,
    sv             : np.ndarray,
    hyperplanes    : list,
    W_out          : np.ndarray,
    b              : list,
    barrier_model,
    dynamics_name  : str,
    use_fast_bound : bool = True,
    early_exit     : bool = True,
    max_refine_depth : int = 8,    # max bisection depth for INCONCLUSIVE cells
) -> VerificationSummary:
    """
    Formally verify B(f(x)) <= 0 on all boundary-adjacent cells.

    After the main pass, INCONCLUSIVE cells are refined by bisection toward
    the furthest vertex. Each bisection reduces r_i by half, cutting the
    remainder by factor 4. Refinement continues until SAFE/UNSAFE or
    max_refine_depth is reached.

    Parameters
    ----------
    BC               : list of (V_k, n) vertex arrays — boundary cells
    sv               : (N_b, total_neurons) activation patterns for BC
    hyperplanes      : list of weight matrices
    W_out            : output layer weights
    b                : list of bias vectors
    barrier_model    : TorchScript network
    dynamics_name    : system name registered in Dynamics.py
    use_fast_bound   : use numpy-only bound first, fall back to full bound
    early_exit       : stop immediately when first UNSAFE cell found
    max_refine_depth : max bisection depth per INCONCLUSIVE cell

    Returns
    -------
    VerificationSummary
    """
    symbols, f_sym = load_dynamics(dynamics_name)
    hb             = HessianBounder(symbols, f_sym)
    summary        = VerificationSummary(
        mode="barrier", dynamics_name=dynamics_name
    )

    print(f"\nBarrier verification: {len(BC)} boundary cells, "
          f"dynamics='{dynamics_name}'")
    t0 = time.perf_counter()

    # Track INCONCLUSIVE cells for refinement
    inconclusive_cells = []   # list of (vertices, sv_i, p_i, cell_idx)

    for i, vertices in enumerate(BC):
        vertices  = np.asarray(vertices, dtype=float)
        sv_i      = sv[i].ravel()
        p_i       = compute_local_gradient(sv_i, hyperplanes, W_out)
        r_i, centroid = hb.cell_radius(vertices)

        f_val, f_jac = _compute_jacobian(f_sym, symbols, centroid)
        cond_vals, distances = _eval_barrier_at_vertices(
            vertices, centroid, f_jac, f_val, barrier_model
        )

        # Debug: print first 3 cells
        if i < 3:
            print(f"  [debug cell {i}] "
                  f"crossings={len(distances)} "
                  f"B_next: min={cond_vals.min():.4f} max={cond_vals.max():.4f} "
                  f"dist: min={distances.min():.4f} max={distances.max():.4f} "
                  f"r_i={r_i:.4f}")

        if use_fast_bound:
            M      = hb.bound_fast(p_i, vertices)
            result = _verify_cell(cond_vals, M, i, distances)
            if result.label == Label.INCONCLUSIVE:
                M      = hb.bound(p_i, vertices)
                result = _verify_cell(cond_vals, M, i, distances)
        else:
            M      = hb.bound(p_i, vertices)
            result = _verify_cell(cond_vals, M, i, distances)

        summary.results.append(result)
        if result.label == Label.SAFE:
            summary.n_safe += 1
        elif result.label == Label.UNSAFE:
            summary.n_unsafe += 1
            if early_exit:
                print(f"\n  !! UNSAFE cell found at index {i} — stopping early.")
                print(f"     max B(f(x*)) = {result.max_condition:.6f}")
                print(f"     remainder    = {result.remainder:.6f}")
                break
        else:
            summary.n_inconclusive += 1
            inconclusive_cells.append((vertices, sv_i, p_i, i))

        if (i + 1) % 50 == 0 or (i + 1) == len(BC):
            print(f"  [{i+1}/{len(BC)}] "
                  f"safe={summary.n_safe} "
                  f"unsafe={summary.n_unsafe} "
                  f"inconclusive={summary.n_inconclusive}")

    # ── Refinement pass for INCONCLUSIVE cells ──────────────────────────────
    if inconclusive_cells and summary.n_unsafe == 0:
        print(f"\n  Refining {len(inconclusive_cells)} INCONCLUSIVE cells "
              f"(max depth={max_refine_depth})...")
        resolved_safe   = 0
        resolved_unsafe = 0
        still_inconclusive = 0

        for vertices, sv_i, p_i, cell_idx in inconclusive_cells:
            label = _refine_inconclusive_barrier(
                vertices, sv_i, p_i, hb, f_sym, symbols,
                barrier_model, max_refine_depth, use_fast_bound
            )
            if label == Label.SAFE:
                resolved_safe += 1
                summary.n_safe += 1
                summary.n_inconclusive -= 1
            elif label == Label.UNSAFE:
                resolved_unsafe += 1
                summary.n_unsafe += 1
                summary.n_inconclusive -= 1
                if early_exit:
                    print(f"\n  !! UNSAFE cell found during refinement "
                          f"(original cell {cell_idx}) — stopping.")
                    break
            else:
                still_inconclusive += 1

        print(f"  Refinement complete: "
              f"resolved_safe={resolved_safe} "
              f"resolved_unsafe={resolved_unsafe} "
              f"still_inconclusive={still_inconclusive}")

    summary.runtime_s = time.perf_counter() - t0
    summary.print_summary()
    return summary

    summary.runtime_s = time.perf_counter() - t0
    summary.print_summary()
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Cell refinement
# ═══════════════════════════════════════════════════════════════════════════

def _split_cell(
    vertices : np.ndarray,   # (V, n)
    d        : np.ndarray,   # (n,) cutting direction
    offset   : float,        # hyperplane: d^T x = offset
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a polytope along the hyperplane d^T x = offset.

    Vertices on the positive side (d^T x >= offset) go to child_pos.
    Vertices on the negative side (d^T x <= offset) go to child_neg.
    New vertices are interpolated on crossing edges — exactly as in
    Stage 3 of Algorithm 1 in the paper.

    Returns
    -------
    child_pos : (V+, n) array
    child_neg : (V-, n) array
    """
    signed = vertices @ d - offset    # (V,) signed distances
    V, n   = vertices.shape

    pos_mask = signed >= 0
    neg_mask = signed <= 0

    new_verts = []
    for u in range(V):
        for v in range(u + 1, V):
            if signed[u] * signed[v] < 0:   # sign change — edge crosses plane
                t      = signed[u] / (signed[u] - signed[v])
                x_star = vertices[u] + t * (vertices[v] - vertices[u])
                new_verts.append(x_star)

    if new_verts:
        new_arr   = np.array(new_verts)
        child_pos = np.vstack([vertices[pos_mask], new_arr])
        child_neg = np.vstack([vertices[neg_mask], new_arr])
    else:
        child_pos = vertices[pos_mask]
        child_neg = vertices[neg_mask]

    return child_pos, child_neg


def _refine_inconclusive_barrier(
    vertices      : np.ndarray,   # (V, n)
    sv_i          : np.ndarray,   # activation pattern (inherited unchanged)
    p_i           : np.ndarray,   # local gradient (inherited unchanged)
    hb            : HessianBounder,
    f_sym         : list,
    symbols       : tuple,
    barrier_model,
    max_depth     : int,
    use_fast_bound: bool,
) -> Label:
    """
    Recursively bisect an INCONCLUSIVE barrier cell until SAFE or UNSAFE.

    Cutting strategy: bisect along the direction from centroid to the
    furthest vertex. This halves r_i at each step, reducing the remainder
    by factor 4 per bisection regardless of M_i tightness.

    The sign vector sv_i and local gradient p_i are inherited unchanged —
    the split is verification-only, not part of the network partition.

    Returns the resolved label (SAFE or UNSAFE) or INCONCLUSIVE if
    max_depth is reached without resolution.
    """
    if max_depth == 0:
        return Label.INCONCLUSIVE

    # Find cutting hyperplane: toward furthest vertex
    r_i, centroid = hb.cell_radius(vertices)
    dists  = np.linalg.norm(vertices - centroid, axis=1)
    v_star = vertices[np.argmax(dists)]
    d      = v_star - centroid
    offset = centroid @ d + 0.5 * (d @ d)   # midpoint along d

    child_pos, child_neg = _split_cell(vertices, d, offset)

    labels = []
    for child in [child_pos, child_neg]:
        if len(child) < child.shape[1]:
            # Degenerate child — skip
            labels.append(Label.SAFE)
            continue

        # Recompute condition on child
        r_child, centroid_child = hb.cell_radius(child)
        f_val, f_jac = _compute_jacobian(f_sym, symbols, centroid_child)
        cond_vals, distances = _eval_barrier_at_vertices(
            child, centroid_child, f_jac, f_val, barrier_model
        )

        if use_fast_bound:
            M = hb.bound_fast(p_i, child)
            result = _verify_cell(cond_vals, M, -1, distances)
            if result.label == Label.INCONCLUSIVE:
                M      = hb.bound(p_i, child)
                result = _verify_cell(cond_vals, M, -1, distances)
        else:
            M      = hb.bound(p_i, child)
            result = _verify_cell(cond_vals, M, -1, distances)

        if result.label == Label.INCONCLUSIVE:
            # Recurse
            result_label = _refine_inconclusive_barrier(
                child, sv_i, p_i, hb, f_sym, symbols,
                barrier_model, max_depth - 1, use_fast_bound
            )
        else:
            result_label = result.label

        labels.append(result_label)

        # Early exit on UNSAFE
        if result_label == Label.UNSAFE:
            return Label.UNSAFE

    # Both children must be SAFE for the parent to be SAFE
    if all(l == Label.SAFE for l in labels):
        return Label.SAFE
    if any(l == Label.UNSAFE for l in labels):
        return Label.UNSAFE
    return Label.INCONCLUSIVE


def _refine_inconclusive_lyapunov(
    vertices      : np.ndarray,
    sv_i          : np.ndarray,
    p_i           : np.ndarray,
    hb            : HessianBounder,
    f_sym         : list,
    symbols       : tuple,
    lyapunov_model,
    max_depth     : int,
    use_fast_bound: bool,
) -> Label:
    """
    Recursively bisect an INCONCLUSIVE Lyapunov cell until SAFE or UNSAFE.

    Same cutting strategy as barrier refinement — toward furthest vertex.
    Delta_V condition checked on all vertices of each child cell.
    """
    if max_depth == 0:
        return Label.INCONCLUSIVE

    r_i, centroid = hb.cell_radius(vertices)
    dists  = np.linalg.norm(vertices - centroid, axis=1)
    v_star = vertices[np.argmax(dists)]
    d      = v_star - centroid
    offset = centroid @ d + 0.5 * (d @ d)

    child_pos, child_neg = _split_cell(vertices, d, offset)

    labels = []
    for child in [child_pos, child_neg]:
        if len(child) < child.shape[1]:
            labels.append(Label.SAFE)
            continue

        r_child, centroid_child = hb.cell_radius(child)
        f_val, f_jac = _compute_jacobian(f_sym, symbols, centroid_child)
        cond_vals = _eval_delta_V_at_vertices(
            child, centroid_child, f_jac, f_val, lyapunov_model
        )

        if use_fast_bound:
            M = hb.bound_fast(p_i, child)
            result = _verify_cell(cond_vals, M, -1,
                                  distances=np.full(len(cond_vals), r_child))
            if result.label == Label.INCONCLUSIVE:
                M      = hb.bound(p_i, child)
                result = _verify_cell(cond_vals, M, -1,
                                      distances=np.full(len(cond_vals), r_child))
        else:
            M      = hb.bound(p_i, child)
            result = _verify_cell(cond_vals, M, -1,
                                  distances=np.full(len(cond_vals), r_child))

        if result.label == Label.INCONCLUSIVE:
            result_label = _refine_inconclusive_lyapunov(
                child, sv_i, p_i, hb, f_sym, symbols,
                lyapunov_model, max_depth - 1, use_fast_bound
            )
        else:
            result_label = result.label

        labels.append(result_label)

        if result_label == Label.UNSAFE:
            return Label.UNSAFE

    if all(l == Label.SAFE for l in labels):
        return Label.SAFE
    if any(l == Label.UNSAFE for l in labels):
        return Label.UNSAFE
    return Label.INCONCLUSIVE




def verify_lyapunov(
    enumerate_poly   : list,
    sv_all           : np.ndarray,
    hyperplanes      : list,
    W_out            : np.ndarray,
    b                : list,
    lyapunov_model,
    dynamics_name    : str,
    use_fast_bound   : bool = True,
    early_exit       : bool = True,
    max_refine_depth : int = 8,
) -> VerificationSummary:
    """
    Formally verify Delta_V(x) = V(f(x)) - V(x) <= 0 on all cells.

    After the main pass, INCONCLUSIVE cells are refined by bisection.
    Each bisection halves r_i, reducing the remainder by factor 4.

    Parameters
    ----------
    enumerate_poly   : list of (V_k, n) vertex arrays — all enumerated cells
    sv_all           : (N, total_neurons) activation patterns
    hyperplanes      : list of weight matrices
    W_out            : output layer weights
    b                : list of bias vectors
    lyapunov_model   : TorchScript network
    dynamics_name    : system name registered in Dynamics.py
    use_fast_bound   : use bound_fast first, fall back to full bound
    early_exit       : stop immediately when first UNSAFE cell found
    max_refine_depth : max bisection depth per INCONCLUSIVE cell

    Returns
    -------
    VerificationSummary
    """
    symbols, f_sym = load_dynamics(dynamics_name)
    hb             = HessianBounder(symbols, f_sym)
    summary        = VerificationSummary(
        mode="lyapunov", dynamics_name=dynamics_name
    )

    print(f"\nLyapunov verification: {len(enumerate_poly)} cells, "
          f"dynamics='{dynamics_name}'")
    t0 = time.perf_counter()

    inconclusive_cells = []   # (vertices, sv_i, p_i, cell_idx)

    for i, vertices in enumerate(enumerate_poly):
        vertices  = np.asarray(vertices, dtype=float)
        sv_i      = sv_all[i].ravel()
        p_i       = compute_local_gradient(sv_i, hyperplanes, W_out)
        r_i, centroid = hb.cell_radius(vertices)

        f_val, f_jac = _compute_jacobian(f_sym, symbols, centroid)
        cond_vals    = _eval_delta_V_at_vertices(
            vertices, centroid, f_jac, f_val, lyapunov_model
        )

        if use_fast_bound:
            M      = hb.bound_fast(p_i, vertices)
            result = _verify_cell(cond_vals, M, i,
                                  distances=np.full(len(cond_vals), r_i))
            if result.label == Label.INCONCLUSIVE:
                M      = hb.bound(p_i, vertices)
                result = _verify_cell(cond_vals, M, i,
                                      distances=np.full(len(cond_vals), r_i))
        else:
            M      = hb.bound(p_i, vertices)
            result = _verify_cell(cond_vals, M, i,
                                  distances=np.full(len(cond_vals), r_i))

        summary.results.append(result)
        if result.label == Label.SAFE:
            summary.n_safe += 1
        elif result.label == Label.UNSAFE:
            summary.n_unsafe += 1
            if early_exit:
                print(f"\n  !! UNSAFE cell found at index {i} — stopping early.")
                print(f"     max Delta_V(v_k) = {result.max_condition:.6f}")
                print(f"     remainder        = {result.remainder:.6f}")
                break
        else:
            summary.n_inconclusive += 1
            inconclusive_cells.append((vertices, sv_i, p_i, i))

        if (i + 1) % 1000 == 0 or (i + 1) == len(enumerate_poly):
            elapsed = time.perf_counter() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(enumerate_poly) - i - 1) / rate
            print(f"  [{i+1}/{len(enumerate_poly)}] "
                  f"safe={summary.n_safe} "
                  f"unsafe={summary.n_unsafe} "
                  f"inconclusive={summary.n_inconclusive} "
                  f"| {rate:.0f} cells/s | ETA {eta:.0f}s")

    # ── Refinement pass for INCONCLUSIVE cells ──────────────────────────────
    if inconclusive_cells and summary.n_unsafe == 0:
        print(f"\n  Refining {len(inconclusive_cells)} INCONCLUSIVE cells "
              f"(max depth={max_refine_depth})...")
        resolved_safe      = 0
        resolved_unsafe    = 0
        still_inconclusive = 0

        for idx, (vertices, sv_i, p_i, cell_idx) in \
                enumerate(inconclusive_cells):
            label = _refine_inconclusive_lyapunov(
                vertices, sv_i, p_i, hb, f_sym, symbols,
                lyapunov_model, max_refine_depth, use_fast_bound
            )
            if label == Label.SAFE:
                resolved_safe += 1
                summary.n_safe += 1
                summary.n_inconclusive -= 1
            elif label == Label.UNSAFE:
                resolved_unsafe += 1
                summary.n_unsafe += 1
                summary.n_inconclusive -= 1
                if early_exit:
                    print(f"\n  !! UNSAFE cell found during refinement "
                          f"(original cell {cell_idx}) — stopping.")
                    break
            else:
                still_inconclusive += 1

            if (idx + 1) % 100 == 0:
                print(f"  [refine {idx+1}/{len(inconclusive_cells)}] "
                      f"safe={resolved_safe} "
                      f"unsafe={resolved_unsafe} "
                      f"inconclusive={still_inconclusive}")

        print(f"  Refinement complete: "
              f"resolved_safe={resolved_safe} "
              f"resolved_unsafe={resolved_unsafe} "
              f"still_inconclusive={still_inconclusive}")

    summary.runtime_s = time.perf_counter() - t0
    summary.print_summary()
    return summary

    summary.runtime_s = time.perf_counter() - t0
    summary.print_summary()
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from dynamics import load_dynamics
    from hessian_bound import HessianBounder, compute_local_gradient

    print("Smoke test: _verify_cell labels")

    # SAFE: max(cond + remainder) <= 0
    # distances = [0.1, 0.1, 0.1], M_i=1.0 -> remainders = 0.5*1.0*0.01 = 0.005
    r = _verify_cell(np.array([-0.5, -0.3, -0.1]), M_i=1.0, cell_idx=0,
                     distances=np.array([0.1, 0.1, 0.1]))
    assert r.label == Label.SAFE, f"Expected SAFE, got {r.label}"
    print(f"  SAFE case:         {r.label}  max_cond={r.max_condition:.2f}  rem={r.remainder:.4f}")

    # UNSAFE: min(cond - remainder) > 0
    r = _verify_cell(np.array([0.5, 0.8, 1.0]), M_i=1.0, cell_idx=1,
                     distances=np.array([0.1, 0.1, 0.1]))
    assert r.label == Label.UNSAFE, f"Expected UNSAFE, got {r.label}"
    print(f"  UNSAFE case:       {r.label}  max_cond={r.max_condition:.2f}  rem={r.remainder:.4f}")

    # INCONCLUSIVE: straddles zero within remainder
    # distances = [0.5, 0.5], M_i=1.0 -> remainders = 0.5*1.0*0.25 = 0.125
    r = _verify_cell(np.array([-0.05, 0.05]), M_i=1.0, cell_idx=2,
                     distances=np.array([0.5, 0.5]))
    assert r.label == Label.INCONCLUSIVE, f"Expected INCONCLUSIVE, got {r.label}"
    print(f"  INCONCLUSIVE case: {r.label}  max_cond={r.max_condition:.2f}  rem={r.remainder:.4f}")

    print("\nSmoke test: _compute_jacobian on Arch3")
    symbols, f_sym = load_dynamics("arch3")
    centroid = np.array([0.5, 0.3])
    f_val, f_jac = _compute_jacobian(f_sym, symbols, centroid)
    print(f"  f(centroid) = {f_val}")
    print(f"  J_f shape   = {f_jac.shape}")
    assert f_jac.shape == (2, 2)

    print("\nSmoke test: _split_cell")
    verts = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    d = np.array([1., 0.])   # cut at x=0.5
    pos, neg = _split_cell(verts, d, 0.5)
    assert pos.shape[1] == 2 and neg.shape[1] == 2
    assert (pos[:, 0] >= 0.5 - 1e-10).all(), f"pos side wrong: {pos}"
    assert (neg[:, 0] <= 0.5 + 1e-10).all(), f"neg side wrong: {neg}"
    print(f"  pos vertices: {pos.shape[0]}  neg vertices: {neg.shape[0]}")

    print("\nAll smoke tests passed.")
