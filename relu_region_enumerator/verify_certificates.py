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
    from dynamics import load_dynamics


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

def _eval_barrier_at_vertices(
    vertices   : np.ndarray,      # (V, n)
    centroid   : np.ndarray,      # (n,)
    f_jac      : np.ndarray,      # (n, n) Jacobian of f at centroid
    f_val      : np.ndarray,      # (n,)   f(centroid)
    barrier_model,                # TorchScript network
) -> np.ndarray:
    """
    Evaluate B(f_lin(v_k)) only at vertices where B(v_k) <= 0.

    The barrier condition is:
        B(x) <= 0  =>  B(f(x)) <= 0

    This only needs to hold in the unsafe region (B(x) <= 0).
    Vertices with B(v_k) > 0 are in the safe region — no check needed.

    f_lin(x) = f(centroid) + J_f @ (x - centroid)

    Returns
    -------
    B_next : array of B(f_lin(v_k)) for vertices where B(v_k) <= 0.
             Returns [-1.0] (trivially safe) if no vertices are in
             the unsafe region.
    """
    model_dtype = next(barrier_model.parameters()).dtype

    # Step 1: evaluate B at current vertices
    with torch.no_grad():
        B_curr = barrier_model(
            torch.tensor(vertices.astype(np.float32), dtype=model_dtype)
        ).numpy().ravel()   # (V,)

    # Step 2: filter to vertices in unsafe region (B(x) <= 0)
    mask = B_curr <= 0.0
    if not mask.any():
        # All vertices are in safe region — condition trivially holds
        return np.array([-1.0])

    unsafe_verts = vertices[mask]              # (V_unsafe, n)
    delta        = unsafe_verts - centroid     # (V_unsafe, n)
    f_lin_vk     = f_val + delta @ f_jac.T    # (V_unsafe, n)

    # Step 3: evaluate B(f_lin(v_k)) at unsafe vertices
    with torch.no_grad():
        B_next = barrier_model(
            torch.tensor(f_lin_vk.astype(np.float32), dtype=model_dtype)
        ).numpy().ravel()   # (V_unsafe,)

    return B_next


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
    condition_vals : np.ndarray,   # (V,) condition at vertices
    remainder      : float,
    M_i            : float,
    r_i            : float,
    cell_idx       : int,
) -> CellResult:
    """
    Label one cell given condition values at vertices and remainder bound.

    SAFE        : max(condition_vals) + remainder <= 0
    UNSAFE      : min(condition_vals) - remainder >  0
    INCONCLUSIVE: neither
    """
    max_cond = float(condition_vals.max())
    min_cond = float(condition_vals.min())

    if max_cond + remainder <= 0.0:
        label = Label.SAFE
    elif min_cond - remainder > 0.0:
        label = Label.UNSAFE
    else:
        label = Label.INCONCLUSIVE

    return CellResult(
        label         = label,
        cell_idx      = cell_idx,
        M_i           = M_i,
        r_i           = r_i,
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
    early_exit     : bool = True,   # stop immediately on first UNSAFE cell
) -> VerificationSummary:
    """
    Formally verify B(f(x)) <= 0 on all boundary-adjacent cells.

    This is strictly stronger than Ren et al.'s geometric boundary check:
    we certify the decrease condition under the true nonlinear dynamics.

    Parameters
    ----------
    BC             : list of (V_k, n) vertex arrays — boundary cells
    sv             : (N_b, total_neurons) activation patterns for BC
    hyperplanes    : list of weight matrices
    W_out          : output layer weights
    b              : list of bias vectors
    barrier_model  : TorchScript network
    dynamics_name  : system name registered in Dynamics.py
    use_fast_bound : use numpy-only bound first, fall back to full bound
    early_exit     : if True, stop immediately when first UNSAFE cell found

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

    for i, vertices in enumerate(BC):
        vertices  = np.asarray(vertices, dtype=float)
        sv_i      = sv[i].ravel()
        p_i       = compute_local_gradient(sv_i, hyperplanes, W_out)
        r_i, centroid = hb.cell_radius(vertices)

        # Jacobian of dynamics at centroid
        f_val, f_jac = _compute_jacobian(f_sym, symbols, centroid)

        # Condition values at vertices: B(f_lin(v_k)) for unsafe vertices only
        cond_vals = _eval_barrier_at_vertices(
            vertices, centroid, f_jac, f_val, barrier_model
        )

        # Debug: print first 3 cells to sanity-check values
        if i < 3:
            model_dtype = next(barrier_model.parameters()).dtype
            with torch.no_grad():
                B_curr = barrier_model(
                    torch.tensor(vertices.astype(np.float32), dtype=model_dtype)
                ).numpy().ravel()
            print(f"  [debug cell {i}] "
                  f"B_curr: min={B_curr.min():.4f} max={B_curr.max():.4f} "
                  f"unsafe_verts={int((B_curr<=0).sum())}/{len(B_curr)} "
                  f"B_next: min={cond_vals.min():.4f} max={cond_vals.max():.4f} "
                  f"r_i={r_i:.4f}")

        # Hessian bound — fast first if requested
        if use_fast_bound:
            M_fast    = hb.bound_fast(p_i, vertices)
            remainder = 0.5 * M_fast * r_i ** 2
            result    = _verify_cell(cond_vals, remainder, M_fast, r_i, i)

            # Only call full bound if inconclusive
            if result.label == Label.INCONCLUSIVE:
                M_i       = hb.bound(p_i, vertices)
                remainder = 0.5 * M_i * r_i ** 2
                result    = _verify_cell(cond_vals, remainder, M_i, r_i, i)
        else:
            M_i       = hb.bound(p_i, vertices)
            remainder = 0.5 * M_i * r_i ** 2
            result    = _verify_cell(cond_vals, remainder, M_i, r_i, i)

        summary.results.append(result)
        if result.label == Label.SAFE:
            summary.n_safe += 1
        elif result.label == Label.UNSAFE:
            summary.n_unsafe += 1
            if early_exit:
                print(f"\n  !! UNSAFE cell found at index {i} — stopping early.")
                print(f"     max B(f(v_k)) = {result.max_condition:.6f}")
                print(f"     remainder     = {result.remainder:.6f}")
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


# ═══════════════════════════════════════════════════════════════════════════
# Public API: Lyapunov decrease verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_lyapunov(
    enumerate_poly : list,
    sv_all         : np.ndarray,
    hyperplanes    : list,
    W_out          : np.ndarray,
    b              : list,
    lyapunov_model,
    dynamics_name  : str,
    use_fast_bound : bool = True,
    early_exit     : bool = True,   # stop immediately on first UNSAFE cell
) -> VerificationSummary:
    """
    Formally verify Delta_V(x) = V(f(x)) - V(x) <= 0 on all cells.

    Parameters
    ----------
    enumerate_poly : list of (V_k, n) vertex arrays — all enumerated cells
    sv_all         : (N, total_neurons) activation patterns
    hyperplanes    : list of weight matrices
    W_out          : output layer weights
    b              : list of bias vectors
    lyapunov_model : TorchScript network
    dynamics_name  : system name registered in Dynamics.py
    use_fast_bound : use bound_fast first, fall back to full bound
    early_exit     : if True, stop immediately when first UNSAFE cell found

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

    for i, vertices in enumerate(enumerate_poly):
        vertices  = np.asarray(vertices, dtype=float)
        sv_i      = sv_all[i].ravel()
        p_i       = compute_local_gradient(sv_i, hyperplanes, W_out)
        r_i, centroid = hb.cell_radius(vertices)

        # Jacobian of dynamics at centroid
        f_val, f_jac = _compute_jacobian(f_sym, symbols, centroid)

        # Condition values: Delta_V_lin(v_k) = V(f_lin(v_k)) - V(v_k)
        cond_vals = _eval_delta_V_at_vertices(
            vertices, centroid, f_jac, f_val, lyapunov_model
        )

        # Hessian bound
        if use_fast_bound:
            M_fast    = hb.bound_fast(p_i, vertices)
            remainder = 0.5 * M_fast * r_i ** 2
            result    = _verify_cell(cond_vals, remainder, M_fast, r_i, i)
            if result.label == Label.INCONCLUSIVE:
                M_i       = hb.bound(p_i, vertices)
                remainder = 0.5 * M_i * r_i ** 2
                result    = _verify_cell(cond_vals, remainder, M_i, r_i, i)
        else:
            M_i       = hb.bound(p_i, vertices)
            remainder = 0.5 * M_i * r_i ** 2
            result    = _verify_cell(cond_vals, remainder, M_i, r_i, i)

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
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from Dynamics import load_dynamics
    from hessian_bound import HessianBounder, compute_local_gradient

    print("Smoke test: _verify_cell labels")

    # SAFE: max_cond + remainder <= 0
    r = _verify_cell(np.array([-0.5, -0.3, -0.1]), remainder=0.05, M_i=1.0, r_i=0.1, cell_idx=0)
    assert r.label == Label.SAFE, f"Expected SAFE, got {r.label}"
    print(f"  SAFE case:         {r.label}  max_cond={r.max_condition:.2f}  rem={r.remainder:.2f}")

    # UNSAFE: min_cond - remainder > 0
    r = _verify_cell(np.array([0.5, 0.8, 1.0]), remainder=0.05, M_i=1.0, r_i=0.1, cell_idx=1)
    assert r.label == Label.UNSAFE, f"Expected UNSAFE, got {r.label}"
    print(f"  UNSAFE case:       {r.label}  max_cond={r.max_condition:.2f}  rem={r.remainder:.2f}")

    # INCONCLUSIVE: straddles zero within remainder
    r = _verify_cell(np.array([-0.05, 0.05]), remainder=0.1, M_i=1.0, r_i=0.1, cell_idx=2)
    assert r.label == Label.INCONCLUSIVE, f"Expected INCONCLUSIVE, got {r.label}"
    print(f"  INCONCLUSIVE case: {r.label}  max_cond={r.max_condition:.2f}  rem={r.remainder:.2f}")

    print("\nSmoke test: _compute_jacobian on Arch3")
    symbols, f_sym = load_dynamics("arch3")
    centroid = np.array([0.5, 0.3])
    f_val, f_jac = _compute_jacobian(f_sym, symbols, centroid)
    print(f"  f(centroid) = {f_val}")
    print(f"  J_f shape   = {f_jac.shape}")
    assert f_jac.shape == (2, 2)

    print("\nAll smoke tests passed.")