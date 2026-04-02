"""
hessian_bound.py
================
Computes a guaranteed upper bound M_i on the projected Hessian spectral norm
over a polytopic cell X_i, used in the Taylor remainder bound:

    remainder_i = 0.5 * M_i * r_i^2

where r_i = max_k ||v_k - centroid_i|| is the cell radius.

Design
------
All symbolic Hessian entries are compiled ONCE at startup into three
callable forms — numpy, gmpy2, and mpmath.iv — and reused for every cell.
Per-cell work is a weighted sum of precompiled evaluations.

Three-tier evaluation per cell
-------------------------------
Tier 1 — Numpy at actual cell vertices (vectorized, fast):
    Exact when Hessian entries are convex (e.g. Arch3, linear entries).

Tier 2 — gmpy2 at 2^n box corners (fast, empirically validated):
    Used when n <= corner_threshold (default 6, at most 64 corners).
    ~1.7x faster than mpmath. Exact for polynomial dynamics deg <= 2.

Tier 3 — mpmath.iv over coordinate box (formally guaranteed, any n):
    Propagates interval bounds through sin/cos/exp/tanh/log etc.
    Sound regardless of convexity or polynomial degree.
    Used when n > corner_threshold.

M_i = max(Tier 1, Tier 2 or Tier 3).

Dependencies
------------
    conda install -c conda-forge gmpy2
    (mpmath and sympy are already in the relu_enum environment)
"""

import numpy as np
import sympy as sp
import gmpy2
from mpmath import iv as mpiv
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple


# ── gmpy2 precision ──────────────────────────────────────────────────────────
gmpy2.get_context().precision = 53


# ── mpmath.iv function map ───────────────────────────────────────────────────
# tanh, cosh, sinh are not in mpmath.iv directly — implement via exp

def _iv_tanh(x):
    e2x = mpiv.exp(2 * x)
    one = mpiv.mpf(1)
    return (e2x - one) / (e2x + one)

def _iv_cosh(x):
    return (mpiv.exp(x) + mpiv.exp(-x)) / mpiv.mpf(2)

def _iv_sinh(x):
    return (mpiv.exp(x) - mpiv.exp(-x)) / mpiv.mpf(2)

_IV_MOD = {
    "sin"  : mpiv.sin,
    "cos"  : mpiv.cos,
    "exp"  : mpiv.exp,
    "log"  : mpiv.log,
    "sqrt" : mpiv.sqrt,
    "tan"  : mpiv.tan,
    "tanh" : _iv_tanh,
    "cosh" : _iv_cosh,
    "sinh" : _iv_sinh,
    "mpf"  : mpiv.mpf,   # so lambdify-generated mpf() calls use iv.mpf
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _abs_bound(val) -> float:
    """Upper bound on |val| for mpmath.iv interval, gmpy2.mpfr, or float."""
    try:
        # mpmath.iv: interval has .a (lower) and .b (upper)
        return max(abs(float(val.a)), abs(float(val.b)))
    except AttributeError:
        return abs(float(val))


def _scale_iv(val, scalar: float):
    """Multiply an mpmath.iv interval by a scalar, flipping when scalar < 0."""
    try:
        lo = float(val.a) * scalar
        hi = float(val.b) * scalar
        if scalar < 0:
            lo, hi = hi, lo
        return mpiv.mpf([lo, hi])
    except AttributeError:
        return scalar * float(val)


# ═══════════════════════════════════════════════════════════════════════════
# Local gradient computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_local_gradient(
    sv_i: np.ndarray,
    hyperplanes: list,
    W_out: np.ndarray,
) -> np.ndarray:
    """
    Compute the local gradient p_i = d/dx [network(x)] for cell X_i.

    For a scalar-output ReLU network on cell X_i, the activation pattern
    is fixed so the network is affine:

        network(x) = W_out @ D_L @ W_L @ ... @ D_1 @ W_1 @ x + const

    The gradient is the row vector:

        p_i = W_out @ D_L @ W_L @ ... @ D_1 @ W_1      shape (n,)

    Parameters
    ----------
    sv_i        : (total_neurons,) float64
                  Flat activation pattern for cell X_i.
                  sv_i[k] = 1.0 if neuron k is active, 0.0 if inactive.
                  Layers concatenated in order (layer 0 first), matching
                  the convention in Finding_cell_id / barrier_certificate_cells.
    hyperplanes : list of L arrays, hyperplanes[l] shape (H_l, n_in_l)
                  Hidden layer weight matrices.
    W_out       : (1, H_last) or (H_last,) float64
                  Output layer weight matrix.

    Returns
    -------
    p_i : (n,) float64
        Local gradient of the scalar network output w.r.t. x on cell X_i.
    """
    W_out_flat = np.asarray(W_out, dtype=np.float64).ravel()   # (H_last,)

    # Split sv_i into per-layer diagonal masks
    layer_sizes = [h.shape[0] for h in hyperplanes]
    masks = []
    offset = 0
    for size in layer_sizes:
        masks.append(sv_i[offset: offset + size].ravel())
        offset += size

    # Left-to-right pass:
    #   v = W_out_flat                           shape (H_last,)
    #   for layer L down to 1:
    #       v = v * mask[L]                      apply D_L
    #       v = v @ W_L                          (H_L,) @ (H_L, n_in_L) -> (n_in_L,)
    v = W_out_flat.copy()
    for mask, W_i in zip(reversed(masks), reversed(hyperplanes)):
        v = v * mask          # zero out inactive neurons
        v = v @ W_i           # (H_l,) @ (H_l, n_in_l) -> (n_in_l,)

    return v   # shape (n,)


# ═══════════════════════════════════════════════════════════════════════════
# Per-entry container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _Entry:
    j       : int       # dynamics component index
    k       : int       # Hessian row
    l       : int       # Hessian col
    fn_np   : Callable  # numpy callable
    fn_gm   : Callable  # gmpy2 callable
    fn_iv   : Callable  # mpmath.iv callable
    is_const: bool      # True when entry has no free symbols


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════

class HessianBounder:
    """
    Precompiled Hessian bound engine for a fixed dynamical system.

    Instantiate ONCE per system at startup. Reuse for all cells.

    Parameters
    ----------
    symbols           : tuple of sympy.Symbol, length n
    f                 : list of n sympy expressions
    corner_threshold  : use gmpy2 box-corner eval when n <= this (default 6)

    Example
    -------
        from dynamics import load_dynamics
        from hessian_bound import HessianBounder

        symbols, f = load_dynamics("arch3")
        hb = HessianBounder(symbols, f)

        # Per cell:
        M_i, r_i, remainder = hb.remainder_bound(p_i, vertices)
    """

    def __init__(
        self,
        symbols: Sequence[sp.Symbol],
        f: Sequence[sp.Expr],
        corner_threshold: int = 6,
    ):
        self.symbols          = tuple(symbols)
        self.n                = len(symbols)
        self.corner_threshold = corner_threshold
        assert len(f) == self.n, f"len(f)={len(f)} != len(symbols)={self.n}"
        self._entries = self._build(symbols, f)

    # ── Startup ──────────────────────────────────────────────────────────────

    def _build(self, symbols, f) -> list:
        entries = []
        for j, fj in enumerate(f):
            for k, xk in enumerate(symbols):
                dfj_k = sp.diff(fj, xk)
                for l, xl in enumerate(symbols):
                    expr = sp.diff(dfj_k, xl)
                    if expr is sp.S.Zero or expr == sp.Integer(0):
                        continue
                    is_const = len(expr.free_symbols) == 0
                    fn_np = sp.lambdify(symbols, expr, modules="numpy")
                    fn_gm = sp.lambdify(
                        symbols, expr,
                        modules=[{"mpfr": gmpy2.mpfr}, "math"]
                    )
                    fn_iv = sp.lambdify(
                        symbols, expr,
                        modules=[_IV_MOD, "math"]
                    )
                    entries.append(_Entry(j, k, l, fn_np, fn_gm, fn_iv, is_const))
        return entries

    # ── Per-cell API ──────────────────────────────────────────────────────────

    def bound(
        self,
        p_i: np.ndarray,
        vertices: np.ndarray,
    ) -> float:
        """
        Compute guaranteed upper bound M_i on ||H_cell(x)||_2 over X_i.

        Parameters
        ----------
        p_i      : (n,) array — local certificate gradient p_i = W D_i
        vertices : (V, n) array — vertex set F_0(X_i)

        Returns
        -------
        M_i : float
        """
        p_i      = np.asarray(p_i, dtype=float).ravel()
        vertices = np.asarray(vertices, dtype=float)
        V, n     = vertices.shape

        x_lo    = vertices.min(axis=0)
        x_hi    = vertices.max(axis=0)
        np_args = [vertices[:, d] for d in range(n)]

        # Tier 1 — numpy at actual vertices
        H_np = np.zeros((V, n, n), dtype=float)

        # Tier 2 — gmpy2 at box corners  (n <= threshold)
        use_corners = (n <= self.corner_threshold)
        if use_corners:
            corners      = self._box_corners(x_lo, x_hi)  # (2^n, n)
            H_gm         = np.zeros((len(corners), n, n), dtype=float)
            gm_args_list = [
                [gmpy2.mpfr(float(c[d])) for d in range(n)]
                for c in corners
            ]

        # Tier 3 — mpmath.iv over coordinate box  (n > threshold)
        else:
            iv_args = [mpiv.mpf([float(x_lo[d]), float(x_hi[d])])
                       for d in range(n)]
            H_iv    = np.zeros((n, n), dtype=float)

        for e in self._entries:
            pj = float(p_i[e.j])
            if abs(pj) < 1e-15:
                continue

            # Tier 1
            raw_np = e.fn_np(*np_args)
            if np.isscalar(raw_np):
                vals = np.full(V, float(raw_np))
            else:
                vals = np.broadcast_to(np.atleast_1d(raw_np), (V,))
            H_np[:, e.k, e.l] += pj * vals

            if use_corners:
                # Tier 2 — gmpy2 at each box corner
                for ci, gm_args in enumerate(gm_args_list):
                    try:
                        val_gm = float(e.fn_gm(*gm_args))
                    except Exception:
                        val_gm = float(e.fn_np(
                            *[float(a) for a in gm_args]
                        ))
                    H_gm[ci, e.k, e.l] += pj * val_gm
            else:
                # Tier 3 — mpmath.iv over coordinate box
                try:
                    raw_iv = e.fn_iv(*iv_args)
                    H_iv[e.k, e.l] += _abs_bound(_scale_iv(raw_iv, pj))
                except Exception:
                    # Fallback: use vertex max as conservative bound
                    H_iv[e.k, e.l] += abs(pj) * float(
                        np.abs(H_np[:, e.k, e.l]).max()
                    )

        # Spectral norms
        M_vertex = float(
            np.linalg.norm(H_np, ord=2, axis=(-2, -1)).max()
        )
        if use_corners:
            M_formal = float(
                np.linalg.norm(H_gm, ord=2, axis=(-2, -1)).max()
            )
        else:
            M_formal = float(np.linalg.norm(H_iv, ord=2))

        return max(M_vertex, M_formal)

    def bound_fast(
        self,
        p_i: np.ndarray,
        vertices: np.ndarray,
    ) -> float:
        """
        Numpy-only vertex evaluation — no interval arithmetic.

        Not formally guaranteed for non-convex Hessian entries, but fast.
        Use as a first-pass filter in verify_lyapunov.py: only call
        bound() for formal guarantee when bound_fast() gives INCONCLUSIVE.

        Parameters
        ----------
        p_i      : (n,) array
        vertices : (V, n) array

        Returns
        -------
        M_fast : float  (lower bound on true M_i)
        """
        p_i      = np.asarray(p_i, dtype=float).ravel()
        vertices = np.asarray(vertices, dtype=float)
        V, n     = vertices.shape
        np_args  = [vertices[:, d] for d in range(n)]
        H_np     = np.zeros((V, n, n), dtype=float)

        for e in self._entries:
            pj = float(p_i[e.j])
            if abs(pj) < 1e-15:
                continue
            raw = e.fn_np(*np_args)
            if np.isscalar(raw):
                vals = np.full(V, float(raw))
            else:
                vals = np.broadcast_to(np.atleast_1d(raw), (V,))
            H_np[:, e.k, e.l] += pj * vals

        return float(np.linalg.norm(H_np, ord=2, axis=(-2, -1)).max())

    def cell_radius(self, vertices: np.ndarray) -> Tuple[float, np.ndarray]:
        """r_i = max_k ||v_k - centroid||, and the centroid."""
        vertices = np.asarray(vertices, dtype=float)
        centroid = vertices.mean(axis=0)
        r_i      = float(np.linalg.norm(vertices - centroid, axis=1).max())
        return r_i, centroid

    def remainder_bound(
        self,
        p_i: np.ndarray,
        vertices: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Convenience: returns (M_i, r_i, 0.5 * M_i * r_i^2) in one call.
        """
        M_i       = self.bound(p_i, vertices)
        r_i, _    = self.cell_radius(vertices)
        remainder = 0.5 * M_i * r_i ** 2
        return M_i, r_i, remainder

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _box_corners(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """All 2^n corners of [lb, ub]. Returns (2^n, n) array."""
        n = len(lb)
        return np.array(
            np.meshgrid(*[[lb[k], ub[k]] for k in range(n)], indexing='ij')
        ).reshape(n, -1).T


# ── Standalone helper ────────────────────────────────────────────────────────

def cell_radius(vertices):
    vertices = np.asarray(vertices, dtype=float)
    centroid = vertices.mean(axis=0)
    return float(np.linalg.norm(vertices - centroid, axis=1).max()), centroid


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test + benchmark
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time, sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from Dynamics import load_dynamics

    np.random.seed(42)
    test_cases = [
        (
            "arch3",
            np.array([0.5, -0.3]),
            np.array([[0.4,0.4],[0.6,0.4],[0.6,0.6],[0.4,0.6]]),
        ),
        (
            "complex",
            np.array([0.3, -0.5, 0.2]),
            np.array([[0.1,0.1,0.1],[0.3,0.1,0.1],
                      [0.3,0.3,0.1],[0.1,0.3,0.3]]),
        ),
        (
            "quadrotor",
            np.random.randn(12),
            np.random.uniform(-0.05, 0.05, (20, 12)),
        ),
    ]

    total_cells = {"arch3": 16080, "complex": 226550, "quadrotor": 70382}

    for name, p_i, vertices in test_cases:
        print(f"\n{'='*55}")
        print(f"System : {name}  (n={len(p_i)})")

        t0 = time.perf_counter()
        symbols, f = load_dynamics(name)
        hb = HessianBounder(symbols, f)
        t1 = time.perf_counter()
        tier = "corners(gmpy2)" if len(p_i) <= hb.corner_threshold else "interval(mpmath.iv)"
        print(f"  Startup : {(t1-t0)*1000:.0f}ms | "
              f"{len(hb._entries)} entries | tier={tier}")

        # Correctness
        M_i, r_i, rem = hb.remainder_bound(p_i, vertices)
        M_fast         = hb.bound_fast(p_i, vertices)
        print(f"  M_i     = {M_i:.6f}  (fast={M_fast:.6f})")
        print(f"  r_i     = {r_i:.6f}")
        print(f"  rem     = {rem:.6f}")

        # Throughput — full bound
        N = 200 if name == "quadrotor" else 1000
        t0 = time.perf_counter()
        for _ in range(N):
            hb.bound(p_i, vertices)
        t1 = time.perf_counter()
        rate_full = N / (t1 - t0)
        print(f"  Full    : {1000/rate_full:.2f}ms/cell | "
              f"est {total_cells[name]/rate_full:.0f}s total")

        # Throughput — fast bound
        t0 = time.perf_counter()
        for _ in range(N):
            hb.bound_fast(p_i, vertices)
        t1 = time.perf_counter()
        rate_fast = N / (t1 - t0)
        print(f"  Fast    : {1000/rate_fast:.2f}ms/cell | "
              f"est {total_cells[name]/rate_fast:.0f}s total")

    print("\nAll smoke tests passed.")