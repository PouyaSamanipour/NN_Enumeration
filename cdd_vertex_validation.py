"""
cdd_vertex_validation.py
========================
Validates that the vertex sets produced by the enumerator are exactly
correct using cddlib as ground truth.

For each enumerated cell, the validation proceeds in two directions:

  Direction 1 — V-rep -> H-rep -> V-rep round-trip (cdd canonical form)
  ---------------------------------------------------------------------
  Convert the enumerated vertices to an H-rep via cddlib, then back to
  a V-rep.  The resulting extreme points must match the original vertices
  (up to permutation and floating-point tolerance).  Any extra or missing
  vertex indicates a redundant or missing vertex in the enumeration.

  Direction 2 — H-rep from activation pattern -> V-rep via cdd
  -------------------------------------------------------------
  Build the H-rep directly from the activation pattern halfspaces
  (neuron hyperplanes + domain bounds).  Convert to V-rep via cddlib.
  The resulting vertices must match the enumerated vertices exactly.
  This is the gold standard: cddlib independently computes the true
  vertex set of the polytope defined by the activation pattern.

Both checks together confirm:
  - No redundant vertices (Direction 1)
  - No missing vertices (Direction 2)
  - Correct halfspace signs (Direction 2)

Usage
-----
    from cdd_vertex_validation import run_cdd_vertex_validation
    report = run_cdd_vertex_validation(
        cells, H1, b1, H2, b2, TH, n_check=50
    )
"""

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_hrep_matrix(A: np.ndarray, b: np.ndarray):
    """Build a cddlib H-rep matrix from A x <= b, compatible with both
    pycddlib 2.x and 3.x APIs."""
    import cdd
    mat_data = np.hstack([b[:, None], -A])
    try:
        mat = cdd.matrix_from_array(mat_data, rep_type=cdd.RepType.INEQUALITY)
    except AttributeError:
        mat = cdd.Matrix(mat_data.tolist(), number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY
    return mat


def _make_vrep_matrix(V: np.ndarray):
    """Build a cddlib V-rep matrix from vertex array, compatible with both
    pycddlib 2.x and 3.x APIs."""
    import cdd
    V_mat = np.hstack([np.ones((len(V), 1)), V])
    try:
        mat = cdd.matrix_from_array(V_mat, rep_type=cdd.RepType.GENERATOR)
    except AttributeError:
        mat = cdd.Matrix(V_mat.tolist(), number_type='float')
        mat.rep_type = cdd.RepType.GENERATOR
    return mat


def _safe_vertices_from_hrep(A: np.ndarray, b: np.ndarray,
                              timeout: int = 10) -> Optional[np.ndarray]:
    """
    Run _vertices_from_hrep in a subprocess (spawn) to isolate cddlib segfaults.
    cdd is imported only inside the child process — never in the parent —
    so fork-after-import cannot cause corruption.
    Returns None if cddlib crashes, times out, or the polytope is infeasible.
    """
    import multiprocessing as mp
    import pickle

    def _worker(A_bytes, b_bytes, queue):
        # Import cdd here so it is never loaded in the parent process
        import cdd as _cdd
        import numpy as _np
        import warnings as _warnings

        def make_hrep(A, b):
            mat_data = _np.hstack([b[:, None], -A])
            try:
                mat = _cdd.matrix_from_array(mat_data,
                                              rep_type=_cdd.RepType.INEQUALITY)
            except AttributeError:
                mat = _cdd.Matrix(mat_data.tolist(), number_type='float')
                mat.rep_type = _cdd.RepType.INEQUALITY
            return mat

        try:
            A = pickle.loads(A_bytes)
            b = pickle.loads(b_bytes)
            mat  = make_hrep(A, b)
            poly = _cdd.Polyhedron(mat)
            gen  = poly.get_generators()
            gen_arr = _np.array(gen)
            if len(gen_arr) == 0:
                queue.put(pickle.dumps(None))
                return
            vertex_rows = gen_arr[gen_arr[:, 0] > 0.5]
            if len(vertex_rows) == 0:
                queue.put(pickle.dumps(None))
                return
            queue.put(pickle.dumps(vertex_rows[:, 1:]))
        except Exception as e:
            queue.put(pickle.dumps(None))

    ctx = mp.get_context('spawn')   # spawn avoids fork-after-import segfaults
    q   = ctx.Queue()
    p   = ctx.Process(target=_worker,
                      args=(pickle.dumps(A), pickle.dumps(b), q))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()
        p.join()
        warnings.warn("cddlib timed out — skipping cell")
        return None

    if p.exitcode != 0:
        warnings.warn(f"cddlib segfaulted (exitcode={p.exitcode}) — skipping cell")
        return None

    return pickle.loads(q.get()) if not q.empty() else None


def _vertices_from_hrep(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute vertices of polytope {x : A x <= b} using cddlib.
    Called only inside child processes — cdd imported lazily.
    Returns (V, n) array of extreme points, or None if infeasible/unbounded.
    """
    try:
        mat  = _make_hrep_matrix(A, b)
        poly_obj = __import__('cdd').Polyhedron(mat)
        gen  = poly_obj.get_generators()
    except Exception as e:
        warnings.warn(f"cddlib H->V conversion failed: {e}")
        return None

    gen_arr = np.array(gen)
    if len(gen_arr) == 0:
        return None

    vertex_rows = gen_arr[gen_arr[:, 0] > 0.5]
    if len(vertex_rows) == 0:
        return None

    return vertex_rows[:, 1:]


def _vertices_match(V1: np.ndarray, V2: np.ndarray, tol: float = 1e-5) -> bool:
    """
    Check if two vertex sets are equal up to permutation and tolerance.
    Returns True if every vertex in V1 has a match in V2 and vice versa.
    """
    if len(V1) != len(V2):
        return False

    used = np.zeros(len(V2), dtype=bool)
    for v in V1:
        dists = np.linalg.norm(V2 - v, axis=1)
        best  = int(np.argmin(dists))
        if dists[best] > tol or used[best]:
            return False
        used[best] = True
    return True


def _build_activation_hrep(
    s1         : np.ndarray,   # (m1,) binary
    s2         : Optional[np.ndarray],   # (m2,) binary or None for 1-layer
    H1         : np.ndarray,   # (m1, n)
    b1         : np.ndarray,   # (m1,)
    H2         : Optional[np.ndarray],   # (m2, m1) or None
    b2         : Optional[np.ndarray],   # (m2,) or None
    TH         : list,
    slack      : float = 0.0,
) -> tuple:
    """
    Build H-rep [A_all, b_all] for the polytope defined by activation
    pattern (s1, s2) within the domain [-TH_i, TH_i].

    Returns (A_all, b_all) such that the polytope is {x : A_all x <= b_all}.
    """
    n = H1.shape[1]

    # Domain constraints: x_i <= TH_i and -x_i <= TH_i
    A_dom = np.vstack([np.eye(n), -np.eye(n)])
    b_dom = np.array(TH + TH, dtype=np.float64)

    # Layer 1 sign constraints: sign_j * (H1_j @ x + b1_j) >= slack
    signs1 = np.where(s1 == 1, 1.0, -1.0)
    A_l1   = -(signs1[:, None] * H1)       # A x <= b form
    b_l1   = signs1 * b1 - slack

    A_all = np.vstack([A_dom, A_l1])
    b_all = np.concatenate([b_dom, b_l1])

    if s2 is not None and H2 is not None and b2 is not None:
        # Effective layer-2 hyperplanes in input space
        D1     = np.diag(s1.astype(float))
        H2_eff = H2 @ D1 @ H1
        b2_eff = H2 @ D1 @ b1 + b2

        signs2 = np.where(s2 == 1, 1.0, -1.0)
        A_l2   = -(signs2[:, None] * H2_eff)
        b_l2   = signs2 * b2_eff - slack

        A_all = np.vstack([A_all, A_l2])
        b_all = np.concatenate([b_all, b_l2])

    return A_all, b_all


# ═══════════════════════════════════════════════════════════════════════════
# Per-cell validation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CellValidationResult:
    cell_idx          : int
    n_verts_enum      : int      # vertices from enumerator
    n_verts_cdd       : int      # vertices from cddlib H->V
    roundtrip_ok      : bool     # Direction 1: V->H->V round-trip
    hrep_ok           : bool     # Direction 2: activation H-rep -> V matches enum
    n_redundant       : int      # vertices in enum not in cdd V-rep
    n_missing         : int      # vertices in cdd V-rep not in enum
    max_vertex_error  : float    # max distance between matched vertices


def validate_cell_cdd(
    vertices   : np.ndarray,   # (V, n) enumerated vertices
    s1         : np.ndarray,   # (m1,) layer-1 activation pattern
    s2         : Optional[np.ndarray],   # (m2,) layer-2 pattern or None
    H1         : np.ndarray,
    b1         : np.ndarray,
    H2         : Optional[np.ndarray],
    b2         : Optional[np.ndarray],
    TH         : list,
    cell_idx   : int,
    tol        : float = 1e-5,
) -> CellValidationResult:
    """
    Validate a single cell's vertex set against cddlib ground truth.
    """
    n = vertices.shape[1]

    # ── Direction 1: V-rep -> H-rep -> V-rep round-trip ──────────────────────
    try:
        mat_v       = _make_vrep_matrix(vertices)
        poly_v      = cdd.Polyhedron(mat_v)
        H_from_v    = poly_v.get_inequalities()
        H_arr       = np.array(H_from_v)
        A_rt        = -H_arr[:, 1:]
        b_rt        =  H_arr[:,  0]
        verts_rt    = _safe_vertices_from_hrep(A_rt, b_rt)
        roundtrip_ok = (verts_rt is not None and
                        _vertices_match(vertices, verts_rt, tol=tol))
    except Exception as e:
        warnings.warn(f"Cell {cell_idx} round-trip failed: {e}")
        roundtrip_ok = False
        verts_rt     = None

    # ── Direction 2: activation H-rep -> V-rep via cdd ───────────────────────
    A_act, b_act = _build_activation_hrep(s1, s2, H1, b1, H2, b2, TH)
    verts_cdd    = _safe_vertices_from_hrep(A_act, b_act)

    if verts_cdd is None:
        hrep_ok     = False
        n_verts_cdd = 0
        n_redundant = 0
        n_missing   = len(vertices)
        max_err     = float('inf')
    else:
        hrep_ok     = _vertices_match(vertices, verts_cdd, tol=tol)
        n_verts_cdd = len(verts_cdd)

        # Count redundant (in enum but not in cdd) and missing (in cdd but not enum)
        n_redundant = 0
        n_missing   = 0
        max_err     = 0.0

        used_cdd  = np.zeros(len(verts_cdd),  dtype=bool)
        used_enum = np.zeros(len(vertices),    dtype=bool)

        for k, v in enumerate(vertices):
            dists = np.linalg.norm(verts_cdd - v, axis=1)
            best  = int(np.argmin(dists))
            if dists[best] <= tol:
                used_cdd[best]  = True
                used_enum[k]    = True
                max_err = max(max_err, dists[best])
            else:
                n_redundant += 1   # vertex in enum has no match in cdd

        n_missing = int((~used_cdd).sum())   # cdd vertices not matched by enum

    return CellValidationResult(
        cell_idx         = cell_idx,
        n_verts_enum     = len(vertices),
        n_verts_cdd      = n_verts_cdd,
        roundtrip_ok     = roundtrip_ok,
        hrep_ok          = hrep_ok,
        n_redundant      = n_redundant,
        n_missing        = n_missing,
        max_vertex_error = max_err,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Batch validation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CddValidationReport:
    n_checked         : int
    n_roundtrip_fail  : int = 0
    n_hrep_fail       : int = 0
    n_redundant_total : int = 0
    n_missing_total   : int = 0
    max_vertex_error  : float = 0.0
    runtime_s         : float = 0.0
    results           : List[CellValidationResult] = field(default_factory=list)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"CDD Vertex Validation Report")
        print(f"{'='*60}")
        print(f"  Cells checked          : {self.n_checked}")
        print(f"  Round-trip failures    : {self.n_roundtrip_fail}"
              f"  (V->H->V mismatch)")
        print(f"  H-rep failures         : {self.n_hrep_fail}"
              f"  (activation H-rep -> V mismatch)")
        print(f"  Total redundant verts  : {self.n_redundant_total}"
              f"  (in enum, not in cdd)")
        print(f"  Total missing verts    : {self.n_missing_total}"
              f"  (in cdd, not in enum)")
        print(f"  Max vertex error       : {self.max_vertex_error:.2e}")
        print(f"  Runtime                : {self.runtime_s:.1f} s")
        if (self.n_roundtrip_fail == 0 and self.n_hrep_fail == 0 and
                self.n_redundant_total == 0 and self.n_missing_total == 0):
            print(f"\n  All {self.n_checked} cells PASSED cddlib vertex validation.")
        else:
            print(f"\n  !! {self.n_hrep_fail} cells have incorrect vertex sets.")
        print(f"{'='*60}")


def run_cdd_vertex_validation(
    cells      : list,         # list of (V_k, n) vertex arrays from enumerator
    H1         : np.ndarray,   # (m1, n) layer-1 weights
    b1         : np.ndarray,   # (m1,)
    H2         : Optional[np.ndarray] = None,   # (m2, m1) layer-2 weights
    b2         : Optional[np.ndarray] = None,   # (m2,)
    TH         : list          = None,
    n_check    : int           = None,   # None = check all
    tol        : float         = 1e-5,
    print_every: int           = 20,
) -> CddValidationReport:
    """
    Validate enumerated cell vertices against cddlib ground truth.

    For each cell:
      1. Extract activation pattern (s1, s2) from centroid
      2. Run Direction 1: V->H->V round-trip
      3. Run Direction 2: build H-rep from activation pattern, get V via cdd

    Parameters
    ----------
    cells    : enumerated polytopes (list of vertex arrays)
    H1, b1   : layer-1 weights
    H2, b2   : layer-2 weights (None for single-layer networks)
    TH       : domain half-widths
    n_check  : number of cells to check (None = all)
    tol      : vertex matching tolerance
    """
    n        = H1.shape[1]
    n_check  = len(cells) if n_check is None else min(n_check, len(cells))
    TH       = TH or [1.0] * n

    report = CddValidationReport(n_checked=n_check)
    t0     = time.perf_counter()

    for cell_idx in range(n_check):
        vertices = np.asarray(cells[cell_idx], dtype=np.float64)
        centroid = vertices.mean(axis=0)

        # Extract activation pattern from centroid
        z1 = H1 @ centroid + b1
        s1 = (z1 > 0).astype(int)

        s2 = None
        if H2 is not None and b2 is not None:
            a1 = np.maximum(z1, 0)
            z2 = H2 @ a1 + b2
            s2 = (z2 > 0).astype(int)

        r = validate_cell_cdd(
            vertices, s1, s2, H1, b1, H2, b2, TH,
            cell_idx=cell_idx, tol=tol,
        )
        report.results.append(r)

        if not r.roundtrip_ok:
            report.n_roundtrip_fail += 1
        if not r.hrep_ok:
            report.n_hrep_fail += 1
            if r.n_redundant > 0 or r.n_missing > 0:
                print(f"  [FAIL] Cell {cell_idx}: "
                      f"redundant={r.n_redundant}  missing={r.n_missing}  "
                      f"enum_verts={r.n_verts_enum}  cdd_verts={r.n_verts_cdd}")

        report.n_redundant_total += r.n_redundant
        report.n_missing_total   += r.n_missing
        report.max_vertex_error   = max(report.max_vertex_error,
                                        r.max_vertex_error
                                        if r.max_vertex_error != float('inf')
                                        else 0.0)

        if (cell_idx + 1) % print_every == 0 or (cell_idx + 1) == n_check:
            elapsed = time.perf_counter() - t0
            rate    = (cell_idx + 1) / elapsed
            eta     = (n_check - cell_idx - 1) / rate
            print(f"  [{cell_idx+1}/{n_check}] "
                  f"hrep_fail={report.n_hrep_fail} "
                  f"roundtrip_fail={report.n_roundtrip_fail} "
                  f"| {rate:.1f} cells/s | ETA {eta:.0f}s")

    report.runtime_s = time.perf_counter() - t0
    report.print_summary()
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Integration with lp_ground_truth_deep.py
# ═══════════════════════════════════════════════════════════════════════════

def run_trial_with_cdd(n, m1, m2, seed, domain, slack, tol=1e-5):
    """
    Extend the existing LP ground truth trial with cddlib vertex validation.
    Runs after the enumerator and checks all enumerated cells.
    """
    from cdd_vertex_validation import run_cdd_vertex_validation
    import numpy as np

    rng = np.random.default_rng(seed)
    H1  = rng.standard_normal((m1, n))
    b1  = rng.uniform(-0.5, 0.5, m1)
    H2  = rng.standard_normal((m2, m1))
    b2  = rng.uniform(-0.5, 0.5, m2)
    TH  = [domain] * n

    # Import enumerator from existing script
    try:
        from lp_ground_truth_deep import run_enumerator_deep
        cells, n_enum, t_enum = run_enumerator_deep(H1, b1, H2, b2, TH)
    except ImportError:
        print("lp_ground_truth_deep not found — provide cells manually.")
        return None

    print(f"\n  CDD vertex validation: {n_enum} cells, tol={tol}")
    report = run_cdd_vertex_validation(
        cells, H1, b1, H2, b2, TH,
        n_check=None, tol=tol,
    )
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import cdd
    print("Smoke test: unit square")

    # Unit square vertices
    verts = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])

    # H-rep: 0 <= x_i <= 1
    A = np.array([[ 1.,  0.],
                  [-1.,  0.],
                  [ 0.,  1.],
                  [ 0., -1.]])
    b = np.array([1., 0., 1., 0.])

    verts_cdd = _vertices_from_hrep(A, b)
    print(f"  cdd V-rep ({len(verts_cdd)} vertices):")
    print(f"  {verts_cdd}")
    assert _vertices_match(verts, verts_cdd), "Unit square V-rep mismatch"
    print("  Unit square: PASS")

    # Round-trip
    mat_rt   = _make_vrep_matrix(verts)
    poly     = cdd.Polyhedron(mat_rt)
    H_rt  = np.array(poly.get_inequalities())
    A_rt  = -H_rt[:, 1:]
    b_rt  =  H_rt[:,  0]
    verts_rt = _vertices_from_hrep(A_rt, b_rt)
    assert _vertices_match(verts, verts_rt), "Round-trip mismatch"
    print("  Round-trip: PASS")

    print("\nSmoke test passed.")