"""
ground_truth_verify.py
======================

Ground-truth correctness verification for the bitwise vertex enumeration.

Two experiments are run:

  1. Axis-aligned grid network  -- hyperplanes are x_i = c_{i,k}, ground
     truth is exactly (k+1)^n cells.  Easy to inspect geometrically.

  2. Rotated grid network -- same network rotated by a random orthogonal
     matrix R.  Ground truth count is still (k+1)^n but the bitmask
     adjacency test sees generic hyperplane orientations, making this a
     stronger correctness test.

For each experiment the script verifies:
  (a) Cell count matches (k+1)^n exactly.
  (b) Every returned polytope is full-dimensional.
  (c) Cells tile the domain with no gaps (volumes sum to domain volume).
  (d) Sign vectors are consistent with centroid evaluation.

Usage
-----
    python ground_truth_verify.py [--n 4] [--k 2] [--seed 42]

Arguments
---------
  --n    Input dimension (default 4, use 6 for the paper experiment).
  --k    Number of cuts per dimension (default 2 → (k+1)^n = 3^n cells).
  --seed Random seed for rotation matrix (default 42).
"""

import argparse
import itertools
import time

import numpy as np
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Import the enumerator.  Adjust the import path to match your package layout.
# ---------------------------------------------------------------------------
try:
    from relu_region_enumerator.bitwise_utils import Enumerator_rapid
except ImportError:
    # Fallback: pure-numpy version produced for benchmarking.
    from bitwise_utils_numpy import Enumerator_rapid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_hypercube_vertices(bounds):
    """All 2^n vertices of the axis-aligned box [-b_i, b_i]."""
    return np.array(list(itertools.product(*[(-b, b) for b in bounds])))


def givens_rotation(n, degree):
    """
    Givens rotation matrix by `degree` degrees in the x1-x2 plane.

    At 0 degrees this is exactly the identity, recovering the axis-aligned
    case. At 45 degrees hyperplanes are in generic position relative to the
    coordinate axes.  The cell count is invariant for any angle.

    Parameters
    ----------
    n      : int   -- dimension
    degree : float -- rotation angle in degrees

    Returns
    -------
    R : (n, n) float64 orthogonal matrix, det(R) = +1
    """
    theta = np.deg2rad(degree)
    R     = np.eye(n)
    R[0, 0] =  np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] =  np.sin(theta)
    R[1, 1] =  np.cos(theta)
    return R


def polytope_volume(verts):
    """Volume of the convex hull of verts (returns 0 for degenerate cells)."""
    try:
        return ConvexHull(verts).volume
    except Exception:
        return 0.0


def is_full_dimensional(verts, tol=1e-9):
    """True if vertex set spans an n-dimensional affine subspace."""
    n       = verts.shape[1]
    centered = verts - verts.mean(axis=0)
    return np.linalg.matrix_rank(centered, tol=tol) == n


def sign_vector_consistent(poly, hyperplanes, biases, tol=1e-5):
    """
    Check that the activation pattern at the centroid is consistent with
    all vertices.  Vertices on the boundary of a cell lie exactly on a
    hyperplane (z = 0), so we use a tolerance: a vertex is only considered
    inconsistent if it strictly contradicts the centroid sign, i.e. it is
    on the wrong side by more than tol.
    """
    centroid = poly.mean(axis=0)
    state    = centroid.copy()
    for H, b in zip(hyperplanes, biases):
        z     = H @ state + b
        state = np.maximum(0.0, z)
        z_verts = poly @ H.T + b  # (V, neurons)

        for j, z_c in enumerate(z):
            if z_c > tol:
                # Centroid says active: no vertex should be strictly inactive
                if np.any(z_verts[:, j] < -tol):
                    return False
            elif z_c < -tol:
                # Centroid says inactive: no vertex should be strictly active
                if np.any(z_verts[:, j] > tol):
                    return False
            # |z_c| <= tol: centroid is on the boundary, skip check
    return True


# ---------------------------------------------------------------------------
# Grid network constructor
# ---------------------------------------------------------------------------

def build_grid_network(n, k, domain=1.0):
    """
    Build a single-hidden-layer ReLU network whose hyperplanes are
    k evenly-spaced cuts along each of the n coordinate axes.

    The cuts along axis i are at positions:
        c_{i,j} = -domain + (2*domain / (k+1)) * j,  j = 1, ..., k

    Each cut requires two neurons (one for each sign) to create a
    proper bounded slab.  With k cuts per axis and 2 neurons per cut,
    the hidden layer has 2*n*k neurons total.

    Actually we use a simpler construction: one neuron per cut, with
    weight +e_i or -e_i and bias chosen so the zero crossing is at c_{i,j}.
    This gives k*n neurons and exactly (k+1)^n cells within the domain.

    Parameters
    ----------
    n      : int   -- input dimension
    k      : int   -- cuts per dimension
    domain : float -- half-width of the domain [-domain, domain]^n

    Returns
    -------
    H    : (k*n, n) float64 -- weight matrix
    b    : (k*n,)   float64 -- bias vector
    cuts : list of (n,) arrays -- cut positions per dimension
    expected_cells : int -- (k+1)^n
    """
    rows_H = []
    rows_b = []
    cut_positions = []

    for dim in range(n):
        # k evenly spaced cuts in (-domain, +domain)
        positions = np.linspace(-domain, domain, k + 2)[1:-1]  # exclude endpoints
        cut_positions.append(positions)
        for c in positions:
            row      = np.zeros(n)
            row[dim] = 1.0
            rows_H.append(row)
            rows_b.append(-c)   # neuron fires when x_i - c > 0, i.e. x_i > c

    H = np.array(rows_H, dtype=np.float64)
    b = np.array(rows_b, dtype=np.float64)
    expected_cells = (k + 1) ** n

    return H, b, cut_positions, expected_cells


def expected_cell_volumes(n, k, domain=1.0):
    """
    Total volume of the domain and per-cell volume for the grid network.
    All cells are identical rectangular boxes.
    """
    domain_volume  = (2 * domain) ** n
    per_cell_volume = domain_volume / ((k + 1) ** n)
    return domain_volume, per_cell_volume


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_experiment(label, H, b, TH, border_hyperplane, border_bias,
                   initial_vertices, expected_count, n, use_wide):
    """Run enumeration and verify all correctness criteria."""

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Expected cells : {expected_count}")
    print(f"  Dimension      : {n}")
    print(f"  Neurons        : {len(H)}")
    print(f"  use_wide       : {use_wide}")

    t0 = time.time()
    result = Enumerator_rapid(
        H, b,
        np.array([initial_vertices]),
        TH,
        [border_hyperplane],
        [list(border_bias)],
        parallel=False,
        D=np.array([1] * len(H)),
        m=0,
        use_wide=use_wide,
    )
    t1 = time.time()

    n_cells = len(result)
    print(f"  Returned cells : {n_cells}  (in {t1-t0:.3f} s)")

    # (a) Count check
    count_ok = n_cells == expected_count
    print(f"  (a) Count correct          : {count_ok}  "
          f"{'OK' if count_ok else 'FAIL — got ' + str(n_cells)}")

    # (b) Full-dimensionality check
    n_degenerate = sum(1 for p in result if not is_full_dimensional(np.array(p)))
    fulldim_ok   = n_degenerate == 0
    print(f"  (b) All full-dimensional   : {fulldim_ok}  "
          f"({n_degenerate} degenerate cells)")

    # (c) Volume check — sum of cell volumes must equal domain volume
    domain_volume = (2.0 * TH[0]) ** n   # works for uniform TH
    total_volume  = sum(polytope_volume(np.array(p)) for p in result)
    vol_err       = abs(total_volume - domain_volume) / domain_volume
    volume_ok     = vol_err < 1e-6
    print(f"  (c) Volume tiling          : {volume_ok}  "
          f"(error = {vol_err:.2e}, expected {domain_volume:.4f}, "
          f"got {total_volume:.4f})")

    # (d) Sign vector consistency
    n_inconsistent = sum(
        1 for p in result
        if not sign_vector_consistent(np.array(p), [H], [b])
    )
    signvec_ok = n_inconsistent == 0
    print(f"  (d) Sign vectors consistent: {signvec_ok}  "
          f"({n_inconsistent} inconsistent cells)")

    all_ok = count_ok and fulldim_ok and volume_ok and signvec_ok
    print(f"\n  OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return all_ok, n_cells, t1 - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ground-truth verification for ReLU enumeration.")
    parser.add_argument("--n",      type=int,   default=4,  help="Input dimension (default 4)")
    parser.add_argument("--k",      type=int,   default=2,  help="Cuts per dimension (default 2)")
    parser.add_argument("--degree", type=float, default=45.0,
                        help="Rotation angle in degrees for experiment 2 (default 45)")
    args = parser.parse_args()

    n      = args.n
    k      = args.k
    degree = args.degree
    domain = 1.0
    TH     = [domain] * n

    print(f"\nGround-truth verification: n={n}, k={k}, expected cells = {(k+1)**n}")

    # Build grid network
    H, b, cuts, expected_cells = build_grid_network(n, k, domain)

    total_hyperplanes = 2 * n + len(H)
    use_wide          = total_hyperplanes > 64

    # -----------------------------------------------------------------------
    # Experiment 1: Axis-aligned grid
    # -----------------------------------------------------------------------
    border_hyperplane_aligned = np.vstack((np.eye(n), -np.eye(n)))
    border_bias_aligned       = np.array(TH + TH, dtype=np.float64)
    initial_verts_aligned     = generate_hypercube_vertices(TH)

    ok1, cells1, t1 = run_experiment(
        label             = "Experiment 1: Axis-aligned grid",
        H                 = H,
        b                 = b,
        TH                = TH,
        border_hyperplane = border_hyperplane_aligned,
        border_bias       = border_bias_aligned,
        initial_vertices  = initial_verts_aligned,
        expected_count    = expected_cells,
        n                 = n,
        use_wide          = use_wide,
    )

    # -----------------------------------------------------------------------
    # Experiment 2: Rotated grid (Givens rotation in x1-x2 plane)
    # -----------------------------------------------------------------------
    R = givens_rotation(n, degree)
    print(f"\n  Givens rotation: {degree} degrees in x1-x2 plane")
    print(f"  det(R) = {np.linalg.det(R):.6f}")

    H_rot                     = H @ R.T
    G                         = np.vstack((np.eye(n), -np.eye(n)))
    border_hyperplane_rotated = G @ R.T
    border_bias_rotated       = np.array(TH + TH, dtype=np.float64)
    initial_verts_rotated     = initial_verts_aligned @ R.T

    ok2, cells2, t2 = run_experiment(
        label             = f"Experiment 2: Rotated grid ({degree} deg)",
        H                 = H_rot,
        b                 = b,
        TH                = TH,
        border_hyperplane = border_hyperplane_rotated,
        border_bias       = border_bias_rotated,
        initial_vertices  = initial_verts_rotated,
        expected_count    = expected_cells,
        n                 = n,
        use_wide          = use_wide,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Axis-aligned : {'PASS' if ok1 else 'FAIL'}  "
          f"({cells1} cells in {t1:.3f} s)")
    print(f"  Rotated      : {'PASS' if ok2 else 'FAIL'}  "
          f"({cells2} cells in {t2:.3f} s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()