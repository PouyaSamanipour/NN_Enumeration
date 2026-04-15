"""
lp_ground_truth_deep.py
=======================

Ground-truth correctness verification for DEEP ReLU networks via LP feasibility.

Extends lp_ground_truth.py to two hidden layers. For each joint sign pattern
(s1, s2) over both layers, the effective hyperplanes in the input space are
computed analytically and a single LP feasibility check is performed.

This validates the full enumeration pipeline including:
  - finding_deep_hype (effective parameter propagation)
  - Layer-by-layer cell splitting in core.py
  - Bitwise adjacency test across layers

Architecture: n inputs -> m1 neurons (layer 1) -> m2 neurons (layer 2) -> output
Total patterns: 2^(m1 + m2)

Keep m1 + m2 <= 18 for tractability (2^18 = 262144 patterns).
Recommended: n=3, m1=6, m2=6 (4096 patterns, fast)
             n=3, m1=8, m2=6 (16384 patterns, moderate)
             n=4, m1=6, m2=6 (4096 patterns, higher dim)

Usage
-----
    python lp_ground_truth_deep.py [--n 3] [--m1 6] [--m2 6] [--seed 0]
                                   [--domain 1.0] [--slack 1e-5] [--trials 5]
"""

import argparse
import itertools
from random import seed
import time

import numpy as np
from scipy.optimize import linprog

try:
    from relu_region_enumerator.core import enumeration_function
    HAS_CORE = True
    print("Using full pipeline (core.py).")
except ImportError:
    HAS_CORE = False
    print("core.py not available. Using Enumerator_rapid directly.")

try:
    from relu_region_enumerator.bitwise_utils import Enumerator_rapid
except ImportError:
    from bitwise_utils_numpy import Enumerator_rapid


# ---------------------------------------------------------------------------
# Network generator
# ---------------------------------------------------------------------------

def random_deep_network(n, m1, m2, seed):
    """Sample a random two-hidden-layer ReLU network.

    Layer 1: (m1, n)  weight matrix, input dimension n
    Layer 2: (m2, m1) weight matrix, input dimension m1

    Parameters
    ----------
    n, m1, m2 : int -- dimensions
    seed      : int -- random seed

    Returns
    -------
    H1 : (m1, n)  float64
    b1 : (m1,)    float64
    H2 : (m2, m1) float64
    b2 : (m2,)    float64
    """
    rng = np.random.default_rng(seed)
    H1  = rng.standard_normal((m1, n))
    b1  = rng.uniform(-0.5, 0.5, m1)
    H2  = rng.standard_normal((m2, m1))
    b2  = rng.uniform(-0.5, 0.5, m2)
    return H1, b1, H2, b2


# ---------------------------------------------------------------------------
# Effective hyperplane computation for deep networks
# ---------------------------------------------------------------------------

def effective_hyperplanes(s1, H1, b1, H2, b2):
    """Compute effective hyperplanes in input space for a two-layer network.

    Within the region defined by layer-1 sign pattern s1, layer 2 sees
    effective weight matrix and bias:
        H2_eff = H2 @ D1 @ H1
        b2_eff = H2 @ D1 @ b1 + b2
    where D1 = diag(max(s1, 0)) is the activation matrix for s1.

    Parameters
    ----------
    s1 : (m1,) int array -- layer 1 sign pattern, entries in {0, 1}
    H1, b1 : layer 1 weights
    H2, b2 : layer 2 weights

    Returns
    -------
    H2_eff : (m2, n) float64 -- effective layer-2 weights in input space
    b2_eff : (m2,)   float64 -- effective layer-2 biases
    """
    D1     = np.diag(s1.astype(float))   # diagonal activation matrix
    H2_eff = H2 @ D1 @ H1
    b2_eff = H2 @ D1 @ b1 + b2
    return H2_eff, b2_eff


# ---------------------------------------------------------------------------
# LP feasibility for a joint sign pattern (s1, s2)
# ---------------------------------------------------------------------------

def is_feasible_deep(s1, s2, H1, b1, H2, b2, G, g, slack=1e-5):
    """Check if joint sign pattern (s1, s2) has a full-dimensional feasible region.

    The feasibility LP combines:
      - Layer 1 sign constraints on H1 x + b1
      - Layer 2 sign constraints on H2_eff x + b2_eff (effective in input space)
      - Domain constraints G x <= g

    Parameters
    ----------
    s1 : (m1,) int -- layer 1 pattern, entries {0,1}
    s2 : (m2,) int -- layer 2 pattern, entries {0,1}
    slack : float  -- interior margin

    Returns
    -------
    bool
    """
    n = H1.shape[1]

    # Effective layer-2 hyperplanes in input space
    H2_eff, b2_eff = effective_hyperplanes(s1, H1, b1, H2, b2)

    # Build combined constraint matrix
    # Layer 1: signs1_j * (H1_j x + b1_j) >= slack
    signs1  = np.where(s1 == 1, 1.0, -1.0)
    A_l1    = -(signs1[:, None] * H1)
    b_l1    = signs1 * b1 - slack

    # Layer 2: signs2_j * (H2_eff_j x + b2_eff_j) >= slack
    signs2  = np.where(s2 == 1, 1.0, -1.0)
    A_l2    = -(signs2[:, None] * H2_eff)
    b_l2    = signs2 * b2_eff - slack

    A_ub = np.vstack([A_l1, A_l2, G])
    b_ub = np.concatenate([b_l1, b_l2, g])

    result = linprog(
        np.zeros(n), A_ub=A_ub, b_ub=b_ub,
        bounds=[(None, None)] * n,
        method="highs",
        options={"disp": False},
    )

    return result.status == 0


# ---------------------------------------------------------------------------
# LP ground truth: enumerate all 2^(m1+m2) joint patterns
# ---------------------------------------------------------------------------

def lp_ground_truth_deep(H1, b1, H2, b2, G, g, slack=1e-5):
    """Enumerate all feasible joint sign patterns via LP feasibility.

    Returns
    -------
    feasible_patterns : list of (s1, s2) tuples
    n_feasible        : int
    t_lp              : float
    """
    m1 = H1.shape[0]
    m2 = H2.shape[0]
    feasible_patterns = []

    t0 = time.time()
    for p1 in itertools.product([0, 1], repeat=m1):
        s1 = np.array(p1, dtype=np.int32)
        for p2 in itertools.product([0, 1], repeat=m2):
            s2 = np.array(p2, dtype=np.int32)
            if is_feasible_deep(s1, s2, H1, b1, H2, b2, G, g, slack=slack):
                feasible_patterns.append((tuple(p1), tuple(p2)))
    t_lp = time.time() - t0

    return feasible_patterns, len(feasible_patterns), t_lp


# ---------------------------------------------------------------------------
# Run enumerator (two-layer, using Enumerator_rapid directly)
# ---------------------------------------------------------------------------

def run_enumerator_deep(H1, b1, H2, b2, TH):
    """Run the bitwise enumerator on a two-layer network.

    Replicates the core.py layer-by-layer logic without file I/O.

    Parameters
    ----------
    H1, b1 : layer 1
    H2, b2 : layer 2
    TH     : list of float -- domain half-widths

    Returns
    -------
    cells  : list of vertex arrays (final cells after both layers)
    n_cells: int
    t_enum : float
    """
    from relu_region_enumerator.bitwise_utils import finding_deep_hype

    n  = H1.shape[1]
    m1 = H1.shape[0]
    m2 = H2.shape[0]

    total_hyperplanes = 2 * n + m1 + m2
    use_wide = total_hyperplanes > 64

    initial_verts = np.array(
        list(itertools.product(*[(-t, t) for t in TH])),
        dtype=np.float64,
    )

    border_hyperplane = np.vstack((np.eye(n), -np.eye(n)))
    border_bias       = list(TH) + list(TH)

    hyperplanes = [H1, H2]
    biases      = [b1, b2]

    t0 = time.time()

    # --- Layer 0 ---
    enumerate_poly = Enumerator_rapid(
        H1, b1,
        np.array([initial_verts]),
        TH,
        [border_hyperplane.copy()],
        [border_bias.copy()],
        parallel=False,
        D=np.array([1] * m1),
        m=0,
        use_wide=use_wide,
    )

    # --- Layer 1 ---
    enumerate_poly_l2 = []
    for j, cell in enumerate(enumerate_poly):
        hype1, bias1, border_hyperplane1, border_bias1 = finding_deep_hype(
            hyperplanes, biases,
            cell,
            border_hyperplane, border_bias,
            1, n,
        )
        result = Enumerator_rapid(
            hype1, bias1,
            np.array([cell]),
            TH,
            [border_hyperplane1],
            [border_bias1],
            parallel=False,
            D=np.array([1] * m2),
            m=1,
            use_wide=use_wide,
        )
        enumerate_poly_l2.extend(result)

    t_enum = time.time() - t0
    return enumerate_poly_l2, len(enumerate_poly_l2), t_enum


# ---------------------------------------------------------------------------
# Sign vector extraction from enumerated cells (two layers)
# ---------------------------------------------------------------------------

def get_sign_vectors_deep(cells, H1, b1, H2, b2):
    """Compute joint sign vectors (s1, s2) from cell centroids."""
    sign_vecs = set()
    for poly in cells:
        poly_arr = np.array(poly)
        centroid = poly_arr.mean(axis=0)

        # Layer 1
        z1 = H1 @ centroid + b1
        s1 = tuple((z1 > 0).astype(int))

        # Layer 2 (using layer 1 output)
        a1 = np.maximum(z1, 0)
        z2 = H2 @ a1 + b2
        s2 = tuple((z2 > 0).astype(int))

        sign_vecs.add((s1, s2))
    return sign_vecs


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(n, m1, m2, seed, domain, slack):
    print(f"\n  Seed {seed}: n={n}, m1={m1}, m2={m2}, "
          f"domain=[-{domain},{domain}]^{n}, "
          f"total patterns=2^{m1+m2}={2**(m1+m2)}")
 
    H1, b1, H2, b2 = random_deep_network(n, m1, m2, seed)
    TH = [domain] * n
    G  = np.vstack((np.eye(n), -np.eye(n)))
    g  = np.array(TH + TH, dtype=np.float64)
 
    # --- LP ground truth ---
    print(f"    Running LP feasibility on {2**(m1+m2)} joint patterns...",
          end=" ", flush=True)
    # feasible_patterns, n_lp, t_lp = lp_ground_truth_deep(
    #     H1, b1, H2, b2, G, g, slack=slack
    # )
    # print(f"{n_lp} feasible  ({t_lp:.2f} s)")
 
    # --- Enumerator ---
    print(f"    Running deep enumerator...", end=" ", flush=True)
    cells, n_enum, t_enum = run_enumerator_deep(H1, b1, H2, b2, TH)
    print(f"{n_enum} cells  ({t_enum:.3f} s)")
 
    # --- Count comparison ---
    # count_match = n_enum == n_lp
 
    # # --- Sign vector comparison ---
    # lp_sv_set   = set(feasible_patterns)
    # enum_sv_set = get_sign_vectors_deep(cells, H1, b1, H2, b2)
 
    # missing_from_enum = lp_sv_set   - enum_sv_set
    # extra_in_enum     = enum_sv_set - lp_sv_set
    # sv_match = (len(missing_from_enum) == 0 and len(extra_in_enum) == 0)
 
    # print(f"    (a) Count match        : {'PASS' if count_match else 'FAIL'}  "
    #       f"(LP={n_lp}, Enum={n_enum})")
    # print(f"    (b) Sign vector match  : {'PASS' if sv_match else 'FAIL'}  "
    #       f"(missing={len(missing_from_enum)}, extra={len(extra_in_enum)})")
 
    # if not sv_match:
    #     if missing_from_enum:
    #         print(f"    Patterns in LP but not enum ({len(missing_from_enum)}):")
    #         for p in list(missing_from_enum)[:3]:
    #             print(f"      s1={p[0]}, s2={p[1]}")
    #     if extra_in_enum:
    #         print(f"    Patterns in enum but not LP ({len(extra_in_enum)}):")
    #         for p in list(extra_in_enum)[:3]:
    #             print(f"      s1={p[0]}, s2={p[1]}")
 
    # # --- CDD vertex validation ---
    # from cdd_vertex_validation import run_cdd_vertex_validation
    # print(f"    Running cddlib vertex validation on {n_enum} cells...")
    # cdd_report = run_cdd_vertex_validation(
    #     cells, H1, b1, H2, b2, TH,
    #     n_check=None,
    #     tol=1e-5,
    #     print_every=max(1, n_enum // 5),
    # )
    # cdd_pass = (cdd_report.n_hrep_fail    == 0 and
    #             cdd_report.n_roundtrip_fail == 0 and
    #             cdd_report.n_missing_total  == 0 and
    #             cdd_report.n_redundant_total == 0)
    # print(f"    (c) CDD vertex match   : {'PASS' if cdd_pass else 'FAIL'}  "
    #       f"(hrep_fail={cdd_report.n_hrep_fail}, "
    #       f"missing={cdd_report.n_missing_total}, "
    #       f"redundant={cdd_report.n_redundant_total})")
 
    # overall = count_match and sv_match
    # print(f"    OVERALL: {'PASS' if overall else 'FAIL'}")
 
    # return {
    #     "seed"              : seed,
    #     "n_lp"              : n_lp,
    #     "n_enum"            : n_enum,
    #     "t_lp"              : t_lp,
    #     "t_enum"            : t_enum,
    #     "count_match"       : count_match,
    #     "sv_match"          : sv_match,
    #     "missing_from_enum" : len(missing_from_enum),
    #     "extra_in_enum"     : len(extra_in_enum),
    #     "pass"              : overall,
    # }
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LP feasibility ground-truth for deep (2-layer) ReLU enumeration."
    )
    parser.add_argument("--n",      type=int,   default=3,    help="Input dimension (default 3)")
    parser.add_argument("--m1",     type=int,   default=6,    help="Layer 1 neurons (default 6)")
    parser.add_argument("--m2",     type=int,   default=6,    help="Layer 2 neurons (default 6)")
    parser.add_argument("--seed",   type=int,   default=0,    help="Base random seed (default 0)")
    parser.add_argument("--domain", type=float, default=1.0,  help="Domain half-width (default 1.0)")
    parser.add_argument("--slack",  type=float, default=1e-5, help="LP interior slack (default 1e-5)")
    parser.add_argument("--trials", type=int,   default=5,    help="Number of trials (default 5)")
    args = parser.parse_args()

    total_patterns = 2 ** (args.m1 + args.m2)
    if args.m1 + args.m2 > 20:
        print(f"Warning: m1+m2={args.m1+args.m2} gives {total_patterns} patterns. "
              f"Consider keeping m1+m2 <= 20.")

    print(f"\nDeep LP ground truth: n={args.n}, m1={args.m1}, m2={args.m2}, "
          f"{args.trials} trials, slack={args.slack}")
    print(f"Total joint patterns per trial: 2^{args.m1+args.m2} = {total_patterns}")
    print("=" * 60)

    results = []
    for trial in range(args.trials):
        seed = args.seed + trial
        r    = run_trial(args.n, args.m1, args.m2, seed, args.domain, args.slack)
        results.append(r)

    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Seed':>6}  {'LP':>6}  {'Enum':>6}  {'t_LP':>8}  "
          f"{'t_Enum':>8}  {'Count':>6}  {'SigVec':>6}  {'Result':>6}")
    print("  " + "-" * 70)
    for r in results:
        print(f"  {r['seed']:>6}  {r['n_lp']:>6}  {r['n_enum']:>6}  "
              f"{r['t_lp']:>8.2f}s  {r['t_enum']:>8.3f}s  "
              f"{'OK' if r['count_match'] else 'FAIL':>6}  "
              f"{'OK' if r['sv_match'] else 'FAIL':>6}  "
              f"{'PASS' if r['pass'] else 'FAIL':>6}")

    n_pass = sum(r["pass"] for r in results)
    print(f"\n  Passed: {n_pass} / {args.trials}")
    print(f"{'='*75}")

if __name__ == "__main__":
    main()
