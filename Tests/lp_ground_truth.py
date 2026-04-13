"""
lp_ground_truth.py
==================

Ground-truth correctness verification via LP feasibility enumeration.

For a randomly initialized single-hidden-layer ReLU network with n=3 inputs
and m neurons, this script:

  1. Enumerates all 2^m sign patterns and checks each for LP feasibility
     over a bounded domain D.  A pattern is feasible iff there exists a
     full-dimensional (interior) point in D satisfying the sign constraints.
     This gives the exact ground-truth cell count, independent of any
     enumeration algorithm.

  2. Runs the bitwise vertex enumerator on the same network and domain.

  3. Compares the two counts and reports pass/fail.

The LP feasibility check is the standard approach used in LP-based enumeration
methods (e.g. Ren et al.).  Using it as ground truth here is valid because:
  - It operates on sign patterns directly, with no geometric data structures.
  - It is algorithmically completely independent of the bitwise enumerator.
  - For small m (<=20), exhaustive enumeration of all 2^m patterns is tractable.

Usage
-----
    python lp_ground_truth.py [--n 3] [--m 10] [--seed 0] [--domain 1.0]
                              [--slack 1e-6] [--trials 5]

Arguments
---------
  --n       Input dimension (default 3).
  --m       Number of neurons (default 10). Keep m <= 20 for tractability.
  --seed    Random seed for network weights (default 0).
  --domain  Domain half-width for [-domain, domain]^n (default 1.0).
  --slack   Minimum interior slack for LP feasibility (default 1e-6).
            A pattern is feasible only if the LP finds a point strictly
            inside all constraints by at least this amount.
  --trials  Number of random networks to test (default 5).
            Each trial uses seed + trial_index as the random seed.
"""

import argparse
import itertools
import time

import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Import enumerator — adjust path if needed.
# ---------------------------------------------------------------------------
try:
    from relu_region_enumerator.bitwise_utils import Enumerator_rapid
    print("Using Numba-accelerated enumerator.")
except ImportError:
    try:
        from bitwise_utils_numpy import Enumerator_rapid
        print("Using pure-NumPy enumerator.")
    except ImportError:
        raise ImportError(
            "Could not import Enumerator_rapid. "
            "Run from the NN_Enumeration directory or adjust the import path."
        )


# ---------------------------------------------------------------------------
# Network generator
# ---------------------------------------------------------------------------

def random_network(n, m, seed):
    """Sample a random single-hidden-layer ReLU network.

    Weights drawn from N(0,1), biases from Uniform(-0.5, 0.5).
    Generic position holds almost surely for continuous distributions.

    Parameters
    ----------
    n    : int -- input dimension
    m    : int -- number of hidden neurons
    seed : int -- random seed

    Returns
    -------
    H : (m, n) float64 -- weight matrix
    b : (m,)   float64 -- bias vector
    """
    rng = np.random.default_rng(seed)
    H   = rng.standard_normal((m, n))
    b   = rng.uniform(-0.5, 0.5, m)
    return H, b


# ---------------------------------------------------------------------------
# LP feasibility check
# ---------------------------------------------------------------------------

def is_feasible(pattern, H, b, G, g, slack=1e-6):
    """Check if a sign pattern has a full-dimensional feasible region.

    Solves the LP:
        find x in R^n
        s.t.  s_j * (H_j x + b_j) >= slack   for all j   [sign constraints]
              G x <= g                                      [domain constraints]

    where s_j = +1 if pattern[j] == 1 (active) else -1.

    The slack > 0 requirement ensures we find a point strictly in the
    interior, excluding lower-dimensional faces.

    Parameters
    ----------
    pattern : (m,) int array -- sign pattern, entries in {0, 1}
                                (1 = active neuron, 0 = inactive)
    H       : (m, n) float64
    b       : (m,)   float64
    G       : (q, n) float64 -- domain inequality matrix
    g       : (q,)   float64 -- domain inequality rhs
    slack   : float  -- minimum interior margin

    Returns
    -------
    bool -- True if pattern is feasible with interior point
    """
    m, n = H.shape

    # Build constraint matrix A_ub x <= b_ub.
    # Sign constraint: s_j * (H_j x + b_j) >= slack
    #   => -s_j * H_j x <= s_j * b_j - slack
    signs = np.where(pattern == 1, 1.0, -1.0)  # (m,)

    A_sign = -(signs[:, None] * H)             # (m, n)
    b_sign = signs * b - slack                  # (m,)

    A_ub = np.vstack([A_sign, G])              # (m+q, n)
    b_ub = np.concatenate([b_sign, g])         # (m+q,)

    # Objective: minimize 0 (feasibility only).
    c = np.zeros(n)

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        bounds=[(None, None)] * n,
        method="highs",
        options={"disp": False},
    )

    return result.status == 0   # 0 = optimal (feasible)


# ---------------------------------------------------------------------------
# LP ground truth: enumerate all 2^m patterns
# ---------------------------------------------------------------------------

def lp_ground_truth(H, b, G, g, slack=1e-6):
    """Enumerate all feasible sign patterns via LP feasibility checks.

    Parameters
    ----------
    H, b  : network weights and biases
    G, g  : domain inequality Gx <= g
    slack : interior margin

    Returns
    -------
    feasible_patterns : list of (m,) int arrays -- all feasible patterns
    n_feasible        : int -- exact ground-truth cell count
    t_lp              : float -- total LP time in seconds
    """
    m = H.shape[0]
    feasible_patterns = []

    t0 = time.time()
    for pattern_tuple in itertools.product([0, 1], repeat=m):
        pattern = np.array(pattern_tuple, dtype=np.int32)
        if is_feasible(pattern, H, b, G, g, slack=slack):
            feasible_patterns.append(pattern)
    t_lp = time.time() - t0

    return feasible_patterns, len(feasible_patterns), t_lp


# ---------------------------------------------------------------------------
# Run enumerator
# ---------------------------------------------------------------------------

def run_enumerator(H, b, TH):
    """Run the bitwise enumerator on a single-hidden-layer network.

    Parameters
    ----------
    H  : (m, n) float64
    b  : (m,)   float64
    TH : list of float -- domain half-widths

    Returns
    -------
    cells  : list of vertex arrays
    n_cells: int
    t_enum : float -- enumeration time in seconds
    """
    n = H.shape[1]
    m = H.shape[0]

    total_hyperplanes = 2 * n + m
    use_wide = total_hyperplanes > 64

    # Initial hypercube vertices.
    initial_verts = np.array(
        list(itertools.product(*[(-t, t) for t in TH])),
        dtype=np.float64,
    )

    # Domain boundary hyperplanes: [I; -I] x <= TH.
    border_hyperplane = np.vstack((np.eye(n), -np.eye(n)))
    border_bias       = list(TH) + list(TH)

    t0 = time.time()
    cells = Enumerator_rapid(
        H, b,
        np.array([initial_verts]),
        TH,
        [border_hyperplane],
        [border_bias],
        parallel=False,
        D=np.array([1] * m),
        m=0,
        use_wide=use_wide,
    )
    t_enum = time.time() - t0

    return cells, len(cells), t_enum


# ---------------------------------------------------------------------------
# Sign vector extraction from enumerated cells
# ---------------------------------------------------------------------------

def get_sign_vectors(cells, H, b):
    """Compute the sign vector of each enumerated cell from its centroid.

    Returns
    -------
    sign_vecs : set of tuples -- unique sign vectors (as 0/1 tuples)
    """
    sign_vecs = set()
    for poly in cells:
        poly_arr = np.array(poly)
        centroid = poly_arr.mean(axis=0)
        z        = H @ centroid + b
        sv       = tuple((z > 0).astype(int))
        sign_vecs.add(sv)
    return sign_vecs


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(n, m, seed, domain, slack):
    """Run one full ground-truth comparison trial.

    Parameters
    ----------
    n, m   : network dimensions
    seed   : random seed
    domain : domain half-width
    slack  : LP interior margin

    Returns
    -------
    dict with trial results
    """
    print(f"\n  Seed {seed}: n={n}, m={m}, domain=[-{domain},{domain}]^{n}")

    H, b = random_network(n, m, seed)
    TH   = [domain] * n

    # Domain for LP: Gx <= g  where G=[I;-I], g=[TH;TH]
    G = np.vstack((np.eye(n), -np.eye(n)))
    g = np.array(TH + TH, dtype=np.float64)

    # --- Ground truth via LP ---
    print(f"    Running LP feasibility on {2**m} sign patterns...", end=" ", flush=True)
    feasible_patterns, n_lp, t_lp = lp_ground_truth(H, b, G, g, slack=slack)
    print(f"{n_lp} feasible  ({t_lp:.2f} s)")

    # --- Enumerator ---
    print(f"    Running bitwise enumerator...", end=" ", flush=True)
    cells, n_enum, t_enum = run_enumerator(H, b, TH)
    print(f"{n_enum} cells  ({t_enum:.3f} s)")

    # --- Count comparison ---
    count_match = n_enum == n_lp

    # --- Sign vector comparison ---
    # Every feasible LP pattern should appear as a cell in the enumerator,
    # and vice versa.
    lp_sv_set   = set(tuple(p) for p in feasible_patterns)
    enum_sv_set = get_sign_vectors(cells, H, b)

    missing_from_enum = lp_sv_set - enum_sv_set    # LP found but enumerator missed
    extra_in_enum     = enum_sv_set - lp_sv_set    # Enumerator found but LP missed

    sv_match = (len(missing_from_enum) == 0 and len(extra_in_enum) == 0)

    # --- Report ---
    print(f"    (a) Count match        : {'PASS' if count_match else 'FAIL'}  "
          f"(LP={n_lp}, Enum={n_enum})")
    print(f"    (b) Sign vector match  : {'PASS' if sv_match else 'FAIL'}  "
          f"(missing={len(missing_from_enum)}, extra={len(extra_in_enum)})")

    if not count_match or not sv_match:
        if missing_from_enum:
            print(f"    Patterns in LP but not enum ({len(missing_from_enum)}):")
            for p in list(missing_from_enum)[:5]:
                print(f"      {p}")
        if extra_in_enum:
            print(f"    Patterns in enum but not LP ({len(extra_in_enum)}):")
            for p in list(extra_in_enum)[:5]:
                print(f"      {p}")

    overall = count_match and sv_match
    print(f"    OVERALL: {'PASS' if overall else 'FAIL'}")

    return {
        "seed"              : seed,
        "n_lp"              : n_lp,
        "n_enum"            : n_enum,
        "t_lp"              : t_lp,
        "t_enum"            : t_enum,
        "count_match"       : count_match,
        "sv_match"          : sv_match,
        "missing_from_enum" : len(missing_from_enum),
        "extra_in_enum"     : len(extra_in_enum),
        "pass"              : overall,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LP feasibility ground-truth verification for ReLU enumeration."
    )
    parser.add_argument("--n",      type=int,   default=3,    help="Input dimension (default 3)")
    parser.add_argument("--m",      type=int,   default=10,   help="Number of neurons (default 10, max ~20)")
    parser.add_argument("--seed",   type=int,   default=0,    help="Base random seed (default 0)")
    parser.add_argument("--domain", type=float, default=1.0,  help="Domain half-width (default 1.0)")
    parser.add_argument("--slack",  type=float, default=1e-6, help="LP interior slack (default 1e-6)")
    parser.add_argument("--trials", type=int,   default=5,    help="Number of random networks (default 5)")
    args = parser.parse_args()

    if args.m > 20:
        print(f"Warning: m={args.m} gives {2**args.m} patterns. "
              f"LP enumeration may be slow. Consider m <= 20.")

    print(f"\nLP feasibility ground truth: n={args.n}, m={args.m}, "
          f"{args.trials} trials, slack={args.slack}")
    print(f"Total sign patterns per trial: 2^{args.m} = {2**args.m}")
    print("=" * 60)

    results = []
    for trial in range(args.trials):
        seed = args.seed + trial
        r    = run_trial(args.n, args.m, seed, args.domain, args.slack)
        results.append(r)

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Seed':>6}  {'LP':>6}  {'Enum':>6}  {'t_LP':>8}  "
          f"{'t_Enum':>8}  {'Count':>6}  {'SigVec':>6}  {'Result':>6}")
    print("  " + "-" * 56)
    for r in results:
        print(f"  {r['seed']:>6}  {r['n_lp']:>6}  {r['n_enum']:>6}  "
              f"{r['t_lp']:>8.2f}s  {r['t_enum']:>8.3f}s  "
              f"{'OK' if r['count_match'] else 'FAIL':>6}  "
              f"{'OK' if r['sv_match'] else 'FAIL':>6}  "
              f"{'PASS' if r['pass'] else 'FAIL':>6}")

    n_pass = sum(r["pass"] for r in results)
    print(f"\n  Passed: {n_pass} / {args.trials}")
    print("=" * 60)

    # --- Speedup note ---
    avg_t_lp   = np.mean([r["t_lp"]   for r in results])
    avg_t_enum = np.mean([r["t_enum"] for r in results])
    if avg_t_enum > 0:
        print(f"\n  Avg LP time   : {avg_t_lp:.2f} s")
        print(f"  Avg Enum time : {avg_t_enum:.3f} s")
        print(f"  Speedup       : {avg_t_lp / avg_t_enum:.1f}x "
              f"(note: LP here checks all 2^m patterns, not a fair speed comparison)")


if __name__ == "__main__":
    main()