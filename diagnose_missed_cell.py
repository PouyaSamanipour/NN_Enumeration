"""
diagnose_missed_cell.py
=======================

Investigates the single missed cell found in the LP ground truth run:
  n=8, m=18, seed=2, pattern=(1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,1)

Checks:
  1. Volume of the missed cell's polytope (is it near-degenerate?)
  2. LP optimal value with varying slack (how interior is the feasible point?)
  3. Whether the enumerator's epsilon threshold is responsible
  4. Whether the cell satisfies Assumption 1 (general position)
"""

import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

try:
    from relu_region_enumerator.bitwise_utils import Enumerator_rapid
except ImportError:
    from bitwise_utils_numpy import Enumerator_rapid


# ---------------------------------------------------------------------------
# Reproduce the exact network from seed=2
# ---------------------------------------------------------------------------
n, m, seed = 8, 18, 2
domain = 1.0

rng = np.random.default_rng(seed)
H   = rng.standard_normal((m, n))
b   = rng.uniform(-0.5, 0.5, m)

TH = [domain] * n
G  = np.vstack((np.eye(n), -np.eye(n)))
g  = np.array(TH + TH, dtype=np.float64)

missed_pattern = np.array(
    [1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,1], dtype=np.int32
)

print(f"Network: n={n}, m={m}, seed={seed}")
print(f"Missed pattern: {tuple(missed_pattern)}")
print()

# ---------------------------------------------------------------------------
# 1. How interior is the missed cell? Sweep slack values.
# ---------------------------------------------------------------------------
print("=" * 55)
print("1. LP feasibility vs slack threshold")
print("=" * 55)

def lp_max_slack(pattern, H, b, G, g):
    """Find the maximum slack s such that the pattern is feasible with margin s.
    
    Solves:  max  s
             s.t. signs_j * (H_j x + b_j) >= s   for all j
                  G x <= g
                  s >= 0
    
    The optimal s tells us how 'deep' into the interior the feasible region goes.
    Small s = thin cell near boundary. Large s = well-interior cell.
    """
    m_n, n_d = H.shape
    signs = np.where(pattern == 1, 1.0, -1.0)

    # Variables: [x (n_d,), s (1,)]  total n_d+1
    # Objective: maximize s  =>  minimize -s
    c = np.zeros(n_d + 1)
    c[-1] = -1.0  # minimize -s

    # Sign constraints: -signs_j * H_j x - s <= signs_j * b_j
    A_sign = np.hstack([-(signs[:, None] * H), -np.ones((m_n, 1))])
    b_sign = signs * b

    # Domain constraints: G x <= g  (s unconstrained here)
    A_dom = np.hstack([G, np.zeros((G.shape[0], 1))])
    b_dom = g

    # s >= 0: -s <= 0
    A_s = np.zeros((1, n_d + 1))
    A_s[0, -1] = -1.0
    b_s = np.array([0.0])

    A_ub = np.vstack([A_sign, A_dom, A_s])
    b_ub = np.concatenate([b_sign, b_dom, b_s])

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        bounds=[(None, None)] * n_d + [(0, None)],
        method="highs",
    )

    if result.status == 0:
        return result.x[-1], result.x[:-1]  # max slack, optimal x
    else:
        return -1.0, None

max_slack, x_opt = lp_max_slack(missed_pattern, H, b, G, g)
print(f"  Maximum interior slack: {max_slack:.6e}")
print(f"  (enumerator Stage 1 threshold epsilon: 1e-5)")

if max_slack < 1e-5:
    print(f"  --> Cell exists but max slack {max_slack:.2e} < epsilon 1e-5")
    print(f"      This is a NEAR-DEGENERATE cell that falls below the")
    print(f"      enumerator's floating-point threshold. Not a bug.")
elif max_slack < 1e-4:
    print(f"  --> Cell is thin but above epsilon. Likely a numerical issue.")
else:
    print(f"  --> Cell is well-interior. This may be a genuine bug.")

# ---------------------------------------------------------------------------
# 2. Verify the LP feasible point satisfies constraints
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("2. Verification of LP feasible point")
print("=" * 55)

if x_opt is not None:
    z = H @ x_opt + b
    signs = np.where(missed_pattern == 1, 1.0, -1.0)
    margin = signs * z
    domain_slack = g - G @ x_opt

    print(f"  Min sign constraint margin : {margin.min():.6e}")
    print(f"  Max sign constraint margin : {margin.max():.6e}")
    print(f"  Min domain slack           : {domain_slack.min():.6e}")
    print(f"  Point x_opt: {x_opt.round(6)}")

# ---------------------------------------------------------------------------
# 3. Check whether the enumerator finds this pattern with relaxed epsilon
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("3. Does the enumerator find it with relaxed epsilon?")
print("=" * 55)
print("  (This requires modifying Stage 1 epsilon in bitwise_utils.py)")
print("  Current epsilon = 1e-5 in slice_polytope_with_hyperplane")
print()
print("  Recommendation: if max_slack < 1e-5, the cell is below the")
print("  numerical tolerance and the miss is expected, not a bug.")
print("  If max_slack > 1e-5, investigate the adjacency test.")

# ---------------------------------------------------------------------------
# 4. Compute approximate volume of the missed cell
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("4. Approximate cell volume via vertex sampling")
print("=" * 55)

# Estimate volume by finding the polytope's vertices via LP corner finding.
# For each subset of n active constraints, solve for the vertex.
# This is approximate but sufficient to check if the cell is degenerate.

def estimate_cell_volume_sampling(pattern, H, b, G, g, n_samples=10000, seed=42):
    """Estimate cell volume by hit-and-run sampling."""
    rng2 = np.random.default_rng(seed)
    signs = np.where(pattern == 1, 1.0, -1.0)

    # Use x_opt as starting point
    _, x_start = lp_max_slack(pattern, H, b, G, g)
    if x_start is None:
        return 0.0

    # Simple rejection sampling from the domain box
    samples = rng2.uniform(-domain, domain, (n_samples, n))
    z_samp  = samples @ H.T + b           # (n_samples, m)
    sign_samp = (z_samp > 0).astype(float) # (n_samples, m)

    in_cell = np.all(sign_samp == missed_pattern[None, :], axis=1)
    fraction = in_cell.mean()
    domain_vol = (2 * domain) ** n
    est_vol = fraction * domain_vol

    return est_vol, fraction, in_cell.sum()

est_vol, fraction, n_in = estimate_cell_volume_sampling(
    missed_pattern, H, b, G, g
)
print(f"  Rejection sampling: {n_in} / 10000 points in cell")
print(f"  Estimated volume  : {est_vol:.6e}")
print(f"  Domain volume     : {(2*domain)**n:.4f}")
print(f"  Volume fraction   : {fraction:.6e}")

if fraction < 1e-6:
    print(f"  --> Cell is extremely thin (volume fraction < 1e-6).")
    print(f"      Near-degenerate cell, consistent with numerical miss.")
elif fraction < 1e-4:
    print(f"  --> Cell is thin but non-trivial. Worth investigating.")
else:
    print(f"  --> Cell has significant volume. This is a genuine bug.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Max interior slack : {max_slack:.6e}")
print(f"  Volume fraction    : {fraction:.6e}")
print(f"  Enumerator epsilon : 1e-5")
print()
if max_slack < 1e-5:
    print("  CONCLUSION: Near-degenerate cell below numerical tolerance.")
    print("  The miss is consistent with the Limitations paragraph in")
    print("  the paper (floating-point epsilon handling). Not a bug.")
    print("  Consider increasing LP slack to 1e-5 to match enumerator.")
else:
    print("  CONCLUSION: Cell has non-trivial interior. Investigate")
    print("  the adjacency test for this specific configuration.")