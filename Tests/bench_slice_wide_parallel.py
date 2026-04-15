"""
bench_slice_wide_parallel.py
============================
Benchmark sequential vs parallel slice_polytope_wide.

Generates synthetic polytope data at several sizes that cover the range
seen in practice (small early-layer polytopes up to large late-layer ones),
then compares wall-clock time and verifies that both versions produce
identical results.

Run:
    python Tests/bench_slice_wide_parallel.py
"""

import sys, time
import numpy as np
import numba as nb
from numba import njit, prange

sys.path.insert(0, ".")
from relu_region_enumerator.bitwise_utils import (
    _wide_popcount_ge,
    _wide_mask_or_and,
    _slice_polytope_wide_seq,
    _slice_polytope_wide_par,
    slice_polytope_wide,          # dispatcher
)

# ─────────────────────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────────────────────

def make_test_case(n_verts, n_dim, num_words, hyperplane_idx, rng):
    """Random polytope with a roughly 50/50 hyperplane split."""
    verts      = rng.standard_normal((n_verts, n_dim)).astype(np.float64)
    h_val      = rng.standard_normal(n_verts).astype(np.float64)
    # ensure at least some vertices on each side
    h_val[:n_verts // 2]  -= 0.5
    h_val[n_verts // 2:]  += 0.5

    # random uint64 masks with ~(n_dim-1) bits set per vertex
    masks = np.zeros((n_verts, num_words), dtype=np.uint64)
    for v in range(n_verts):
        bits = rng.choice(num_words * 64, size=n_dim + 2, replace=False)
        for b in bits:
            masks[v, b // 64] |= np.uint64(1) << np.uint64(b % 64)

    return verts, h_val, masks


# ─────────────────────────────────────────────────────────────
# Correctness check
# ─────────────────────────────────────────────────────────────

def results_equal(r_seq, r_par, tol=1e-10):
    """Check that sequential and parallel results contain the same vertices."""
    cv_seq = r_seq[2]
    cv_par = r_par[2]
    if len(cv_seq) != len(cv_par):
        return False, f"created_verts count differs: {len(cv_seq)} vs {len(cv_par)}"
    if len(cv_seq) > 0:
        cv_seq_s = np.sort(cv_seq, axis=0)
        cv_par_s = np.sort(cv_par, axis=0)
        if not np.allclose(cv_seq_s, cv_par_s, atol=tol):
            diff = np.abs(cv_seq_s - cv_par_s).max()
            return False, f"max vertex diff = {diff:.2e}"
    for side in range(2):
        s_seq = np.sort(r_seq[0][side], axis=0)
        s_par = np.sort(r_par[0][side], axis=0)
        if s_seq.shape != s_par.shape:
            return False, f"polytope[{side}] size differs"
        if not np.allclose(s_seq, s_par, atol=tol):
            return False, f"polytope[{side}] vertex mismatch"
    return True, "OK"


# ─────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────

def bench(label, fn, *args, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    return result, times


def main():
    rng = np.random.default_rng(42)

    # Warm up Numba JIT (compile both versions once, not timed)
    print("Warming up JIT (compiling seq, par, and dispatcher)...")
    warm_verts, warm_hval, warm_masks = make_test_case(50, 8, 1, 10, rng)
    _slice_polytope_wide_seq(warm_verts, warm_hval, warm_masks, 10, 8)
    _slice_polytope_wide_par(warm_verts, warm_hval, warm_masks, 10, 8)
    slice_polytope_wide(warm_verts, warm_hval, warm_masks, 10, 8)
    print("Done.\n")

    test_cases = [
        # (label,           n_verts, n_dim, num_words, h_idx)
        ("small  V=100  n=4",   100,   4, 1,  5),
        ("medium V=500  n=8",   500,   8, 1, 20),
        ("large  V=2000 n=8",  2000,   8, 1, 40),
        ("large  V=2000 n=12", 2000,  12, 1, 50),
        ("xlarge V=5000 n=8",  5000,   8, 1, 60),
        ("wide   V=1000 n=8 w=2", 1000, 8, 2, 70),
    ]

    header = (f"{'Case':<28}  {'Seq (ms)':>10}  {'Par (ms)':>10}"
              f"  {'Dispatch (ms)':>13}  {'Speedup':>8}  {'Correct':>8}")
    print(header)
    print("-" * len(header))

    all_correct = True
    for label, n_verts, n_dim, num_words, h_idx in test_cases:
        verts, h_val, masks = make_test_case(n_verts, n_dim, num_words, h_idx, rng)

        r_seq,  t_seq  = bench(label, _slice_polytope_wide_seq, verts, h_val, masks, h_idx, n_dim)
        r_par,  t_par  = bench(label, _slice_polytope_wide_par, verts, h_val, masks, h_idx, n_dim)
        r_disp, t_disp = bench(label, slice_polytope_wide,      verts, h_val, masks, h_idx, n_dim)

        ok_sp, msg_sp = results_equal(r_seq, r_par)
        ok_sd, msg_sd = results_equal(r_seq, r_disp)
        ok = ok_sp and ok_sd
        msg = msg_sp if not ok_sp else msg_sd
        if not ok:
            all_correct = False

        med_seq  = np.median(t_seq)  * 1000
        med_par  = np.median(t_par)  * 1000
        med_disp = np.median(t_disp) * 1000
        speedup  = med_seq / med_par if med_par > 0 else float('inf')

        print(f"{label:<28}  {med_seq:>10.2f}  {med_par:>10.2f}"
              f"  {med_disp:>13.2f}  {speedup:>7.2f}x"
              f"  {'✓' if ok else '✗ ' + msg}")

    print()
    if all_correct:
        print("All correctness checks PASSED.")
    else:
        print("SOME CORRECTNESS CHECKS FAILED.")

    print(f"\nNumba threads available: {nb.get_num_threads()}")


if __name__ == "__main__":
    main()
