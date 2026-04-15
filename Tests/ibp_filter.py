"""
ibp_filter.py
=============
Numba-parallel IBP pre-filter for layer-0 cells.

Computes certified [B_lo, B_hi] bounds on the barrier function B(x)
over each cell's bounding box. Cells where B_lo > 0 or B_hi < 0
cannot contain B(x)=0 and are pruned before layer-1 expansion.

Supports any number of hidden layers via the general version.
The 2-layer specific version is faster due to unrolled loops.

Integration in core.py:
    from ibp_filter import ibp_prefilter

    # After layer-0 enumeration, before layer-1:
    layer_0_cells = ibp_prefilter(layer_0_cells, W, b)
"""

import numpy as np
from numba import njit, prange
import time


# ─────────────────────────────────────────────
# Weight preprocessing — called once at startup
# ─────────────────────────────────────────────

def precompute_ibp_weights(W, b):
    """
    Split each weight matrix into positive and negative parts.
    Called ONCE before the enumeration loop.

    Parameters
    ----------
    W : list of (m_out, m_in) float64 arrays
    b : list of (m_out,) float64 arrays

    Returns
    -------
    W_pos : list of arrays — max(W[i], 0)
    W_neg : list of arrays — min(W[i], 0)
    b     : list of arrays — unchanged
    """
    W_pos = [np.maximum(Wi, 0.0).astype(np.float64) for Wi in W]
    W_neg = [np.minimum(Wi, 0.0).astype(np.float64) for Wi in W]
    b_arr = [bi.astype(np.float64) for bi in b]
    return W_pos, W_neg, b_arr


# ─────────────────────────────────────────────
# Core Numba kernel — parallel over all cells
# Works for any number of hidden layers via
# pre-stacked weight tensors
# ─────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _ibp_kernel(bbox_lo, bbox_hi, W_pos_list, W_neg_list, b_list, n_layers):
    """
    Numba parallel IBP over N cells.

    Parameters
    ----------
    bbox_lo  : (N, n_in) float64
    bbox_hi  : (N, n_in) float64
    W_pos_list, W_neg_list : tuple of 2D arrays per layer
    b_list   : tuple of 1D arrays per layer
    n_layers : int — total layers including output

    Returns
    -------
    B_lo : (N,) float64
    B_hi : (N,) float64
    """
    N = bbox_lo.shape[0]
    B_lo_out = np.empty(N, dtype=np.float64)
    B_hi_out = np.empty(N, dtype=np.float64)

    for c in prange(N):
        # Current interval — starts as input bounding box
        lo = bbox_lo[c].copy()
        hi = bbox_hi[c].copy()

        for layer in range(n_layers):
            Wp = W_pos_list[layer]
            Wn = W_neg_list[layer]
            bv = b_list[layer]
            m_out = Wp.shape[0]
            m_in  = Wp.shape[1]

            new_lo = np.empty(m_out)
            new_hi = np.empty(m_out)

            for j in range(m_out):
                lo_j = bv[j]
                hi_j = bv[j]
                for i in range(m_in):
                    lo_j += Wp[j, i] * lo[i] + Wn[j, i] * hi[i]
                    hi_j += Wp[j, i] * hi[i] + Wn[j, i] * lo[i]
                new_lo[j] = lo_j
                new_hi[j] = hi_j

            # Apply ReLU on all layers except the last (output)
            if layer < n_layers - 1:
                for j in range(m_out):
                    new_lo[j] = max(0.0, new_lo[j])
                    new_hi[j] = max(0.0, new_hi[j])

            lo = new_lo
            hi = new_hi

        # Output is scalar (1,) — store result
        B_lo_out[c] = lo[0]
        B_hi_out[c] = hi[0]

    return B_lo_out, B_hi_out


# ─────────────────────────────────────────────
# Bounding box computation — parallel
# ─────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _compute_bboxes(all_verts, cell_start, cell_end, N, n):
    """
    Compute bounding boxes for all cells in parallel.

    Parameters
    ----------
    all_verts  : (total_verts, n) float64 — all vertices concatenated
    cell_start : (N,) int64 — start index of each cell in all_verts
    cell_end   : (N,) int64 — end index (exclusive)
    N, n       : int

    Returns
    -------
    bbox_lo, bbox_hi : (N, n) float64
    """
    bbox_lo = np.empty((N, n), dtype=np.float64)
    bbox_hi = np.empty((N, n), dtype=np.float64)

    for c in prange(N):
        for d in range(n):
            lo =  1e300
            hi = -1e300
            for v in range(cell_start[c], cell_end[c]):
                val = all_verts[v, d]
                if val < lo: lo = val
                if val > hi: hi = val
            bbox_lo[c, d] = lo
            bbox_hi[c, d] = hi

    return bbox_lo, bbox_hi


def compute_bboxes(cell_vertices_list):
    """
    Compute bounding boxes for a list of vertex arrays.

    Parameters
    ----------
    cell_vertices_list : list of (n_verts, n) float64 arrays

    Returns
    -------
    bbox_lo, bbox_hi : (N, n) float64
    """
    N = len(cell_vertices_list)
    n = cell_vertices_list[0].shape[1]

    # Stack all vertices and compute offsets
    sizes      = np.array([v.shape[0] for v in cell_vertices_list], dtype=np.int64)
    cell_start = np.zeros(N, dtype=np.int64)
    cell_end   = np.zeros(N, dtype=np.int64)
    cell_start[1:] = np.cumsum(sizes[:-1])
    cell_end       = cell_start + sizes

    all_verts = np.vstack(cell_vertices_list).astype(np.float64)

    return _compute_bboxes(all_verts, cell_start, cell_end, N, n)


# ─────────────────────────────────────────────
# Numba warmup — call once at module load
# ─────────────────────────────────────────────

def warmup_numba(n_in=3, m_hidden=8, n_layers=3):
    """
    Warm up Numba JIT compilation with a tiny dummy network.
    Call this once at startup before the main enumeration.
    """
    print("  Warming up IBP Numba kernel...", end=" ", flush=True)
    t0 = time.time()

    dummy_lo = np.zeros((1, n_in), dtype=np.float64)
    dummy_hi = np.ones((1, n_in), dtype=np.float64)

    # Build dummy weight tuples for Numba
    W_pos_t = tuple([np.ones((m_hidden, n_in), dtype=np.float64),
                     np.ones((m_hidden, m_hidden), dtype=np.float64),
                     np.ones((1, m_hidden), dtype=np.float64)])
    W_neg_t = tuple([np.zeros((m_hidden, n_in), dtype=np.float64),
                     np.zeros((m_hidden, m_hidden), dtype=np.float64),
                     np.zeros((1, m_hidden), dtype=np.float64)])
    b_t     = tuple([np.zeros(m_hidden, dtype=np.float64),
                     np.zeros(m_hidden, dtype=np.float64),
                     np.zeros(1, dtype=np.float64)])

    _ibp_kernel(dummy_lo, dummy_hi, W_pos_t, W_neg_t, b_t, 3)

    # Also warm up bbox computation
    dummy_verts = np.random.randn(8, n_in).astype(np.float64)
    _compute_bboxes(dummy_verts,
                    np.array([0], dtype=np.int64),
                    np.array([8], dtype=np.int64),
                    1, n_in)

    print(f"done ({time.time()-t0:.2f}s)")


# ─────────────────────────────────────────────
# Main interface — drop this into core.py
# ─────────────────────────────────────────────

def ibp_prefilter(cell_vertices_list, W_pos, W_neg, b, threshold=0.0,
                  verbose=True):
    """
    IBP pre-filter: remove layer-0 cells that cannot contain B(x)=0.

    Parameters
    ----------
    cell_vertices_list : list of (n_verts, n) float64 arrays
                         — output of layer-0 Enumerator_rapid
    W_pos : list of (m_out, m_in) float64 — positive parts of weights
    W_neg : list of (m_out, m_in) float64 — negative parts of weights
    b     : list of (m_out,) float64      — biases
    threshold : float — tolerance around B=0 (default 0)
    verbose   : bool

    Returns
    -------
    filtered_cells : list of vertex arrays (subset of input)
    keep_mask      : (N,) bool array
    """
    N = len(cell_vertices_list)
    if N == 0:
        return cell_vertices_list, np.array([], dtype=bool)

    n_layers = len(W_pos)

    # Convert weight lists to tuples for Numba
    W_pos_t = tuple(W_pos)
    W_neg_t = tuple(W_neg)
    b_t     = tuple(b)

    # Compute bounding boxes
    t0 = time.time()
    bbox_lo, bbox_hi = compute_bboxes(cell_vertices_list)
    t_bbox = time.time() - t0

    # Run IBP kernel
    t1 = time.time()
    B_lo, B_hi = _ibp_kernel(bbox_lo, bbox_hi, W_pos_t, W_neg_t, b_t, n_layers)
    t_ibp = time.time() - t1

    # Filter: prune if certified entirely positive or entirely negative
    keep_mask = ~((B_lo > threshold) | (B_hi < -threshold))

    n_kept   = int(keep_mask.sum())
    n_pruned = N - n_kept

    if verbose:
        print(f"  IBP pre-filter: {N} cells → {n_kept} kept, "
              f"{n_pruned} pruned ({100.0*n_pruned/N:.1f}%) "
              f"[bbox: {t_bbox:.3f}s, ibp: {t_ibp:.3f}s]")

    filtered_cells = [cell_vertices_list[i] for i in range(N) if keep_mask[i]]
    return filtered_cells, keep_mask


# ─────────────────────────────────────────────
# Convenience wrapper: precompute + filter
# ─────────────────────────────────────────────

def ibp_prefilter_from_weights(cell_vertices_list, W, b,
                                threshold=0.0, verbose=True):
    """
    Same as ibp_prefilter but takes raw W, b lists and
    handles weight preprocessing internally.

    Use this when calling from outside core.py.
    For core.py integration, precompute W_pos/W_neg once
    and call ibp_prefilter directly.
    """
    W_pos, W_neg, b_arr = precompute_ibp_weights(W, b)
    return ibp_prefilter(cell_vertices_list, W_pos, W_neg, b_arr,
                         threshold, verbose)


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("IBP pre-filter self-test")
    print("=" * 55)

    np.random.seed(0)
    n, m1, m2 = 8, 16, 16  # hi-ord8 dimensions

    W = [
        np.random.randn(m1, n ).astype(np.float64),
        np.random.randn(m2, m1).astype(np.float64),
        np.random.randn(1,  m2).astype(np.float64),
    ]
    b = [
        np.random.randn(m1).astype(np.float64),
        np.random.randn(m2).astype(np.float64),
        np.zeros(1,         dtype=np.float64),
    ]

    W_pos, W_neg, b_arr = precompute_ibp_weights(W, b)

    # Warm up Numba
    warmup_numba(n_in=n, m_hidden=m1, n_layers=3)

    # ── Test 1: simulated layer-0 cells ──
    print("\nTest 1: 11,355 simulated layer-0 cells (hi-ord8 scale)")
    N = 11355
    cells = [np.random.uniform(-2, 2, (8, n)).astype(np.float64)
             for _ in range(N)]

    t0 = time.time()
    filtered, mask = ibp_prefilter(cells, W_pos, W_neg, b_arr, verbose=True)
    print(f"  Total time: {time.time()-t0:.3f}s")

    # ── Test 2: cells far from boundary (should be pruned) ──
    print("\nTest 2: cells with large positive B (should be pruned)")
    # Make network output large positive: shift bias
    b_shifted = [b_arr[0].copy(), b_arr[1].copy(),
                 np.array([10.0], dtype=np.float64)]  # large positive bias
    W_pos2, W_neg2, b2 = precompute_ibp_weights(W, b_shifted)
    cells2 = [np.random.uniform(-0.1, 0.1, (8, n)).astype(np.float64)
              for _ in range(100)]
    filtered2, mask2 = ibp_prefilter(cells2, W_pos2, W_neg2, b2, verbose=True)

    # ── Test 3: soundness check ──
    print("\nTest 3: Soundness — no genuine boundary cells pruned")
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(n, m1), nn.ReLU(),
                nn.Linear(m1, m2), nn.ReLU(),
                nn.Linear(m2, 1)
            )
        def forward(self, x): return self.layers(x)

    net = Net().double()
    with torch.no_grad():
        net.layers[0].weight.data = torch.tensor(W[0])
        net.layers[0].bias.data   = torch.tensor(b[0])
        net.layers[2].weight.data = torch.tensor(W[1])
        net.layers[2].bias.data   = torch.tensor(b[1])
        net.layers[4].weight.data = torch.tensor(W[2])
        net.layers[4].bias.data   = torch.tensor(b[2])

    false_prunes = 0
    for i in range(N):
        if mask[i]:
            continue  # kept — skip
        # Pruned cell — verify B doesn't cross zero
        verts = cells[i]
        with torch.no_grad():
            B_v = net(torch.tensor(verts)).numpy().flatten()
        if B_v.min() < 0 and B_v.max() > 0:
            false_prunes += 1

    print(f"  False prunes: {false_prunes} / {(~mask).sum()} pruned cells")
    if false_prunes == 0:
        print("  SOUNDNESS PASSED ✓")
    else:
        print("  SOUNDNESS FAILED ✗ — investigate!")

    print("\n" + "=" * 55)
    print("Integration template for core.py:")
    print("=" * 55)
    print("""
# At top of core.py:
from ibp_filter import precompute_ibp_weights, ibp_prefilter, warmup_numba

# Once, before enumeration starts:
W_pos, W_neg, b_arr = precompute_ibp_weights(W, b)
warmup_numba(n_in=n, m_hidden=W[0].shape[0], n_layers=len(W))

# After layer-0 enumeration:
layer_0_cells, _ = ibp_prefilter(
    layer_0_cells, W_pos, W_neg, b_arr,
    threshold=1e-4,   # match your boundary tolerance
    verbose=True
)
print(f'Layer-0 after IBP filter: {len(layer_0_cells)} cells')

# Then proceed with layer-1 expansion as normal
""")