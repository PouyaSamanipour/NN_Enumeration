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

    Pre-compiles every specialization that will be needed at runtime:
    - _ibp_kernel    for each (n_rem) from n_layers down to 2
    - _eval_partial_nb for each n_known from 1 to n_layers-1
    - _eval_network_nb, _vertex_sign_check_nb, _compute_bboxes,
      _compute_bboxes_subset, _count_false_prunes_nb (once each)

    This avoids JIT recompilation during the enumeration loop when large
    data is already in memory (which can cause an OOM kill for deep networks).
    """
    print("  Warming up IBP Numba kernel...", end=" ", flush=True)
    t0 = time.time()

    # Build per-layer shapes matching the real network structure.
    # Layer 0    : (m_hidden, n_in)
    # Layers 1..n_layers-2 : (m_hidden, m_hidden)
    # Layer n_layers-1     : (1, m_hidden)   -- output
    def _w(l):
        in_sz = n_in if l == 0 else m_hidden
        out_sz = 1 if l == n_layers - 1 else m_hidden
        return (out_sz, in_sz)

    W_pos_t = tuple([np.ones(_w(l),  dtype=np.float64) for l in range(n_layers)])
    W_neg_t = tuple([np.zeros(_w(l), dtype=np.float64) for l in range(n_layers)])
    b_t     = tuple([np.zeros(_w(l)[0], dtype=np.float64) for l in range(n_layers)])
    W_full_t = tuple([(Wp + Wn) for Wp, Wn in zip(W_pos_t, W_neg_t)])

    dummy_lo    = np.zeros((1, n_in),    dtype=np.float64)
    dummy_hi    = np.ones((1, n_in),     dtype=np.float64)
    dummy_verts = np.random.randn(8, n_in).astype(np.float64)

    # ── _eval_network_nb and _eval_partial_nb (one compilation each per n_known)
    _eval_network_nb(dummy_verts, W_full_t, b_t, n_layers)
    for n_known in range(1, n_layers):
        _eval_partial_nb(dummy_verts, W_full_t, b_t, n_known)

    # ── _ibp_kernel: compile for each (n_rem, input_dim) pair that will occur.
    # At layer i (n_known = i+1), IBP receives a (M, m_hidden) input and runs
    # on layers i+1 .. n_layers-1.  Special case: n_known=0 uses raw (M, n_in).
    # We always call with n_rem >= 2 (n_rem=1 would be output-only — trivial).
    _ibp_kernel(dummy_lo, dummy_hi, W_pos_t, W_neg_t, b_t, n_layers)  # n_known=0
    dummy_lo_h = np.zeros((1, m_hidden), dtype=np.float64)
    dummy_hi_h = np.ones((1, m_hidden),  dtype=np.float64)
    for n_known in range(1, n_layers - 1):      # n_rem = n_layers - n_known >= 2
        n_rem   = n_layers - n_known
        Wp_rem  = tuple(W_pos_t[n_known:])
        Wn_rem  = tuple(W_neg_t[n_known:])
        b_rem   = tuple(b_t[n_known:])
        _ibp_kernel(dummy_lo_h, dummy_hi_h, Wp_rem, Wn_rem, b_rem, n_rem)

    # ── shape-independent kernels (compile once)
    B_dummy = _eval_network_nb(dummy_verts, W_full_t, b_t, n_layers)
    _vertex_sign_check_nb(B_dummy,
                          np.array([0], dtype=np.int64),
                          np.array([8], dtype=np.int64), 1)
    _compute_bboxes(dummy_verts,
                    np.array([0], dtype=np.int64),
                    np.array([8], dtype=np.int64),
                    1, n_in)
    _compute_bboxes_subset(dummy_verts,
                           np.array([0], dtype=np.int64),
                           np.array([8], dtype=np.int64),
                           np.array([0], dtype=np.int64), n_in)
    _count_false_prunes_nb(B_dummy,
                           np.array([0], dtype=np.int64),
                           np.array([8], dtype=np.int64), 1)

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
# Soundness check — Numba parallel forward pass
# ─────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _eval_partial_nb(all_verts, W_list, b_list, n_known):
    """
    Propagate all_verts through the first n_known layers exactly (all hidden,
    so ReLU always applied). Returns intermediate activations (V, m_out).

    Used to get a tight starting interval for IBP on the remaining layers:
    for a cell at enumeration layer i, layers 0..i-1 have fixed activation
    patterns, so propagating vertices through them is exact.

    Parameters
    ----------
    all_verts : (V, n_in) float64
    W_list    : tuple of 2D arrays — full weight list
    b_list    : tuple of 1D arrays — full bias list
    n_known   : int — number of layers to propagate exactly (>0)

    Returns
    -------
    h : (V, m_out) float64 — activations after layer n_known-1
    """
    V     = all_verts.shape[0]
    m_out = W_list[n_known - 1].shape[0]
    h     = np.empty((V, m_out), dtype=np.float64)

    for v in prange(V):
        x = all_verts[v].copy()
        for layer in range(n_known):
            W  = W_list[layer]
            bv = b_list[layer]
            m  = W.shape[0]
            y  = np.empty(m)
            for j in range(m):
                s = bv[j]
                for k in range(W.shape[1]):
                    s += W[j, k] * x[k]
                y[j] = s
            # All n_known layers are hidden — always apply ReLU
            for j in range(m):
                if y[j] < 0.0:
                    y[j] = 0.0
            x = y
        h[v] = x

    return h


@njit(parallel=True, cache=True)
def _eval_network_nb(all_verts, W_list, b_list, n_layers):
    """
    Evaluate the barrier network at every vertex (exact forward pass, no IBP).

    Parameters
    ----------
    all_verts : (V_total, n_in) float64 — all vertices concatenated
    W_list    : tuple of (m_out, m_in) float64 arrays per layer
    b_list    : tuple of (m_out,) float64 arrays per layer
    n_layers  : int

    Returns
    -------
    B_vals : (V_total,) float64
    """
    V = all_verts.shape[0]
    B_vals = np.empty(V, dtype=np.float64)

    for v in prange(V):
        x = all_verts[v].copy()
        for layer in range(n_layers):
            W  = W_list[layer]
            bv = b_list[layer]
            m_out = W.shape[0]
            y = np.empty(m_out)
            for j in range(m_out):
                s = bv[j]
                for k in range(W.shape[1]):
                    s += W[j, k] * x[k]
                y[j] = s
            # ReLU on all layers except the last
            if layer < n_layers - 1:
                for j in range(m_out):
                    if y[j] < 0.0:
                        y[j] = 0.0
            x = y
        B_vals[v] = x[0]

    return B_vals


@njit(parallel=True, cache=True)
def _count_false_prunes_nb(B_vals, cell_start, cell_end, N):
    """
    For each pruned cell, check whether B changes sign across its vertices.
    A sign change (min < 0 and max > 0) means the cell straddles B=0 — a
    false prune.

    Parameters
    ----------
    B_vals     : (V_total,) float64
    cell_start : (N,) int64
    cell_end   : (N,) int64
    N          : int

    Returns
    -------
    flags : (N,) int64 — 1 if false prune, 0 otherwise
    """
    flags = np.zeros(N, dtype=np.int64)
    for c in prange(N):
        lo =  1e300
        hi = -1e300
        for v in range(cell_start[c], cell_end[c]):
            val = B_vals[v]
            if val < lo: lo = val
            if val > hi: hi = val
        if lo < 0.0 and hi > 0.0:
            flags[c] = 1
    return flags


@njit(parallel=True, cache=True)
def _compute_bboxes_subset(all_verts, cell_start, cell_end, indices, n):
    """
    Compute bounding boxes for a subset of cells from a pre-stacked vertex array.
    Avoids re-stacking: reuses all_verts / cell_start / cell_end from the eval step.

    Parameters
    ----------
    all_verts  : (V_total, n) float64 — all vertices concatenated
    cell_start : (N,) int64
    cell_end   : (N,) int64
    indices    : (M,) int64 — indices of the cells to process
    n          : int — input dimension

    Returns
    -------
    bbox_lo, bbox_hi : (M, n) float64
    """
    M = len(indices)
    bbox_lo = np.empty((M, n), dtype=np.float64)
    bbox_hi = np.empty((M, n), dtype=np.float64)
    for m in prange(M):
        c = indices[m]
        for d in range(n):
            lo =  1e300
            hi = -1e300
            for v in range(cell_start[c], cell_end[c]):
                val = all_verts[v, d]
                if val < lo: lo = val
                if val > hi: hi = val
            bbox_lo[m, d] = lo
            bbox_hi[m, d] = hi
    return bbox_lo, bbox_hi


@njit(parallel=True, cache=True)
def _vertex_sign_check_nb(B_vals, cell_start, cell_end, N):
    """
    Classify each cell based on the sign of B at its vertices.

    Returns
    -------
    status : (N,) int8
        +1  all vertices positive  (same-sign, needs IBP to try pruning)
        -1  all vertices negative  (same-sign, needs IBP to try pruning)
         0  mixed signs            (definite boundary cell, keep immediately)
    """
    status = np.empty(N, dtype=np.int8)
    for c in prange(N):
        lo =  1e300
        hi = -1e300
        for v in range(cell_start[c], cell_end[c]):
            val = B_vals[v]
            if val < lo: lo = val
            if val > hi: hi = val
        if lo < 0.0 and hi > 0.0:
            status[c] = 0    # mixed sign — definite boundary
        elif hi <= 0.0:
            status[c] = -1   # all non-positive
        else:
            status[c] = 1    # all non-negative
    return status


def vertex_ibp_filter(cell_vertices_list, W_pos, W_neg, b,
                      is_last_layer=False, n_known=0,
                      threshold=0.0, verbose=True):
    """
    Two-step filter combining vertex sign check and IBP.

    Step 1 — vertex sign check (exact forward pass at polytope vertices):
        Mixed signs  → definite boundary cell → keep, skip IBP.
        Uniform sign → uncertain (B may cross zero in interior) → go to step 2.

    Step 2 — IBP on bounding box (only for uniform-sign cells):
        When n_known > 0: propagate vertices through the first n_known layers
        exactly (activation pattern is fixed there), then compute the bounding
        box in that intermediate space and apply IBP only for remaining layers.
        This is tighter than IBP from the raw input bounding box.
        B_lo > 0 or B_hi < 0 → certified no crossing → prune.
        Otherwise             → keep conservatively.

    Note: when is_last_layer=True, B is affine on each cell so the vertex
    check is exact — uniform-sign cells are pruned directly, no IBP needed.

    Parameters
    ----------
    cell_vertices_list : list of (n_verts, n) float64 arrays
    W_pos, W_neg       : lists of weight arrays (positive/negative split)
    b                  : list of bias arrays
    is_last_layer      : bool  — skip IBP step when True
    n_known            : int   — layers with fixed activation (0 = use raw input bbox)
    threshold          : float — tolerance around B=0

    Returns
    -------
    filtered_cells : list of vertex arrays
    keep_mask      : (N,) bool array
    """
    N = len(cell_vertices_list)
    if N == 0:
        return cell_vertices_list, np.array([], dtype=bool)

    n_layers = len(W_pos)
    W_full_t = tuple([(Wp + Wn).astype(np.float64)
                      for Wp, Wn in zip(W_pos, W_neg)])
    b_t = tuple(b)

    # ── Build vertex index arrays ──────────────────────────────────────────
    sizes      = np.array([v.shape[0] for v in cell_vertices_list], dtype=np.int64)
    cell_start = np.zeros(N, dtype=np.int64)
    cell_end   = np.zeros(N, dtype=np.int64)
    cell_start[1:] = np.cumsum(sizes[:-1])
    cell_end        = cell_start + sizes
    all_verts  = np.vstack(cell_vertices_list).astype(np.float64)

    # ── Step 1: evaluate B at all vertices ────────────────────────────────
    t0     = time.time()
    B_vals = _eval_network_nb(all_verts, W_full_t, b_t, n_layers)
    t_eval = time.time() - t0

    status = _vertex_sign_check_nb(B_vals, cell_start, cell_end, N)
    # status == 0  → mixed sign  → keep immediately
    # status != 0  → same sign   → IBP step (or prune directly if last layer)

    mixed_mask = (status == 0)
    same_mask  = (status != 0)
    n_mixed    = int(mixed_mask.sum())
    n_same     = int(same_mask.sum())

    t_ibp        = 0.0
    n_ibp_pruned = 0

    if is_last_layer or n_same == 0:
        # Vertex check is exact here → prune all uniform-sign cells directly
        keep_mask    = mixed_mask
        n_ibp_pruned = n_same
    else:
        # ── Step 2: IBP on uniform-sign cells ─────────────────────────────
        same_indices = np.where(same_mask)[0].astype(np.int64)

        t1 = time.time()
        if n_known > 0:
            # Propagate ONLY the same-sign cell vertices through n_known layers.
            # Propagating all_verts (which includes already-kept mixed-sign
            # cells) wastes memory proportional to n_mixed.  For deep networks
            # this can trigger OOM when IBP runs at multiple intermediate layers.
            vert_keep = np.zeros(len(all_verts), dtype=np.bool_)
            for k in same_indices:
                vert_keep[cell_start[k]:cell_end[k]] = True
            same_all_verts = all_verts[vert_keep]

            # Build per-cell offsets for the same-sign subset.
            same_sizes     = sizes[same_indices]
            same_start_sub = np.zeros(len(same_indices), dtype=np.int64)
            same_end_sub   = np.zeros(len(same_indices), dtype=np.int64)
            if len(same_indices) > 1:
                same_start_sub[1:] = np.cumsum(same_sizes[:-1])
            same_end_sub[:] = same_start_sub + same_sizes

            h_verts  = _eval_partial_nb(same_all_verts, W_full_t, b_t, n_known)
            m_hidden = h_verts.shape[1]
            del same_all_verts  # free immediately — no longer needed

            bbox_lo, bbox_hi = _compute_bboxes(
                h_verts, same_start_sub, same_end_sub, len(same_indices), m_hidden
            )
            del h_verts  # free immediately

            # IBP only on remaining layers
            W_pos_t = tuple(W_pos[n_known:])
            W_neg_t = tuple(W_neg[n_known:])
            b_rem_t = tuple(b[n_known:])
            n_rem   = n_layers - n_known
        else:
            # No known layers — bbox in raw input space
            n_in    = all_verts.shape[1]
            bbox_lo, bbox_hi = _compute_bboxes_subset(
                all_verts, cell_start, cell_end, same_indices, n_in
            )
            W_pos_t = tuple(W_pos)
            W_neg_t = tuple(W_neg)
            b_rem_t = b_t
            n_rem   = n_layers

        B_lo, B_hi = _ibp_kernel(bbox_lo, bbox_hi, W_pos_t, W_neg_t, b_rem_t, n_rem)
        t_ibp = time.time() - t1

        ibp_keep     = ~((B_lo > threshold) | (B_hi < -threshold))
        n_ibp_pruned = int((~ibp_keep).sum())

        keep_mask = mixed_mask.copy()
        for idx, k in enumerate(same_indices):
            if ibp_keep[idx]:
                keep_mask[k] = True

    n_kept   = int(keep_mask.sum())
    n_pruned = N - n_kept

    if verbose:
        print(f"  Vertex+IBP filter: {N} cells → {n_kept} kept, {n_pruned} pruned "
              f"({100.0*n_pruned/N:.1f}%) "
              f"[{n_mixed} mixed-sign kept, {n_ibp_pruned} pruned by IBP/"
              f"{'exact' if is_last_layer else 'ibp'} "
              f"| eval {t_eval:.3f}s, ibp {t_ibp:.3f}s]")

    filtered_cells = [cell_vertices_list[i] for i in range(N) if keep_mask[i]]
    return filtered_cells, keep_mask


def exact_boundary_filter(cell_vertices_list, W_full_t, b_full_t, verbose=True):
    """
    Exact boundary filter for the final enumeration layer.

    B(x) is affine on every final linear region, so evaluating B at the
    polytope vertices is exact. Keeps only cells where B changes sign
    (min(B) <= 0 <= max(B)) — these are the true boundary cells.

    Parameters
    ----------
    cell_vertices_list : list of (n_verts, n) float64 arrays
    W_full_t           : tuple of (m_out, m_in) float64 arrays — full weights
    b_full_t           : tuple of (m_out,) float64 arrays — biases
    verbose            : bool

    Returns
    -------
    filtered_cells : list of vertex arrays (boundary cells only)
    keep_mask      : (N,) bool array
    """
    N = len(cell_vertices_list)
    if N == 0:
        return cell_vertices_list, np.array([], dtype=bool)

    n_layers = len(W_full_t)

    sizes      = np.array([v.shape[0] for v in cell_vertices_list], dtype=np.int64)
    cell_start = np.zeros(N, dtype=np.int64)
    cell_end   = np.zeros(N, dtype=np.int64)
    cell_start[1:] = np.cumsum(sizes[:-1])
    cell_end        = cell_start + sizes
    all_verts  = np.vstack(cell_vertices_list).astype(np.float64)

    t0     = time.time()
    B_vals = _eval_network_nb(all_verts, W_full_t, b_full_t, n_layers)
    t_eval = time.time() - t0

    keep_mask = _vertex_sign_check_nb(B_vals, cell_start, cell_end, N) == 0

    n_kept   = int(keep_mask.sum())
    n_pruned = N - n_kept

    if verbose:
        print(f"  Exact filter: {N} → {n_kept} boundary cells, "
              f"{n_pruned} pruned ({100.0*n_pruned/N:.1f}%) "
              f"[eval: {t_eval:.3f}s]")

    filtered_cells = [cell_vertices_list[i] for i in range(N) if keep_mask[i]]
    return filtered_cells, keep_mask


def ibp_soundness_check(pruned_cells, W_pos, W_neg, b):
    """
    Verify that none of the IBP-pruned cells actually straddles B(x)=0.
    Uses a Numba-parallel exact forward pass — no PyTorch required.

    Parameters
    ----------
    pruned_cells : list of (n_verts, n) float64 arrays
    W_pos, W_neg : lists of weight arrays (positive/negative split)
    b            : list of bias arrays

    Returns
    -------
    n_false_prunes : int  (should always be 0 if IBP is sound)
    """
    if not pruned_cells:
        return 0

    n_layers = len(W_pos)

    # Reconstruct full weights: W = W_pos + W_neg
    W_full_t = tuple([(Wp + Wn).astype(np.float64)
                      for Wp, Wn in zip(W_pos, W_neg)])
    b_t = tuple(b)

    # Build concatenated vertex array + per-cell offsets
    N          = len(pruned_cells)
    sizes      = np.array([v.shape[0] for v in pruned_cells], dtype=np.int64)
    cell_start = np.zeros(N, dtype=np.int64)
    cell_end   = np.zeros(N, dtype=np.int64)
    cell_start[1:] = np.cumsum(sizes[:-1])
    cell_end        = cell_start + sizes
    all_verts  = np.vstack(pruned_cells).astype(np.float64)

    # Parallel forward pass at every vertex
    B_vals = _eval_network_nb(all_verts, W_full_t, b_t, n_layers)

    # Parallel sign-change check per cell
    flags = _count_false_prunes_nb(B_vals, cell_start, cell_end, N)

    return int(flags.sum())


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