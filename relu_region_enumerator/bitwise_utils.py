
"""
bitwise_utils.py
================

Core geometric primitives for vertex-based ReLU region enumeration.

The key computational idea: each vertex of an enumerated polytope is annotated
with a bitmask that encodes which active hyperplanes (from the boundary of the
current input domain plus all previously processed ReLU hyperplanes) it lies
on.  Two vertices are adjacent in the polytope edge-graph if and only if their
shared bitmask has at least (n-1) bits set, where n is the input dimension.
This reduces the O(n^3) LP-feasibility adjacency test to a single bitwise AND
plus a popcount, i.e. O(1) per vertex pair.

Functions
---------
generate_mask                       -- Assign hyperplane-incidence bitmasks to vertices (JIT, parallel).
slice_polytope_with_hyperplane      -- Split a polytope with a ReLU hyperplane using bitmask adjacency (JIT).
slice_polytope_with_hyperplane_jit  -- Alternate JIT slicer with slightly relaxed tolerances.
slice_polytope_parallel             -- Two-pass parallel slicer for large polytopes (JIT, parallel).
Polytope_formation_hd               -- Assemble the two child polytopes from the slicing results.
Enumerator_rapid                    -- Enumerate all linear regions of one neural-network layer.
Enumerator_rapid_face               -- Enumerate sub-regions of a ZLS face by sequential hyperplane cuts.
generate_mask_wide                  -- Wide-mask variant of generate_mask for > 64 hyperplanes.
slice_polytope_wide                 -- Wide-mask slicer (pure NumPy, no JIT).
RaggedPolytopeStorage               -- In-memory ragged array for variable-vertex polytopes.
HybridRaggedStorage                 -- RaggedPolytopeStorage with automatic disk overflow.
"""

import os
import tempfile
import time

from click import Tuple
import numpy as np
from numba import njit, prange
from numba import types  # noqa: F401  (kept for potential extension)
from numba.typed import List  # noqa: F401

from .ibp_filter import vertex_ibp_filter


# ---------------------------------------------------------------------------
# Bitmask generation
# ---------------------------------------------------------------------------

@njit(parallel=True)
def generate_mask(vertices, hyperplanes, b, tolerance=1e-5):
    """Compute a hyperplane-incidence bitmask for each vertex.

    A bit h is set in the mask for vertex v if |hyperplanes[h] · v + b[h]|
    <= tolerance, meaning v lies (numerically) on the h-th hyperplane.

    Parameters
    ----------
    vertices   : (V, n) float64 array  -- polytope vertices.
    hyperplanes: (H, n) float64 array  -- hyperplane normal vectors.
    b          : (H,)   float64 array  -- hyperplane offsets.
    tolerance  : float                 -- incidence tolerance (default 1e-7).

    Returns
    -------
    masks : (V,) uint64 array  -- bitmask per vertex.
    """
    n_verts  = vertices.shape[0]
    n_planes = hyperplanes.shape[0]
    masks = np.zeros(n_verts, dtype=np.uint64)

    for i in prange(n_verts):
        cntr=0
        mask = np.uint64(0)
        v    = vertices[i]
        for h in range(n_planes):
            val = hyperplanes[h, :] @ v + b[h]
            if np.abs(val) <= tolerance:
                mask = mask | (np.uint64(1) << np.uint64(h))
                cntr += 1
        masks[i] = mask
        # if cntr < np.shape(hyperplanes)[1]:
        #     print("Warning: bitmask result is sparse, check the tolerance")
    return masks


@njit
def generate_mask_serial(vertices, hyperplanes, b, tolerance=1e-5):
    """Serial version of generate_mask — faster for small vertex counts.

    Avoids numba thread-pool overhead that dominates when V is small
    (typical for refinement sub-cells).  Use generate_mask for V >= 200.
    """
    n_verts  = vertices.shape[0]
    n_planes = hyperplanes.shape[0]
    masks = np.zeros(n_verts, dtype=np.uint64)
    for i in range(n_verts):
        cntr=0
        mask = np.uint64(0)
        v    = vertices[i]
        for h in range(n_planes):
            val = hyperplanes[h, :] @ v + b[h]
            if np.abs(val) <= tolerance:
                mask = mask | (np.uint64(1) << np.uint64(h))
                cntr += 1
        masks[i] = mask
        # if cntr < np.shape(hyperplanes)[1]:
        #     print("Warning: bitmask result is sparse, check the tolerance")
    return masks


# ---------------------------------------------------------------------------
# Polytope slicing — serial JIT version (small polytopes, len(masks) < 1000)
# ---------------------------------------------------------------------------

@njit
def slice_polytope_with_hyperplane(enumerate_poly, hyperplane_val, masks, i, n):
    """Slice a polytope with a ReLU hyperplane using bitmask edge adjacency.

    Vertices with |hyperplane_val| <= 1e-5 are treated as on-boundary for
    assembly purposes.  Intersection points are generated only for vertex
    pairs (u strictly inside, v strictly outside) whose shared bitmask has
    at least (n-1) bits set — the bitwise adjacency condition.

    Parameters
    ----------
    enumerate_poly : (V, n) float64 array -- current polytope vertices.
    hyperplane_val : (V,)   float64 array -- signed hyperplane value per vertex.
    masks          : (V,)   uint64  array -- hyperplane-incidence bitmasks.
    i              : int -- global index of the current ReLU hyperplane
                           (used to set the new bit in intersection masks).
    n              : int -- input dimension.

    Returns
    -------
    polytopes  : list of two (V_k, n) arrays -- child polytopes [inside, outside].
    mask_lists : list of two (V_k,) uint64 arrays -- corresponding vertex masks.
    created_verts : (E, n) float64 array -- newly created boundary vertices.
    """
    ONE = np.uint64(1)
    req_shared = n - 1

    # Strict classification: vertices clearly on one side (used for edge search).
    strict_is_inside  = hyperplane_val < 1e-5
    strict_is_outside = hyperplane_val > -1e-5
    strict_index_in   = np.where(strict_is_inside)[0]
    strict_index_out  = np.where(strict_is_outside)[0]

    # Relaxed classification: vertices assigned to each child polytope.
    index_in  = np.where(hyperplane_val <= 1e-5)[0]
    index_out = np.where(hyperplane_val >= -1e-5)[0]

    # Pre-allocate worst-case intersection buffers.
    max_new = len(index_in) * len(index_out)
    new_verts_buffer = np.zeros((max_new, n))
    new_masks_buffer = np.zeros(max_new, dtype=np.uint64)

    count = 0
    for k in range(len(strict_index_in)):
        u_idx  = strict_index_in[k]
        mask_u = masks[u_idx]

        for l in range(len(strict_index_out)):
            v_idx = strict_index_out[l]

            # Only process edges that strictly cross the hyperplane.
            if hyperplane_val[u_idx] < -1e-9 and hyperplane_val[v_idx] > 1e-9:
                mask_v      = masks[v_idx]
                shared_mask = mask_u & mask_v

                # Popcount with early exit: count shared hyperplanes.
                n_shared = 0
                temp = shared_mask
                while temp > 0 and n_shared < req_shared:
                    if temp & ONE:
                        n_shared += 1
                    temp >>= ONE

                # Bitmask adjacency condition: edge iff shared >= n-1 hyperplanes.
                if n_shared >= req_shared:
                    d1 = hyperplane_val[u_idx]
                    d2 = hyperplane_val[v_idx]
                    t  = -d1 / (d2 - d1)
                    intersection = enumerate_poly[u_idx] + t * (
                        enumerate_poly[v_idx] - enumerate_poly[u_idx]
                    )
                    new_verts_buffer[count] = intersection
                    # Intersection lies on the new hyperplane; inherit shared bits
                    # plus any bit for the new hyperplane already carried by u.
                    new_masks_buffer[count] = shared_mask | (mask_u & (ONE << np.uint64(i)))
                    count += 1

    created_verts = new_verts_buffer[:count]
    created_masks = new_masks_buffer[:count]

    verts_in  = np.vstack((enumerate_poly[index_in],  created_verts))
    masks_in  = np.concatenate((masks[index_in],  created_masks))
    verts_out = np.vstack((enumerate_poly[index_out], created_verts))
    masks_out = np.concatenate((masks[index_out], created_masks))

    return [verts_in, verts_out], [masks_in, masks_out], created_verts


# ---------------------------------------------------------------------------
# Polytope slicing — alternate JIT version (relaxed tolerance fallback)
# ---------------------------------------------------------------------------

@njit
def slice_polytope_with_hyperplane_jit(enumerate_poly, hyperplane_val, masks, i, n):
    """Alternate slicer with slightly relaxed strict-classification tolerances.

    Identical in structure to :func:`slice_polytope_with_hyperplane` but uses
    stricter inner thresholds (1e-9) for edge traversal, which can recover
    intersection points that the primary slicer misses near degenerate vertices.

    Called automatically by :func:`Enumerator_rapid` when the primary slicer
    returns fewer than (n-1) intersection points.

    Parameters / Returns: same as :func:`slice_polytope_with_hyperplane`.
    """
    ONE = np.uint64(1)
    req_shared = n - 1

    strict_is_inside  = hyperplane_val < 1e-5
    strict_is_outside = hyperplane_val > -1e-5
    strict_index_in   = np.where(strict_is_inside)[0]
    strict_index_out  = np.where(strict_is_outside)[0]

    index_in  = np.where(hyperplane_val <= 1e-5)[0]
    index_out = np.where(hyperplane_val >= -1e-5)[0]

    max_new = len(index_in) * len(index_out)
    new_verts_buffer = np.zeros((max_new, n))
    new_masks_buffer = np.zeros(max_new, dtype=np.uint64)

    count = 0
    for k in range(len(strict_index_in)):
        u_idx  = strict_index_in[k]
        mask_u = masks[u_idx]

        for l in range(len(strict_index_out)):
            v_idx = strict_index_out[l]

            if hyperplane_val[u_idx] < -1e-9 and hyperplane_val[v_idx] > 1e-9:
                mask_v      = masks[v_idx]
                shared_mask = mask_u & mask_v

                n_shared = 0
                temp = shared_mask
                while temp > 0 and n_shared < req_shared:
                    if temp & ONE:
                        n_shared += 1
                    temp >>= ONE

                if n_shared >= req_shared:
                    d1 = hyperplane_val[u_idx]
                    d2 = hyperplane_val[v_idx]
                    t  = -d1 / (d2 - d1)
                    intersection = enumerate_poly[u_idx] + t * (
                        enumerate_poly[v_idx] - enumerate_poly[u_idx]
                    )
                    new_verts_buffer[count] = intersection
                    new_masks_buffer[count] = shared_mask | (mask_u & (ONE << np.uint64(i)))
                    count += 1

    created_verts = new_verts_buffer[:count]
    created_masks = new_masks_buffer[:count]

    verts_in  = np.vstack((enumerate_poly[index_in],  created_verts))
    masks_in  = np.concatenate((masks[index_in],  created_masks))
    verts_out = np.vstack((enumerate_poly[index_out], created_verts))
    masks_out = np.concatenate((masks[index_out], created_masks))

    return [verts_in, verts_out], [masks_in, masks_out], created_verts


# ---------------------------------------------------------------------------
# Polytope slicing — two-pass parallel JIT version (large polytopes)
# ---------------------------------------------------------------------------

@njit(parallel=True)
def slice_polytope_parallel(enumerate_poly, hyperplane_val, masks, i, n):
    """Two-pass parallel slicer for polytopes with many vertices (>= 1000 masks).

    Pass 1 (parallel): count valid intersections per inside-vertex to compute
    write offsets without race conditions.
    Pass 2 (parallel): write intersection points and masks into pre-allocated
    exact-sized buffers using the offsets from Pass 1.

    This avoids the worst-case over-allocation of the serial slicer and is
    more cache-friendly for large polytopes.

    Parameters / Returns: same as :func:`slice_polytope_with_hyperplane`.
    """
    ONE = np.uint64(1)
    req_shared = n - 1

    strict_is_inside  = hyperplane_val < -1e-10
    strict_is_outside = hyperplane_val > 1e-10
    strict_index_in   = np.where(strict_is_inside)[0]
    strict_index_out  = np.where(strict_is_outside)[0]

    index_in  = np.where(hyperplane_val <= 1e-5)[0]
    index_out = np.where(hyperplane_val >= -1e-5)[0]

    new_plane_bit  = ONE << np.uint64(i)
    num_strict_in  = len(strict_index_in)
    num_strict_out = len(strict_index_out)

    if num_strict_in == 0 or num_strict_out == 0:
        return (
            [enumerate_poly[index_in], enumerate_poly[index_out]],
            [masks[index_in], masks[index_out]],
            np.empty((0, n)),
        )

    # ------------------------------------------------------------------
    # Pass 1: count valid intersections for each inside vertex (parallel).
    # ------------------------------------------------------------------
    u_counts = np.zeros(num_strict_in, dtype=np.int64)

    for k in prange(num_strict_in):
        u_idx  = strict_index_in[k]
        mask_u = masks[u_idx]
        local_count = 0

        for l in range(num_strict_out):
            v_idx       = strict_index_out[l]
            shared_mask = mask_u & masks[v_idx]

            # Optimised popcount with early exit (clears lowest set bit each iter).
            n_shared = 0
            temp = shared_mask
            while temp > 0 and n_shared < req_shared:
                temp &= temp - ONE
                n_shared += 1

            if n_shared >= req_shared:
                local_count += 1

        u_counts[k] = local_count

    total_new = np.sum(u_counts)

    if total_new == 0:
        return (
            [enumerate_poly[index_in], enumerate_poly[index_out]],
            [masks[index_in], masks[index_out]],
            np.empty((0, n)),
        )

    # ------------------------------------------------------------------
    # Prefix-sum offsets (serial).
    # ------------------------------------------------------------------
    offsets = np.zeros(num_strict_in, dtype=np.int64)
    current_offset = 0
    for k in range(num_strict_in):
        offsets[k] = current_offset
        current_offset += u_counts[k]

    new_verts_buffer = np.empty((total_new, n), dtype=np.float64)
    new_masks_buffer = np.empty(total_new, dtype=np.uint64)

    # ------------------------------------------------------------------
    # Pass 2: write intersections at pre-computed offsets (parallel).
    # ------------------------------------------------------------------
    for k in prange(num_strict_in):
        u_idx  = strict_index_in[k]
        mask_u = masks[u_idx]
        u_vec  = enumerate_poly[u_idx]
        val_u  = hyperplane_val[u_idx]
        write_idx = offsets[k]

        for l in range(num_strict_out):
            v_idx       = strict_index_out[l]
            mask_v      = masks[v_idx]
            shared_mask = mask_u & mask_v

            n_shared = 0
            temp = shared_mask
            while temp > 0 and n_shared < req_shared:
                temp &= temp - ONE
                n_shared += 1

            if n_shared >= req_shared:
                v_vec = enumerate_poly[v_idx]
                t     = -val_u / (hyperplane_val[v_idx] - val_u)
                new_verts_buffer[write_idx] = u_vec + t * (v_vec - u_vec)
                new_masks_buffer[write_idx] = shared_mask | new_plane_bit
                write_idx += 1

    verts_in  = np.vstack((enumerate_poly[index_in],  new_verts_buffer))
    masks_in  = np.concatenate((masks[index_in],  new_masks_buffer))
    verts_out = np.vstack((enumerate_poly[index_out], new_verts_buffer))
    masks_out = np.concatenate((masks[index_out], new_masks_buffer))

    return [verts_in, verts_out], [masks_in, masks_out], new_verts_buffer


# ---------------------------------------------------------------------------
# Child-polytope assembly
# ---------------------------------------------------------------------------

def Polytope_formation_hd(original_polytope, hyperplane_val, Th, intersection_test, polytops_test):
    """Assemble the two child polytopes produced by slicing.

    Separates the original vertices by sign of hyperplane_val, then appends
    the newly created boundary intersection points to each side.

    Parameters
    ----------
    original_polytope : (V, n) array  -- parent polytope vertices.
    hyperplane_val    : (V,)   array  -- signed hyperplane value per vertex.
    Th                : list          -- domain bounds (unused here; kept for API consistency).
    intersection_test : (E, n) array  -- newly created boundary vertices.
    polytops_test     : list          -- raw sliced vertex sets (unused here).

    Returns
    -------
    [poly1, poly2] : list of two (V_k, n) arrays.

    Raises
    ------
    Warning : if fewer than (n-1) intersection points are provided.
    """
    n = len(original_polytope[0])

    if len(intersection_test) < n - 1:
        raise Warning(
            f"Number of intersection points ({len(intersection_test)}) "
            f"must be at least {n - 1}."
        )

    # Vertices with hyperplane_val >= -eps go to poly1 (inside/on boundary).
    poly1 = np.vstack((original_polytope[hyperplane_val >= -1e-13], intersection_test))
    # Vertices with hyperplane_val <=  eps go to poly2 (outside/on boundary).
    poly2 = np.vstack((original_polytope[hyperplane_val <=  1e-13], intersection_test))

    if len(poly1) < n + 1:
        print("Warning: poly1 may have too few vertices for a full-dimensional polytope.")
    if len(poly2) < n + 1:
        print("Warning: poly2 may have too few vertices for a full-dimensional polytope.")

    return [poly1, poly2]


# ---------------------------------------------------------------------------
# Layer-level enumerator
# ---------------------------------------------------------------------------

def Enumerator_rapid(
    hyperplanes,
    b,
    original_polytope_test,
    TH,
    boundary_hyperplanes,
    border_bias,
    parallel,
    D,
    m,
    use_wide=False,
    W_pos_ibp=None,
    W_neg_ibp=None,
    b_arr_ibp=None,
    is_last_layer=False,
    mask_tolerance=1e-10,
):
    """Enumerate all linear regions produced by one layer of ReLU hyperplanes.

    For each ReLU neuron hyperplane in the layer, every current polytope is
    tested for sign variation across that hyperplane.  Polytopes that strictly
    straddle the hyperplane are sliced into two children using the bitmask
    adjacency test; the rest pass through unchanged.

    After slicing, the hyperplane is added to the boundary set so that it
    contributes a bit to vertex masks in subsequent iterations.

    Dispatcher logic
    ----------------
    use_wide=False (total hyperplanes <= 64):
      * len(masks) < 1000  -> :func:`slice_polytope_with_hyperplane` (serial JIT, uint64).
      * len(masks) >= 1000 -> :func:`slice_polytope_parallel` (parallel JIT, uint64).
      * If the primary slicer yields fewer than (n-1) intersections,
        :func:`slice_polytope_with_hyperplane_jit` is called as a fallback.

    use_wide=True (total hyperplanes > 64):
      * :func:`generate_mask_wide` produces (V, num_words) uint64 masks.
      * :func:`slice_polytope_wide` handles all slicing (serial, no parallel path).

    Parameters
    ----------
    hyperplanes          : (H, n) array -- ReLU neuron weight matrices for this layer.
    b                    : (H,)   array -- corresponding biases.
    original_polytope_test : list of arrays -- initial polytope set (typically one hypercube).
    TH                   : list  -- per-dimension domain bounds.
    boundary_hyperplanes : list  -- accumulated boundary hyperplane normals (mutated in place).
    border_bias          : list  -- accumulated boundary hyperplane offsets (mutated in place).
    parallel             : bool  -- reserved for future use (currently unused).
    D                    : array -- reserved for activation pattern tracking (currently unused).
    m                    : int   -- layer index (informational).
    use_wide             : bool  -- if True, use multi-word uint64 masks (default False).

    Returns
    -------
    enumerate_poly : list of (V_k, n) arrays -- all enumerated polytopes for this layer.
    """
    enumerate_poly = list(original_polytope_test)

    for i in range(len(hyperplanes)):
        # print(f"Processing hyperplane {i+1}/{len(hyperplanes)} with {len(enumerate_poly)} polytopes...")
        if not enumerate_poly:
            break
        intact_poly = []
        poly_dummy  = []
        n = len(enumerate_poly[0][0])

        # ---- Sign-variation screening ----
        sgn_var = []
        for k in enumerate_poly:
            dum = np.dot(k, hyperplanes[i].T) + b[i]
            if np.min(dum) < -1e-5 and np.max(dum) > 1e-5:
                sgn_var.append(np.max(dum) * np.min(dum))
            else:
                sgn_var.append(0.0)

        # Global bit index of the current hyperplane across the boundary set.
        global_bit_index = i + len(boundary_hyperplanes[0])

        # Build bh/bb once per i — they don't change within the j loop.
        bh = np.array(boundary_hyperplanes[0])
        bb = np.array(border_bias[0])

        # ---- Slicing loop ----
        for j in range(len(enumerate_poly)):
            if sgn_var[j] < -1e-9:
                hyperplane_val = np.dot(enumerate_poly[j], hyperplanes[i].T) + b[i]
                verts = np.array(enumerate_poly[j])
                bh_n,bb_n=finding_side_polytope(bh,verts,bb)
                # sides,hypes=finding_side_old(bh,verts,bb)
                # if len(bh_n)!=len(hypes):
                    # print("check")
                if len(bh_n)<=64:
                    use_wide=False
                if use_wide:
                    # Wide path: multi-word uint64 masks, pure Python/NumPy.
                    masks_w = generate_mask_wide(verts, bh_n, bb_n, tolerance=mask_tolerance)
                    polytops_test, _, created_verts = slice_polytope_wide(
                        verts,
                        np.array(hyperplane_val),
                        masks_w,
                        global_bit_index,
                        n,
                    )
                else:
                    # Fast path: scalar uint64 masks, Numba JIT.
                    # Use serial version for small polytopes — parallel thread
                    # overhead dominates when vertex count is low.
                    if len(verts) < 200:
                        masks = generate_mask_serial(verts, bh_n, bb_n, tolerance=mask_tolerance)
                    else:
                        masks = generate_mask(verts, bh_n, bb_n, tolerance=mask_tolerance)
                    if len(masks) < 1000:
                        polytops_test, _, created_verts = slice_polytope_with_hyperplane(
                            verts,
                            np.array(hyperplane_val),
                            masks,
                            global_bit_index,
                            n,
                        )
                    else:
                        polytops_test, _, created_verts = slice_polytope_parallel(
                            verts,
                            np.array(hyperplane_val),
                            masks,
                            global_bit_index,
                            n,
                        )

                    # Fallback slicer if primary returned too few intersections.
                    if len(created_verts) <= n - 1:
                        polytops_test, _, created_verts = slice_polytope_with_hyperplane_jit(
                            verts,
                            np.array(hyperplane_val),
                            masks,
                            global_bit_index,
                            n,
                        )

                if len(created_verts) > n - 1:
                    fv_in = _dedup_verts(np.asarray(polytops_test[0], dtype=np.float64))
                    fv_out = _dedup_verts(np.asarray(polytops_test[1], dtype=np.float64))
                    res=[]
                    res.append(fv_in)
                    res.append(fv_out)
                    result = res
                else:
                    result = [enumerate_poly[j]]
                    print(f"Warning: Slicer returned fewer than (n-1) intersection points; "                          f"this may lead to incorrect enumeration results for this hyperplane.")

                
                poly_dummy.extend(result)
            else:
                intact_poly.append(enumerate_poly[j])
        if W_pos_ibp is not None and len(poly_dummy) > 1:
            is_final_hyperplane = (i == len(hyperplanes) - 1)
            poly_dummy, _ = vertex_ibp_filter(
                poly_dummy, W_pos_ibp, W_neg_ibp, b_arr_ibp,
                is_last_layer=(is_last_layer and is_final_hyperplane),
                n_known=m + 1 if is_final_hyperplane else m,
                verbose=False,
            )
        intact_poly.extend(poly_dummy)

        # Per-hyperplane IBP filter on all polytopes (split and non-split).
        # On the final hyperplane: layer m is fully processed → n_known = m+1
        # and is_last_layer is forwarded from the caller, enabling the exact
        # vertex sign check on the final layer instead of IBP approximation.


        boundary_hyperplanes[0] = np.vstack(
            (boundary_hyperplanes[0], hyperplanes[i])
        ).tolist()
        border_bias[0] = border_bias[0] + [b[i]]
        enumerate_poly = intact_poly

    return enumerate_poly


def _dedup_verts(verts: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Remove near-duplicate rows from a vertex array (L-inf grid snap)."""
    if len(verts) <= 1:
        return verts
    scale = 1.0 / tol
    rounded = np.round(verts * scale).astype(np.int64)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return verts[np.sort(idx)]


def Enumerator_rapid_face(
    hyperplanes,
    b,
    x_stars_list,
    n,
    mask_tolerance=1e-5,
    use_wide=False,
):
    """Enumerate sub-regions of a ZLS face by sequentially cutting with hyperplanes.

    Each face carries its own dedicated boundary hyperplanes (bh, bb) so that
    cuts applied to one face never contaminate sibling faces.

    Parameters
    ----------
    hyperplanes  : (H, n) array -- cutting hyperplane normals
    b            : (H,)   array -- cutting hyperplane biases (hval = v @ h + b_i)
    x_stars_list : list of (verts, bh, bb) tuples
                     verts : (K, n) float64 -- face vertices
                     bh    : (B, n) float64 -- dedicated boundary normals
                     bb    : (B,)   float64 -- dedicated boundary biases
    n            : int -- ambient input dimension
    mask_tolerance : float -- hyperplane-incidence tolerance for mask generation
    use_wide     : bool -- use multi-word uint64 masks when total hyperplanes > 64

    Returns
    -------
    list of (verts, bh, bb) tuples -- one per surviving sub-face
    """
    enumerate_faces = [
        (np.asarray(v, dtype=np.float64),
         np.asarray(bh, dtype=np.float64),
         np.asarray(bb, dtype=np.float64))
        for v, bh, bb in x_stars_list
    ]

    for i in range(len(hyperplanes)):
        if not enumerate_faces:
            break

        next_faces = []
        for face_v, face_bh, face_bb in enumerate_faces:
            hval = face_v @ hyperplanes[i] + b[i]

            if hval.min() >= -1e-5 or hval.max() <= 1e-5:
                next_faces.append((face_v, face_bh, face_bb))
                continue

            # Bit index for the new cut = number of face's own boundary hyperplanes
            global_bit_index = len(face_bh)
            current_use_wide = use_wide or (global_bit_index >= 64)

            if current_use_wide:
                words_needed = (global_bit_index + 64) // 64
                masks_w = generate_mask_face_wide(face_v, face_bh, face_bb, tolerance=mask_tolerance)
                if masks_w.shape[1] < words_needed:
                    pad     = np.zeros((len(masks_w), words_needed - masks_w.shape[1]),
                                       dtype=np.uint64)
                    masks_w = np.hstack([masks_w, pad])
                (verts_in, verts_out), _, created = slice_face_wide(
                    face_v, hval, masks_w, global_bit_index, n
                )
            else:
                masks_n, flag = generate_mask_face_serial(face_v, face_bh, face_bb, tolerance=mask_tolerance)
                # if flag == 1:
                #     print("Warning: mask generation is sparse, check the tolerance")
                (verts_in, verts_out), _, created = slice_face_with_hyperplane(
                    face_v, hval, masks_n, global_bit_index, n
                )

            if len(created) == 0:
                next_faces.append((face_v, face_bh, face_bb))
                print("Warning: slicer returned no intersection points; skipping slice.")
                continue

            # Extended boundary for this face = face's own walls + new cut
            bh_ext = np.vstack([face_bh, hyperplanes[i]])
            bb_ext = np.append(face_bb, float(b[i]))

            if len(verts_in) == 0 and len(verts_out) == 0:
                print("Warning: slicer returned no vertices for either side; skipping slice.")
                continue

            if len(verts_in) > 0:
                # st1=time.time()
                fv_in = _dedup_verts(np.asarray(verts_in, dtype=np.float64))
                # if len(fv_in)!=len(verts_in):
                #     print(f"Deduplication reduced vertex count from {len(verts_in)} to {len(fv_in)} for the inside face.")
                # print(f"Deduplication took {time.time() - st1:.5f} seconds for {len(verts_in)} vertices.")
                # st1=time.time()
                # fv_in1 = _dedup_verts_nb(np.asarray(verts_in, dtype=np.float64))
                # print(f"Numba deduplication took {time.time() - st1:.5f} seconds for {len(verts_in)} vertices.")

                # if len(fv_in) > 1000:
                #     print("Warning: slicer returned a large number of vertices for the inside face; check the mask tolerance.")
                bh_in, bb_in = finding_side(bh_ext, fv_in, bb_ext)
                next_faces.append((fv_in, bh_in, bb_in))

            if len(verts_out) > 0:
                fv_out = _dedup_verts(np.asarray(verts_out, dtype=np.float64))
                # if len(fv_out)!=len(verts_out):
                #     print(f"Deduplication reduced vertex count from {len(verts_out)} to {len(fv_out)} for the outside face.")
                # if len(fv_out) > 1000:
                #     print("Warning: slicer returned a large number of vertices for the outside face; check the mask tolerance.")
                bh_out, bb_out = finding_side(bh_ext, fv_out, bb_ext)
                next_faces.append((fv_out, bh_out, bb_out))

        enumerate_faces = next_faces

    return enumerate_faces


# ---------------------------------------------------------------------------
# Wide-mask primitives  (used when total hyperplanes > 64)
# ---------------------------------------------------------------------------

@njit(parallel=True)
def generate_mask_wide(vertices, hyperplanes, b, tolerance=1e-5):
    """Compute multi-word hyperplane-incidence bitmasks for each vertex.

    Identical semantics to :func:`generate_mask` but stores each mask as a
    row of ``ceil(H/64)`` uint64 words rather than a single uint64 scalar.
    This supports an arbitrary number of hyperplanes without overflow.

    Parameters
    ----------
    vertices   : (V, n) float64 array
    hyperplanes: (H, n) float64 array
    b          : (H,)   float64 array
    tolerance  : float

    Returns
    -------
    masks : (V, num_words) uint64 array
    """
    n_verts   = vertices.shape[0]
    n_planes  = hyperplanes.shape[0]
    num_words = (n_planes + 63) // 64
    masks = np.zeros((n_verts, num_words), dtype=np.uint64)

    for i in prange(n_verts):
        v = vertices[i]
        cntr = 0
        for h in range(n_planes):
            val = hyperplanes[h, :] @ v + b[h]
            if np.abs(val) <= tolerance:
                cntr += 1
                word = np.uint64(h // 64)
                bit  = np.uint64(h % 64)
                masks[i, word] |= np.uint64(1) << bit
        # if cntr < np.shape(hyperplanes)[1]:
        #     print("Warning: bitmask result is sparse, check the tolerance")
    return masks



@njit
def _wide_popcount_ge(mask_u, mask_v, req_shared, num_words):
    """Return True if bitwise AND of two wide masks has >= req_shared bits set.
 
    Uses Kernighan's bit-clearing trick word by word with early exit.
 
    Parameters
    ----------
    mask_u, mask_v : (num_words,) uint64 arrays
    req_shared     : int -- minimum shared bit count required
    num_words      : int
 
    Returns
    -------
    bool
    """
    count = 0
    ONE = np.uint64(1)
    for w in range(num_words):
        temp = mask_u[w] & mask_v[w]
        while temp > np.uint64(0):
            temp &= temp - ONE
            count += 1
            if count >= req_shared:
                return True
    return False
 
 
@njit
def _wide_mask_or_and(shared_u, mask_u, bit_index, num_words):
    """Compute ``shared_u | (mask_u & (1 << bit_index))`` for wide masks.
 
    This replicates the intersection-mask assignment from the narrow slicer:
    the new vertex inherits shared bits plus any pre-existing bit for the
    current hyperplane that vertex u already carried.
 
    Parameters
    ----------
    shared_u  : (num_words,) uint64 array -- mask_u & mask_v
    mask_u    : (num_words,) uint64 array
    bit_index : int -- global hyperplane index
    num_words : int
 
    Returns
    -------
    (num_words,) uint64 array
    """
    result = shared_u.copy()
    word = np.uint64(bit_index // 64)
    bit  = np.uint64(bit_index % 64)
    result[word] |= mask_u[word] & (np.uint64(1) << bit)
    return result
 
 
@njit
def _slice_polytope_wide_seq(enumerate_poly, hyperplane_val, masks_w, i, n):
    """Sequential (single-threaded) wide-mask polytope slicer.

    Called by :func:`slice_polytope_wide` for small polytopes where thread
    overhead would outweigh the parallelism benefit.
    """
    num_words  = masks_w.shape[1]
    req_shared = n - 1

    strict_index_in  = np.where(hyperplane_val < -1e-9)[0]
    strict_index_out = np.where(hyperplane_val >  1e-9)[0]
    index_in  = np.where(hyperplane_val <= 1e-5)[0]
    index_out = np.where(hyperplane_val >= -1e-5)[0]

    max_new = len(index_in) * len(index_out)
    new_verts_buffer = np.zeros((max_new, n),         dtype=np.float64)
    new_masks_buffer = np.zeros((max_new, num_words),  dtype=np.uint64)
    count = 0

    for k in range(len(strict_index_in)):
        u_idx  = strict_index_in[k]
        mask_u = masks_w[u_idx]

        for l in range(len(strict_index_out)):
            v_idx = strict_index_out[l]

            if hyperplane_val[u_idx] < -1e-9 and hyperplane_val[v_idx] > 1e-9:
                mask_v      = masks_w[v_idx]
                shared_mask = mask_u & mask_v   # element-wise, shape (num_words,)

                if _wide_popcount_ge(mask_u, mask_v, req_shared, num_words):
                    d1 = hyperplane_val[u_idx]
                    d2 = hyperplane_val[v_idx]
                    t  = -d1 / (d2 - d1)
                    new_verts_buffer[count] = enumerate_poly[u_idx] + t * (
                        enumerate_poly[v_idx] - enumerate_poly[u_idx]
                    )
                    new_masks_buffer[count] = _wide_mask_or_and(
                        shared_mask, mask_u, i, num_words
                    )
                    count += 1

    created_verts = new_verts_buffer[:count]
    created_masks = new_masks_buffer[:count]

    n_in  = len(index_in)
    n_out = len(index_out)

    verts_in  = np.empty((n_in  + count, n),         dtype=np.float64)
    masks_in  = np.empty((n_in  + count, num_words),  dtype=np.uint64)
    verts_out = np.empty((n_out + count, n),         dtype=np.float64)
    masks_out = np.empty((n_out + count, num_words),  dtype=np.uint64)

    verts_in[:n_in]   = enumerate_poly[index_in]
    masks_in[:n_in]   = masks_w[index_in]
    verts_in[n_in:]   = created_verts
    masks_in[n_in:]   = created_masks

    verts_out[:n_out] = enumerate_poly[index_out]
    masks_out[:n_out] = masks_w[index_out]
    verts_out[n_out:] = created_verts
    masks_out[n_out:] = created_masks

    return [verts_in, verts_out], [masks_in, masks_out], created_verts


@njit
def _compact_wide(verts_buf, masks_buf, valid, n, num_words):
    """Compact pre-allocated buffers: keep only rows where valid[idx] is True."""
    count = 0
    for idx in range(len(valid)):
        if valid[idx]:
            count += 1
    out_v = np.empty((count, n),          dtype=np.float64)
    out_m = np.empty((count, num_words),   dtype=np.uint64)
    j = 0
    for idx in range(len(valid)):
        if valid[idx]:
            out_v[j] = verts_buf[idx]
            out_m[j] = masks_buf[idx]
            j += 1
    return out_v, out_m


import numba as nb

@njit(parallel=True)
def _slice_polytope_wide_par(enumerate_poly, hyperplane_val, masks_w, i, n):
    """Parallel wide-mask slicer using prange on the outer (u) loop.

    Each (k, l) pair writes to a pre-determined slot k*n_out+l so there is
    no data race.  A sequential compaction pass follows to remove empty slots.
    Called by :func:`slice_polytope_wide` for large polytopes.
    """
    num_words  = masks_w.shape[1]
    req_shared = n - 1

    strict_index_in  = np.where(hyperplane_val < -1e-9)[0]
    strict_index_out = np.where(hyperplane_val >  1e-9)[0]
    index_in  = np.where(hyperplane_val <= 1e-5)[0]
    index_out = np.where(hyperplane_val >= -1e-5)[0]

    n_in_s  = len(strict_index_in)
    n_out_s = len(strict_index_out)
    max_new = n_in_s * n_out_s

    new_verts_buffer = np.zeros((max_new, n),          dtype=np.float64)
    new_masks_buffer = np.zeros((max_new, num_words),   dtype=np.uint64)
    valid            = np.zeros(max_new,                dtype=nb.boolean)

    for k in prange(n_in_s):                   # parallel over u-vertices
        u_idx  = strict_index_in[k]
        mask_u = masks_w[u_idx]
        d1     = hyperplane_val[u_idx]

        for l in range(n_out_s):               # serial over v-vertices per thread
            v_idx    = strict_index_out[l]
            flat_idx = k * n_out_s + l

            if d1 < -1e-9 and hyperplane_val[v_idx] > 1e-9:
                mask_v = masks_w[v_idx]
                if _wide_popcount_ge(mask_u, mask_v, req_shared, num_words):
                    d2 = hyperplane_val[v_idx]
                    t  = -d1 / (d2 - d1)
                    for dim in range(n):
                        new_verts_buffer[flat_idx, dim] = (
                            enumerate_poly[u_idx, dim]
                            + t * (enumerate_poly[v_idx, dim]
                                   - enumerate_poly[u_idx, dim])
                        )
                    new_masks_buffer[flat_idx] = _wide_mask_or_and(
                        mask_u & mask_v, mask_u, i, num_words
                    )
                    valid[flat_idx] = True

    created_verts, created_masks = _compact_wide(
        new_verts_buffer, new_masks_buffer, valid, n, num_words
    )

    n_in  = len(index_in)
    n_out = len(index_out)
    count = len(created_verts)

    verts_in  = np.empty((n_in  + count, n),          dtype=np.float64)
    masks_in  = np.empty((n_in  + count, num_words),   dtype=np.uint64)
    verts_out = np.empty((n_out + count, n),           dtype=np.float64)
    masks_out = np.empty((n_out + count, num_words),   dtype=np.uint64)

    verts_in[:n_in]   = enumerate_poly[index_in]
    masks_in[:n_in]   = masks_w[index_in]
    verts_in[n_in:]   = created_verts
    masks_in[n_in:]   = created_masks

    verts_out[:n_out] = enumerate_poly[index_out]
    masks_out[:n_out] = masks_w[index_out]
    verts_out[n_out:] = created_verts
    masks_out[n_out:] = created_masks

    return [verts_in, verts_out], [masks_in, masks_out], created_verts


# Threshold: use parallel version when the double-loop work exceeds this many
# iterations.  Below this, thread-spawn overhead dominates.
_PAR_THRESHOLD = 10_000


def slice_polytope_wide(enumerate_poly, hyperplane_val, masks_w, i, n):
    """Slice a polytope using wide (multi-word) bitmask adjacency.

    Mirrors the logic of :func:`slice_polytope_with_hyperplane` exactly,
    but operates on (V, num_words) mask arrays instead of scalar uint64.

    Dispatches to a parallel implementation (prange over u-vertices) when
    ``n_in * n_out > _PAR_THRESHOLD`` (≈ V > 200 with a 50/50 split), and
    falls back to the sequential version for small polytopes where thread
    overhead would dominate.

    Parameters
    ----------
    enumerate_poly : (V, n) float64 array
    hyperplane_val : (V,)   float64 array
    masks_w        : (V, num_words) uint64 array
    i              : int -- global bit index of the current hyperplane
    n              : int -- input dimension

    Returns
    -------
    polytopes     : list of two (V_k, n) arrays -- [inside, outside]
    mask_lists    : list of two (V_k, num_words) arrays
    created_verts : (E, n) float64 array
    """
    n_in_s  = int(np.sum(hyperplane_val < -1e-9))
    n_out_s = int(np.sum(hyperplane_val >  1e-9))
    if n_in_s * n_out_s > _PAR_THRESHOLD:
        return _slice_polytope_wide_par(enumerate_poly, hyperplane_val, masks_w, i, n)
    return _slice_polytope_wide_seq(enumerate_poly, hyperplane_val, masks_w, i, n)

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

class RaggedPolytopeStorage:
    """Contiguous in-memory storage for a collection of variable-vertex polytopes.

    All polytope vertex arrays are concatenated into a single flat numpy array
    with an integer offset array that records each polytope's start/end index.
    This avoids the Python-list-of-arrays overhead for large collections.

    Attributes
    ----------
    vertices       : (total_V, n) float64 array or None before any addition.
    offsets        : list of int -- cumulative vertex counts (len = n_polytopes + 1).
    n_dim          : int or None -- input dimension (set on first addition).
    total_vertices : int         -- total vertex count across all polytopes.
    """

    def __init__(self):
        self.vertices       = None
        self.offsets        = [0]
        self.n_dim          = None
        self.total_vertices = 0

    def add_polytope(self, vertices):
        """Append a single polytope to the storage.

        Parameters
        ----------
        vertices : array-like (V, n) -- vertex array for the new polytope.
        """
        vertices = np.asarray(vertices, dtype=np.float64)

        if self.n_dim is None:
            self.n_dim     = vertices.shape[1]
            self.vertices  = vertices.copy()
        else:
            self.vertices = np.vstack([self.vertices, vertices])

        self.total_vertices += len(vertices)
        self.offsets.append(self.total_vertices)

    def get_polytope(self, idx):
        """Return the vertex array of the idx-th polytope.

        Parameters
        ----------
        idx : int

        Returns
        -------
        (V_idx, n) float64 array -- view into the flat vertex array.
        """
        start = self.offsets[idx]
        end   = self.offsets[idx + 1]
        return self.vertices[start:end]

    def memory_usage_mb(self):
        """Approximate memory footprint in megabytes."""
        if self.vertices is None:
            return 0.0
        vertex_mem = self.vertices.nbytes / (1024 * 1024)
        offset_mem = len(self.offsets) * 8 / (1024 * 1024)
        return vertex_mem + offset_mem

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, idx):
        return self.get_polytope(idx)


class HybridRaggedStorage:
    """Ragged storage that spills to disk when the in-memory limit is reached.

    Polytopes are accumulated in a :class:`RaggedPolytopeStorage` until its
    memory footprint exceeds ``memory_limit_mb``, at which point the current
    batch is serialised to a pair of ``.npy`` files in ``temp_dir`` and
    the in-memory buffer is reset.  Subsequent additions continue in batches
    of 1000 polytopes.

    This is useful for high-dimensional networks where thousands of polytopes
    cannot be held in RAM simultaneously.

    Parameters
    ----------
    memory_limit_mb : float -- RAM budget in MB before first spill (default 500).
    temp_dir        : str or None -- directory for temp files; uses system temp if None.
    """

    def __init__(self, memory_limit_mb=500, temp_dir=None):
        self.memory_limit_mb  = memory_limit_mb
        self.temp_dir         = temp_dir or tempfile.mkdtemp()
        self.memory_storage   = RaggedPolytopeStorage()
        self.disk_chunks      = []
        self.mode             = "memory"
        self.total_polytopes  = 0

    def add_polytope(self, vertices):
        """Append a polytope, spilling to disk if the memory budget is exceeded.

        Parameters
        ----------
        vertices : array-like (V, n)
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        self.memory_storage.add_polytope(vertices)
        self.total_polytopes += 1

        if self.mode == "memory":
            if self.memory_storage.memory_usage_mb() > self.memory_limit_mb:
                print(
                    f"  Memory limit reached "
                    f"({self.memory_storage.memory_usage_mb():.1f} MB). "
                    f"Spilling {len(self.memory_storage)} polytopes to disk."
                )
                self._flush_to_disk()
                self.mode = "disk"
        else:
            # In disk mode, flush every 1000 polytopes to bound memory use.
            if len(self.memory_storage) >= 1000:
                self._flush_to_disk()

    def _flush_to_disk(self):
        """Serialise the current in-memory buffer to disk and reset it."""
        if len(self.memory_storage) == 0:
            return

        chunk_idx    = len(self.disk_chunks)
        vertices_file = os.path.join(self.temp_dir, f"vertices_{chunk_idx}.npy")
        offsets_file  = os.path.join(self.temp_dir, f"offsets_{chunk_idx}.npy")

        np.save(vertices_file, self.memory_storage.vertices)
        np.save(offsets_file,  np.array(self.memory_storage.offsets, dtype=np.int64))

        self.disk_chunks.append(
            {
                "vertices_file": vertices_file,
                "offsets_file":  offsets_file,
                "count":         len(self.memory_storage),
            }
        )
        print(f"  Flushed chunk {chunk_idx} ({len(self.memory_storage)} polytopes) to disk.")

        import gc
        self.memory_storage = RaggedPolytopeStorage()
        gc.collect()

    def get_polytope(self, idx):
        """Return the vertex array of the idx-th polytope (may load from disk).

        Parameters
        ----------
        idx : int

        Returns
        -------
        (V_idx, n) float64 array
        """
        cumulative = 0
        for chunk in self.disk_chunks:
            if idx < cumulative + chunk["count"]:
                vertices = np.load(chunk["vertices_file"])
                offsets  = np.load(chunk["offsets_file"])
                local_idx = idx - cumulative
                return vertices[offsets[local_idx]:offsets[local_idx + 1]]
            cumulative += chunk["count"]

        # Index falls in the current in-memory buffer.
        local_idx = idx - cumulative
        return self.memory_storage[local_idx]

    def cleanup(self):
        """Remove all temporary disk files created by this storage object."""
        for chunk in self.disk_chunks:
            for key in ("vertices_file", "offsets_file"):
                if os.path.exists(chunk[key]):
                    os.remove(chunk[key])
        if os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
            os.rmdir(self.temp_dir)

    def __len__(self):
        return self.total_polytopes

    def __getitem__(self, idx):
        return self.get_polytope(idx)



def finding_deep_hype(hyperplanes,b,S_prime,border_hyperplane,border_bias,i,n):
    S_init=np.eye(np.shape(hyperplanes[0])[0])
    hype=S_init@hyperplanes[0]
    bias=(S_init@b[0])

    # border_hyperplane=np.vstack((border_hyperplane,hyperplanes[0]))
    # border_bias=border_bias+b[0].tolist()
    m=2*n
    for j in range(0,i):
        border_hyperplane=np.vstack((border_hyperplane,hype))
        border_bias=np.hstack((np.array(border_bias),bias))
        mid_point=np.mean(S_prime,axis=0)
        S=np.sign(np.maximum(border_hyperplane@mid_point+border_bias,0))
        hype=hyperplanes[j+1]@np.diag(S[m:])@hype    
        bias=hyperplanes[j+1]@np.diag(S[m:])@bias+b[j+1]
        m=m+(np.shape(hyperplanes[j])[0])
    return hype,bias,border_hyperplane,border_bias.tolist()



# """
# bitwise_utils_numpy.py
# ======================

# Pure-NumPy version of bitwise_utils.py — identical logic, zero Numba.

# Use this to benchmark how much Numba contributes on top of the core
# algorithmic speedup (LP removal). The only changes from the Numba version
# are:

#   - All @njit / @njit(parallel=True) decorators removed.
#   - prange(...) replaced with range(...).
#   - np.uint64 scalar arithmetic replaced with Python int where Numba
#     required explicit casts that NumPy handles natively.
#   - Wide-mask functions left as plain Python loops over numpy arrays.
#   - Everything else is bit-for-bit identical to bitwise_utils.py.
# """

# import os
# import tempfile

# import numpy as np


# # ---------------------------------------------------------------------------
# # Bitmask generation
# # ---------------------------------------------------------------------------

# def generate_mask(vertices, hyperplanes, b, tolerance=1e-7):
#     """Compute a hyperplane-incidence bitmask for each vertex.

#     A bit h is set in the mask for vertex v if |hyperplanes[h] · v + b[h]|
#     <= tolerance, meaning v lies (numerically) on the h-th hyperplane.

#     Parameters
#     ----------
#     vertices   : (V, n) float64 array
#     hyperplanes: (H, n) float64 array
#     b          : (H,)   float64 array
#     tolerance  : float

#     Returns
#     -------
#     masks : (V,) int64 array  -- bitmask per vertex (Python int, no uint64 overflow).
#     """
#     n_verts  = vertices.shape[0]
#     n_planes = hyperplanes.shape[0]
#     masks = np.zeros(n_verts, dtype=np.int64)

#     # Compute all signed distances at once: (V, H)
#     vals = vertices @ hyperplanes.T + b  # shape (V, H)
#     on_plane = np.abs(vals) <= tolerance  # shape (V, H) bool

#     for h in range(n_planes):
#         masks[on_plane[:, h]] |= (1 << h)

#     return masks


# # ---------------------------------------------------------------------------
# # Polytope slicing — serial pure-NumPy version
# # ---------------------------------------------------------------------------

# def slice_polytope_with_hyperplane(enumerate_poly, hyperplane_val, masks, i, n):
#     """Slice a polytope with a ReLU hyperplane using bitmask edge adjacency.

#     Parameters
#     ----------
#     enumerate_poly : (V, n) float64 array
#     hyperplane_val : (V,)   float64 array
#     masks          : (V,)   int64 array
#     i              : int -- global index of the current ReLU hyperplane
#     n              : int -- input dimension

#     Returns
#     -------
#     polytopes     : list of two (V_k, n) arrays -- [inside, outside]
#     mask_lists    : list of two (V_k,) int64 arrays
#     created_verts : (E, n) float64 array
#     """
#     req_shared = n - 1

#     strict_index_in  = np.where(hyperplane_val < -1e-9)[0]
#     strict_index_out = np.where(hyperplane_val >  1e-9)[0]
#     index_in  = np.where(hyperplane_val <= 1e-5)[0]
#     index_out = np.where(hyperplane_val >= -1e-5)[0]

#     max_new = len(index_in) * len(index_out)
#     new_verts_buffer = np.zeros((max_new, n), dtype=np.float64)
#     new_masks_buffer = np.zeros(max_new,      dtype=np.int64)

#     count = 0
#     for k in range(len(strict_index_in)):
#         u_idx  = strict_index_in[k]
#         mask_u = int(masks[u_idx])

#         for l in range(len(strict_index_out)):
#             v_idx = strict_index_out[l]

#             if hyperplane_val[u_idx] < -1e-9 and hyperplane_val[v_idx] > 1e-9:
#                 mask_v      = int(masks[v_idx])
#                 shared_mask = mask_u & mask_v

#                 # Kernighan popcount with early exit.
#                 n_shared = 0
#                 temp = shared_mask
#                 while temp > 0 and n_shared < req_shared:
#                     temp &= temp - 1
#                     n_shared += 1

#                 if n_shared >= req_shared:
#                     d1 = hyperplane_val[u_idx]
#                     d2 = hyperplane_val[v_idx]
#                     t  = -d1 / (d2 - d1)
#                     new_verts_buffer[count] = (
#                         enumerate_poly[u_idx]
#                         + t * (enumerate_poly[v_idx] - enumerate_poly[u_idx])
#                     )
#                     new_masks_buffer[count] = shared_mask | (mask_u & (1 << i))
#                     count += 1

#     created_verts = new_verts_buffer[:count]
#     created_masks = new_masks_buffer[:count]

#     verts_in  = np.vstack((enumerate_poly[index_in],  created_verts))
#     masks_in  = np.concatenate((masks[index_in],  created_masks))
#     verts_out = np.vstack((enumerate_poly[index_out], created_verts))
#     masks_out = np.concatenate((masks[index_out], created_masks))

#     return [verts_in, verts_out], [masks_in, masks_out], created_verts


# # Alias: the Numba version had a jit fallback with slightly different tolerance
# # handling. Here both are the same function since there is no JIT overhead.
# slice_polytope_with_hyperplane_jit = slice_polytope_with_hyperplane


# # ---------------------------------------------------------------------------
# # Polytope slicing — "parallel" version (serial in pure NumPy)
# # ---------------------------------------------------------------------------

# def slice_polytope_parallel(enumerate_poly, hyperplane_val, masks, i, n):
#     """Serial fallback for the parallel JIT slicer.

#     In the Numba version this used prange for parallelism. Here it is
#     identical to slice_polytope_with_hyperplane — included so the caller
#     (Enumerator_rapid) needs no changes.
#     """
#     return slice_polytope_with_hyperplane(enumerate_poly, hyperplane_val, masks, i, n)


# # ---------------------------------------------------------------------------
# # Child-polytope assembly  (unchanged from Numba version)
# # ---------------------------------------------------------------------------

# def Polytope_formation_hd(original_polytope, hyperplane_val, Th, intersection_test, polytops_test):
#     """Assemble the two child polytopes produced by slicing."""
#     n = len(original_polytope[0])

#     if len(intersection_test) < n - 1:
#         raise Warning(
#             f"Number of intersection points ({len(intersection_test)}) "
#             f"must be at least {n - 1}."
#         )

#     poly1 = np.vstack((original_polytope[hyperplane_val >= -1e-13], intersection_test))
#     poly2 = np.vstack((original_polytope[hyperplane_val <=  1e-13], intersection_test))

#     if len(poly1) < n + 1:
#         print("Warning: poly1 may have too few vertices for a full-dimensional polytope.")
#     if len(poly2) < n + 1:
#         print("Warning: poly2 may have too few vertices for a full-dimensional polytope.")

#     return [poly1, poly2]


# # ---------------------------------------------------------------------------
# # Layer-level enumerator  (unchanged logic from Numba version)
# # ---------------------------------------------------------------------------

# def Enumerator_rapid(
#     hyperplanes,
#     b,
#     original_polytope_test,
#     TH,
#     boundary_hyperplanes,
#     border_bias,
#     parallel,
#     D,
#     m,
#     use_wide=False,
# ):
#     """Enumerate all linear regions produced by one layer of ReLU hyperplanes.

#     Identical to the Numba version; dispatcher thresholds kept the same so
#     timing comparisons are apple-to-apple.
#     """
#     enumerate_poly = list(original_polytope_test)

#     for i in range(len(hyperplanes)):
#         intact_poly = []
#         poly_dummy  = []
#         n = len(enumerate_poly[0][0])

#         # Sign-variation screening.
#         sgn_var = []
#         for k in enumerate_poly:
#             dum = np.dot(k, hyperplanes[i].T) + b[i]
#             if np.min(dum) < -1e-5 and np.max(dum) > 1e-5:
#                 sgn_var.append(np.max(dum) * np.min(dum))
#             else:
#                 sgn_var.append(0.0)

#         global_bit_index = i + len(boundary_hyperplanes[0])

#         # Total bits needed = all boundary hyperplanes accumulated so far
#         # PLUS all remaining neurons in this layer (including the current one).
#         # This ensures the mask array is wide enough for every bit index that
#         # _wide_mask_or_and will ever write during this Enumerator_rapid call.
#         total_bits_needed = len(boundary_hyperplanes[0]) + len(hyperplanes)
#         num_words_needed  = (total_bits_needed + 63) // 64

#         for j in range(len(enumerate_poly)):
#             if sgn_var[j] < -1e-9:
#                 hyperplane_val = np.dot(enumerate_poly[j], hyperplanes[i].T) + b[i]
#                 verts = np.array(enumerate_poly[j])
#                 bh    = np.array(boundary_hyperplanes[0])
#                 bb    = np.array(border_bias[0])

#                 if use_wide:
#                     masks_w = generate_mask_wide(
#                         verts, bh, bb, tolerance=1e-10, num_words=num_words_needed
#                     )
#                     polytops_test, _, created_verts = slice_polytope_wide(
#                         verts, np.array(hyperplane_val), masks_w, global_bit_index, n,
#                     )
#                 else:
#                     masks = generate_mask(verts, bh, bb, tolerance=1e-10)
#                     # Keep the same dispatch threshold as the Numba version.
#                     if len(masks) < 1000:
#                         polytops_test, _, created_verts = slice_polytope_with_hyperplane(
#                             verts, np.array(hyperplane_val), masks, global_bit_index, n,
#                         )
#                     else:
#                         polytops_test, _, created_verts = slice_polytope_parallel(
#                             verts, np.array(hyperplane_val), masks, global_bit_index, n,
#                         )

#                     # Fallback (same condition as Numba version).
#                     if len(created_verts) <= n - 1:
#                         polytops_test, _, created_verts = slice_polytope_with_hyperplane_jit(
#                             verts, np.array(hyperplane_val), masks, global_bit_index, n,
#                         )

#                 if len(created_verts) > n - 1:
#                     result = Polytope_formation_hd(
#                         enumerate_poly[j],
#                         np.array(hyperplane_val),
#                         TH,
#                         np.array(created_verts),
#                         polytops_test,
#                     )
#                 else:
#                     result = [enumerate_poly[j]]

#                 poly_dummy.extend(result)
#             else:
#                 intact_poly.append(enumerate_poly[j])

#         intact_poly.extend(poly_dummy)
#         boundary_hyperplanes[0] = np.vstack(
#             (boundary_hyperplanes[0], hyperplanes[i])
#         ).tolist()
#         border_bias[0] = border_bias[0] + [b[i]]
#         enumerate_poly = intact_poly

#     return enumerate_poly


# # ---------------------------------------------------------------------------
# # Wide-mask primitives  (> 64 hyperplanes)
# # ---------------------------------------------------------------------------

# def generate_mask_wide(vertices, hyperplanes, b, tolerance=1e-7, num_words=None):
#     """Multi-word hyperplane-incidence bitmasks (pure NumPy).

#     Parameters
#     ----------
#     vertices    : (V, n) float64 array
#     hyperplanes : (H, n) float64 array  -- current boundary hyperplanes only
#     b           : (H,)   float64 array
#     tolerance   : float
#     num_words   : int or None
#         Total number of uint64 words needed to hold ALL hyperplane bits,
#         including future neurons not yet in `hyperplanes`. Must satisfy
#         num_words >= (global_bit_index_max + 1 + 63) // 64.
#         If None, inferred from hyperplanes.shape[0] (may be too small for
#         wide-mask mode — always pass explicitly from Enumerator_rapid).
#     """
#     n_verts  = vertices.shape[0]
#     n_planes = hyperplanes.shape[0]
#     if num_words is None:
#         num_words = (n_planes + 63) // 64
#     masks = np.zeros((n_verts, num_words), dtype=np.uint64)

#     vals     = vertices @ hyperplanes.T + b   # (V, H)
#     on_plane = np.abs(vals) <= tolerance       # (V, H) bool

#     for h in range(n_planes):
#         word = h // 64
#         bit  = h % 64
#         masks[on_plane[:, h], word] |= np.uint64(1 << bit)

#     return masks


# def _wide_popcount_ge(mask_u, mask_v, req_shared, num_words):
#     """Return True if bitwise AND of two wide masks has >= req_shared bits set."""
#     count = 0
#     for w in range(num_words):
#         temp = int(mask_u[w]) & int(mask_v[w])
#         while temp > 0:
#             temp &= temp - 1
#             count += 1
#             if count >= req_shared:
#                 return True
#     return False


# def _wide_mask_or_and(shared_u, mask_u, bit_index, num_words):
#     """Compute shared_u | (mask_u & (1 << bit_index)) for wide masks."""
#     result = shared_u.copy()
#     word = bit_index // 64
#     bit  = bit_index % 64
#     result[word] |= mask_u[word] & np.uint64(1 << bit)
#     return result


# def slice_polytope_wide(enumerate_poly, hyperplane_val, masks_w, i, n):
#     """Slice a polytope using wide (multi-word) bitmask adjacency (pure NumPy)."""
#     num_words  = masks_w.shape[1]
#     req_shared = n - 1

#     strict_index_in  = np.where(hyperplane_val < -1e-9)[0]
#     strict_index_out = np.where(hyperplane_val >  1e-9)[0]
#     index_in  = np.where(hyperplane_val <= 1e-5)[0]
#     index_out = np.where(hyperplane_val >= -1e-5)[0]

#     max_new = len(index_in) * len(index_out)
#     new_verts_buffer = np.zeros((max_new, n),          dtype=np.float64)
#     new_masks_buffer = np.zeros((max_new, num_words),   dtype=np.uint64)
#     count = 0

#     for k in range(len(strict_index_in)):
#         u_idx  = strict_index_in[k]
#         mask_u = masks_w[u_idx]

#         for l in range(len(strict_index_out)):
#             v_idx = strict_index_out[l]

#             if hyperplane_val[u_idx] < -1e-9 and hyperplane_val[v_idx] > 1e-9:
#                 mask_v      = masks_w[v_idx]
#                 shared_mask = mask_u & mask_v

#                 if _wide_popcount_ge(mask_u, mask_v, req_shared, num_words):
#                     d1 = hyperplane_val[u_idx]
#                     d2 = hyperplane_val[v_idx]
#                     t  = -d1 / (d2 - d1)
#                     new_verts_buffer[count] = (
#                         enumerate_poly[u_idx]
#                         + t * (enumerate_poly[v_idx] - enumerate_poly[u_idx])
#                     )
#                     new_masks_buffer[count] = _wide_mask_or_and(
#                         shared_mask, mask_u, i, num_words
#                     )
#                     count += 1

#     created_verts = new_verts_buffer[:count]
#     created_masks = new_masks_buffer[:count]

#     n_in  = len(index_in)
#     n_out = len(index_out)

#     verts_in  = np.empty((n_in  + count, n), dtype=np.float64)
#     masks_in  = np.empty((n_in  + count, num_words), dtype=np.uint64)
#     verts_out = np.empty((n_out + count, n), dtype=np.float64)
#     masks_out = np.empty((n_out + count, num_words), dtype=np.uint64)

#     verts_in[:n_in]   = enumerate_poly[index_in]
#     masks_in[:n_in]   = masks_w[index_in]
#     verts_in[n_in:]   = created_verts
#     masks_in[n_in:]   = created_masks

#     verts_out[:n_out] = enumerate_poly[index_out]
#     masks_out[:n_out] = masks_w[index_out]
#     verts_out[n_out:] = created_verts
#     masks_out[n_out:] = created_masks

#     return [verts_in, verts_out], [masks_in, masks_out], created_verts


# # ---------------------------------------------------------------------------
# # Storage helpers  (unchanged — no Numba involvement)
# # ---------------------------------------------------------------------------

# class RaggedPolytopeStorage:
#     """Contiguous in-memory storage for variable-vertex polytopes."""

#     def __init__(self):
#         self.vertices       = None
#         self.offsets        = [0]
#         self.n_dim          = None
#         self.total_vertices = 0

#     def add_polytope(self, vertices):
#         vertices = np.asarray(vertices, dtype=np.float64)
#         if self.n_dim is None:
#             self.n_dim    = vertices.shape[1]
#             self.vertices = vertices.copy()
#         else:
#             self.vertices = np.vstack([self.vertices, vertices])
#         self.total_vertices += len(vertices)
#         self.offsets.append(self.total_vertices)

#     def get_polytope(self, idx):
#         return self.vertices[self.offsets[idx]:self.offsets[idx + 1]]

#     def memory_usage_mb(self):
#         if self.vertices is None:
#             return 0.0
#         return (self.vertices.nbytes + len(self.offsets) * 8) / (1024 * 1024)

#     def __len__(self):
#         return len(self.offsets) - 1

#     def __getitem__(self, idx):
#         return self.get_polytope(idx)


# class HybridRaggedStorage:
#     """Ragged storage with disk overflow (unchanged from Numba version)."""

#     def __init__(self, memory_limit_mb=500, temp_dir=None):
#         self.memory_limit_mb = memory_limit_mb
#         self.temp_dir        = temp_dir or tempfile.mkdtemp()
#         self.memory_storage  = RaggedPolytopeStorage()
#         self.disk_chunks     = []
#         self.mode            = "memory"
#         self.total_polytopes = 0

#     def add_polytope(self, vertices):
#         vertices = np.asarray(vertices, dtype=np.float64)
#         self.memory_storage.add_polytope(vertices)
#         self.total_polytopes += 1

#         if self.mode == "memory":
#             if self.memory_storage.memory_usage_mb() > self.memory_limit_mb:
#                 self._flush_to_disk()
#                 self.mode = "disk"
#         else:
#             if len(self.memory_storage) >= 1000:
#                 self._flush_to_disk()

#     def _flush_to_disk(self):
#         if len(self.memory_storage) == 0:
#             return
#         chunk_idx     = len(self.disk_chunks)
#         vertices_file = os.path.join(self.temp_dir, f"vertices_{chunk_idx}.npy")
#         offsets_file  = os.path.join(self.temp_dir, f"offsets_{chunk_idx}.npy")
#         np.save(vertices_file, self.memory_storage.vertices)
#         np.save(offsets_file,  np.array(self.memory_storage.offsets, dtype=np.int64))
#         self.disk_chunks.append(
#             {"vertices_file": vertices_file, "offsets_file": offsets_file,
#              "count": len(self.memory_storage)}
#         )
#         import gc
#         self.memory_storage = RaggedPolytopeStorage()
#         gc.collect()

#     def get_polytope(self, idx):
#         cumulative = 0
#         for chunk in self.disk_chunks:
#             if idx < cumulative + chunk["count"]:
#                 vertices  = np.load(chunk["vertices_file"])
#                 offsets   = np.load(chunk["offsets_file"])
#                 local_idx = idx - cumulative
#                 return vertices[offsets[local_idx]:offsets[local_idx + 1]]
#             cumulative += chunk["count"]
#         return self.memory_storage[idx - cumulative]

#     def cleanup(self):
#         for chunk in self.disk_chunks:
#             for key in ("vertices_file", "offsets_file"):
#                 if os.path.exists(chunk[key]):
#                     os.remove(chunk[key])
#         if os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
#             os.rmdir(self.temp_dir)

#     def __len__(self):
#         return self.total_polytopes

#     def __getitem__(self, idx):
#         return self.get_polytope(idx)


# # ---------------------------------------------------------------------------
# # Deep hyperplane recovery  (unchanged)
# # ---------------------------------------------------------------------------

# def finding_deep_hype(hyperplanes, b, S_prime, border_hyperplane, border_bias, i, n):
#     S_init = np.eye(np.shape(hyperplanes[0])[0])
#     hype   = S_init @ hyperplanes[0]
#     bias   = S_init @ b[0]

#     m = 2 * n
#     for j in range(0, i):
#         border_hyperplane = np.vstack((border_hyperplane, hype))
#         border_bias       = np.hstack((np.array(border_bias), bias))
#         mid_point         = np.mean(S_prime, axis=0)
#         S = np.sign(np.maximum(border_hyperplane @ mid_point + border_bias, 0))
#         hype = hyperplanes[j + 1] @ np.diag(S[m:]) @ hype
#         bias = hyperplanes[j + 1] @ np.diag(S[m:]) @ bias + b[j + 1]
#         m   += np.shape(hyperplanes[j])[0]

#     return hype, bias, border_hyperplane, border_bias.tolist()


def get_cell_hyperplanes_input_space(
    sv_i           : np.ndarray,   # (total_neurons,) activation pattern
    layer_W        : list,         # layer_W[l] shape (H_l, n_in_l)
    layer_b        : list,         # layer_b[l] shape (H_l,)
    boundary_H     : np.ndarray,   # (B, n) domain boundary hyperplane normals
    boundary_b     : np.ndarray,   # (B,)   domain boundary offsets
):
    """
    Compute all hyperplanes forming cell i in input space.

    Combines:
      1. Network hyperplanes — each neuron's activation boundary mapped
         back to input space via the fixed activation pattern sv_i.
      2. Domain boundary hyperplanes — safe set or domain constraints.

    The output feeds directly into generate_mask() and
    slice_polytope_with_hyperplane() for correct polytope splitting.

    Returns
    -------
    H_all : (total_neurons + B, n)  hyperplane normals in input space
    b_all : (total_neurons + B,)    hyperplane offsets
    """
    n = layer_W[0].shape[1]

    # Split activation pattern into per-layer masks
    layer_sizes = [W.shape[0] for W in layer_W]
    masks = []
    offset = 0
    for size in layer_sizes:
        masks.append(sv_i[offset: offset + size])
        offset += size

    H_planes = []
    b_planes = []

    A_prev = np.eye(n)
    b_prev = np.zeros(n)

    for W, b, mask in zip(layer_W, layer_b, masks):
        # Pre-activation of this layer in input space
        H_l = W @ A_prev        # (H_l, n)
        o_l = W @ b_prev + b    # (H_l,)

        H_planes.append(H_l)
        b_planes.append(o_l)

        # Update linear map for next layer: apply D_l
        A_prev = mask[:, None] * H_l   # (H_l, n)
        b_prev = mask * o_l            # (H_l,)

    # Stack network hyperplanes with domain boundaries
    H_all = np.vstack([*H_planes, boundary_H])
    b_all = np.concatenate([*b_planes, boundary_b])

    return H_all, b_all





@njit
def generate_mask_face_serial(vertices, hyperplanes, b, tolerance=1e-5):
    """Serial bitmask generation for vertices of an (n-1)-dimensional face.
    Sparsity check expects at least n-1 bits set per vertex."""
    n_verts  = vertices.shape[0]
    n_planes = hyperplanes.shape[0]
    n_dim    = vertices.shape[1]
    masks    = np.zeros(n_verts, dtype=np.uint64)
    flag=0

    for i in range(n_verts):
        mask = np.uint64(0)
        cntr = 0
        v    = vertices[i]
        for h in range(n_planes):
            val = hyperplanes[h, :] @ v + b[h]
            if np.abs(val) <= tolerance:
                mask |= (np.uint64(1) << np.uint64(h))
                cntr += 1
        masks[i] = mask
        if cntr < n_dim - 1:
            flag=1
            # print(np.ascontiguousarray(hyperplanes) @ v + b)
    return masks,flag


@njit(parallel=True)
def generate_mask_face(vertices, hyperplanes, b, tolerance=1e-7):
    """Parallel bitmask generation for vertices of an (n-1)-dimensional face.
    Sparsity check expects at least n-1 bits set per vertex."""
    n_verts  = vertices.shape[0]
    n_planes = hyperplanes.shape[0]
    n_dim    = vertices.shape[1]
    masks    = np.zeros(n_verts, dtype=np.uint64)
    flag=0
    for i in prange(n_verts):
        mask = np.uint64(0)
        cntr = 0
        v    = vertices[i]
        for h in range(n_planes):
            val = hyperplanes[h, :] @ v + b[h]
            if np.abs(val) <= tolerance:
                mask |= (np.uint64(1) << np.uint64(h))
                cntr += 1
        masks[i] = mask
        if cntr < n_dim - 1:
            # print(np.ascontiguousarray(hyperplanes) @ v + b)
            # print("Warning: face vertex has fewer than n-1 incidences, check tolerance")
            falg=1
    return masks,falg


@njit
def slice_face_with_hyperplane(face_verts, hyperplane_val, masks, i, n):
    """Slice an (n-1)-dimensional face with a hyperplane (narrow uint64 masks).

    Adjacency condition: n-2 shared hyperplanes (one fewer than full polytope
    slicer) because the face hyperplane is already implicitly shared.

    Parameters
    ----------
    face_verts     : (V, n) float64 array
    hyperplane_val : (V,)   float64 array
    masks          : (V,)   uint64 array
    i              : int -- global bit index of the cutting hyperplane
    n              : int -- ambient input dimension

    Returns
    -------
    [verts_in, verts_out], [masks_in, masks_out], created_verts
    """
    ONE        = np.uint64(1)
    req_shared = n - 2   # key difference from full polytope slicer

    strict_index_in  = np.where(hyperplane_val < -1e-9)[0]
    strict_index_out = np.where(hyperplane_val >  1e-9)[0]
    index_in  = np.where(hyperplane_val <= 1e-5)[0]
    index_out = np.where(hyperplane_val >= -1e-5)[0]

    max_new = len(index_in) * len(index_out)
    new_verts_buffer = np.zeros((max_new, n), dtype=np.float64)
    new_masks_buffer = np.zeros(max_new,      dtype=np.uint64)

    count = 0
    for k in range(len(strict_index_in)):
        u_idx  = strict_index_in[k]
        mask_u = masks[u_idx]

        for l in range(len(strict_index_out)):
            v_idx = strict_index_out[l]

            if hyperplane_val[u_idx] < -1e-9 and hyperplane_val[v_idx] > 1e-9:
                mask_v      = masks[v_idx]
                shared_mask = mask_u & mask_v

                n_shared = 0
                temp = shared_mask
                while temp > np.uint64(0) and n_shared < req_shared:
                    temp &= temp - ONE
                    n_shared += 1

                if n_shared >= req_shared:
                    d1 = hyperplane_val[u_idx]
                    d2 = hyperplane_val[v_idx]
                    t  = -d1 / (d2 - d1)
                    new_verts_buffer[count] = (
                        face_verts[u_idx]
                        + t * (face_verts[v_idx] - face_verts[u_idx])
                    )
                    new_masks_buffer[count] = shared_mask | (mask_u & (ONE << np.uint64(i)))
                    count += 1

    created_verts = new_verts_buffer[:count]
    created_masks = new_masks_buffer[:count]

    verts_in  = np.vstack((face_verts[index_in],  created_verts))
    masks_in  = np.concatenate((masks[index_in],  created_masks))
    verts_out = np.vstack((face_verts[index_out], created_verts))
    masks_out = np.concatenate((masks[index_out], created_masks))

    return [verts_in, verts_out], [masks_in, masks_out], created_verts


# ---------------------------------------------------------------------------
# Wide-mask variants  (> 64 hyperplanes)
# ---------------------------------------------------------------------------

@njit(parallel=True)
def generate_mask_face_wide(vertices, hyperplanes, b, tolerance=1e-5):
    """Multi-word bitmask generation for vertices of an (n-1)-dimensional face.
    Sparsity check expects at least n-1 bits set per vertex."""
    n_verts   = vertices.shape[0]
    n_planes  = hyperplanes.shape[0]
    n_dim     = vertices.shape[1]
    num_words = (n_planes + 63) // 64
    masks     = np.zeros((n_verts, num_words), dtype=np.uint64)

    for i in prange(n_verts):
        v    = vertices[i]
        cntr = 0
        for h in range(n_planes):
            val = hyperplanes[h, :] @ v + b[h]
            if np.abs(val) <= tolerance:
                cntr += 1
                word = np.uint64(h // 64)
                bit  = np.uint64(h % 64)
                masks[i, word] |= np.uint64(1) << bit
        if cntr < n_dim - 1:
            print("Warning: face vertex has fewer than n-1 incidences, check tolerance")
    return masks


@njit
def _slice_face_wide_seq(face_verts, hyperplane_val, masks_w, i, n):
    """Sequential wide-mask face slicer. Adjacency condition: n-2 shared bits."""
    num_words  = masks_w.shape[1]
    req_shared = n - 2   # key difference from full polytope slicer

    strict_index_in  = np.where(hyperplane_val < -1e-9)[0]
    strict_index_out = np.where(hyperplane_val >  1e-9)[0]
    index_in  = np.where(hyperplane_val <= 1e-5)[0]
    index_out = np.where(hyperplane_val >= -1e-5)[0]

    max_new = len(index_in) * len(index_out)
    new_verts_buffer = np.zeros((max_new, n),          dtype=np.float64)
    new_masks_buffer = np.zeros((max_new, num_words),   dtype=np.uint64)
    count = 0

    for k in range(len(strict_index_in)):
        u_idx  = strict_index_in[k]
        mask_u = masks_w[u_idx]

        for l in range(len(strict_index_out)):
            v_idx = strict_index_out[l]

            if hyperplane_val[u_idx] < -1e-9 and hyperplane_val[v_idx] > 1e-9:
                mask_v      = masks_w[v_idx]
                shared_mask = mask_u & mask_v

                if _wide_popcount_ge(mask_u, mask_v, req_shared, num_words):
                    d1 = hyperplane_val[u_idx]
                    d2 = hyperplane_val[v_idx]
                    t  = -d1 / (d2 - d1)
                    new_verts_buffer[count] = (
                        face_verts[u_idx]
                        + t * (face_verts[v_idx] - face_verts[u_idx])
                    )
                    new_masks_buffer[count] = _wide_mask_or_and(
                        shared_mask, mask_u, i, num_words
                    )
                    count += 1

    created_verts = new_verts_buffer[:count]
    created_masks = new_masks_buffer[:count]

    n_in  = len(index_in)
    n_out = len(index_out)

    verts_in  = np.empty((n_in  + count, n),          dtype=np.float64)
    masks_in  = np.empty((n_in  + count, num_words),   dtype=np.uint64)
    verts_out = np.empty((n_out + count, n),           dtype=np.float64)
    masks_out = np.empty((n_out + count, num_words),   dtype=np.uint64)

    verts_in[:n_in]   = face_verts[index_in]
    masks_in[:n_in]   = masks_w[index_in]
    verts_in[n_in:]   = created_verts
    masks_in[n_in:]   = created_masks

    verts_out[:n_out] = face_verts[index_out]
    masks_out[:n_out] = masks_w[index_out]
    verts_out[n_out:] = created_verts
    masks_out[n_out:] = created_masks

    return [verts_in, verts_out], [masks_in, masks_out], created_verts


@njit(parallel=True)
def _slice_face_wide_par(face_verts, hyperplane_val, masks_w, i, n):
    """Parallel wide-mask face slicer (prange over u-vertices).
    Adjacency condition: n-2 shared bits."""
    num_words  = masks_w.shape[1]
    req_shared = n - 2   # key difference from full polytope slicer

    strict_index_in  = np.where(hyperplane_val < -1e-9)[0]
    strict_index_out = np.where(hyperplane_val >  1e-9)[0]
    index_in  = np.where(hyperplane_val <= 1e-5)[0]
    index_out = np.where(hyperplane_val >= -1e-5)[0]

    n_in_s  = len(strict_index_in)
    n_out_s = len(strict_index_out)
    max_new = n_in_s * n_out_s

    new_verts_buffer = np.zeros((max_new, n),          dtype=np.float64)
    new_masks_buffer = np.zeros((max_new, num_words),   dtype=np.uint64)
    valid            = np.zeros(max_new,                dtype=nb.boolean)

    for k in prange(n_in_s):
        u_idx  = strict_index_in[k]
        mask_u = masks_w[u_idx]
        d1     = hyperplane_val[u_idx]

        for l in range(n_out_s):
            v_idx    = strict_index_out[l]
            flat_idx = k * n_out_s + l

            if d1 < -1e-9 and hyperplane_val[v_idx] > 1e-9:
                mask_v = masks_w[v_idx]
                if _wide_popcount_ge(mask_u, mask_v, req_shared, num_words):
                    d2 = hyperplane_val[v_idx]
                    t  = -d1 / (d2 - d1)
                    for dim in range(n):
                        new_verts_buffer[flat_idx, dim] = (
                            face_verts[u_idx, dim]
                            + t * (face_verts[v_idx, dim] - face_verts[u_idx, dim])
                        )
                    new_masks_buffer[flat_idx] = _wide_mask_or_and(
                        mask_u & mask_v, mask_u, i, num_words
                    )
                    valid[flat_idx] = True

    created_verts, created_masks = _compact_wide(
        new_verts_buffer, new_masks_buffer, valid, n, num_words
    )

    n_in  = len(index_in)
    n_out = len(index_out)
    count = len(created_verts)

    verts_in  = np.empty((n_in  + count, n),          dtype=np.float64)
    masks_in  = np.empty((n_in  + count, num_words),   dtype=np.uint64)
    verts_out = np.empty((n_out + count, n),           dtype=np.float64)
    masks_out = np.empty((n_out + count, num_words),   dtype=np.uint64)

    verts_in[:n_in]   = face_verts[index_in]
    masks_in[:n_in]   = masks_w[index_in]
    verts_in[n_in:]   = created_verts
    masks_in[n_in:]   = created_masks

    verts_out[:n_out] = face_verts[index_out]
    masks_out[:n_out] = masks_w[index_out]
    verts_out[n_out:] = created_verts
    masks_out[n_out:] = created_masks

    return [verts_in, verts_out], [masks_in, masks_out], created_verts


def slice_face_wide(face_verts, hyperplane_val, masks_w, i, n):
    """Dispatch wide-mask face slicer: parallel for large faces, serial otherwise.

    Parameters
    ----------
    face_verts     : (V, n) float64 array
    hyperplane_val : (V,)   float64 array
    masks_w        : (V, num_words) uint64 array
    i              : int -- global bit index of the cutting hyperplane
    n              : int -- ambient input dimension

    Returns
    -------
    [verts_in, verts_out], [masks_in, masks_out], created_verts
    """
    n_in_s  = int(np.sum(hyperplane_val < -1e-9))
    n_out_s = int(np.sum(hyperplane_val >  1e-9))
    if n_in_s * n_out_s > _PAR_THRESHOLD:
        return _slice_face_wide_par(face_verts, hyperplane_val, masks_w, i, n)
    return _slice_face_wide_seq(face_verts, hyperplane_val, masks_w, i, n)


@njit
def finding_side_polytope(boundary_hyperplanes, enumerate_poly, border_bias):
    n = boundary_hyperplanes.shape[1]
    mask = np.zeros(len(boundary_hyperplanes), dtype=np.bool_)
    for j in range(len(boundary_hyperplanes)):
        dum = boundary_hyperplanes[j] @ enumerate_poly.T + border_bias[j]
        if np.sum(np.abs(dum) < 1e-3) >= n:
            mask[j] = True
    return boundary_hyperplanes[mask], border_bias[mask]



@njit
def finding_side(boundary_hyperplanes, enumerate_poly, border_bias):
    n = boundary_hyperplanes.shape[1]
    mask = np.zeros(len(boundary_hyperplanes), dtype=np.bool_)
    for j in range(len(boundary_hyperplanes)):
        dum = boundary_hyperplanes[j] @ enumerate_poly.T + border_bias[j]
        if np.sum(np.abs(dum) < 1e-5) >= n - 1:
            mask[j] = True
    return boundary_hyperplanes[mask], border_bias[mask]
# @njit
# def finding_side(boundary_hyperplanes,enumerate_poly,border_bias):

#     b_h=[]
#     b_b=[]
#     n=len(boundary_hyperplanes[0])
#     for j,i in enumerate(boundary_hyperplanes):
#         dum=np.dot(np.ascontiguousarray(i),np.ascontiguousarray(enumerate_poly.T))+border_bias[j]
#         if np.sum(np.abs(dum)<1e-5)>=n-1:
#             b_h.append(i)
#             b_b.append(border_bias[j])

#     return b_h,b_b


@njit(cache=True)
def _dedup_verts_nb(verts, tol=1e-8):
    """Remove near-duplicate rows using pairwise L-inf distance."""
    n_verts = verts.shape[0]
    ndim    = verts.shape[1]
    if n_verts <= 1:
        return verts
    keep = np.ones(n_verts, dtype=np.bool_)
    for i in range(1, n_verts):
        for j in range(i):
            if not keep[j]:
                continue
            d = 0.0
            for k in range(ndim):
                diff = verts[i, k] - verts[j, k]
                if diff < 0.0:
                    diff = -diff
                if diff > d:
                    d = diff
            if d < tol:
                keep[i] = False
                break
    count = 0
    for i in range(n_verts):
        if keep[i]:
            count += 1
    out = np.empty((count, ndim), dtype=np.float64)
    idx = 0
    for i in range(n_verts):
        if keep[i]:
            out[idx] = verts[i]
            idx += 1
    return out


@njit
def finding_side_old(boundary_hyperplanes,enumerate_poly,border_bias):
    # side=list()
    # hyp_f=List()
    side=List()
    # side=[]
    hyp_f=[]
    n=len(boundary_hyperplanes[0])
    # test=np.reshape(border_bias,(len(border_bias),1))
    # test=border_bias.reshape((len(border_bias),-1))
    dum=np.dot(boundary_hyperplanes,enumerate_poly.T)+border_bias.reshape((len(border_bias),-1))
    # dum=np.dot(boundary_hyperplanes,(np.array(enumerate_poly)).T)+test
    for j,i in enumerate(dum):
        res=[k for k,l in enumerate(i) if np.abs(l)<1e-10]
        if len(res)>=n:
            if res not in side:
                side.append(((res)))
                hyp_f.append((np.append(boundary_hyperplanes[j],border_bias[j])))
                # vertices=(dum[j])[dum[j]<1e-10 and dum[j]>-1e-10]
    return side,hyp_f