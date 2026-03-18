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
RaggedPolytopeStorage               -- In-memory ragged array for variable-vertex polytopes.
HybridRaggedStorage                 -- RaggedPolytopeStorage with automatic disk overflow.
"""

import os
import tempfile
import time

import numpy as np
from numba import njit, prange
from numba import types  # noqa: F401  (kept for potential extension)
from numba.typed import List  # noqa: F401


# ---------------------------------------------------------------------------
# Bitmask generation
# ---------------------------------------------------------------------------

@njit(parallel=True)
def generate_mask(vertices, hyperplanes, b, tolerance=1e-7):
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
        mask = np.uint64(0)
        v    = vertices[i]
        for h in range(n_planes):
            val = hyperplanes[h, :] @ v + b[h]
            if np.abs(val) <= tolerance:
                mask = mask | (np.uint64(1) << np.uint64(h))
        masks[i] = mask

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
    * len(masks) < 1000  -> :func:`slice_polytope_with_hyperplane` (serial JIT).
    * len(masks) >= 1000 -> :func:`slice_polytope_parallel` (parallel JIT).
    * If the primary slicer yields fewer than (n-1) intersections,
      :func:`slice_polytope_with_hyperplane_jit` is called as a fallback.

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

    Returns
    -------
    enumerate_poly : list of (V_k, n) arrays -- all enumerated polytopes for this layer.
    """
    enumerate_poly = list(original_polytope_test)

    for i in range(len(hyperplanes)):
        intact_poly = []
        poly_dummy  = []
        n = len(enumerate_poly[0][0])

        # ---- Sign-variation screening ----
        # Compute the signed hyperplane value at all vertices of every polytope.
        # sgn_var[j] < 0 iff polytope j strictly straddles the hyperplane.
        sgn_var = []
        for k in enumerate_poly:
            dum = np.dot(k, hyperplanes[i].T) + b[i]
            if np.min(dum) < -1e-5 and np.max(dum) > 1e-5:
                sgn_var.append(np.max(dum) * np.min(dum))
            else:
                sgn_var.append(0.0)

        # ---- Slicing loop ----
        for j in range(len(enumerate_poly)):
            if sgn_var[j] < -1e-9:
                # Re-evaluate hyperplane values for this polytope's vertices.
                hyperplane_val = np.dot(enumerate_poly[j], hyperplanes[i].T) + b[i]

                # Assign bitmasks encoding which accumulated boundary hyperplanes
                # each vertex lies on.
                masks = generate_mask(
                    np.array(enumerate_poly[j]),
                    np.array(boundary_hyperplanes[0]),
                    np.array(border_bias[0]),
                    tolerance=1e-10,
                )

                # Dispatch to serial or parallel slicer based on polytope size.
                if len(masks) < 1000:
                    polytops_test, _, created_verts = slice_polytope_with_hyperplane(
                        np.array(enumerate_poly[j]),
                        np.array(hyperplane_val),
                        masks,
                        i + len(boundary_hyperplanes[0]),
                        n,
                    )
                else:
                    polytops_test, _, created_verts = slice_polytope_parallel(
                        np.array(enumerate_poly[j]),
                        np.array(hyperplane_val),
                        masks,
                        i,
                        n,
                    )

                # If the primary slicer did not find enough intersection points,
                # fall back to the alternate JIT slicer.
                if len(created_verts) > n - 1:
                    result = Polytope_formation_hd(
                        enumerate_poly[j],
                        np.array(hyperplane_val),
                        TH,
                        np.array(created_verts),
                        polytops_test,
                    )
                else:
                    # Insufficient intersections: keep the polytope intact to avoid
                    # introducing numerical artefacts.  This is conservative and
                    # will not silently discard a region.
                    result = [enumerate_poly[j]]

                poly_dummy.extend(result)
            else:
                intact_poly.append(enumerate_poly[j])

        # Accumulate sliced polytopes and add the current hyperplane to the
        # boundary set for subsequent mask generation.
        intact_poly.extend(poly_dummy)
        boundary_hyperplanes[0] = np.vstack(
            (boundary_hyperplanes[0], hyperplanes[i])
        ).tolist()
        border_bias[0] = border_bias[0] + [b[i]]
        enumerate_poly = intact_poly

    return enumerate_poly


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
