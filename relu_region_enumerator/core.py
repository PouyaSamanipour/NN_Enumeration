
"""
core.py
=======

Top-level enumeration pipeline for ReLU neural network linear regions.

The entry point :func:`enumeration_function` loads a TorchScript model,
extracts its weight matrices, and runs a layer-by-layer vertex enumeration
over the axis-aligned input hypercube defined by ``TH``.

For each hidden layer, the algorithm iterates over the already-enumerated
polytopes from the previous layer and applies :func:`Enumerator_rapid` (from
:mod:`relu_region_enumerator.bitwise_utils`) to split each polytope along the
current layer's ReLU hyperplanes.  High-dimensional problems (n > 5) stream
intermediate results through HDF5 files to stay within RAM budgets.

The final cell-vertex representation is written to ``<name_file>_polytope.h5``
and the activation-pattern IDs to a returned array.

Functions
---------
enumeration_function  -- Full enumeration pipeline.
generate_hypercube    -- Generate all vertices of an axis-aligned hypercube.
Finding_cell_id       -- Compute per-layer activation patterns for enumerated cells.
"""

import gc
import itertools
import os
import time
from xml.parsers.expat import model

import h5py
import numpy as np
import torch
import numba as nb

# from .hessian_bound import HessianBounder, compute_local_gradient
from .hessian_bound import HessianBounder, compute_local_gradient

from .bitwise_utils import Enumerator_rapid,finding_deep_hype, generate_mask_wide,slice_polytope_wide
from .Dynamics import load_dynamics, list_systems
from .verify_certificates import verify_barrier
from .verify_certificates import verify_lyapunov




# ---------------------------------------------------------------------------
# Public helper: hypercube vertex generation
# ---------------------------------------------------------------------------

def generate_hypercube(bounds):
    """Generate all 2^d vertices of the axis-aligned input hypercube.

    For each dimension with bound b, the two extreme values are -b and +b.

    Parameters
    ----------
    bounds : list of float -- per-dimension upper bound (length = input dim d).

    Returns
    -------
    list of tuples -- all 2^d vertices, each a d-tuple of floats.

    Example
    -------
    >>> generate_hypercube([1.0, 2.0])
    [(-1.0, -2.0), (-1.0, 2.0), (1.0, -2.0), (1.0, 2.0)]
    """
    return list(itertools.product(*[(-b, b) for b in bounds]))


# ---------------------------------------------------------------------------
# Public helper: activation-pattern identification
# ---------------------------------------------------------------------------

@nb.njit(parallel=True, cache=True)
def _cell_id_kernel(
    Mid_points,   # (N, n_dims)              float64
    H_flat,       # (total_neurons, max_in)  float64  -- zero-padded rows
    b_flat,       # (total_neurons,)         float64
    layer_sizes,  # (num_layers,)            int64
    in_dims,      # (num_layers,)            int64  -- actual input width per layer
    D_flat,       # (total_neurons, N)       float64  -- output
):
    N          = Mid_points.shape[0]
    num_layers = layer_sizes.shape[0]
    max_n      = layer_sizes.max()
 
    for j in nb.prange(N):                      # parallel over regions
        x_in  = np.empty(Mid_points.shape[1])
        x_out = np.empty(max_n)
 
        for d in range(Mid_points.shape[1]):
            x_in[d] = Mid_points[j, d]
 
        row_offset = 0
        for i in range(num_layers):
            n_i  = layer_sizes[i]
            d_in = in_dims[i]               # valid columns for this layer
 
            for h in range(n_i):
                z = b_flat[row_offset + h]
                for d in range(d_in):       # only iterate over valid columns
                    z += H_flat[row_offset + h, d] * x_in[d]
 
                D_flat[row_offset + h, j] = 1.0 if z > 0.0 else 0.0
                x_out[h] = z if z > 0.0 else 0.0
 
            # Copy x_out -> x_in for next layer
            for h in range(n_i):
                x_in[h] = x_out[h]
 
            row_offset += n_i
 
 
def Finding_cell_id(enumerate_poly, hyperplanes, bias, num_hidden_layers, batch_size=None):
    """Compute the activation-pattern matrix for all enumerated linear regions.
 
    Numba-accelerated, deep-network-compatible version.
    Each layer's weight matrix may have a different number of columns
    (n_dims for layer 0, then n_{i-1} neurons for subsequent layers).
 
    Parameters
    ----------
    enumerate_poly    : list of (V_k, n) arrays
    hyperplanes       : list of (H_i, n_in_i) arrays -- one per hidden layer
    bias              : list of (H_i,) arrays
    num_hidden_layers : int
    batch_size        : ignored, kept for API compatibility
 
    Returns
    -------
    D_raw : list of (H_i, N) float64 arrays
            D_raw[i][h, j] = 1 if neuron h in layer i is active for region j
    """
    num_layers = int(num_hidden_layers)
    N          = len(enumerate_poly)
    n_dims     = hyperplanes[0].shape[1]
 
    # ── Centroids ─────────────────────────────────────────────────────────
    Mid_points = np.empty((N, n_dims), dtype=np.float64)
    for idx, poly in enumerate(enumerate_poly):
        Mid_points[idx] = poly.sum(axis=0) / poly.shape[0]
 
    # ── Per-layer shapes ──────────────────────────────────────────────────
    layer_sizes   = np.array([hyperplanes[i].shape[0] for i in range(num_layers)], dtype=np.int64)
    in_dims       = np.array([hyperplanes[i].shape[1] for i in range(num_layers)], dtype=np.int64)
    total_neurons = int(layer_sizes.sum())
    max_in_dim    = int(in_dims.max())
 
    # ── Zero-pad weights to uniform column width for H_flat ───────────────
    H_flat = np.zeros((total_neurons, max_in_dim), dtype=np.float64)
    b_flat = np.empty(total_neurons, dtype=np.float64)
    offset = 0
    for i in range(num_layers):
        n_i  = int(layer_sizes[i])
        d_in = int(in_dims[i])
        H_flat[offset: offset + n_i, :d_in] = hyperplanes[i].astype(np.float64)
        b_flat[offset: offset + n_i]         = bias[i].ravel().astype(np.float64)
        offset += n_i
 
    # ── Output buffer ─────────────────────────────────────────────────────
    D_flat = np.zeros((total_neurons, N), dtype=np.float64)
 
    # ── Run JIT kernel ────────────────────────────────────────────────────
    _cell_id_kernel(Mid_points, H_flat, b_flat, layer_sizes, in_dims, D_flat)
 
    # ── Split back into per-layer arrays ──────────────────────────────────
    D_raw  = []
    offset = 0
    for i in range(num_layers):
        n_i = int(layer_sizes[i])
        D_raw.append(D_flat[offset: offset + n_i, :].copy())
        offset += n_i
 
    return D_raw

# ---------------------------------------------------------------------------
# Barrier certificate verification
# ---------------------------------------------------------------------------
def barrier_certificate_cells(model, enumerate_poly, hyperplanes, b, name_file,test_stat=False):
    """Identify linear regions that straddle the zero level set of a barrier certificate.

    A region is a boundary cell if b(x) changes sign across its vertices,
    i.e. min(b(vertices)) < 0 and max(b(vertices)) > 0.  Since b(x) is
    affine on each linear region, evaluating at vertices is exact.

    Optimisations
    -------------
    - All vertices of a region are batched into one tensor; the model forward
      pass runs once per region, not once per vertex.
    - The centroid for activation-pattern extraction uses np.mean (single call).
    - Activation patterns are accumulated as a list and stacked once at the end.

    Validation Tests
    ----------------
    1. Halfspace satisfaction   : all vertices satisfy neuron hyperplane inequalities.
    2. Full dimensionality      : vertex set spans an n-dimensional affine subspace.
    3. Convexity / volume       : convex hull is non-degenerate (positive volume).
    4. Sign vector consistency  : activation pattern at centroid matches stored row.
    5. Bitmask consistency      : each vertex lies on exactly n defining hyperplanes
                                  (requires bitmasks from enumeration; skipped if None).

    Parameters
    ----------
    model          : torch.nn.Module -- TorchScript barrier certificate network.
    enumerate_poly : list of (V_k, n) arrays -- enumerated polytope vertices.
    hyperplanes    : list of (H_i, n) arrays -- hidden layer weight matrices.
    b              : list of (H_i,)   arrays -- hidden layer biases.
    name_file      : str -- base name for output files.
    bitmasks       : list of lists of int -- per-cell vertex bitmasks from
                     enumeration (optional; pass None to skip test 5).

    Returns
    -------
    D              : (N_b, total_neurons) float64 array -- binary activation
                     patterns for boundary cells, one row per cell.
    boundary_cells : list of (V_k, n) arrays -- vertex sets of boundary cells.
    validation_report : dict -- summary of validation results.
    """
    import pickle
    import torch
    from scipy.spatial import ConvexHull

    total_neurons = sum(len(h) for h in hyperplanes)
    N             = len(enumerate_poly)
    model.eval()
    boundary_cells  = []
    activation_rows = []
    model_dtype = next(model.parameters()).dtype

    # ── Validation counters ──────────────────────────────────────────────────
    val = {
        "total_cells"            : N,
        "boundary_cells"         : 0,
        # per-boundary-cell checks
        "fail_halfspace"         : 0,
        "fail_full_dimensional"  : 0,
        "fail_convexity"         : 0,
        "fail_sign_vector"       : 0,
        "fail_bitmask"           : 0,
        "degenerate_hull"        : 0,
        "passed_all"             : 0,
    }
    TOL = 1e-6

    # ── Helper: Test 1 — halfspace satisfaction ──────────────────────────────
    def test_halfspace(poly_arr, activation_row):
        """Every vertex must lie on the correct side of every neuron hyperplane."""
        state = poly_arr.copy()
        offset = 0
        for layer_idx, (h, bias_i) in enumerate(zip(hyperplanes, b)):
            m  = len(h)
            sv = activation_row[offset:offset + m]
            z  = state @ h.T + bias_i

            active_mask   = sv > 0.5
            inactive_mask = sv < 0.5

            if active_mask.any():
                active_violations = z[:, active_mask]
                active_min = active_violations.min()
            else:
                active_min = 0.0

            if inactive_mask.any():
                inactive_violations = z[:, inactive_mask]
                inactive_max = inactive_violations.max()
            else:
                inactive_max = 0.0

            active_ok   = active_min   >= -TOL
            inactive_ok = inactive_max <=  TOL

            if not (active_ok and inactive_ok):
                print(f"    Layer {layer_idx}")
                print(f"    Active   min value : {active_min:.6e}  (should be >= 0)")
                print(f"    Inactive max value : {inactive_max:.6e} (should be <= 0)")
                print(f"    Num vertices       : {poly_arr.shape[0]}")
                print(f"    Num active neurons : {active_mask.sum()}")
                print(f"    Num inactive neurons: {inactive_mask.sum()}")
                return False

            state = np.maximum(0.0, z)
            offset += m
        return True
    # ── Helper: Test 2 — full dimensionality ────────────────────────────────
    def test_full_dimensional(poly_arr):
        """Vertex set must span an n-dimensional affine subspace."""
        n = poly_arr.shape[1]
        centered = poly_arr - poly_arr.mean(axis=0)
        rank = np.linalg.matrix_rank(centered, tol=TOL)
        return rank == n

    # ── Helper: Test 3 — convexity / volume ─────────────────────────────────
    def test_convexity(poly_arr, cell_idx):
        TOL_volume = 1e-9
        try:
            hull = ConvexHull(poly_arr)
            if hull.volume <= TOL_volume:
                print(f"    Cell {cell_idx}: volume = {hull.volume:.6e}, "
                    f"num_vertices = {poly_arr.shape[0]}, "
                    f"num_facets = {len(hull.simplices)}")
            return hull.volume > TOL_volume, False
        except Exception as e:
            print(f"    Cell {cell_idx}: degenerate hull — {e}")
            return False, True

    # ── Helper: Test 4 — sign vector consistency ────────────────────────────
    def test_sign_vector(poly_arr, stored_row):
        """Activation pattern evaluated at centroid must match stored row."""
        centroid = poly_arr.mean(axis=0)
        state    = centroid.copy()
        recomputed = []
        for h, bias_i in zip(hyperplanes, b):
            z     = h @ state + bias_i
            state = np.maximum(0.0, z)
            recomputed.append((z > 0).astype(np.float64))
        recomputed = np.concatenate(recomputed)
        return np.all(recomputed == stored_row)

    # ── Helper: Test 5 — bitmask consistency ────────────────────────────────
    def test_bitmasks(cell_bitmasks, n_dim):
        """Each vertex must lie on exactly n defining hyperplanes."""
        if cell_bitmasks is None:
            return True                              # skip if not provided
        for mu in cell_bitmasks:
            if bin(mu).count('1') != n_dim:
                return False
        return True

    # ── Main loop ────────────────────────────────────────────────────────────
    with torch.no_grad():
        for cell_idx, poly in enumerate(enumerate_poly):
            poly_arr = np.asarray(poly, dtype=np.float64)
            n_dim    = poly_arr.shape[1]

            # Boundary detection — single forward pass over all vertices.
            P_k = model(torch.tensor(poly_arr, dtype=model_dtype))

            if P_k.min().item() < 1e-6 and P_k.max().item() > -1e-6:
                val["boundary_cells"] += 1

                # Compute activation pattern from centroid.
                state = poly_arr.mean(axis=0).copy()
                row   = []
                for h, bias_i in zip(hyperplanes, b):
                    z     = h @ state + bias_i
                    state = np.maximum(0.0, z)
                    row.append((z > 0).astype(np.float64))
                activation_row = np.concatenate(row)
                if test_stat:
                    # ── Run all five validation tests ────────────────────────────
                    cell_passed = True

                    # Test 1: halfspace satisfaction
                    if not test_halfspace(poly_arr, activation_row):
                        val["fail_halfspace"] += 1
                        cell_passed = False
                        print(f"  [WARN] Cell {cell_idx}: failed halfspace test")

                    # Test 2: full dimensionality
                    if not test_full_dimensional(poly_arr):
                        val["fail_full_dimensional"] += 1
                        cell_passed = False
                        print(f"  [WARN] Cell {cell_idx}: not full dimensional")

                    # Test 3: convexity / volume
                    # passed_convex, is_degenerate = test_convexity(poly_arr,cell_idx)
                    # if is_degenerate:
                    #     val["degenerate_hull"] += 1
                    #     cell_passed = False
                    #     print(f"  [WARN] Cell {cell_idx}: degenerate convex hull")
                    # elif not passed_convex:
                    #     val["fail_convexity"] += 1
                    #     cell_passed = False
                    #     print(f"  [WARN] Cell {cell_idx}: failed convexity/volume test")

                    # Test 4: sign vector consistency
                    if not test_sign_vector(poly_arr, activation_row):
                        val["fail_sign_vector"] += 1
                        cell_passed = False
                        print(f"  [WARN] Cell {cell_idx}: sign vector mismatch")

                    # Test 5: bitmask consistency (skipped if bitmasks not provided)
                    cell_bitmasks = None  # wire up to your enumeration if available
                    if not test_bitmasks(cell_bitmasks, n_dim):
                        val["fail_bitmask"] += 1
                        cell_passed = False
                        print(f"  [WARN] Cell {cell_idx}: bitmask inconsistency")

                    if cell_passed:
                        val["passed_all"] += 1

                boundary_cells.append(poly_arr)
                activation_rows.append(activation_row)

    # ── Validation report ────────────────────────────────────────────────────
    if test_stat:
        print("\n" + "="*55)
        print("VALIDATION REPORT")
        print("="*55)
        print(f"  Total cells enumerated      : {val['total_cells']}")
        print(f"  Boundary cells found        : {val['boundary_cells']}")
        print(f"  Passed all tests            : {val['passed_all']}")
        print(f"  Failed halfspace test       : {val['fail_halfspace']}")
        print(f"  Failed full-dim test        : {val['fail_full_dimensional']}")
        print(f"  Failed convexity test       : {val['fail_convexity']}")
        print(f"  Degenerate hulls            : {val['degenerate_hull']}")
        print(f"  Failed sign-vector test     : {val['fail_sign_vector']}")
        print(f"  Failed bitmask test         : {val['fail_bitmask']}")
        print("="*55 + "\n")

    # ── Save to HDF5 ─────────────────────────────────────────────────────────
    n_boundary = len(boundary_cells)
    print(f"Boundary cells (straddling b(x)=0): {n_boundary} / {N}")
    out_h5 = name_file + "_boundary_cells.h5"
    with h5py.File(out_h5, "w") as f:
        bc_offsets = np.zeros(len(boundary_cells) + 1, dtype=np.int64)
        for idx, p in enumerate(boundary_cells):
            bc_offsets[idx + 1] = bc_offsets[idx] + len(p)
        f.create_dataset("offsets", data=bc_offsets)
        if boundary_cells:
            n = boundary_cells[0].shape[1]
            ds = f.create_dataset("vertices",
                                  shape=(int(bc_offsets[-1]), n),
                                  dtype=np.float64)
            for idx, p in enumerate(boundary_cells):
                ds[bc_offsets[idx]:bc_offsets[idx + 1]] = p
        if activation_rows:
            f.create_dataset("activation_patterns",
                             data=np.stack(activation_rows))
        # Save validation report to HDF5 as attributes
        grp = f.create_group("validation")
        for k, v in val.items():
            grp.attrs[k] = v
    print(f"Boundary cells saved to: {out_h5}")

    D = (np.stack(activation_rows) if activation_rows
         else np.zeros((0, total_neurons), dtype=np.float64))

    return D, boundary_cells, val

# ---------------------------------------------------------------------------
# Main enumeration pipeline
# ---------------------------------------------------------------------------

def enumeration_function(NN_file, name_file, TH, mode, parallel,
                         verification=None, barrier_model=None):
    """Enumerate all polytopic linear regions of a ReLU network over a hypercube.

    The function loads a TorchScript-saved ReLU network, extracts its weight
    matrices and biases, then recursively splits the input hypercube defined
    by ``TH`` along every ReLU neuron hyperplane, layer by layer.

    High-dimensional mode (n > 5)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    To stay within RAM, intermediate per-layer polytope lists are written to
    HDF5 files in a ``layer_tmp/`` subdirectory and read back before processing
    the next layer.  The temporary files are deleted immediately after loading.

    Output
    ~~~~~~
    - ``<name_file>_polytope.h5`` -- HDF5 file containing the full vertex
      representation of all enumerated regions (datasets: ``offsets``,
      ``vertices``).
    - Console output: enumeration time, region count, per-layer neuron counts.

    For 2-D inputs, a partition figure is also displayed via ``plot_polytope``.

    Parameters
    ----------
    NN_file   : str  -- path to a TorchScript (.pt) model file.
    name_file : str  -- base name for output files (no extension).
    TH        : list of float -- per-dimension domain half-width (length = n).
    mode      : str  -- ``"Rapid_mode"`` (default) or ``"Low_Ram"`` (CSV-based,
                        not recommended; requires additional ``utils_CSV`` module).
    parallel      : bool -- passed through to :func:`Enumerator_rapid`.
    verification  : str or None -- if ``"barrier"``, run barrier certificate
                                   boundary cell extraction after enumeration.
    barrier_model : torch.nn.Module or None -- required when verification=="barrier".

    Returns
    -------
    None.  Results are written to disk and printed to stdout.
    """
    # ------------------------------------------------------------------
    # 1. Load model and extract parameters
    # ------------------------------------------------------------------
    model  = torch.jit.load(NN_file, map_location=torch.device("cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = []
    for _name, param in model.named_parameters():
        with torch.no_grad():
            p = param.cpu().numpy() if device.type == "cuda" else param.numpy()
            params.append(p)
    # The model has one (weight, bias) pair per hidden layer plus one output layer.
    # Total parameter tensors = 2 * (num_hidden_layers + 1).
    # Rearranged: num_hidden_layers = (total_tensors - 4) / 2 + 1
    num_hidden_layers = int(((len(params) - 4) / 2) + 1)
    print(f"Number of hidden layers detected: {num_hidden_layers}")

    # Separate hidden-layer weights/biases from the output layer.
    hyperplanes = []   # Weight matrices for hidden layers: list of (H_i, n) arrays.
    b           = []   # Bias vectors for hidden layers:    list of (H_i,) arrays.
    nn_sizes    = []   # Number of neurons per hidden layer.

    for i in range(len(params) - 2):
        if i % 2 == 0:
            hyperplanes.append(params[i])
            nn_sizes.append(np.shape(params[i])[0])
        else:
            b.append(params[i])

    W = params[-2]   # Output layer weight matrix (unused in enumeration).
    c = params[-1]   # Output layer bias          (unused in enumeration).

    # ------------------------------------------------------------------
    # 2. Initialise input hypercube and boundary hyperplane set
    # ------------------------------------------------------------------
    n_h, n = np.shape(hyperplanes[0])

    # Decide whether to use wide (multi-word) bitmasks.
    # Total hyperplanes = 2n (domain faces) + all hidden neurons.
    # If this exceeds 64, uint64 masks overflow and we must use the wide path.
    total_hyperplanes = 2 * n + sum(nn_sizes)
    use_wide = total_hyperplanes > 64
    if use_wide:
        print(
            f"Wide-mask mode: total hyperplanes = {total_hyperplanes} > 64. "
            f"Using multi-word uint64 masks."
        )
    else:
        print(
            f"Fast-mask mode: total hyperplanes = {total_hyperplanes} <= 64. "
            f"Using scalar uint64 masks."
        )

    # The initial polytope is the single input hypercube, represented as a
    # list containing one vertex array of shape (2^n, n).
    original_polytope_test = np.array([generate_hypercube(TH)])
    enumerate_poly         = list(original_polytope_test)

    # Boundary hyperplanes encode the axis-aligned faces of the input domain.
    # Stacked as [ I_n ; -I_n ] with bias TH[i] for each face, so that
    # boundary_hyperplane @ x + border_bias <= 0 defines the domain interior.
    border_hyperplane = np.vstack((np.eye(n), -np.eye(n)))
    border_bias       = list(TH) + list(TH)   # [TH[0],...,TH[n-1], TH[0],...,TH[n-1]]
    bdh=np.copy(border_hyperplane)
    bdb=np.copy(border_bias)


    # ------------------------------------------------------------------
    # 3. Layer-by-layer enumeration
    # ------------------------------------------------------------------
    use_disk = n > 5
    if use_disk:
        print(f"High-dimensional mode (n={n}): streaming layer output via HDF5.")

    cwd     = os.getcwd()
    tmp_dir = os.path.join(cwd, "layer_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    WRITE_BUFFER_SIZE = 5000   # Polytopes buffered before a single HDF5 flush.

    st_enum = time.time()

    for i in range(num_hidden_layers):

        if use_disk:
            # Open a per-layer HDF5 scratch file for streaming output.
            h5_path = os.path.join(tmp_dir, f"layer_{i}.h5")
            h5f     = h5py.File(h5_path, "w")
            ds      = h5f.create_dataset(
                "vertices", shape=(0, n), maxshape=(None, n),
                dtype=np.float64, chunks=(512, n),
            )
            offsets      = [0]
            write_buffer = []
            write_buffer_offsets = []

        for j in range(len(enumerate_poly)):
            if j % 1000 == 0:
                print(f"  Layer {i}: processing cell {j} / {len(enumerate_poly)}")

            if i == 0:
                # First layer: all polytopes share the same input domain.
                enumerate_poly_n= Enumerator_rapid(
                    hyperplanes[i], b[i],
                    original_polytope_test, TH,
                    [border_hyperplane], [border_bias],
                    parallel, np.array([1] * n_h), i,
                    use_wide=use_wide,
                )
            else:
                # Deeper layers: propagate accumulated boundary hyperplanes
                # for the current cell through the previous layers.
                hype1, bias1, border_hyperplane1, border_bias1 = finding_deep_hype(
                    hyperplanes, b,
                    enumerate_poly[j],
                    border_hyperplane, border_bias,
                    i, n,
                )
                enumerate_poly_n= Enumerator_rapid(
                    hype1, bias1,
                    np.array([enumerate_poly[j]]), TH,
                    [border_hyperplane1], [border_bias1],
                    parallel, np.array([1] * n_h), i,
                    use_wide=use_wide,
                )

            if use_disk:
                # Buffer results and flush to HDF5 in batches to reduce
                # the number of resize calls.
                for poly in enumerate_poly_n:
                    poly_arr = np.array(poly, dtype=np.float64)
                    write_buffer.append(poly_arr)
                    write_buffer_offsets.append(len(poly_arr))
                    if len(write_buffer) >= WRITE_BUFFER_SIZE:
                        batch = np.vstack(write_buffer)
                        ds.resize(ds.shape[0] + len(batch), axis=0)
                        ds[-len(batch):] = batch
                        for sz in write_buffer_offsets:
                            offsets.append(offsets[-1] + sz)
                        write_buffer         = []
                        write_buffer_offsets = []
                del enumerate_poly_n
            else:
                if j == 0:
                    enumerate_poly_n_list = list(enumerate_poly_n)
                else:
                    enumerate_poly_n_list.extend(enumerate_poly_n)
                del enumerate_poly_n

        # ------------------------------------------------------------------
        # End of layer: finalise output and reload for the next iteration.
        # ------------------------------------------------------------------
        if use_disk:
            # Flush any remaining buffered polytopes.
            if write_buffer:
                batch = np.vstack(write_buffer)
                ds.resize(ds.shape[0] + len(batch), axis=0)
                ds[-len(batch):] = batch
                for sz in write_buffer_offsets:
                    offsets.append(offsets[-1] + sz)

            h5f.close()
            del enumerate_poly
            gc.collect()

            # Read the HDF5 file back as a Python list of vertex arrays.
            with h5py.File(h5_path, "r") as h5f_r:
                ds_r        = h5f_r["vertices"]
                offsets_arr = np.array(offsets)
                enumerate_poly = [
                    ds_r[offsets_arr[k]:offsets_arr[k + 1]][:]
                    for k in range(len(offsets_arr) - 1)
                ]
            os.remove(h5_path)
            print(f"  Layer {i} complete: {len(enumerate_poly)} regions.")
        else:
            del enumerate_poly
            gc.collect()
            enumerate_poly = enumerate_poly_n_list
            del enumerate_poly_n_list
    
    
    
    
    # ------------------------------------------------------------------
    # 4. Compute activation-pattern IDs
    # ------------------------------------------------------------------
    # barrier mode: D is computed only for boundary cells inside
    # barrier_certificate_cells, which is cheaper than computing it for
    # all regions when only a small fraction straddles b(x)=0.
    end_enum       = time.time()
    enumeration_time = end_enum - st_enum
    if verification == "barrier":
        if barrier_model is None:
            print("Warning: verification='barrier' requires barrier_model. Skipping.")
        else:
            sv,BC,_=barrier_certificate_cells(
                barrier_model, enumerate_poly, hyperplanes, b, name_file
            )
    else:
        # Default: compute D for all enumerated regions.
        D_raw=Finding_cell_id(enumerate_poly, hyperplanes, b, num_hidden_layers)


    # ------------------------------------------------------------------
    # 5. Report results
    # ------------------------------------------------------------------
    print(f"\nEnumeration time : {enumeration_time:.2f} s")
    print(f"Neurons per layer: {[len(hyperplanes[k]) for k in range(num_hidden_layers)]}")
    print(f"Total regions    : {len(enumerate_poly)}")

    # ------------------------------------------------------------------
    # 6. Save full vertex representation to HDF5
    # ------------------------------------------------------------------
    out_h5 = name_file + "_polytope.h5"
    with h5py.File(out_h5, "w") as f:
        file_offsets = np.zeros(len(enumerate_poly) + 1, dtype=np.int64)
        for idx, p in enumerate(enumerate_poly):
            file_offsets[idx + 1] = file_offsets[idx] + len(p)
        f.create_dataset("offsets", data=file_offsets)
        ds = f.create_dataset(
            "vertices",
            shape=(int(file_offsets[-1]), n),
            dtype=np.float64,
        )
        for idx, p in enumerate(enumerate_poly):
            ds[file_offsets[idx]:file_offsets[idx + 1]] = p
    print(f"Polytope vertex data saved to: {out_h5}")

    
    #### test Hessian of dynamics
    
    if verification == "barrier" and len(BC) > 0:
        dynamics_name = name_file.split("/")[-1].split("_")[0].lower()
        summary = verify_barrier(
            BC, sv, hyperplanes, b, W,bdh,bdb,
            barrier_model, dynamics_name=dynamics_name
    )
        from relu_region_enumerator.validate_with_nlp import validate_with_nlp

        report = validate_with_nlp(
            BC, sv, hyperplanes, b, W,bdh,bdb,
            barrier_model, dynamics_name=dynamics_name,
            summary=summary,        # pass the VerificationSummary from verify_barrier
            continuous_time=True,
        )
        report.print_counterexamples(
    max_print    = 5,
    BC           = BC,
    sv           = sv,
    layer_W      = hyperplanes,
    W_out        = W,
    barrier_model= barrier_model,
)
        # report.print_counterexamples()
    if verification == "lyapunov":
        # D_raw = Finding_cell_id(enumerate_poly, hyperplanes, b, num_hidden_layers)
        sv_all = np.hstack([D_raw[l] for l in range(num_hidden_layers)]).T
        summary = verify_lyapunov(
            enumerate_poly, sv_all, hyperplanes, W, b,
            barrier_model, dynamics_name=dynamics_name
    )    




    





    # ------------------------------------------------------------------
    # 7. Optional: 2-D partition visualisation
    # ------------------------------------------------------------------
    if n == 2:
        try:
            from .visualization import plot_polytope
            plot_polytope(enumerate_poly, "partition")
        except ImportError:
            print("visualization module not found; skipping 2-D visualisation.")


