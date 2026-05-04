
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

from .bitwise_utils import Enumerator_rapid, finding_deep_hype, generate_mask_wide,slice_polytope_wide
from .bitwise_utils_efficient import Enumerator_rapid_h5
from .Dynamics import load_dynamics, list_systems
from .verify_certificate_face import verify_barrier
from .verify_certificates import verify_lyapunov
from .ibp_filter import precompute_ibp_weights, vertex_ibp_filter, warmup_numba


def _remove_file_with_retry(path, retries=40, delay_s=0.05, strict=False):
    """Remove a file with retries to tolerate transient Windows locks.

    Returns True when removed (or already absent), False when still locked and
    strict=False.
    """
    for attempt in range(retries):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if attempt == retries - 1:
                if strict:
                    raise
                print(f"  [Cleanup] Skipping locked temp file: {path}")
                return False
            time.sleep(delay_s)


def _open_h5_lock_tolerant(path, mode, retries=5, delay_s=0.05, **kwargs):
    """Open HDF5 file with a Windows lock-violation fallback.

    Some Windows setups intermittently fail with:
    "unable to lock file ... Win32 GetLastError() = 33".
    When that happens, retry with HDF5 file-locking disabled for this process.
    """
    try:
        return h5py.File(path, mode, **kwargs)
    except OSError as exc:
        msg = str(exc).lower()
        if os.name != "nt" or "unable to lock file" not in msg:
            raise

    # In write/append modes, clear stale file first if a previous run left one.
    if any(flag in mode for flag in ("w", "a", "x", "+")) and os.path.exists(path):
        _remove_file_with_retry(path, retries=40, delay_s=delay_s, strict=False)

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    last_exc = None
    for _ in range(retries):
        try:
            return h5py.File(path, mode, **kwargs)
        except OSError as exc:
            last_exc = exc
            if "unable to lock file" not in str(exc).lower():
                raise
            time.sleep(delay_s)
    raise last_exc




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
def barrier_certificate_cells(model, enumerate_poly, hyperplanes, b, name_file,
                              test_stat=False, pre_filtered=False):
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

            # Boundary detection — skipped when pre_filtered=True because the
            # vertex_ibp_filter already guaranteed every cell straddles B=0.
            if not pre_filtered:
                P_k = model(torch.tensor(poly_arr, dtype=model_dtype))
                is_boundary = P_k.min().item() < 1e-5 and P_k.max().item() > 1e-5
            else:
                is_boundary = True

            if is_boundary:
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
                         verification=None, barrier_model=None,
                         ibp_filter=True):
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

    # Precompute IBP weights once (barrier network = enumeration network).
    _run_ibp = (ibp_filter and verification == "barrier" and barrier_model is not None)
    if _run_ibp:
        W_ibp_full = list(hyperplanes) + [W]
        b_ibp_full = [bi.ravel() for bi in b] + [c.ravel()]
        W_pos_ibp, W_neg_ibp, b_arr_ibp = precompute_ibp_weights(
            W_ibp_full, b_ibp_full
        )
        warmup_numba(n_in=n, m_hidden=hyperplanes[0].shape[0],
                     n_layers=len(W_ibp_full))

    st_enum = time.time()

    # ── Disk-streaming state ──────────────────────────────────────────────────
    # For the use_disk path, instead of loading all cells into RAM between
    # layers, we keep a pointer to the HDF5 file that holds the current layer's
    # input cells.  The j-loop reads one cell at a time from this file.
    # At i==0 there is only one input cell (the hypercube), so no HDF5 needed.
    _in_h5_path    = None   # path to input HDF5 for current layer (None → RAM)
    _in_offsets    = None   # list of vertex-row offsets inside that file
    _in_N          = 1      # number of input cells (starts at 1: the hypercube)

    FILTER_BATCH   = 5_000  # cells per IBP-filter batch (controls peak RAM)

    saved_to_h5 = False

    for i in range(num_hidden_layers):

        if use_disk:
            # Open a per-layer HDF5 scratch file for streaming output.
            # rdcc_nbytes=0: disable chunk cache so every write goes straight to
            # disk — prevents B-tree corruption when reading the file in the
            # next layer on WSL / networked filesystems.
            h5_path = os.path.join(tmp_dir, f"layer_{i}.h5")
            h5f     = _open_h5_lock_tolerant(h5_path, "w", rdcc_nbytes=0)
            ds      = h5f.create_dataset(
                "vertices", shape=(0, n), maxshape=(None, n),
                dtype=np.float64, chunks=(512, n),
            )
            offsets              = [0]
            write_buffer         = []
            write_buffer_offsets = []

            # Open the input file for this layer (i >= 1 only; i==0 uses RAM).
            if _in_h5_path is not None:
                _h5f_in  = h5py.File(_in_h5_path, "r", rdcc_nbytes=0)
                _ds_in   = _h5f_in["vertices"]
                _offs_in = np.array(_in_offsets)
            else:
                _h5f_in = None

        N_j = _in_N if (use_disk and _in_h5_path is not None) else len(enumerate_poly)

        for j in range(N_j):
            if j % 1000 == 0:
                print(f"  Layer {i}: processing cell {j} / {N_j}")

            _is_last_layer = _run_ibp and (i == num_hidden_layers - 1)
            if i == num_hidden_layers - 1 and not use_disk:
                saved_to_h5 = True
                out_h5_temp = name_file + "_polytope.h5"
                # if i == 0:
                #     offsets, n_regions = Enumerator_rapid_h5(
                #         hyperplanes[i], b[i],
                #         original_polytope_test, TH,
                #         [border_hyperplane], [border_bias],
                #         parallel, np.array([1] * n_h), i,
                #         out_h5_path=out_h5_temp,
                #         use_wide=use_wide,
                #         W_pos_ibp=W_pos_ibp if _run_ibp else None,
                #         W_neg_ibp=W_neg_ibp if _run_ibp else None,
                #         b_arr_ibp=b_arr_ibp if _run_ibp else None,
                #         is_last_layer=_is_last_layer,
                #     )
                # else:
                #     if use_disk and _h5f_in is not None:
                #         cell_j = _ds_in[_offs_in[j]:_offs_in[j + 1]][:]
                #     else:
                #         cell_j = enumerate_poly[j]
                #     hype1, bias1, border_hyperplane1, border_bias1 = finding_deep_hype(
                #         hyperplanes, b,
                #         cell_j,
                #         border_hyperplane, border_bias,
                #         i, n,
                #     )
                #     offsets, n_regions = Enumerator_rapid_h5(
                #         hype1, bias1,
                #         np.array([cell_j]), TH,
                #         [border_hyperplane1], [border_bias1],
                #         False,  # serial
                #         np.array([1] * n_h), i,
                #         out_h5_path=out_h5_temp,
                #         use_wide=use_wide,
                #         W_pos_ibp=W_pos_ibp if _run_ibp else None,
                #         W_neg_ibp=W_neg_ibp if _run_ibp else None,
                #         b_arr_ibp=b_arr_ibp if _run_ibp else None,
                #         is_last_layer=_is_last_layer,
                #     )
                # continue
            if i == 0:
                # First layer: all polytopes share the same input domain.
                enumerate_poly_n = Enumerator_rapid(
                    hyperplanes[i], b[i],
                    original_polytope_test, TH,
                    [border_hyperplane], [border_bias],
                    parallel, np.array([1] * n_h), i,
                    use_wide=use_wide,
                    W_pos_ibp=W_pos_ibp if _run_ibp else None,
                    W_neg_ibp=W_neg_ibp if _run_ibp else None,
                    b_arr_ibp=b_arr_ibp if _run_ibp else None,
                    is_last_layer=_is_last_layer,
                )
            else:
                # Deeper layers: read cell j from HDF5 (disk path) or RAM.
                if use_disk and _h5f_in is not None:
                    cell_j = _ds_in[_offs_in[j]:_offs_in[j + 1]][:]
                else:
                    cell_j = enumerate_poly[j]

                hype1, bias1, border_hyperplane1, border_bias1 = finding_deep_hype(
                    hyperplanes, b,
                    cell_j,
                    border_hyperplane, border_bias,
                    i, n,
                )
                enumerate_poly_n = Enumerator_rapid(
                    hype1, bias1,
                    np.array([cell_j]), TH,
                    [border_hyperplane1], [border_bias1],
                    False,  # serial: single polytope → parallel overhead > benefit
                    np.array([1] * n_h), i,
                    use_wide=use_wide,
                    W_pos_ibp=W_pos_ibp if _run_ibp else None,
                    W_neg_ibp=W_neg_ibp if _run_ibp else None,
                    b_arr_ibp=b_arr_ibp if _run_ibp else None,
                    is_last_layer=_is_last_layer,
                )

            if use_disk:
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
        # End of layer: finalise output, apply IBP filter, update disk state.
        # ------------------------------------------------------------------
        _is_last = _run_ibp and (i == num_hidden_layers - 1)

        if use_disk:
            # Close the input HDF5 (no longer needed) and delete it.
            if _h5f_in is not None:
                _h5f_in.close()
                _h5f_in = None
            if _in_h5_path is not None and os.path.exists(_in_h5_path):
                _remove_file_with_retry(_in_h5_path)
                _in_h5_path = None

            # Flush any remaining buffered output polytopes.
            if write_buffer:
                batch = np.vstack(write_buffer)
                ds.resize(ds.shape[0] + len(batch), axis=0)
                ds[-len(batch):] = batch
                for sz in write_buffer_offsets:
                    offsets.append(offsets[-1] + sz)
            h5f.flush()   # ensure all data reaches the OS buffer before close
            h5f.close()
            # On WSL / networked filesystems the OS buffer may not be synced yet.
            # fsync the file descriptor so the next h5py.File(...,"r") sees valid data.
            try:
                _sync_fd = os.open(h5_path, os.O_RDONLY)
                os.fsync(_sync_fd)
                os.close(_sync_fd)
            except OSError:
                pass
            del enumerate_poly
            gc.collect()

            offsets_arr = np.array(offsets)
            N_layer     = len(offsets_arr) - 1
            print(f"  Layer {i} complete: {N_layer} regions.")

            if _run_ibp:
                # ── Streaming IBP filter → write survivors to filtered_i.h5 ──
                # Reads layer_i.h5 in FILTER_BATCH-sized chunks; each batch is
                # filtered and its survivors are immediately written to the new
                # HDF5 file.  Peak RAM ≈ FILTER_BATCH × avg_verts × n × 8 B.
                filtered_h5_path    = os.path.join(tmp_dir, f"filtered_{i}.h5")
                filtered_offsets    = [0]
                n_kept_total        = 0
                t0_f                = time.time()

                with _open_h5_lock_tolerant(h5_path, "r", rdcc_nbytes=0) as h5f_r, \
                     _open_h5_lock_tolerant(filtered_h5_path, "w", rdcc_nbytes=0) as h5f_w:
                    ds_r  = h5f_r["vertices"]
                    ds_w  = h5f_w.create_dataset(
                        "vertices", shape=(0, n), maxshape=(None, n),
                        dtype=np.float64, chunks=(512, n),
                    )
                    fbuf      = []
                    fbuf_szs  = []

                    for b_start in range(0, N_layer, FILTER_BATCH):
                        b_end       = min(b_start + FILTER_BATCH, N_layer)
                        batch_cells = [
                            ds_r[offsets_arr[k]:offsets_arr[k + 1]][:]
                            for k in range(b_start, b_end)
                        ]
                        filtered_batch, _ = vertex_ibp_filter(
                            batch_cells,
                            W_pos_ibp, W_neg_ibp, b_arr_ibp,
                            is_last_layer=_is_last,
                            n_known=i + 1,
                            verbose=False,
                        )
                        for cell in filtered_batch:
                            fbuf.append(cell)
                            fbuf_szs.append(len(cell))
                            n_kept_total += 1
                            if len(fbuf) >= WRITE_BUFFER_SIZE:
                                blk = np.vstack(fbuf)
                                ds_w.resize(ds_w.shape[0] + len(blk), axis=0)
                                ds_w[-len(blk):] = blk
                                for sz in fbuf_szs:
                                    filtered_offsets.append(filtered_offsets[-1] + sz)
                                fbuf     = []
                                fbuf_szs = []
                        del batch_cells, filtered_batch

                    # Flush remainder.
                    if fbuf:
                        blk = np.vstack(fbuf)
                        ds_w.resize(ds_w.shape[0] + len(blk), axis=0)
                        ds_w[-len(blk):] = blk
                        for sz in fbuf_szs:
                            filtered_offsets.append(filtered_offsets[-1] + sz)

                # Ensure dataset/file proxy objects are released before delete on Windows.
                del ds_r, ds_w
                gc.collect()

                _remove_file_with_retry(h5_path)
                n_pruned = N_layer - n_kept_total
                print(f"  [Filter] Layer {i}: {N_layer} → {n_kept_total} "
                      f"({n_pruned} pruned, {100.0*n_pruned/N_layer:.1f}%) "
                      f"in {time.time()-t0_f:.3f}s")
                if _is_last:
                    print(f"  [Filter] Layer {i} (last): vertex check exact.")

                # Point the next layer's input at the filtered HDF5.
                _in_h5_path = filtered_h5_path
                _in_offsets = filtered_offsets
                _in_N       = n_kept_total

            else:
                # No filter: the raw layer output becomes the next input.
                _in_h5_path = h5_path        # keep the file; do NOT delete yet
                _in_offsets = list(offsets)
                _in_N       = N_layer

            # enumerate_poly is no longer used in the disk path after i==0;
            # set it to a tiny sentinel so the first-layer branch still works.
            enumerate_poly = []

        else:
            del enumerate_poly
            gc.collect()
            enumerate_poly = enumerate_poly_n_list
            del enumerate_poly_n_list

            # ── In-memory IBP filter (low-dim path, n <= 5) ───────────────
            if _run_ibp:
                cells_before = enumerate_poly
                t0_f = time.time()
                enumerate_poly, _ = vertex_ibp_filter(
                    cells_before,
                    W_pos_ibp, W_neg_ibp, b_arr_ibp,
                    is_last_layer=_is_last,
                    n_known=i + 1,
                    verbose=True,
                )
                n_before = len(cells_before)
                n_pruned = n_before - len(enumerate_poly)
                print(f"  [Filter] Layer {i}: {n_before} → {len(enumerate_poly)} "
                      f"({n_pruned} pruned) in {time.time()-t0_f:.3f}s")
                if _is_last:
                    print(f"  [Filter] Layer {i} (last): vertex check exact.")
                del cells_before

    # For the disk path, load the final surviving polytopes back into memory
    # (the last filtered_*.h5 holds the output of the IBP filter on the last layer).
    # Then delete the scratch file.
    if use_disk:
        if _in_h5_path is not None and os.path.exists(_in_h5_path):
            offs_final = np.array(_in_offsets)
            with _open_h5_lock_tolerant(_in_h5_path, "r") as hf:
                ds_final = hf["vertices"]
                enumerate_poly = [
                    ds_final[offs_final[k]:offs_final[k + 1]][:]
                    for k in range(_in_N)
                ]
            _remove_file_with_retry(_in_h5_path)
            _in_h5_path = None
        else:
            # File already gone or never written (0 survivors).
            enumerate_poly = []
    elif saved_to_h5:
        with _open_h5_lock_tolerant(out_h5_temp, "r") as hf:
            ds = hf["vertices"]
            offsets_arr = hf["offsets"][:]
            enumerate_poly = [ds[offsets_arr[k]:offsets_arr[k+1]][:] for k in range(len(offsets_arr)-1)]
    
    
    
    
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
                barrier_model, enumerate_poly, hyperplanes, b, name_file,
                pre_filtered=_run_ibp,
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
    if not saved_to_h5:
        out_h5 = name_file + "_polytope.h5"
        with _open_h5_lock_tolerant(out_h5, "w") as f:
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

    # Free enumerate_poly before verification — BC is a subset of it and
    # both would otherwise live in memory simultaneously during verify_barrier.
    del enumerate_poly
    gc.collect()

    if verification == "barrier" and len(BC) > 0:
        dynamics_name = name_file.split("/")[-1].split("_")[0].lower()
        summary = verify_barrier(
            BC, sv, hyperplanes, b, W, bdh, bdb,
            barrier_model, dynamics_name=dynamics_name,
            TH=TH,
        )
        from relu_region_enumerator.validate_with_nlp import validate_with_nlp


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


