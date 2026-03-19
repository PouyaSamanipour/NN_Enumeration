# """
# core.py
# =======

# Top-level enumeration pipeline for ReLU neural network linear regions.

# The entry point :func:`enumeration_function` loads a TorchScript model,
# extracts its weight matrices, and runs a layer-by-layer vertex enumeration
# over the axis-aligned input hypercube defined by ``TH``.

# For each hidden layer, the algorithm iterates over the already-enumerated
# polytopes from the previous layer and applies :func:`Enumerator_rapid` (from
# :mod:`relu_region_enumerator.bitwise_utils`) to split each polytope along the
# current layer's ReLU hyperplanes.  High-dimensional problems (n > 5) stream
# intermediate results through HDF5 files to stay within RAM budgets.

# The final cell-vertex representation is written to ``<name_file>_polytope.h5``
# and the activation-pattern IDs to a returned array.

# Functions
# ---------
# enumeration_function  -- Full enumeration pipeline.
# generate_hypercube    -- Generate all vertices of an axis-aligned hypercube.
# Finding_cell_id       -- Compute per-layer activation patterns for enumerated cells.
# """
# import pickle
# import gc
# import itertools
# import os
# import time

# import h5py
# import numpy as np
# import torch

# from .bitwise_utils import Enumerator_rapid, finding_deep_hype


# # ---------------------------------------------------------------------------
# # Public helper: hypercube vertex generation
# # ---------------------------------------------------------------------------

# def generate_hypercube(bounds):
#     """Generate all 2^d vertices of the axis-aligned input hypercube.

#     For each dimension with bound b, the two extreme values are -b and +b.

#     Parameters
#     ----------
#     bounds : list of float -- per-dimension upper bound (length = input dim d).

#     Returns
#     -------
#     list of tuples -- all 2^d vertices, each a d-tuple of floats.

#     Example
#     -------
#     >>> generate_hypercube([1.0, 2.0])
#     [(-1.0, -2.0), (-1.0, 2.0), (1.0, -2.0), (1.0, 2.0)]
#     """
#     return list(itertools.product(*[(-b, b) for b in bounds]))


# # ---------------------------------------------------------------------------
# # Public helper: activation-pattern identification
# # ---------------------------------------------------------------------------

# def Finding_cell_id(enumerate_poly, hyperplanes, bias, num_hidden_layers, batch_size=1000):
#     """Compute the activation-pattern matrix for all enumerated linear regions.

#     Each region is represented by its centroid (mean of vertices).  The centroid
#     is forward-propagated through each hidden layer to determine which neurons
#     are active (pre-activation value > 0).

#     Parameters
#     ----------
#     enumerate_poly    : list of (V_k, n) arrays -- enumerated polytope vertices.
#     hyperplanes       : list of (H_i, n) arrays -- weight matrices per hidden layer.
#     bias              : list of (H_i,)   arrays -- bias vectors per hidden layer.
#     num_hidden_layers : int  -- number of hidden layers.
#     batch_size        : int  -- number of centroids processed per batch (default 1000).

#     Returns
#     -------
#     D_raw : list of (H_i, N) float64 arrays -- binary activation patterns,
#             one array per layer.  D_raw[i][h, j] = 1 if neuron h in layer i
#             is active for region j.
#     """
#     N      = len(enumerate_poly)
#     n_dims = len(hyperplanes[0][0])

#     # Compute centroids in one pass to avoid repeated list traversal.
#     Mid_points = np.zeros((N, n_dims))
#     for idx, poly in enumerate(enumerate_poly):
#         Mid_points[idx] = np.sum(poly, axis=0) / len(poly)

#     D_raw = [
#         np.zeros((len(hyperplanes[i]), N), dtype=np.float64)
#         for i in range(int(num_hidden_layers))
#     ]

#     for batch_start in range(0, N, batch_size):
#         batch_end = min(batch_start + batch_size, N)
#         # Shape: (n_dims, batch_size) for efficient matrix-vector products.
#         points = Mid_points[batch_start:batch_end].T

#         for i in range(int(num_hidden_layers)):
#             z = np.dot(hyperplanes[i], points) + bias[i].reshape(-1, 1)
#             # Forward through ReLU: active neurons propagate to the next layer.
#             points = np.maximum(0, z)
#             D_raw[i][:, batch_start:batch_end] = (z > 0).astype(np.float64)

#     return D_raw


# # ---------------------------------------------------------------------------
# # Main enumeration pipeline
# # ---------------------------------------------------------------------------

# def enumeration_function(NN_file, name_file, TH, mode, parallel):
#     """Enumerate all polytopic linear regions of a ReLU network over a hypercube.

#     The function loads a TorchScript-saved ReLU network, extracts its weight
#     matrices and biases, then recursively splits the input hypercube defined
#     by ``TH`` along every ReLU neuron hyperplane, layer by layer.

#     High-dimensional mode (n > 5)
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     To stay within RAM, intermediate per-layer polytope lists are written to
#     HDF5 files in a ``layer_tmp/`` subdirectory and read back before processing
#     the next layer.  The temporary files are deleted immediately after loading.

#     Output
#     ~~~~~~
#     - ``<name_file>_polytope.h5`` -- HDF5 file containing the full vertex
#       representation of all enumerated regions (datasets: ``offsets``,
#       ``vertices``).
#     - Console output: enumeration time, region count, per-layer neuron counts.

#     For 2-D inputs, a partition figure is also displayed via ``plot_polytope``.

#     Parameters
#     ----------
#     NN_file   : str  -- path to a TorchScript (.pt) model file.
#     name_file : str  -- base name for output files (no extension).
#     TH        : list of float -- per-dimension domain half-width (length = n).
#     mode      : str  -- ``"Rapid_mode"`` (default) or ``"Low_Ram"`` (CSV-based,
#                         not recommended; requires additional ``utils_CSV`` module).
#     parallel  : bool -- passed through to :func:`Enumerator_rapid`; reserved
#                         for future parallel dispatching.

#     Returns
#     -------
#     None.  Results are written to disk and printed to stdout.
#     """
#     # ------------------------------------------------------------------
#     # 1. Load model and extract parameters
#     # ------------------------------------------------------------------
#     model  = torch.jit.load(NN_file, map_location=torch.device("cpu"))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     verification="barrier"
#     params = []
#     for _name, param in model.named_parameters():
#         with torch.no_grad():
#             p = param.cpu().numpy() if device.type == "cuda" else param.numpy()
#             params.append(p)

#     # The model has one (weight, bias) pair per hidden layer plus one output layer.
#     # Total parameter tensors = 2 * (num_hidden_layers + 1).
#     # Rearranged: num_hidden_layers = (total_tensors - 4) / 2 + 1
#     num_hidden_layers = int(((len(params) - 4) / 2) + 1)
#     print(f"Number of hidden layers detected: {num_hidden_layers}")

#     # Separate hidden-layer weights/biases from the output layer.
#     hyperplanes = []   # Weight matrices for hidden layers: list of (H_i, n) arrays.
#     b           = []   # Bias vectors for hidden layers:    list of (H_i,) arrays.
#     nn_sizes    = []   # Number of neurons per hidden layer.

#     for i in range(len(params) - 2):
#         if i % 2 == 0:
#             hyperplanes.append(params[i])
#             nn_sizes.append(np.shape(params[i])[0])
#         else:
#             b.append(params[i])

#     W = params[-2]   # Output layer weight matrix (unused in enumeration).
#     c = params[-1]   # Output layer bias          (unused in enumeration).

#     # ------------------------------------------------------------------
#     # 2. Initialise input hypercube and boundary hyperplane set
#     # ------------------------------------------------------------------
#     n_h, n = np.shape(hyperplanes[0])

#     # The initial polytope is the single input hypercube, represented as a
#     # list containing one vertex array of shape (2^n, n).
#     original_polytope_test = np.array([generate_hypercube(TH)])
#     enumerate_poly         = list(original_polytope_test)

#     # Boundary hyperplanes encode the axis-aligned faces of the input domain.
#     # Stacked as [ I_n ; -I_n ] with bias TH[i] for each face, so that
#     # boundary_hyperplane @ x + border_bias <= 0 defines the domain interior.
#     border_hyperplane = np.vstack((np.eye(n), -np.eye(n)))
#     border_bias       = list(TH) + list(TH)   # [TH[0],...,TH[n-1], TH[0],...,TH[n-1]]

#     # ------------------------------------------------------------------
#     # 3. Layer-by-layer enumeration
#     # ------------------------------------------------------------------
#     use_disk = n > 5
#     if use_disk:
#         print(f"High-dimensional mode (n={n}): streaming layer output via HDF5.")

#     cwd     = os.getcwd()
#     tmp_dir = os.path.join(cwd, "layer_tmp")
#     os.makedirs(tmp_dir, exist_ok=True)

#     WRITE_BUFFER_SIZE = 5000   # Polytopes buffered before a single HDF5 flush.

#     st_enum = time.time()

#     for i in range(num_hidden_layers):

#         if use_disk:
#             # Open a per-layer HDF5 scratch file for streaming output.
#             h5_path = os.path.join(tmp_dir, f"layer_{i}.h5")
#             h5f     = h5py.File(h5_path, "w")
#             ds      = h5f.create_dataset(
#                 "vertices", shape=(0, n), maxshape=(None, n),
#                 dtype=np.float64, chunks=(512, n),
#             )
#             offsets      = [0]
#             write_buffer = []
#             write_buffer_offsets = []

#         for j in range(len(enumerate_poly)):
#             if j % 1000 == 0:
#                 print(f"  Layer {i}: processing cell {j} / {len(enumerate_poly)}")

#             if i == 0:
#                 # First layer: all polytopes share the same input domain.
#                 enumerate_poly_n = Enumerator_rapid(
#                     hyperplanes[i], b[i],
#                     original_polytope_test, TH,
#                     [border_hyperplane], [border_bias],
#                     parallel, np.array([1] * n_h), i,
#                 )
#             else:
#                 # Deeper layers: propagate accumulated boundary hyperplanes
#                 # for the current cell through the previous layers.
#                 hype1, bias1, border_hyperplane1, border_bias1 = finding_deep_hype(
#                     hyperplanes, b,
#                     enumerate_poly[j],
#                     border_hyperplane, border_bias,
#                     i, n,
#                 )
#                 enumerate_poly_n = Enumerator_rapid(
#                     hype1, bias1,
#                     np.array([enumerate_poly[j]]), TH,
#                     [border_hyperplane1], [border_bias1],
#                     parallel, np.array([1] * n_h), i,
#                 )

#             if use_disk:
#                 # Buffer results and flush to HDF5 in batches to reduce
#                 # the number of resize calls.
#                 for poly in enumerate_poly_n:
#                     poly_arr = np.array(poly, dtype=np.float64)
#                     write_buffer.append(poly_arr)
#                     write_buffer_offsets.append(len(poly_arr))
#                     if len(write_buffer) >= WRITE_BUFFER_SIZE:
#                         batch = np.vstack(write_buffer)
#                         ds.resize(ds.shape[0] + len(batch), axis=0)
#                         ds[-len(batch):] = batch
#                         for sz in write_buffer_offsets:
#                             offsets.append(offsets[-1] + sz)
#                         write_buffer         = []
#                         write_buffer_offsets = []
#                 del enumerate_poly_n
#             else:
#                 if j == 0:
#                     enumerate_poly_n_list = list(enumerate_poly_n)
#                 else:
#                     enumerate_poly_n_list.extend(enumerate_poly_n)
#                 del enumerate_poly_n

#         # ------------------------------------------------------------------
#         # End of layer: finalise output and reload for the next iteration.
#         # ------------------------------------------------------------------
#         if use_disk:
#             # Flush any remaining buffered polytopes.
#             if write_buffer:
#                 batch = np.vstack(write_buffer)
#                 ds.resize(ds.shape[0] + len(batch), axis=0)
#                 ds[-len(batch):] = batch
#                 for sz in write_buffer_offsets:
#                     offsets.append(offsets[-1] + sz)

#             h5f.close()
#             del enumerate_poly
#             gc.collect()

#             # Read the HDF5 file back as a Python list of vertex arrays.
#             with h5py.File(h5_path, "r") as h5f_r:
#                 ds_r        = h5f_r["vertices"]
#                 offsets_arr = np.array(offsets)
#                 enumerate_poly = [
#                     ds_r[offsets_arr[k]:offsets_arr[k + 1]][:]
#                     for k in range(len(offsets_arr) - 1)
#                 ]
#             os.remove(h5_path)
#             print(f"  Layer {i} complete: {len(enumerate_poly)} regions.")
#         else:
#             del enumerate_poly
#             gc.collect()
#             enumerate_poly = enumerate_poly_n_list
#             del enumerate_poly_n_list
    
#     end_enum       = time.time()
#     enumeration_time = end_enum - st_enum

#     # ------------------------------------------------------------------
#     # 4. Report results
#     # ------------------------------------------------------------------
#     print(f"\nEnumeration time : {enumeration_time:.2f} s")
#     print(f"Neurons per layer: {[len(hyperplanes[k]) for k in range(num_hidden_layers)]}")
#     print(f"Total regions    : {len(enumerate_poly)}")

#     # ------------------------------------------------------------------
#     # 5. Save full vertex representation to HDF5
#     # ------------------------------------------------------------------
#     out_h5 = name_file + "_polytope.h5"
#     with h5py.File(out_h5, "w") as f:
#         file_offsets = np.zeros(len(enumerate_poly) + 1, dtype=np.int64)
#         for idx, p in enumerate(enumerate_poly):
#             file_offsets[idx + 1] = file_offsets[idx] + len(p)
#         f.create_dataset("offsets", data=file_offsets)
#         ds = f.create_dataset(
#             "vertices",
#             shape=(int(file_offsets[-1]), n),
#             dtype=np.float64,
#         )
#         for idx, p in enumerate(enumerate_poly):
#             ds[file_offsets[idx]:file_offsets[idx + 1]] = p
#     print(f"Polytope vertex data saved to: {out_h5}")

#     # ------------------------------------------------------------------
#     # 6. Compute activation-pattern IDs
#     # ------------------------------------------------------------------
#     D_raw= Finding_cell_id(enumerate_poly, hyperplanes, b, num_hidden_layers)

                           
#     # ------------------------------------------------------------------
#     # 7. Optional: 2-D partition visualisation
#     # ------------------------------------------------------------------
#     if n == 2:
#         try:
#             from .visualization import plot_polytope
#             plot_polytope(enumerate_poly, "partition")
#         except ImportError:
#             print("visualization module not found; skipping 2-D visualisation.")


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

import h5py
import numpy as np
import torch

from .bitwise_utils import Enumerator_rapid,finding_deep_hype


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

def Finding_cell_id(enumerate_poly, hyperplanes, bias, num_hidden_layers, batch_size=1000):
    """Compute the activation-pattern matrix for all enumerated linear regions.

    Each region is represented by its centroid (mean of vertices).  The centroid
    is forward-propagated through each hidden layer to determine which neurons
    are active (pre-activation value > 0).

    Parameters
    ----------
    enumerate_poly    : list of (V_k, n) arrays -- enumerated polytope vertices.
    hyperplanes       : list of (H_i, n) arrays -- weight matrices per hidden layer.
    bias              : list of (H_i,)   arrays -- bias vectors per hidden layer.
    num_hidden_layers : int  -- number of hidden layers.
    batch_size        : int  -- number of centroids processed per batch (default 1000).

    Returns
    -------
    D_raw : list of (H_i, N) float64 arrays -- binary activation patterns,
            one array per layer.  D_raw[i][h, j] = 1 if neuron h in layer i
            is active for region j.
    """
    N      = len(enumerate_poly)
    n_dims = len(hyperplanes[0][0])

    # Compute centroids in one pass to avoid repeated list traversal.
    Mid_points = np.zeros((N, n_dims))
    for idx, poly in enumerate(enumerate_poly):
        Mid_points[idx] = np.sum(poly, axis=0) / len(poly)

    D_raw = [
        np.zeros((len(hyperplanes[i]), N), dtype=np.float64)
        for i in range(int(num_hidden_layers))
    ]

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        # Shape: (n_dims, batch_size) for efficient matrix-vector products.
        points = Mid_points[batch_start:batch_end].T

        for i in range(int(num_hidden_layers)):
            z = np.dot(hyperplanes[i], points) + bias[i].reshape(-1, 1)
            # Forward through ReLU: active neurons propagate to the next layer.
            points = np.maximum(0, z)
            D_raw[i][:, batch_start:batch_end] = (z > 0).astype(np.float64)

    return D_raw



# ---------------------------------------------------------------------------
# Barrier certificate verification
# ---------------------------------------------------------------------------

def barrier_certificate_cells(model, enumerate_poly, hyperplanes, b, name_file):
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

    Parameters
    ----------
    model          : torch.nn.Module -- TorchScript barrier certificate network.
    enumerate_poly : list of (V_k, n) arrays -- enumerated polytope vertices.
    hyperplanes    : list of (H_i, n) arrays -- hidden layer weight matrices.
    b              : list of (H_i,)   arrays -- hidden layer biases.
    name_file      : str -- base name for output files.

    Returns
    -------
    D              : (N_b, total_neurons) float64 array -- binary activation
                     patterns for boundary cells, one row per cell.
    boundary_cells : list of (V_k, n) arrays -- vertex sets of boundary cells.
    """
    import pickle
    import torch

    total_neurons = sum(len(h) for h in hyperplanes)
    N             = len(enumerate_poly)

    model.eval()
    boundary_cells  = []
    activation_rows = []

    with torch.no_grad():
        for poly in enumerate_poly:
            poly_arr = np.asarray(poly, dtype=np.float64)

            # Single forward pass over all vertices of this region.
            P_k = model(torch.tensor(poly_arr, dtype=torch.float64))

            if P_k.min().item() < 0 and P_k.max().item() > 0:
                boundary_cells.append(poly_arr)

                # Activation pattern from centroid.
                state = poly_arr.mean(axis=0)
                row   = []
                for h, bias_i in zip(hyperplanes, b):
                    z     = h @ state + bias_i
                    state = np.maximum(0.0, z)
                    row.append((z > 0).astype(np.float64))
                activation_rows.append(np.concatenate(row))

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
            f.create_dataset("activation_patterns", data=np.stack(activation_rows))
    print(f"Boundary cells saved to: {out_h5}")

    D = (np.stack(activation_rows) if activation_rows
         else np.zeros((0, total_neurons), dtype=np.float64))
    D=D.unique(axis=0)  # Remove duplicates if any.
    return D, boundary_cells


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

    # The initial polytope is the single input hypercube, represented as a
    # list containing one vertex array of shape (2^n, n).
    original_polytope_test = np.array([generate_hypercube(TH)])
    enumerate_poly         = list(original_polytope_test)

    # Boundary hyperplanes encode the axis-aligned faces of the input domain.
    # Stacked as [ I_n ; -I_n ] with bias TH[i] for each face, so that
    # boundary_hyperplane @ x + border_bias <= 0 defines the domain interior.
    border_hyperplane = np.vstack((np.eye(n), -np.eye(n)))
    border_bias       = list(TH) + list(TH)   # [TH[0],...,TH[n-1], TH[0],...,TH[n-1]]

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
                enumerate_poly_n = Enumerator_rapid(
                    hyperplanes[i], b[i],
                    original_polytope_test, TH,
                    [border_hyperplane], [border_bias],
                    parallel, np.array([1] * n_h), i,
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
                enumerate_poly_n = Enumerator_rapid(
                    hype1, bias1,
                    np.array([enumerate_poly[j]]), TH,
                    [border_hyperplane1], [border_bias1],
                    parallel, np.array([1] * n_h), i,
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
    if verification == "barrier":
        if barrier_model is None:
            print("Warning: verification='barrier' requires barrier_model. Skipping.")
        else:
            barrier_certificate_cells(
                barrier_model, enumerate_poly, hyperplanes, b, name_file
            )
    else:
        # Default: compute D for all enumerated regions.
        Finding_cell_id(enumerate_poly, hyperplanes, b, num_hidden_layers)
    end_enum       = time.time()
    enumeration_time = end_enum - st_enum

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

    


    # ------------------------------------------------------------------
    # 7. Optional: 2-D partition visualisation
    # ------------------------------------------------------------------
    if n == 2:
        try:
            from .visualization import plot_polytope
            plot_polytope(enumerate_poly, "partition")
        except ImportError:
            print("visualization module not found; skipping 2-D visualisation.")