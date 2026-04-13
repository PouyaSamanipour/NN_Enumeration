"""
trace_refinement_uncertainty.py
================================
For a handful of INCONCLUSIVE-after-Taylor cells, trace how the Taylor
remainder (uncertainty = 0.5 * M_i * r_i^2) evolves across each refinement
depth and each sub-cell.

Output per cell:
  depth=0  remainder=<starting>  worst_cond=<...>  n_refs=<...>
  depth=1  sub-cell 0/N  remainder=<...>  label=<...>
  depth=1  sub-cell 1/N  remainder=<...>  label=<...>
  ...
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import h5py

from relu_region_enumerator.verify_certificate_new import (
    _two_step_label, _find_edge_zls,
    _get_zero_level_set_crossings,
    DynamicsEvaluator, Label,
)
from relu_region_enumerator.hessian_bound import HessianBounder, compute_local_gradient
from relu_region_enumerator.Dynamics import load_dynamics
from relu_region_enumerator.bitwise_utils import (
    get_cell_hyperplanes_input_space,
    Enumerator_rapid,
)

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "NN_files/model_decay_2_10_ren.pt"
BOUNDARY_H5 = "decay_boundary_cells.h5"
TH          = [2.0] * 6
DYNAMICS    = "decay"
MAX_CELLS    = 1000
TARGET_CELLS = [665, 676, 678, 761, 979]  # INCONCLUSIVE cells to trace
MAX_DEPTH    = 4          # max refinement depth to trace

# ── load model ────────────────────────────────────────────────────────────────
print("Loading model …")
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()
model_dtype = next(model.parameters()).dtype
params   = [p.detach().numpy() for _, p in model.named_parameters()]
layer_W  = [params[0], params[2]]
layer_b  = [params[1], params[3]]
W_out    = params[4]
n        = layer_W[0].shape[1]

boundary_H = np.vstack((np.eye(n), -np.eye(n)))
boundary_b = np.array(TH * 2, dtype=np.float64)
np_dtype   = np.float64 if model_dtype == torch.float64 else np.float32
total_neurons = sum(W.shape[0] for W in layer_W)
USE_WIDE      = (total_neurons + len(boundary_H) > 64)

# ── load boundary cells ───────────────────────────────────────────────────────
print("Loading boundary cells …")
with h5py.File(BOUNDARY_H5, "r") as f:
    offsets = f["offsets"][:]
    verts   = f["vertices"][:]
    sv      = f["activation_patterns"][:]
BC = [verts[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
BC = BC[:MAX_CELLS]
sv = sv[:MAX_CELLS]
print(f"  Using first {len(BC)} cells\n")

# ── build dynamics + Hessian bounder ─────────────────────────────────────────
print("Building dynamics …")
symbols, f_sym = load_dynamics(DYNAMICS)
dyn = DynamicsEvaluator(symbols, f_sym)
hb  = HessianBounder(symbols, f_sym)

# ── find ZLS crossings for every cell ────────────────────────────────────────
def get_zls(cell_verts, sv_i):
    with torch.no_grad():
        B_vals = model(
            torch.tensor(cell_verts.astype(np_dtype), dtype=model_dtype)
        ).numpy().ravel().astype(np.float64)
    if B_vals.min() >= -1e-9 or B_vals.max() <= 1e-9:
        return None, None
    # Use bitmask-based crossing — only actual polytope edges, not all O(V²) pairs
    xs, _ = _get_zero_level_set_crossings(
        cell_verts, sv_i, B_vals,
        layer_W, layer_b, boundary_H, boundary_b,
        USE_WIDE, barrier_model=model, model_dtype=model_dtype,
    )
    return xs, B_vals


# ── recursive tracer ─────────────────────────────────────────────────────────
def trace_cell(cell_idx, vertices, sv_i, x_stars, p_i, depth, max_depth, prefix=""):
    """
    Run one step of the Taylor check, print uncertainty, then recurse into
    sub-cells that remain INCONCLUSIVE.
    """
    if len(x_stars) < 2 or depth > max_depth:
        return

    label, M_i, r_i, remainder, worst_cond, ce = _two_step_label(
        x_stars, model, dyn, hb, p_i, cell_idx, continuous_time=True
    )

    tag = f"depth={depth}"
    print(f"{prefix}{tag:10s}  n_xstars={len(x_stars):4d}  "
          f"M={M_i:.3e}  r={r_i:.3e}  "
          f"remainder={remainder:.4e}  worst_cond={worst_cond:.4e}  "
          f"label={label.value}")

    if label != Label.INCONCLUSIVE or depth == max_depth:
        return

    # ── compute n_refs and cut hyperplanes ───────────────────────────────────
    abs_f_lin = max(abs(worst_cond), 1e-10)
    n_refs    = max(2, int(np.ceil(remainder / abs_f_lin)))
    n_refs    = min(n_refs, 32)

    diff   = x_stars[:, None, :] - x_stars[None, :, :]
    dmat   = np.linalg.norm(diff, axis=2)
    flat   = int(np.argmax(dmat))
    idx_a, idx_b = divmod(flat, len(x_stars))
    max_dist = float(dmat[idx_a, idx_b])

    if max_dist < 1e-12:
        print(f"{prefix}  -> x_stars are degenerate, cannot split")
        return

    normal    = (x_stars[idx_b] - x_stars[idx_a]) / max_dist
    base_proj = float(x_stars[idx_a] @ normal)
    step_proj = float((x_stars[idx_b] - x_stars[idx_a]) @ normal) / n_refs
    cuts      = [base_proj + k * step_proj for k in range(1, n_refs)]

    print(f"{prefix}  -> splitting into {n_refs} sub-cells (n_refs={n_refs})")

    H_refine = np.tile(normal, (n_refs - 1, 1)).astype(np.float64)
    b_refine = np.array([-c for c in cuts], dtype=np.float64)

    H_cell, b_cell = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b
    )
    bh_list = H_cell.tolist()
    bb_list = b_cell.tolist()

    try:
        sub_cells = Enumerator_rapid(
            H_refine, b_refine,
            [np.asarray(vertices, dtype=np.float64)],
            TH, [bh_list], [bb_list],
            False, None, 0, USE_WIDE,
        )
    except Exception as exc:
        print(f"{prefix}  -> Enumerator_rapid failed ({exc}), using slab projection")
        sub_cells = None

    # slab fallback
    if sub_cells is None or len(sub_cells) <= 1:
        proj = x_stars @ normal
        for k in range(n_refs):
            lo = cuts[k - 1] if k > 0         else -np.inf
            hi = cuts[k]     if k < n_refs - 1 else  np.inf
            xs_k = x_stars[(proj >= lo - 1e-9) & (proj <= hi + 1e-9)]
            if len(xs_k) < 2:
                continue
            print(f"{prefix}  slab {k+1}/{n_refs}", end="  ")
            trace_cell(cell_idx, vertices, sv_i, xs_k, p_i,
                       depth + 1, max_depth, prefix + "    ")
        return

    # use Enumerator sub-cells
    valid_subs = []
    for sub_verts in sub_cells:
        sub_verts = np.asarray(sub_verts, dtype=np.float64)
        if len(sub_verts) < n + 1:
            continue
        with torch.no_grad():
            B_sub = model(
                torch.tensor(sub_verts.astype(np_dtype), dtype=model_dtype)
            ).numpy().ravel().astype(np.float64)
        if B_sub.min() >= -1e-9 or B_sub.max() <= 1e-9:
            continue
        xs_k, _ = _get_zero_level_set_crossings(
            sub_verts, sv_i, B_sub,
            layer_W, layer_b, boundary_H, boundary_b,
            USE_WIDE, barrier_model=model, model_dtype=model_dtype,
        )
        if len(xs_k) >= 2:
            valid_subs.append((sub_verts, xs_k))

    print(f"{prefix}  -> {len(valid_subs)} non-trivial sub-cells with ZLS crossings")
    for k, (sub_verts, xs_k) in enumerate(valid_subs):
        print(f"{prefix}  sub-cell {k+1}/{len(valid_subs)}", end="  ")
        trace_cell(cell_idx, sub_verts, sv_i, xs_k, p_i,
                   depth + 1, max_depth, prefix + "    ")


# ── trace the target cells ────────────────────────────────────────────────────
for i in TARGET_CELLS:
    cell_verts = BC[i]
    sv_i = sv[i]
    xs, B_vals = get_zls(cell_verts, sv_i)

    if xs is None or len(xs) < 2:
        print(f"CELL {i}: no ZLS crossings — trivially SAFE")
        continue

    p_i    = compute_local_gradient(sv_i, layer_W, W_out)
    label0, M0, r0, rem0, wc0, _ = _two_step_label(
        xs, model, dyn, hb, p_i, i, continuous_time=True
    )

    print("=" * 70)
    print(f"CELL {i}  n_xstars={len(xs)}  initial: M={M0:.3e}  r={r0:.3e}  "
          f"remainder={rem0:.4e}  |f_lin|={abs(wc0):.4e}  "
          f"ratio={rem0/max(abs(wc0),1e-10):.2f}  label={label0.value}")
    print("=" * 70)
    if label0 == Label.INCONCLUSIVE:
        trace_cell(i, cell_verts, sv_i, xs, p_i, depth=0,
                   max_depth=MAX_DEPTH, prefix="  ")
    print()
