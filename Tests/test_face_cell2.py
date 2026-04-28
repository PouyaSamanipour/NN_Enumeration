"""
test_face_cell2.py
==================
Verify cell 2 using the ZLS face refinement strategy
(verify_certificate_face.py) instead of the full-polytope Enumerator_rapid.

Run from the project root:
    conda run -n relu_enum python Tests/test_face_cell2.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import h5py

from relu_region_enumerator.verify_certificate_face import verify_barrier, Label
from relu_region_enumerator.hessian_bound import compute_local_gradient

# ───────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────

BARRIER_PATH = "NN_files/model_quadrotor_ct_B.pt"
BOUNDARY_H5  = "quadrotor_boundary_cells.h5"
TH           = [1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
DYNAMICS     = "quadrotor"

TARGET_CELL  = 63

# ───────────────────────────────────────────────────────────
# Load model
# ───────────────────────────────────────────────────────────

print("Loading barrier model ...")
model  = torch.jit.load(BARRIER_PATH, map_location="cpu"); model.eval()
params = [p.detach().numpy() for _, p in model.named_parameters()]
layer_W = [params[0], params[2]]
layer_b = [params[1], params[3]]
W_out   = params[4]
n       = layer_W[0].shape[1]
print(f"  Input dim={n},  hidden layers={len(layer_W)},  "
      f"neurons={[W.shape[0] for W in layer_W]}")

boundary_H = np.vstack((np.eye(n), -np.eye(n)))
boundary_b = np.array(TH * 2, dtype=np.float64)

# ───────────────────────────────────────────────────────────
# Load only the target cell
# ───────────────────────────────────────────────────────────

print(f"\nLoading cell {TARGET_CELL} from {BOUNDARY_H5} ...")
with h5py.File(BOUNDARY_H5, "r") as f:
    offsets = f["offsets"][:]
    sv_all  = f["activation_patterns"][:]
    tgt_verts = f["vertices"][offsets[TARGET_CELL]:offsets[TARGET_CELL + 1]][:]
    tgt_sv_nn = sv_all[TARGET_CELL]   # shape (30,) — NN neurons only

# Boundary neuron activations: 1 if cell is strictly inside this face, 0 if touching it.
# Uses min over all vertices so cells that touch a boundary face (some verts at H@x+b=0)
# correctly get activation 0 for that constraint.
boundary_vals = boundary_H @ tgt_verts.T + boundary_b[:, None]  # (24, V)
boundary_sv   = (boundary_vals.min(axis=1) > 1e-8).astype(np.float64)  # (24,)
tgt_sv        = np.concatenate([tgt_sv_nn, boundary_sv])          # (54,)

n_nn_neurons  = len(tgt_sv_nn)
n_bnd_neurons = len(boundary_sv)
total_neurons = n_nn_neurons + n_bnd_neurons                       # 54
use_wide      = (total_neurons > 64)

print(f"  Cell {TARGET_CELL}: {tgt_verts.shape[0]} vertices, "
      f"{tgt_verts.shape[1]}D")
print(f"  Memory: {tgt_verts.nbytes / 1024:.1f} KB")
print(f"  Neurons: {n_nn_neurons} (NN) + {n_bnd_neurons} (boundary) = {total_neurons} total")
print(f"  Boundary activations (tight=0): {boundary_sv.astype(int).tolist()}")
print(f"  use_wide={use_wide}")

# ───────────────────────────────────────────────────────────
# Gradient sanity check
# ───────────────────────────────────────────────────────────

p_i = compute_local_gradient(tgt_sv_nn.ravel(), layer_W, W_out)
print(f"\n  p_i norm at cell {TARGET_CELL}: {np.linalg.norm(p_i):.4f}")

# ───────────────────────────────────────────────────────────
# Run face-based verification
# ───────────────────────────────────────────────────────────

print(f"\nVerifying cell {TARGET_CELL} with ZLS face refinement ...")
summary = verify_barrier(
    BC                   = [tgt_verts.copy()],
    sv                   = tgt_sv[np.newaxis, :],
    layer_W              = layer_W,
    layer_b              = layer_b,
    W_out                = W_out,
    boundary_H           = boundary_H,
    boundary_b           = boundary_b,
    barrier_model        = model,
    dynamics_name        = DYNAMICS,
    continuous_time      = True,
    early_exit           = False,
    refinement_max_depth = 20,
    TH                   = TH,
    n_refs               = 5,
)

# ───────────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────────

assert len(summary.results) == 1, "Expected exactly one result"
result = summary.results[0]

print("\n" + "=" * 55)
print(f"RESULT for cell {TARGET_CELL}  [face refinement]")
print("=" * 55)
print(f"  Label      : {result.label.value}")
print(f"  M_i        : {result.M_i:.4e}")
print(f"  r_i        : {result.r_i:.4e}")
print(f"  remainder  : {result.remainder:.4e}")
print(f"  Runtime    : {summary.runtime_s:.2f} s")
print("=" * 55)

if result.label in (Label.SAFE_TAYLOR, Label.SAFE_REFINEMENT):
    print(f"\nPASS: cell {TARGET_CELL} certified SAFE by face refinement.")
elif result.label == Label.UNSAFE:
    print(f"\nFAIL: cell {TARGET_CELL} has a counterexample.")
    if summary.counterexample:
        summary.counterexample.report()
else:
    print(f"\nINCONCLUSIVE: face refinement exhausted (depth=8, n_refs=4).")
    print(f"  ZLS face has {tgt_verts.shape[0]} full-polytope vertices.")
    print(f"  Consider increasing max_depth or n_refs.")
