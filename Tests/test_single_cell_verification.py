"""
test_single_cell_verification.py
==================================
Verify a single boundary cell (cell index TARGET_CELL) in isolation.

Only cells [CONTEXT_CELL, TARGET_CELL] are loaded from the HDF5 file —
no other cell data is kept in memory — to confirm the verification is
cell-local (independent of any neighbouring cell).

Run from the project root:
    conda run -n relu_enum python Tests/test_single_cell_verification.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import h5py

from relu_region_enumerator.verify_certificate_new import verify_barrier, Label
from relu_region_enumerator.hessian_bound import compute_local_gradient

# ───────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────

BARRIER_PATH = "NN_files/model_quadrotor_ct_B.pt"
BOUNDARY_H5  = "quadrotor_boundary_cells.h5"
TH           = [1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
DYNAMICS     = "quadrotor"

CONTEXT_CELL = 2   # loaded into memory but NOT verified — simulates limited context
TARGET_CELL  = 1   # the cell we actually want to verify

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
# Load ONLY the two cells we care about — nothing else
# ───────────────────────────────────────────────────────────

print(f"\nLoading cells {CONTEXT_CELL} and {TARGET_CELL} from {BOUNDARY_H5} ...")
with h5py.File(BOUNDARY_H5, "r") as f:
    offsets = f["offsets"][:]
    sv_all  = f["activation_patterns"][:]

    ctx_verts = f["vertices"][offsets[CONTEXT_CELL]:offsets[CONTEXT_CELL + 1]][:]
    tgt_verts = f["vertices"][offsets[TARGET_CELL]:offsets[TARGET_CELL + 1]][:]

    ctx_sv = sv_all[CONTEXT_CELL]
    tgt_sv = sv_all[TARGET_CELL]

print(f"  Cell {CONTEXT_CELL} (context, not verified): {ctx_verts.shape[0]} vertices")
print(f"  Cell {TARGET_CELL} (target, to verify):      {tgt_verts.shape[0]} vertices")
print(f"  Memory used by both cells: "
      f"{(ctx_verts.nbytes + tgt_verts.nbytes) / 1024:.1f} KB")

# ctx_verts / ctx_sv are in scope but never passed to verify_barrier —
# they represent the "only info we have from cell 2."

# ───────────────────────────────────────────────────────────
# Gradient at the target cell (sanity check)
# ───────────────────────────────────────────────────────────

p_i = compute_local_gradient(tgt_sv.ravel(), layer_W, W_out)
print(f"\n  p_i (local gradient at cell {TARGET_CELL}): "
      f"norm={np.linalg.norm(p_i):.4f}")

# ───────────────────────────────────────────────────────────
# Run verify_barrier on ONLY the target cell
# ───────────────────────────────────────────────────────────

total_neurons = sum(W.shape[0] for W in layer_W)
use_wide      = (total_neurons + len(boundary_H) > 64)
print(f"\n  total_neurons={total_neurons}, use_wide={use_wide}")
print(f"  ZLS vertex guard threshold: 500  "
      f"({'WILL trigger' if use_wide and tgt_verts.shape[0] > 500 else 'will NOT trigger'})")

print(f"\nVerifying cell {TARGET_CELL} in isolation (no vertex guards) ...")
summary = verify_barrier(
    BC                  = [tgt_verts.copy()],
    sv                  = tgt_sv[np.newaxis, :],
    layer_W             = layer_W,
    layer_b             = layer_b,
    W_out               = W_out,
    boundary_H          = boundary_H,
    boundary_b          = boundary_b,
    barrier_model       = model,
    dynamics_name       = DYNAMICS,
    continuous_time     = True,
    early_exit          = False,
    refinement_max_depth= 8,
    TH                  = TH,
    max_slicer_verts    = None,   # disable vertex-count guards for this single cell
)

# ───────────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────────

assert len(summary.results) == 1, "Expected exactly one result"
result = summary.results[0]

print("\n" + "=" * 55)
print(f"RESULT for cell {TARGET_CELL}")
print("=" * 55)
print(f"  Label      : {result.label.value}")
print(f"  M_i        : {result.M_i:.4e}")
print(f"  r_i        : {result.r_i:.4e}")
print(f"  remainder  : {result.remainder:.4e}")
print(f"  Runtime    : {summary.runtime_s:.2f} s")
print("=" * 55)

if result.label == Label.INCONCLUSIVE:
    n_verts = tgt_verts.shape[0]
    print(f"\nNOTE: Cell {TARGET_CELL} has {n_verts} vertices.")
    if use_wide and n_verts > 500:
        print("  The ZLS vertex-count guard (use_wide=True, > 500 verts) "
              "skipped the ZLS computation entirely.")
    else:
        print(f"  use_wide={use_wide} — ZLS guard did not trigger.")
        print("  INCONCLUSIVE is from the refinement: sub-cells are growing")
        print("  in vertex count during cuts (12D geometry expands under slicing).")
        print("  The refinement vertex-count guard (> 500) is catching them.")
elif result.label in (Label.SAFE_TAYLOR, Label.SAFE_REFINEMENT):
    print(f"\nPASS: cell {TARGET_CELL} is certified SAFE independently.")
elif result.label == Label.UNSAFE:
    print(f"\nFAIL: cell {TARGET_CELL} has a counterexample.")
    if summary.counterexample:
        summary.counterexample.report()
