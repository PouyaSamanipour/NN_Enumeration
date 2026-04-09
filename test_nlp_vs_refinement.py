"""
test_nlp_vs_refinement.py
=========================
Validates the adaptive refinement against the NLP fallback on the decay
6D boundary cells.

Checks:
  1. No label conflict: a cell certified SAFE by one method must not be
     UNSAFE by the other (and vice versa).
  2. Counterexample consistency: both methods must find a counterexample
     at the same cell index.
  3. Resolution rate: how many INCONCLUSIVE-after-Taylor cells each method
     closes.
  4. Runtime comparison.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import h5py
import time

from relu_region_enumerator.verify_certificates    import verify_barrier as verify_nlp
from relu_region_enumerator.verify_certificate_new import verify_barrier as verify_ref
from relu_region_enumerator.verify_certificates    import Label as LabelNLP
from relu_region_enumerator.verify_certificate_new import Label as LabelRef

# ── load model ────────────────────────────────────────────────────────────────
MODEL_PATH  = "NN_files/model_decay_2_10_ren.pt"
BOUNDARY_H5 = "decay_boundary_cells.h5"
TH          = [2.0] * 6
DYNAMICS    = "decay"

print("Loading model …")
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()
params  = [p.detach().numpy() for _, p in model.named_parameters()]
layer_W = [params[0], params[2]]
layer_b = [params[1], params[3]]
W_out   = params[4]
n = layer_W[0].shape[1]
boundary_H = np.vstack((np.eye(n), -np.eye(n)))
boundary_b = np.array(TH * 2, dtype=np.float64)

print("Loading boundary cells …")
with h5py.File(BOUNDARY_H5, "r") as f:
    offsets = f["offsets"][:]
    verts   = f["vertices"][:]
    sv      = f["activation_patterns"][:]
BC = [verts[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
print(f"  {len(BC)} boundary cells, {n}D, {sv.shape[1]} neurons")

MAX_CELLS = 1000
BC = BC[:MAX_CELLS]
sv = sv[:MAX_CELLS]
print(f"  Limiting to first {len(BC)} cells for testing\n")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Run NLP fallback
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("RUN 1 — NLP fallback  (verify_certificates.py)")
print("="*60)
summary_nlp = verify_nlp(
    BC, sv, layer_W, layer_b, W_out,
    boundary_H, boundary_b, model,
    dynamics_name=DYNAMICS,
    continuous_time=True,
    early_exit=False,
    nlp_fallback=True,
    nlp_n_starts=5,
)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Run adaptive refinement
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RUN 2 — Adaptive refinement  (verify_certificate_new.py)")
print("="*60)
summary_ref = verify_ref(
    BC, sv, layer_W, layer_b, W_out,
    boundary_H, boundary_b, model,
    dynamics_name=DYNAMICS,
    continuous_time=True,
    early_exit=False,
    refinement_max_depth=3,
    TH=TH,
)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Cell-by-cell comparison
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CELL-BY-CELL COMPARISON")
print("="*60)

nlp_map = {r.cell_idx: r.label for r in summary_nlp.results}
ref_map = {r.cell_idx: r.label for r in summary_ref.results}

# Collapse labels to SAFE / UNSAFE / INCONCLUSIVE for cross-method comparison
def coarse(label):
    v = label.value
    if "SAFE" in v:   return "SAFE"
    if "UNSAFE" in v: return "UNSAFE"
    return "INCONCLUSIVE"

conflicts      = []   # SAFE vs UNSAFE — serious
disagreements  = []   # one resolves, other doesn't — interesting
agreements     = []   # both same coarse label

for idx in sorted(set(nlp_map) | set(ref_map)):
    ln = nlp_map.get(idx)
    lr = ref_map.get(idx)
    if ln is None or lr is None:
        continue
    cn, cr = coarse(ln), coarse(lr)
    if (cn == "SAFE" and cr == "UNSAFE") or (cn == "UNSAFE" and cr == "SAFE"):
        conflicts.append((idx, ln.value, lr.value))
    elif cn != cr:
        disagreements.append((idx, ln.value, lr.value))
    else:
        agreements.append((idx, ln.value, lr.value))

print(f"\n  Cells checked by both  : {len(agreements)+len(disagreements)+len(conflicts)}")
print(f"  Full agreements        : {len(agreements)}")
print(f"  Disagreements (one resolves, other INCONCLUSIVE): {len(disagreements)}")
print(f"  CONFLICTS (SAFE vs UNSAFE) : {len(conflicts)}  ← must be 0")

if conflicts:
    print("\n  !! CONFLICTS FOUND:")
    for idx, ln, lr in conflicts:
        print(f"    cell {idx:5d}  NLP={ln:<22s}  REF={lr}")
else:
    print("\n  No conflicts — both methods are consistent.")

if disagreements:
    print(f"\n  Disagreements (first 10):")
    for idx, ln, lr in disagreements[:10]:
        print(f"    cell {idx:5d}  NLP={ln:<22s}  REF={lr}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Summary table
# ═══════════════════════════════════════════════════════════════════════════
total = len(BC)  # already sliced to MAX_CELLS
print(f"\n{'='*60}")
print(f"{'Metric':<35} {'NLP':>10} {'Refinement':>12}")
print(f"{'='*60}")
print(f"{'SAFE (Taylor, before fallback)':<35} {'—':>10} {'—':>12}")
print(f"{'SAFE (Taylor)':<35} {summary_nlp.n_safe_taylor:>10} {summary_ref.n_safe_taylor:>12}")
print(f"{'SAFE (NLP / Refinement)':<35} {summary_nlp.n_safe_nlp:>10} {summary_ref.n_safe_refinement:>12}")
print(f"{'SAFE total':<35} {summary_nlp.n_safe:>10} {summary_ref.n_safe:>12}")
print(f"{'UNSAFE':<35} {summary_nlp.n_unsafe:>10} {summary_ref.n_unsafe:>12}")
print(f"{'INCONCLUSIVE':<35} {summary_nlp.n_inconclusive:>10} {summary_ref.n_inconclusive:>12}")
print(f"{'Runtime (s)':<35} {summary_nlp.runtime_s:>10.2f} {summary_ref.runtime_s:>12.2f}")
print(f"{'='*60}")

# Counterexample consistency
cx_nlp = summary_nlp.counterexample
cx_ref = summary_ref.counterexample
print(f"\n  NLP counterexample cell       : {cx_nlp.cell_idx if cx_nlp else 'none'}")
print(f"  Refinement counterexample cell: {cx_ref.cell_idx if cx_ref else 'none'}")
if cx_nlp and cx_ref:
    same_cell = cx_nlp.cell_idx == cx_ref.cell_idx
    print(f"  Same cell                     : {same_cell}  {'OK' if same_cell else '!! MISMATCH'}")
