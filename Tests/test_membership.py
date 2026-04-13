"""
test_membership.py
==================
Tests whether the counterexample found by adaptive refinement for cell 853
is actually inside the activation region X(C).

Convention
----------
Network rows  : H[row] @ x + b[row] >= 0  if neuron active  (sv=1)
                H[row] @ x + b[row] <= 0  if neuron inactive (sv=0)
Domain rows   : H[row] @ x + b[row] <= 0  (boundary_H @ x + boundary_b <= 0
                defines domain interior, i.e. x in [-TH, TH]^n)
"""

import sys, os
sys.path.insert(0, '/home/pouya/Codes/NN_Enumeration')

import numpy as np
import torch
import h5py

from relu_region_enumerator.hessian_bound import compute_local_gradient
from relu_region_enumerator.Dynamics import load_dynamics
from relu_region_enumerator.bitwise_utils import get_cell_hyperplanes_input_space

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
MODEL_PATH  = "NN_files/model_decay_2_10_ren.pt"
BOUNDARY_H5 = "decay_boundary_cells.h5"
TH          = [2.0] * 6
CELL_IDX    = 853

# Counterexample from refinement
X_STAR = np.array([-0.17921869,  0.37117925,  0.03658136,
                   -0.07064453, -0.52484272, -0.14869954])

# ═══════════════════════════════════════════════════════════════════════════
# Load
# ═══════════════════════════════════════════════════════════════════════════
print("Loading model ...")
model   = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()
params  = [p.detach().numpy() for _, p in model.named_parameters()]
layer_W = [params[0], params[2]]
layer_b = [params[1], params[3]]
W_out   = params[4]
n       = layer_W[0].shape[1]

boundary_H = np.vstack(( np.eye(n), -np.eye(n)))
boundary_b = np.array(TH * 2, dtype=np.float64)

print("Loading boundary cells ...")
with h5py.File(BOUNDARY_H5, "r") as f:
    offsets = f["offsets"][:]
    verts   = f["vertices"][:]
    sv      = f["activation_patterns"][:]

vertices = verts[offsets[CELL_IDX]:offsets[CELL_IDX+1]]
sv_i     = sv[CELL_IDX].ravel()
p_i      = compute_local_gradient(sv_i, layer_W, W_out)

model_dtype = next(model.parameters()).dtype
np_dtype    = np.float64 if model_dtype == torch.float64 else np.float32

# ═══════════════════════════════════════════════════════════════════════════
# Get hyperplanes
# ═══════════════════════════════════════════════════════════════════════════
H_all, b_all = get_cell_hyperplanes_input_space(
    sv_i, layer_W, layer_b, boundary_H, boundary_b
)
total_neurons = sum(W.shape[0] for W in layer_W)

print(f"\nCell {CELL_IDX}:")
print(f"  Total neurons : {total_neurons}")
print(f"  Domain rows   : {len(b_all) - total_neurons}")
print(f"  sv_i          : {sv_i}")

# ═══════════════════════════════════════════════════════════════════════════
# Membership check
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Membership check for x* = {X_STAR}")
print(f"{'='*60}")

# 1. B(x*) check
with torch.no_grad():
    B_xstar = float(model(
        torch.tensor(X_STAR[None].astype(np_dtype), dtype=model_dtype)
    ).numpy().ravel()[0])
print(f"\nB(x*)  = {B_xstar:.4e}  (should be ~0 for ZLS point)")

# 2. Affine ZLS check
v0 = vertices[0]
with torch.no_grad():
    B_v0 = float(model(
        torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
    ).numpy().ravel()[0])
b_i     = B_v0 - float(p_i @ v0)
zls_val = float(p_i @ X_STAR) + b_i
print(f"p_i·x* + b_i = {zls_val:.4e}  (should be ~0)")

# 3. Lie derivative
symbols, f_sym = load_dynamics("decay")
import sympy as sp
subs   = dict(zip(symbols, X_STAR))
f_vals = np.array([float(fi.subs(subs)) for fi in f_sym])
lie    = float(p_i @ f_vals)
print(f"p_i·f(x*)    = {lie:.6e}  (> 0 = violation)")

# 4. Network neuron constraints
#    Active   (sv=1): H[row]@x + b[row] >= 0
#    Inactive (sv=0): H[row]@x + b[row] <= 0
#    Unified: (2*sv - 1) * (H[row]@x + b[row]) >= 0
# Network rows — use sv_i
preact        = H_all[:total_neurons] @ X_STAR + b_all[:total_neurons]
signs         = 2 * sv_i - 1
signed_preact = signs * preact   # >= 0 if satisfied

# Domain rows — always positive (treated as active, sv=1)
domain_preact = H_all[total_neurons:] @ X_STAR + b_all[total_neurons:]
# domain_preact should be >= 0 (NOT negated)

domain_ok       = domain_preact >= -1e-4
domain_violated = ~domain_ok

print(f"Domain constraints: violated={domain_violated.sum()}, min={domain_preact.min():.4e}")
preact        = H_all[:total_neurons] @ X_STAR + b_all[:total_neurons]
signs         = 2 * sv_i - 1          # +1 for active, -1 for inactive
signed_preact = signs * preact        # should be >= 0 for all rows

neuron_ok       = signed_preact >= -1e-4
neuron_violated = ~neuron_ok

print(f"\nNeuron constraints ({total_neurons} rows):")
print(f"  Satisfied : {neuron_ok.sum()}")
print(f"  Violated  : {neuron_violated.sum()}")
print(f"  Min margin: {signed_preact.min():.4e}")

if neuron_violated.any():
    print(f"\n  Violated rows (row, sv, preact, signed_preact):")
    for row in np.where(neuron_violated)[0]:
        print(f"    row {row:3d}  sv={int(sv_i[row])}  "
              f"preact={preact[row]:+.4e}  "
              f"signed={signed_preact[row]:+.4e}")

# 5. Domain boundary constraints
#    boundary_H @ x + boundary_b <= 0  defines domain interior
#    i.e. -(H[row]@x + b[row]) >= 0
domain_preact   = H_all[total_neurons:] @ X_STAR + b_all[total_neurons:]
domain_signed   = -domain_preact       # should be >= 0
domain_ok       = domain_signed >= -1e-4
domain_violated = ~domain_ok

print(f"\nDomain constraints ({len(domain_signed)} rows):")
print(f"  Satisfied : {domain_ok.sum()}")
print(f"  Violated  : {domain_violated.sum()}")
print(f"  Min margin: {domain_signed.min():.4e}")

if domain_violated.any():
    print(f"\n  Violated rows (row, H@x+b, margin):")
    for row in np.where(domain_violated)[0]:
        print(f"    row {row:3d}  H@x+b={domain_preact[row]:+.4e}  "
              f"margin={domain_signed[row]:+.4e}")

# ═══════════════════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
total_violated = int(neuron_violated.sum()) + int(domain_violated.sum())
if total_violated == 0:
    print("VERDICT: x* IS inside X(C) — counterexample is GENUINE")
    print(f"  Lie derivative = {lie:.6e} > 0 — real violation")
    print(f"  dReal missed it at delta=0.01 — precision issue")
else:
    print(f"VERDICT: x* is OUTSIDE X(C) — counterexample is SPURIOUS")
    print(f"  {total_violated} constraints violated")
    print(f"  The refinement interpolated outside the activation region")
print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════════════════
# Also check all cell vertices for sanity
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nSanity check: are cell vertices inside X(C)?")
with torch.no_grad():
    B_verts = model(
        torch.tensor(vertices.astype(np_dtype), dtype=model_dtype)
    ).numpy().ravel()

n_inside = 0
for k, v in enumerate(vertices):
    preact_v = H_all[:total_neurons] @ v + b_all[:total_neurons]
    sp_v     = (2 * sv_i - 1) * preact_v
    dom_v    = -(H_all[total_neurons:] @ v + b_all[total_neurons:])
    inside   = (sp_v.min() >= -1e-4) and (dom_v.min() >= -1e-4)
    if inside:
        n_inside += 1

print(f"  {n_inside} / {len(vertices)} vertices satisfy all constraints")
print(f"  (all should if H_all is computed correctly)")
# Sanity: check one vertex
v0 = vertices[0]
preact_v0  = H_all[:total_neurons] @ v0 + b_all[:total_neurons]
domain_v0  = H_all[total_neurons:] @ v0 + b_all[total_neurons:]
sp_v0      = (2*sv_i-1) * preact_v0
print("Neuron signed margins for v0:", sp_v0.min())
print("Domain margins for v0:", domain_v0.min())
preact_star        = H_all[:total_neurons] @ X_STAR + b_all[:total_neurons]
signed_preact_star = (2 * sv_i - 1) * preact_star
domain_preact_star = H_all[total_neurons:] @ X_STAR + b_all[total_neurons:]

print(f"Neuron: min={signed_preact_star.min():.4e}, violated={( signed_preact_star < -1e-4).sum()}")
print(f"Domain: min={domain_preact_star.min():.4e}, violated={(domain_preact_star < -1e-4).sum()}")
inside = (signed_preact_star.min() >= -1e-4) and (domain_preact_star.min() >= -1e-4)
print(f"x* inside X(C): {inside}")

x_min = X_STAR.min(axis=0)
x_max = X_STAR.max(axis=0)
print("x_min:", x_min)
print("x_max:", x_max)
print("x*   :", X_STAR)
print("x* >= x_min - 1e-6:", X_STAR >= x_min - 1e-6)
print("x* <= x_max + 1e-6:", X_STAR <= x_max + 1e-6)
print("x* inside box:", np.all(X_STAR >= x_min - 1e-6) and np.all(X_STAR <= x_max + 1e-6))