import torch, numpy as np, h5py, sys
sys.path.insert(0, '/home/pouya/Codes/NN_Enumeration')

from relu_region_enumerator.hessian_bound import compute_local_gradient
from relu_region_enumerator.Dynamics import load_dynamics
from relu_region_enumerator.bitwise_utils import get_cell_hyperplanes_input_space

model = torch.jit.load("NN_files/model_decay_2_10_ren.pt", map_location="cpu")
model.eval()
params  = [p.detach().numpy() for _, p in model.named_parameters()]
layer_W = [params[0], params[2]]
layer_b = [params[1], params[3]]
W_out   = params[4]
n       = layer_W[0].shape[1]
TH      = [2.0] * 6
boundary_H = np.vstack((np.eye(n), -np.eye(n)))
boundary_b = np.array(TH * 2, dtype=np.float64)

with h5py.File("decay_boundary_cells.h5", "r") as f:
    offsets = f["offsets"][:]
    verts   = f["vertices"][:]
    sv      = f["activation_patterns"][:]

idx      = 853
vertices = verts[offsets[idx]:offsets[idx+1]]
sv_i     = sv[idx].ravel()
p_i      = compute_local_gradient(sv_i, layer_W, W_out)

H_cell, d_cell = get_cell_hyperplanes_input_space(
    sv_i, layer_W, layer_b, boundary_H, boundary_b
)

x_star = np.array([-0.17921869, 0.37117925, 0.03658136,
                   -0.07064453, -0.52484272, -0.14869954])

# Check 1: is x* on the zero level set?
model_dtype = next(model.parameters()).dtype
with torch.no_grad():
    B_xstar = float(model(
        torch.tensor(x_star[None].astype(np.float32), dtype=model_dtype)
    ).numpy().ravel()[0])
print(f"B(x*)  = {B_xstar:.2e}  (should be ~0)")

# Check 2: does x* satisfy cell polytope H_cell @ x <= d_cell?

residuals = H_cell @ x_star + d_cell
print("residuals (should all be >= 0):")
print(residuals)
print(f"Min residual: {residuals.min():.4e}")
print(f"Violated (< -1e-6): {(residuals < -1e-6).sum()} rows")
# residuals = H_cell @ x_star - d_cell
# violated  = residuals > 1e-6
# print(f"\nCell polytope check:")
# print(f"  Max residual : {residuals.max():.4e}")
# print(f"  Violated rows: {violated.sum()} / {len(d_cell)}")
# if violated.any():
#     print(f"  Worst violations:")
#     for row in np.where(violated)[0][:5]:
#         print(f"    row {row}: H@x - d = {residuals[row]:.4e}")

# Check 3: does x* satisfy the affine ZLS constraint p_i.x + b_i = 0?
v0  = vertices[0]
with torch.no_grad():
    B_v0 = float(model(
        torch.tensor(v0[None].astype(np.float32), dtype=model_dtype)
    ).numpy().ravel()[0])
b_i = B_v0 - float(p_i @ v0)
zls_val = float(p_i @ x_star) + b_i
print(f"\nAffine ZLS check:")
print(f"  p_i.x* + b_i = {zls_val:.4e}  (should be ~0)")

# Check 4: Lie derivative
symbols, f_sym = load_dynamics("decay")
import sympy as sp
subs   = dict(zip(symbols, x_star))
f_vals = np.array([float(fi.subs(subs)) for fi in f_sym])
lie    = float(p_i @ f_vals)
print(f"\nLie derivative check:")
print(f"  p_i . f(x*) = {lie:.6e}  (> 0 means violation)")