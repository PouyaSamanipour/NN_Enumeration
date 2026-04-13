"""
inspect_not_boundary.py
=======================
For cells classified as NOT_BOUNDARY, report:
  - min B(x) over X(C)
  - max B(x) over X(C)
  - which side of the ZLS the cell is on

Usage:
    python inspect_not_boundary.py complex__boundary_cells.h5 \
                                   verification_patterns.npy  \
                                   NN_files/complex_3d_2x64.pt \
                                   3 3.0
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import sys
from scipy.optimize import linprog

# ─────────────────────────────────────────────
# Network / model loading (same as before)
# ─────────────────────────────────────────────
class BarrierNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

def load_model(model_path, input_dim=3, hidden_dim=64):
    try:
        ts = torch.jit.load(model_path, map_location='cpu').double()
        sd = {name: param.detach() for name, param in ts.named_parameters()}
        model = BarrierNet(input_dim, hidden_dim).double()
        try:
            model.load_state_dict(sd)
        except Exception:
            sd = {(k if k.startswith('layers.') else f'layers.{k}'): v for k, v in sd.items()}
            model.load_state_dict(sd)
        model.eval()
        return model
    except Exception:
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        if not any(k.startswith('layers.') for k in state.keys()):
            state = {f'layers.{k}': v for k, v in state.items()}
        model = BarrierNet(input_dim, hidden_dim)
        model.load_state_dict(state)
        return model.double().eval()

def extract_weights(model):
    linear_layers = [l for l in model.layers.children() if isinstance(l, nn.Linear)]
    W = [l.weight.detach().numpy().astype(np.float64) for l in linear_layers]
    b = [l.bias.detach().numpy().astype(np.float64)   for l in linear_layers]
    return W, b

def get_linear_map(W, b, sign_vec):
    n_hidden = len(W) - 1
    neurons_per_layer = [W[i].shape[0] for i in range(n_hidden)]
    idx = 0
    D = []
    for l in range(n_hidden):
        n_l = neurons_per_layer[l]
        D.append(np.diag(sign_vec[idx:idx + n_l].astype(np.float64)))
        idx += n_l
    if n_hidden == 2:
        W_eff = W[2] @ D[1] @ W[1] @ D[0] @ W[0]
        b_eff = W[2] @ (D[1] @ (W[1] @ (D[0] @ b[0]) + b[1])) + b[2]
    else:
        W_eff = W[1] @ D[0] @ W[0]
        b_eff = W[1] @ (D[0] @ b[0]) + b[1]
    return W_eff.flatten(), float(b_eff.flatten()[0])

def get_region_constraints(W, b, sign_vec, input_dim=3, domain=3.0):
    n_hidden = len(W) - 1
    neurons_per_layer = [W[i].shape[0] for i in range(n_hidden)]
    idx = 0
    D = []
    for l in range(n_hidden):
        n_l = neurons_per_layer[l]
        D.append(np.diag(sign_vec[idx:idx + n_l].astype(np.float64)))
        idx += n_l
    rows_W, rows_r = [], []
    n0 = neurons_per_layer[0]
    for j in range(n0):
        sg = 1.0 if sign_vec[j] == 1 else -1.0
        rows_W.append(sg * W[0][j]); rows_r.append(sg * b[0][j])
    if n_hidden >= 2:
        n1 = neurons_per_layer[1]
        W1_eff = W[1] @ D[0] @ W[0]
        b1_eff = W[1] @ (D[0] @ b[0]) + b[1]
        for j in range(n1):
            sg = 1.0 if sign_vec[n0 + j] == 1 else -1.0
            rows_W.append(sg * W1_eff[j]); rows_r.append(sg * b1_eff[j])
    for i in range(input_dim):
        e = np.zeros(input_dim); e[i] = 1.0
        rows_W.append(e);  rows_r.append(domain)
        rows_W.append(-e); rows_r.append(domain)
    return np.array(rows_W, dtype=np.float64), np.array(rows_r, dtype=np.float64)

# ─────────────────────────────────────────────
# Min/max B(x) over X(C)
# ─────────────────────────────────────────────
def minmax_barrier(w_C, b_C, cons_W, cons_r, input_dim):
    A_ub = -cons_W
    b_ub =  cons_r
    bounds = [(None, None)] * input_dim

    r_max = linprog(-w_C, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                    method='highs', options={'disp': False})
    r_min = linprog( w_C, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                    method='highs', options={'disp': False})

    max_B = (-r_max.fun + b_C) if r_max.status == 0 else float('nan')
    min_B = ( r_min.fun + b_C) if r_min.status == 0 else float('nan')
    return min_B, max_B

# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────
def load_our_cells(h5_path):
    with h5py.File(h5_path, 'r') as f:
        ap = f['activation_patterns']
        cells = ap[:] if isinstance(ap, h5py.Dataset) else np.array([ap[k][:] for k in ap.keys()])
    return cells

def load_ren_cells(npy_path):
    return np.load(npy_path, allow_pickle=True)

def ren_to_tuple(p):
    sv = []
    for k in sorted(k for k in p.keys() if k != max(p.keys())):
        for v in p[k]:
            sv.append(1 if v == 1 else 0)
    return tuple(sv)

def our_to_tuple(cell):
    return tuple(int(round(v)) for v in cell)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(h5_path, npy_path, model_path, input_dim=3, domain=3.0):
    model  = load_model(model_path, input_dim, 64)
    W, b   = extract_weights(model)

    our_set = set(our_to_tuple(c) for c in load_our_cells(h5_path))
    ren_set = set(ren_to_tuple(p) for p in load_ren_cells(npy_path))
    only_ours = our_set - ren_set

    print(f"Inspecting {len(only_ours)} extra cells...\n")

    not_boundary = []

    for cell in only_ours:
        sign_vec       = np.array(cell, dtype=np.float64)
        w_C, b_C       = get_linear_map(W, b, sign_vec)
        cons_W, cons_r = get_region_constraints(W, b, sign_vec, input_dim, domain)
        A_ub = -cons_W
        b_ub =  cons_r

        # Check region non-empty
        r = linprog(np.zeros(input_dim), A_ub=A_ub, b_ub=b_ub,
                    bounds=[(None,None)]*input_dim, method='highs')
        if r.status != 0:
            continue  # empty region — skip

        min_B, max_B = minmax_barrier(w_C, b_C, cons_W, cons_r, input_dim)

        # Classify
        if max_B >= 0 and min_B <= 0:
            verdict = 'BOUNDARY_CELL'
        else:
            verdict = 'NOT_BOUNDARY'
            not_boundary.append((cell, min_B, max_B))

    print(f"NOT_BOUNDARY cells: {len(not_boundary)}")
    print(f"\n{'Cell':>4}  {'min B(x)':>12}  {'max B(x)':>12}  {'Side'}")
    print("-" * 50)

    all_min = []
    all_max = []

    for i, (cell, min_B, max_B) in enumerate(not_boundary):
        if min_B > 0:
            side = "POSITIVE (inside safe set)"
        elif max_B < 0:
            side = "NEGATIVE (outside safe set)"
        else:
            side = "STRADDLES (numerical edge case)"
        print(f"  {i+1:3d}  {min_B:12.6f}  {max_B:12.6f}  {side}")
        all_min.append(min_B)
        all_max.append(max_B)

    if not_boundary:
        print(f"\n{'='*50}")
        print(f"Summary over {len(not_boundary)} NOT_BOUNDARY cells:")
        print(f"  min B(x) range:  [{min(all_min):.6f},  {max(all_min):.6f}]")
        print(f"  max B(x) range:  [{min(all_max):.6f},  {max(all_max):.6f}]")
        n_pos = sum(1 for mn in all_min if mn > 0)
        n_neg = sum(1 for mx in all_max if mx < 0)
        print(f"  Entirely positive (B > 0 everywhere): {n_pos}")
        print(f"  Entirely negative (B < 0 everywhere): {n_neg}")
        print(f"{'='*50}")
        print("""
INTERPRETATION:
  - If B > 0 everywhere in cell -> cell is strictly inside safe set, 
    not a boundary cell. Our enumeration correctly included it but  
    it should not be in the boundary cell list.
  - If B < 0 everywhere in cell -> cell is strictly outside safe set,
    same conclusion.
  - Values very close to 0 -> likely a numerical precision issue where
    the ZLS grazes the cell boundary but LP reports no crossing.
""")

if __name__ == "__main__":
    h5_path    = sys.argv[1] if len(sys.argv) > 1 else "complex_boundary_cells.h5"
    npy_path   = sys.argv[2] if len(sys.argv) > 2 else "verification_patterns.npy"
    model_path = sys.argv[3] if len(sys.argv) > 3 else "NN_files/complex_3d_2x64.pt"
    input_dim  = int(sys.argv[4])   if len(sys.argv) > 4 else 3
    domain     = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
    main(h5_path, npy_path, model_path, input_dim, domain)