"""
validate_extra_cells.py
=======================
Validate extra boundary cells found by our method but not Ren et al.

For each extra cell, two independent LP checks are performed:

  Q1 — Is the activation region X(C) non-empty?
       find x: cons_W @ x >= -cons_r

  Q2 — Does the ZLS B(x)=0 pass through X(C)?
       max  w_C^T x + b_C  over X(C)  -> should be >= 0
       min  w_C^T x + b_C  over X(C)  -> should be <= 0

Verdict per cell:
  BOUNDARY_CELL  — X(C) non-empty AND ZLS passes through it  (genuine)
  NOT_BOUNDARY   — X(C) non-empty BUT ZLS does not cross it   (not a boundary cell)
  EMPTY_REGION   — X(C) is empty                              (spurious, real bug)

Usage:
    python validate_extra_cells.py complex__boundary_cells.h5 \\
                                   verification_patterns.npy  \\
                                   NN_files/complex_3d_2x64.pt \\
                                   3 3.0
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import sys
from scipy.optimize import linprog

# ─────────────────────────────────────────────
# 1. Network definition  (3 -> 64 -> 64 -> 1)
# ─────────────────────────────────────────────
class BarrierNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

# ─────────────────────────────────────────────
# 2. Model loading
# ─────────────────────────────────────────────
def load_model(model_path, input_dim=8, hidden_dim=16):
    print(f"Loading model from {model_path}...")

    # Try TorchScript first
    try:
        ts = torch.jit.load(model_path, map_location='cpu').double()
        sd = {name: param.detach() for name, param in ts.named_parameters()}
        print("TorchScript parameters:")
        for k, v in sd.items():
            print(f"  {k}: {v.shape}")
        model = BarrierNet(input_dim, hidden_dim).double()
        try:
            model.load_state_dict(sd)
        except Exception:
            sd2 = {(k if k.startswith('layers.') else f'layers.{k}'): v
                   for k, v in sd.items()}
            model.load_state_dict(sd2)
        model.eval()
        print("Loaded as TorchScript.")
        return model
    except Exception as e:
        print(f"TorchScript failed ({e}), trying state dict...")

    state = torch.load(model_path, map_location='cpu', weights_only=False)
    if not any(k.startswith('layers.') for k in state.keys()):
        state = {f'layers.{k}': v for k, v in state.items()}
    model = BarrierNet(input_dim, hidden_dim)
    model.load_state_dict(state)
    model = model.double()
    model.eval()
    print("Loaded as state dict.")
    return model

# ─────────────────────────────────────────────
# 3. Extract weight matrices
# ─────────────────────────────────────────────
def extract_weights(model):
    linear_layers = [l for l in model.layers.children()
                     if isinstance(l, nn.Linear)]
    W = [l.weight.detach().numpy().astype(np.float64) for l in linear_layers]
    b = [l.bias.detach().numpy().astype(np.float64)   for l in linear_layers]
    print("\nNetwork architecture:")
    for i, (w, bi) in enumerate(zip(W, b)):
        print(f"  Layer {i}: {w.shape[1]} -> {w.shape[0]}")
    return W, b

# ─────────────────────────────────────────────
# 4. Compute w(C), b(C) — affine map B(x) = w_C^T x + b_C
# ─────────────────────────────────────────────
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
    elif n_hidden == 1:
        W_eff = W[1] @ D[0] @ W[0]
        b_eff = W[1] @ (D[0] @ b[0]) + b[1]
    else:
        raise ValueError(f"Unsupported hidden layers: {n_hidden}")

    return W_eff.flatten(), float(b_eff.flatten()[0])

# ─────────────────────────────────────────────
# 5. Build activation region constraints
#    X(C) = {x : cons_W @ x >= -cons_r}
# ─────────────────────────────────────────────
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

    # Layer 1 constraints
    n0 = neurons_per_layer[0]
    for j in range(n0):
        s = sign_vec[j]
        sg = 1.0 if s == 1 else -1.0
        rows_W.append(sg * W[0][j])
        rows_r.append(sg * b[0][j])

    # Layer 2 constraints
    if n_hidden >= 2:
        n1 = neurons_per_layer[1]
        W1_eff = W[1] @ D[0] @ W[0]
        b1_eff = W[1] @ (D[0] @ b[0]) + b[1]
        for j in range(n1):
            s = sign_vec[n0 + j]
            sg = 1.0 if s == 1 else -1.0
            rows_W.append(sg * W1_eff[j])
            rows_r.append(sg * b1_eff[j])

    # Domain box: -domain <= x_i <= domain
    for i in range(input_dim):
        e = np.zeros(input_dim)
        e[i] = 1.0
        rows_W.append(e);  rows_r.append(domain)
        rows_W.append(-e); rows_r.append(domain)

    return np.array(rows_W, dtype=np.float64), np.array(rows_r, dtype=np.float64)

# ─────────────────────────────────────────────
# 6. Two-question LP validation
# ─────────────────────────────────────────────
def validate_cell(w_C, b_C, cons_W, cons_r, input_dim):
    """
    Q1: Is X(C) non-empty?
    Q2: Does ZLS B(x)=0 pass through X(C)?

    Returns one of: 'BOUNDARY_CELL', 'NOT_BOUNDARY', 'EMPTY_REGION'
    """
    c    = np.zeros(input_dim)
    A_ub = -cons_W          # cons_W @ x >= -cons_r  ->  -cons_W @ x <= cons_r
    b_ub =  cons_r

    # Q1 — feasibility of X(C)
    r = linprog(c, A_ub=A_ub, b_ub=b_ub,
                bounds=[(None, None)] * input_dim,
                method='highs', options={'disp': False})
    if r.status != 0:
        return 'EMPTY_REGION'

    # Q2a — maximize B(x) over X(C)  ->  min -w_C^T x
    r_max = linprog(-w_C, A_ub=A_ub, b_ub=b_ub,
                    bounds=[(None, None)] * input_dim,
                    method='highs', options={'disp': False})
    max_val = -r_max.fun if r_max.status == 0 else -np.inf
    # add bias
    max_B = max_val + b_C

    # Q2b — minimize B(x) over X(C)  ->  min w_C^T x
    r_min = linprog(w_C, A_ub=A_ub, b_ub=b_ub,
                    bounds=[(None, None)] * input_dim,
                    method='highs', options={'disp': False})
    min_val = r_min.fun if r_min.status == 0 else np.inf
    # add bias
    min_B = min_val + b_C

    # ZLS passes through X(C) if max_B >= 0 and min_B <= 0
    if max_B >= 0 and min_B <= 0:
        return 'BOUNDARY_CELL'
    else:
        return 'NOT_BOUNDARY'

# ─────────────────────────────────────────────
# 7. Data loading helpers
# ─────────────────────────────────────────────
def load_our_cells(h5_path):
    with h5py.File(h5_path, 'r') as f:
        ap = f['activation_patterns']
        cells = (ap[:] if isinstance(ap, h5py.Dataset)
                 else np.array([ap[k][:] for k in ap.keys()]))
    print(f"[Ours] {len(cells)} boundary cells")
    return cells

def load_ren_cells(npy_path):
    patterns = np.load(npy_path, allow_pickle=True)
    print(f"[Ren]  {len(patterns)} verification patterns")
    return patterns

def ren_to_tuple(pattern):
    sign_vec = []
    hidden_keys = sorted(k for k in pattern.keys() if k != max(pattern.keys()))
    for k in hidden_keys:
        for v in pattern[k]:
            sign_vec.append(1 if v == 1 else 0)
    return tuple(sign_vec)

def our_to_tuple(cell):
    return tuple(int(round(v)) for v in cell)

def tuple_to_layers(t, neurons_per_layer):
    """Split a flat activation tuple into per-layer lists."""
    layers = []
    idx = 0
    for n in neurons_per_layer:
        layers.append(list(t[idx:idx + n]))
        idx += n
    return layers

def print_pattern(t, neurons_per_layer, label=""):
    """Print an activation pattern layer-by-layer as a compact bit string."""
    layers = tuple_to_layers(t, neurons_per_layer)
    n_active_total = sum(t)
    layer_strs = []
    for i, layer in enumerate(layers):
        bits = ''.join(str(v) for v in layer)
        n_act = sum(layer)
        layer_strs.append(f"L{i+1}[{n_act:2d}/{len(layer)}]:{bits}")
    prefix = f"{label:30s}" if label else ""
    print(f"  {prefix}  " + "  |  ".join(layer_strs) + f"  (total active: {n_active_total}/{len(t)})")

# ─────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────
def main(h5_path, npy_path, model_path, input_dim=8, domain=3.0):

    # model    = load_model(model_path, input_dim=input_dim, hidden_dim=64)
    # W, b     = extract_weights(model)

    our_raw  = load_our_cells(h5_path)
    ren_raw  = load_ren_cells(npy_path)

    our_set  = set(our_to_tuple(c) for c in our_raw)
    ren_set  = set(ren_to_tuple(p) for p in ren_raw)

    only_ours = our_set - ren_set
    only_ren  = ren_set - our_set
    both      = our_set & ren_set

    print(f"\n{'='*60}")
    print(f"Set comparison:")
    print(f"  Ours only  (we found, Ren missed): {len(only_ours):5d}")
    print(f"  Ren only   (Ren found, we missed): {len(only_ren):5d}")
    print(f"  In common:                         {len(both):5d}")
    print(f"{'='*60}")

    # ── Infer neurons-per-layer from the pattern length ──────────────────
    sample = next(iter(our_set | ren_set))
    total_neurons = len(sample)
    # Assume equal-width hidden layers; fall back to single block if unclear
    for n_layers in range(4, 0, -1):
        if total_neurons % n_layers == 0:
            neurons_per_layer = [total_neurons // n_layers] * n_layers
            break

    # ── Show patterns Ren found that we missed ───────────────────────────
    if only_ren:
        print(f"\nPatterns in Ren et al. NOT found by our method ({len(only_ren)} cells):")
        print(f"(neurons per layer: {neurons_per_layer})\n")
        for i, t in enumerate(sorted(only_ren)):
            print_pattern(t, neurons_per_layer, label=f"missing #{i+1:4d}")
        # Summary: active-neuron count distribution per layer
        print(f"\nActive-neuron count distribution for MISSING patterns:")
        for li, n in enumerate(neurons_per_layer):
            offset = sum(neurons_per_layer[:li])
            counts = np.array([sum(t[offset:offset+n]) for t in only_ren])
            print(f"  Layer {li+1}: mean={counts.mean():.1f}  "
                  f"min={counts.min()}  max={counts.max()}  "
                  f"hist={np.bincount(counts, minlength=n+1).tolist()}")
    else:
        print("\nOur method found ALL of Ren's patterns — no missing cells.")

    if not only_ours:
        print("\nNo extra cells to validate (ours ⊆ Ren).")
        return

    print(f"\nRunning two-question LP validation on {len(only_ours)} extra cells (in ours, not Ren)...\n")

    counts = {'BOUNDARY_CELL': 0, 'NOT_BOUNDARY': 0, 'EMPTY_REGION': 0, 'ERROR': 0}

    for i, cell in enumerate(only_ours):
        sign_vec = np.array(cell, dtype=np.float64)
        try:
            w_C, b_C         = get_linear_map(W, b, sign_vec)
            cons_W, cons_r   = get_region_constraints(W, b, sign_vec, input_dim, domain)
            verdict          = validate_cell(w_C, b_C, cons_W, cons_r, input_dim)
        except Exception as e:
            print(f"  Cell {i+1:3d}: ERROR — {e}")
            counts['ERROR'] += 1
            continue

        symbol = {'BOUNDARY_CELL': '✓', 'NOT_BOUNDARY': '~', 'EMPTY_REGION': '✗'}[verdict]
        print(f"  Cell {i+1:3d}: {verdict:15s} {symbol}")
        counts[verdict] += 1

    total = len(only_ours)
    print(f"\n{'='*55}")
    print(f"RESULTS for {total} extra cells:")
    print(f"  BOUNDARY_CELL  (genuine, Ren missed):  {counts['BOUNDARY_CELL']:4d}")
    print(f"  NOT_BOUNDARY   (cell exists, not bdy): {counts['NOT_BOUNDARY']:4d}")
    print(f"  EMPTY_REGION   (spurious, real bug):   {counts['EMPTY_REGION']:4d}")
    print(f"  ERROR:                                 {counts['ERROR']:4d}")
    print(f"{'='*55}")

    if counts['EMPTY_REGION'] == 0 and counts['NOT_BOUNDARY'] == 0:
        print("\n[CONCLUSION] ALL extra cells are genuine boundary cells.")
        print("Ren et al.'s TestOne validity check incorrectly rejected them.")
    elif counts['EMPTY_REGION'] == 0:
        print(f"\n[CONCLUSION] {counts['BOUNDARY_CELL']} genuine boundary cells missed by Ren.")
        print(f"             {counts['NOT_BOUNDARY']} cells exist but do not cross the ZLS.")
        print("             No empty regions — our enumeration has no spurious cells.")
    else:
        print(f"\n[CONCLUSION] {counts['EMPTY_REGION']} truly spurious cells found — investigate enumeration.")

if __name__ == "__main__":
    h5_path    = sys.argv[1] if len(sys.argv) > 1 else "hiord8_boundary_cells.h5"
    npy_path   = sys.argv[2] if len(sys.argv) > 2 else "boundary_list_new.npy"
    model_path = sys.argv[3] if len(sys.argv) > 3 else "NN_files/high_o_commit_2fd0d11_layers_4_size_16_seed_222_certify_True.pt"
    input_dim  = int(sys.argv[4])   if len(sys.argv) > 4 else 3
    domain     = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
    main(h5_path, npy_path, model_path, input_dim, domain)