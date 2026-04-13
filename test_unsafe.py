import torch, numpy as np, h5py, subprocess, os, sys
sys.path.insert(0, '/home/pouya/Codes/NN_Enumeration')

from relu_region_enumerator.hessian_bound import compute_local_gradient
from relu_region_enumerator.Dynamics import load_dynamics
from relu_region_enumerator.bitwise_utils import get_cell_hyperplanes_input_space
from test_refinement_vs_dreal import build_smt2, call_dreal

# ── reload everything ─────────────────────────────────────────────────────
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

symbols, f_sym = load_dynamics("decay")
model_dtype = next(model.parameters()).dtype
np_dtype    = np.float64 if model_dtype == torch.float64 else np.float32

# ── find the UNSAFE cell from refinement ─────────────────────────────────
# Run a quick pass to find it
from relu_region_enumerator.verify_certificate_new import verify_barrier
BC = [verts[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)][:1000]
sv_slice = sv[:1000]

summary = verify_barrier(
    BC, sv_slice, layer_W, layer_b, W_out,
    boundary_H, boundary_b, model,
    dynamics_name="decay",
    continuous_time=True,
    early_exit=False,
    refinement_max_depth=6,
    TH=TH,
)

unsafe_cells = [r for r in summary.results if "UNSAFE" in r.label.value]
print(f"Found {len(unsafe_cells)} UNSAFE cells")
for r in unsafe_cells:
    print(f"  cell {r.cell_idx}  label={r.label.value}")

# ── test dReal at multiple precisions on the UNSAFE cell ──────────────────
for unsafe_result in unsafe_cells:
    idx      = unsafe_result.cell_idx
    vertices = np.asarray(BC[idx], dtype=float)
    sv_i     = sv_slice[idx].ravel()
    p_i      = compute_local_gradient(sv_i, layer_W, W_out)

    with torch.no_grad():
        B_vals = model(
            torch.tensor(vertices.astype(np_dtype), dtype=model_dtype)
        ).numpy().ravel()

    # ZLS crossings
    x_stars_list = []
    for u in np.where(B_vals < 0)[0]:
        for v in np.where(B_vals >= 0)[0]:
            t = -B_vals[u] / (B_vals[v] - B_vals[u])
            x_stars_list.append(vertices[u] + t * (vertices[v] - vertices[u]))
    x_stars = np.array(x_stars_list)

    v0  = vertices[0]
    with torch.no_grad():
        B_v0 = float(model(
            torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
        ).numpy().ravel()[0])
    b_i = B_v0 - float(p_i @ v0)

    H_cell, d_cell = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b
    )

    # Also print the actual violation value from refinement
    if summary.counterexample is not None:
        ce = summary.counterexample
        print(f"\nCounterexample from refinement:")
        print(f"  x*              = {ce.x_star}")
        print(f"  Lie derivative  = {ce.violation_value:.6e}  (> 0 = violation)")

    print(f"\nTesting dReal at multiple precisions on cell {idx}:")
    print(f"  x_stars range: {x_stars.min(axis=0)} to {x_stars.max(axis=0)}")

    for delta in [0.01, 0.001, 0.0001, 0.00001]:
        smt2 = build_smt2(
            p_i, b_i, H_cell, d_cell,
            x_stars.min(axis=0), x_stars.max(axis=0),
            symbols, f_sym, delta=delta
        )

        # Save SMT2
        smt2_path = f"/tmp/unsafe_cell_{idx}_delta{delta}.smt2"
        with open(smt2_path, "w") as f:
            f.write(smt2)

        import time
        t0     = time.perf_counter()
        result = call_dreal(smt2, delta=delta, timeout=120)
        rt     = time.perf_counter() - t0

        print(f"  delta={delta:.5f}  ->  dReal={result:<8s}  time={rt:.3f}s")