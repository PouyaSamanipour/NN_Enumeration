"""
test_refinement_figures.py
==========================
Runs adaptive refinement on all boundary cells and, for cells that are
Taylor-inconclusive, also runs dReal. Saves results to CSV for plotting
in another environment.

Output: refinement_results.csv
  columns: cell_idx, label, ref_time_s, r_i, M_i, remainder,
           dreal_called, dreal_label, dreal_time_s

Usage:
    cd /home/pouya/Codes/NN_Enumeration
    python Tests/test_refinement_figures.py
"""

import sys, os, time, csv, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import h5py
import sympy as sp

from relu_region_enumerator.verify_certificate_new import (
    Label, DynamicsEvaluator,
    _get_zero_level_set_crossings, _two_step_label, _refine_barrier_adaptive,
    _find_farthest_pair_nb,
)
from relu_region_enumerator.hessian_bound import HessianBounder, compute_local_gradient
from relu_region_enumerator.Dynamics import load_dynamics
from relu_region_enumerator.bitwise_utils import get_cell_hyperplanes_input_space, Enumerator_rapid

MODEL_PATH    = "NN_files/model_decay_2_10_ren.pt"
BOUNDARY_H5   = "decay_boundary_cells.h5"
TH            = [2.0] * 6
DYNAMICS      = "decay"
MAX_CELLS     = None        # None = all cells
DREAL_BIN     = "/opt/dreal/4.21.06.2/bin/dreal"
DREAL_DELTA   = 0.01
DREAL_TIMEOUT = 30
OUT_CSV       = "refinement_results.csv"


# ── dReal helpers ─────────────────────────────────────────────────────────────

def _sympy_to_smt2(expr, var_names):
    if expr.is_Number:
        return str(float(expr))
    if expr.is_Symbol:
        return var_names[expr]
    if expr.is_Add:
        return "(+ " + " ".join(_sympy_to_smt2(a, var_names) for a in expr.args) + ")"
    if expr.is_Mul:
        return "(* " + " ".join(_sympy_to_smt2(a, var_names) for a in expr.args) + ")"
    if expr.is_Pow:
        base, exp = expr.args
        b = _sympy_to_smt2(base, var_names)
        if exp == 2: return f"(* {b} {b})"
        if exp == 3: return f"(* {b} (* {b} {b}))"
        if exp == sp.Rational(1, 2): return f"(sqrt {b})"
        return f"(^ {b} {_sympy_to_smt2(exp, var_names)})"
    if expr.is_negative:
        return f"(- {_sympy_to_smt2(-expr, var_names)})"
    func_map = {sp.sin:"sin", sp.cos:"cos", sp.exp:"exp",
                sp.log:"log", sp.sqrt:"sqrt", sp.tanh:"tanh"}
    if expr.func in func_map:
        return f"({func_map[expr.func]} {_sympy_to_smt2(expr.args[0], var_names)})"
    raise ValueError(f"Cannot convert: {expr}")


def call_dreal(vertices, sv_i, p_i, layer_W, layer_b, boundary_H, boundary_b,
               symbols, f_sym, barrier_model, model_dtype, np_dtype):
    n         = len(symbols)
    var_names = {s: f"x{k+1}" for k, s in enumerate(symbols)}
    v0 = vertices[0]
    with torch.no_grad():
        B_v0 = float(barrier_model(
            torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
        ).detach().numpy().ravel()[0])
    b_i = B_v0 - float(p_i @ v0)

    total_neurons = sum(W.shape[0] for W in layer_W)
    H_all, b_all = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b)

    lines = ["(set-logic QF_NRA)", f"(set-info :precision {DREAL_DELTA})"]
    for k in range(n):
        lines.append(f"(declare-fun x{k+1} () Real)")
        lines.append(f"(assert (>= x{k+1} {-float(TH[k]):.10f}))")
        lines.append(f"(assert (<= x{k+1} {float(TH[k]):.10f}))")

    zls = [f"(* {float(p_i[k]):.10f} x{k+1})" for k in range(n) if abs(p_i[k]) > 1e-15]
    zls_sum = ("(+ " + " ".join(zls) + ")") if len(zls) > 1 else (zls[0] if zls else "0.0")
    lines.append(f"(assert (= (+ {zls_sum} {float(b_i):.10f}) 0.0))")

    for row in range(len(b_all)):
        terms = [f"(* {float(H_all[row,k]):.10f} x{k+1})"
                 for k in range(n) if abs(H_all[row, k]) > 1e-15]
        if not terms: continue
        lhs  = "(+ " + " ".join(terms) + ")" if len(terms) > 1 else terms[0]
        expr = f"(+ {lhs} {float(b_all[row]):.10f})"
        if row < total_neurons:
            lines.append(f"(assert ({'>' if sv_i[row]==1 else '<'}= {expr} 0.0))")
        else:
            lines.append(f"(assert (>= {expr} 0.0))")

    lie = [f"(* {float(p_i[k]):.10f} {_sympy_to_smt2(f_sym[k], var_names)})"
           for k in range(n) if abs(p_i[k]) > 1e-15]
    lie_sum = ("(+ " + " ".join(lie) + ")") if len(lie) > 1 else (lie[0] if lie else "0.0")
    lines.append(f"(assert (> {lie_sum} 0.0))")
    lines.append("(check-sat)")
    smt2 = "\n".join(lines)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False, dir='/tmp') as f:
        f.write(smt2); tmp = f.name
    try:
        t0 = time.perf_counter()
        result = subprocess.run(
            [DREAL_BIN, "--precision", str(DREAL_DELTA), tmp],
            capture_output=True, text=True, timeout=DREAL_TIMEOUT)
        dt = time.perf_counter() - t0
        out = (result.stdout + result.stderr).lower()
        if "unsat" in out:   return "SAFE", dt
        elif "sat" in out:   return "UNSAFE", dt
        return "ERROR", dt
    except subprocess.TimeoutExpired:
        return "TIMEOUT", DREAL_TIMEOUT
    finally:
        try: os.unlink(tmp)
        except: pass


# ── Instrumented refinement: records (depth, r_i, M_i) per sub-cell ──────────

def _refine_instrumented(x_stars, worst_cond, remainder, M_i_parent, p_i,
                         barrier_model, dyn, hb, cell_idx, continuous_time,
                         use_wide, max_depth, vertices, sv_i, layer_W, layer_b,
                         boundary_H, boundary_b, TH, model_dtype):
    from collections import deque
    np_dtype = np.float64 if model_dtype == torch.float64 else np.float32
    n        = x_stars.shape[1]
    queue    = deque()
    queue.append((x_stars, vertices, worst_cond, remainder, max_depth,
                  boundary_H, boundary_b))
    records  = []   # (depth_level, r_i, M_i)

    while queue:
        xs, verts_q, wc, _rem, depth, bH, bb = queue.popleft()
        depth_level = max_depth - depth

        if depth == 0 or len(xs) < 2:
            continue

        idx_a, idx_b, max_dist = _find_farthest_pair_nb(
            np.ascontiguousarray(xs, dtype=np.float64))
        if max_dist < 1e-12:
            continue

        normal  = (xs[idx_b] - xs[idx_a]) / max_dist
        n_refs  = 5
        base_p  = float(xs[idx_a] @ normal)
        step_p  = float((xs[idx_b] - xs[idx_a]) @ normal) / n_refs
        cuts    = [base_p + k * step_p for k in range(1, n_refs)]

        H_ref = np.tile(normal, (n_refs - 1, 1)).astype(np.float64)
        b_ref = np.array([-c for c in cuts], dtype=np.float64)

        H_cell, b_cell = get_cell_hyperplanes_input_space(
            sv_i, layer_W, layer_b, bH, bb)
        try:
            sub_cells = Enumerator_rapid(
                H_ref, b_ref,
                [np.asarray(verts_q, dtype=np.float64)],
                TH, [H_cell.tolist()], [b_cell.tolist()],
                False, None, 0, (len(H_cell) + len(H_ref)) > 64,
            )
        except Exception:
            sub_cells = None

        if sub_cells is None or len(sub_cells) <= 1:
            continue

        bH_child = np.vstack([bH, H_ref])
        bb_child = np.concatenate([bb, b_ref])

        for sub_v in sub_cells:
            sub_v = np.asarray(sub_v, dtype=np.float64)
            if len(sub_v) < n + 1:
                continue
            with torch.no_grad():
                B_sub = barrier_model(
                    torch.tensor(sub_v.astype(np_dtype), dtype=model_dtype)
                ).numpy().ravel().astype(np.float64)
            xs_k = sub_v[np.abs(B_sub) < 1e-3]
            if len(xs_k) == 0:
                continue
            lbl, M_i, r_i, rem_i, wc_i, _ = _two_step_label(
                xs_k, barrier_model, dyn, hb, p_i, cell_idx, continuous_time)
            records.append((depth_level, float(r_i), float(M_i)))
            if lbl == Label.UNSAFE:
                return records
            if lbl == Label.INCONCLUSIVE and len(xs_k) >= 2:
                queue.append((xs_k, sub_v, wc_i, rem_i, depth - 1,
                              bH_child, bb_child))
    return records


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model ...")
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()
params   = [p.detach().numpy() for _, p in model.named_parameters()]
layer_W  = [params[0], params[2]]
layer_b  = [params[1], params[3]]
W_out    = params[4]
n        = layer_W[0].shape[1]
model_dtype = next(model.parameters()).dtype
np_dtype    = np.float64 if model_dtype == torch.float64 else np.float32

boundary_H = np.vstack((np.eye(n), -np.eye(n)))
boundary_b = np.array(TH * 2, dtype=np.float64)
use_wide   = (sum(W.shape[0] for W in layer_W) + len(boundary_H)) > 64

# ── Load cells ────────────────────────────────────────────────────────────────
print("Loading boundary cells ...")
with h5py.File(BOUNDARY_H5, "r") as f:
    offsets = f["offsets"][:]
    verts   = f["vertices"][:]
    sv      = f["activation_patterns"][:]

BC = [verts[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
sv = sv[:]
if MAX_CELLS is not None:
    BC = BC[:MAX_CELLS]; sv = sv[:MAX_CELLS]
print(f"  {len(BC)} cells, {n}D")

# ── Load dynamics ─────────────────────────────────────────────────────────────
print("Loading dynamics ...")
symbols, f_sym = load_dynamics(DYNAMICS)
dyn = DynamicsEvaluator(symbols, f_sym)
is_linear = all(sp.diff(fi, xj, xk) == 0
                for fi in f_sym for xj in symbols for xk in symbols)
hb = None if is_linear else HessianBounder(symbols, f_sym)

# ── Run verification ──────────────────────────────────────────────────────────
print("Running verification + dReal on inconclusive cells ...")
rows = []

for i, vertices in enumerate(BC):
    vertices = np.asarray(vertices, dtype=float)
    sv_i = sv[i].ravel()
    p_i  = compute_local_gradient(sv_i, layer_W, W_out)

    t0 = time.perf_counter()

    with torch.no_grad():
        B_vals = model(
            torch.tensor(vertices.astype(np_dtype), dtype=model_dtype)
        ).numpy().ravel()

    x_stars, _, verts_neg, verts_pos = _get_zero_level_set_crossings(
        vertices, sv_i, B_vals,
        layer_W, layer_b, boundary_H, boundary_b, use_wide,
        barrier_model=model, model_dtype=model_dtype,
    )

    if len(x_stars) == 0:
        rows.append(dict(cell_idx=i, label="NO_ZLS",
                         ref_time_s=time.perf_counter()-t0,
                         r_i=0.0, M_i=0.0, remainder=0.0,
                         dreal_called=False, dreal_label="", dreal_time_s=0.0))
        continue

    label, M_i, r_i, remainder, worst_cond, _ = _two_step_label(
        x_stars, model, dyn, hb, p_i, i, True)

    dreal_called = False
    dreal_label  = ""
    dreal_time   = 0.0

    if label == Label.INCONCLUSIVE:
        # run dReal before refinement (same inconclusive x_stars)
        dreal_called = True
        dreal_label, dreal_time = call_dreal(
            vertices, sv_i, p_i, layer_W, layer_b,
            boundary_H, boundary_b, symbols, f_sym,
            model, model_dtype, np_dtype)

        # now run refinement
        v0   = vertices[0]
        B_v0 = float(model(torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
                           ).detach().numpy().ravel()[0])
        q_i    = B_v0 - float(p_i @ v0)
        zls_bH = np.vstack([boundary_H, p_i[None, :]])
        zls_bb = np.append(boundary_b, q_i)
        sub_v  = verts_neg if len(verts_neg) <= len(verts_pos) else verts_pos
        label, M_i, r_i, remainder, _ = _refine_barrier_adaptive(
            x_stars, worst_cond, remainder, M_i, p_i,
            model, dyn, hb, i, True, use_wide, 8,
            sub_v, sv_i, layer_W, layer_b, zls_bH, zls_bb, TH, model_dtype,
        )
        print(f"  cell {i:4d}  ref={label.name:<20s}  dReal={dreal_label:<8s}  dreal_t={dreal_time:.3f}s")

    ref_time = time.perf_counter() - t0 - dreal_time  # refinement time only
    rows.append(dict(cell_idx=i,
                     label=label.name if hasattr(label, "name") else str(label),
                     ref_time_s=ref_time,
                     r_i=float(r_i), M_i=float(M_i), remainder=float(remainder),
                     dreal_called=dreal_called,
                     dreal_label=dreal_label,
                     dreal_time_s=dreal_time))

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(BC)} done")

# ── Save CSV ──────────────────────────────────────────────────────────────────
fields = ["cell_idx","label","ref_time_s","r_i","M_i","remainder",
          "dreal_called","dreal_label","dreal_time_s"]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

label_counts = {}
for r in rows:
    label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
dreal_count = sum(1 for r in rows if r["dreal_called"])
print(f"\nDone. Labels: {label_counts}")
print(f"dReal called on {dreal_count} cells")
print(f"Saved {len(rows)} rows to {OUT_CSV}")

# ── Find cell with most refinement work and record per-depth r_i / M_i ────────
print("\nFinding cell with most refinement ...")

# Use ref_time_s as proxy — the cell that took longest in refinement
refinement_rows = [r for r in rows if r["label"] == "SAFE_REFINEMENT"]
if not refinement_rows:
    print("No SAFE_REFINEMENT cells found.")
else:
    target = max(refinement_rows, key=lambda r: r["ref_time_s"])
    cidx   = target["cell_idx"]
    print(f"  Target cell: {cidx}  (ref_time={target['ref_time_s']:.4f}s)")

    vertices_t = np.asarray(verts[offsets[cidx]:offsets[cidx+1]], dtype=float)
    sv_t = sv[cidx].ravel()
    p_t  = compute_local_gradient(sv_t, layer_W, W_out)

    with torch.no_grad():
        B_t = model(torch.tensor(vertices_t.astype(np_dtype), dtype=model_dtype)
                    ).numpy().ravel()

    x_stars_t, _, verts_neg, verts_pos = _get_zero_level_set_crossings(
        vertices_t, sv_t, B_t,
        layer_W, layer_b, boundary_H, boundary_b, use_wide,
        barrier_model=model, model_dtype=model_dtype,
    )

    _, M_parent, r_parent, remainder_t, worst_cond_t, _ = _two_step_label(
        x_stars_t, model, dyn, hb, p_t, cidx, True)

    v0   = vertices_t[0]
    B_v0 = float(model(torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
                       ).detach().numpy().ravel()[0])
    q_t    = B_v0 - float(p_t @ v0)
    zls_bH = np.vstack([boundary_H, p_t[None, :]])
    zls_bb = np.append(boundary_b, q_t)
    sub_v  = verts_neg if len(verts_neg) <= len(verts_pos) else verts_pos

    records = _refine_instrumented(
        x_stars_t, worst_cond_t, remainder_t, M_parent, p_t,
        model, dyn, hb, cidx, True, use_wide, 8,
        sub_v, sv_t, layer_W, layer_b,
        zls_bH, zls_bb, TH, model_dtype,
    )
    print(f"  Subcell records: {len(records)}")

    # aggregate per depth level
    from collections import defaultdict
    by_depth = defaultdict(list)
    for depth_level, r_i, M_i in records:
        by_depth[depth_level].append((r_i, M_i))

    depth_rows = [dict(depth=-1, n_subcells=1,
                       r_mean=r_parent, r_std=0.0,
                       M_mean=M_parent, M_std=0.0)]
    for d in sorted(by_depth):
        vals   = by_depth[d]
        r_list = [v[0] for v in vals]
        M_list = [v[1] for v in vals]
        depth_rows.append(dict(
            depth=d, n_subcells=len(vals),
            r_mean=float(np.mean(r_list)), r_std=float(np.std(r_list)),
            M_mean=float(np.mean(M_list)), M_std=float(np.std(M_list)),
        ))

    DEPTH_CSV = "refinement_depth_data.csv"
    with open(DEPTH_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["depth","n_subcells",
                                               "r_mean","r_std","M_mean","M_std"])
        writer.writeheader()
        writer.writerows(depth_rows)
    print(f"  Saved per-depth data to {DEPTH_CSV}  (cell {cidx})")
    for dr in depth_rows:
        print(f"    depth={dr['depth']:2d}  n={dr['n_subcells']:4d}"
              f"  r={dr['r_mean']:.6f}±{dr['r_std']:.6f}"
              f"  M={dr['M_mean']:.6f}±{dr['M_std']:.6f}")
