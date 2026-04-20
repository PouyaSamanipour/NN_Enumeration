"""
test_refinement_figures.py
==========================
Generates two sets of figures comparing adaptive refinement vs dReal:

  Figure 1 — Per-cell wall-clock time: refinement vs dReal
      X: cell index  |  Y: time (s)
      Shown only for cells where Taylor was inconclusive (dReal was called).

  Figure 2 — r_i and M_i reduction across refinement subcells
      For a single cell that required refinement (SAFE_REFINEMENT), the
      instrumented refinement loop records (depth, r_i, M_i) for every
      sub-cell evaluated at each depth level, then plots mean ± std.

Usage
-----
    cd /home/pouya/Codes/NN_Enumeration
    python Tests/test_refinement_figures.py

Requirements
------------
    matplotlib, dReal at /opt/dreal/4.21.06.2/bin/dreal
"""

from __future__ import annotations

import sys, os, subprocess, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import h5py
import sympy as sp
import matplotlib.pyplot as plt
from collections import deque, defaultdict

from relu_region_enumerator.verify_certificate_new import (
    Label,
    DynamicsEvaluator,
    _get_zero_level_set_crossings,
    _two_step_label,
    _refine_barrier_adaptive,
    _find_farthest_pair_nb,
)
from relu_region_enumerator.hessian_bound import HessianBounder, compute_local_gradient
from relu_region_enumerator.Dynamics import load_dynamics
from relu_region_enumerator.bitwise_utils import get_cell_hyperplanes_input_space

from relu_region_enumerator.bitwise_utils import Enumerator_rapid

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PATH    = "NN_files/model_decay_2_10_ren.pt"
BOUNDARY_H5   = "decay_boundary_cells.h5"
TH            = [2.0] * 6
DYNAMICS      = "decay"
DREAL_BIN     = "/opt/dreal/4.21.06.2/bin/dreal"
DREAL_DELTA   = 0.01
DREAL_TIMEOUT = 30
MAX_CELLS     = 3000

# ═══════════════════════════════════════════════════════════════════════════
# SMT2 / dReal helpers  (copied from test_refinement_vs_dreal.py)
# ═══════════════════════════════════════════════════════════════════════════

def _sympy_to_smt2(expr, sym_names: dict) -> str:
    if expr.is_Number:
        return str(float(expr))
    if expr.is_Symbol:
        return sym_names[expr]
    if expr.is_Add:
        args = [_sympy_to_smt2(a, sym_names) for a in expr.args]
        return "(+ " + " ".join(args) + ")"
    if expr.is_Mul:
        args = [_sympy_to_smt2(a, sym_names) for a in expr.args]
        return "(* " + " ".join(args) + ")"
    if expr.is_Pow:
        base, exp = expr.args
        b = _sympy_to_smt2(base, sym_names)
        if exp == 2:
            return f"(* {b} {b})"
        if exp == 3:
            return f"(* {b} (* {b} {b}))"
        if exp == sp.Rational(1, 2):
            return f"(sqrt {b})"
        e = _sympy_to_smt2(exp, sym_names)
        return f"(^ {b} {e})"
    if expr.is_negative:
        return f"(- {_sympy_to_smt2(-expr, sym_names)})"
    func_map = {
        sp.sin: "sin", sp.cos: "cos", sp.tan: "tan",
        sp.exp: "exp", sp.log: "log", sp.sqrt: "sqrt",
        sp.tanh: "tanh", sp.sinh: "sinh", sp.cosh: "cosh",
        sp.Abs: "abs",
    }
    func = expr.func
    arg  = expr.args[0] if len(expr.args) == 1 else None
    if func in func_map and arg is not None:
        return f"({func_map[func]} {_sympy_to_smt2(arg, sym_names)})"
    raise ValueError(f"Cannot convert: {expr}")


def build_smt2(p_i, b_i, H_all, b_all, sv_i, total_neurons, TH,
               symbols, f_sym, delta=0.01) -> str:
    n         = len(symbols)
    var_names = {s: f"x{k+1}" for k, s in enumerate(symbols)}
    lines     = []
    lines.append("(set-logic QF_NRA)")
    lines.append(f"(set-info :precision {delta})")
    for k in range(n):
        lines.append(f"(declare-fun x{k+1} () Real)")
        lines.append(f"(assert (>= x{k+1} {-float(TH[k]):.10f}))")
        lines.append(f"(assert (<= x{k+1} {float(TH[k]):.10f}))")
    zls_terms = [f"(* {float(p_i[k]):.10f} x{k+1})"
                 for k in range(n) if abs(p_i[k]) > 1e-15]
    zls_sum = ("(+ " + " ".join(zls_terms) + ")"
               if len(zls_terms) > 1 else zls_terms[0] if zls_terms else "0.0")
    lines.append(f"(assert (= (+ {zls_sum} {float(b_i):.10f}) 0.0))")
    for row in range(total_neurons):
        terms = [f"(* {float(H_all[row,k]):.10f} x{k+1})"
                 for k in range(n) if abs(H_all[row, k]) > 1e-15]
        if not terms:
            continue
        lhs  = "(+ " + " ".join(terms) + ")" if len(terms) > 1 else terms[0]
        expr = f"(+ {lhs} {float(b_all[row]):.10f})"
        if sv_i[row] == 1:
            lines.append(f"(assert (>= {expr} 0.0))")
        else:
            lines.append(f"(assert (<= {expr} 0.0))")
    for row in range(total_neurons, len(b_all)):
        terms = [f"(* {float(H_all[row,k]):.10f} x{k+1})"
                 for k in range(n) if abs(H_all[row, k]) > 1e-15]
        if not terms:
            continue
        lhs  = "(+ " + " ".join(terms) + ")" if len(terms) > 1 else terms[0]
        expr = f"(+ {lhs} {float(b_all[row]):.10f})"
        lines.append(f"(assert (>= {expr} 0.0))")
    lie_terms = []
    for k, fk in enumerate(f_sym):
        coeff = float(p_i[k])
        if abs(coeff) < 1e-15:
            continue
        lie_terms.append(f"(* {coeff:.10f} {_sympy_to_smt2(fk, var_names)})")
    lie_sum = ("(+ " + " ".join(lie_terms) + ")"
               if len(lie_terms) > 1 else lie_terms[0] if lie_terms else "0.0")
    lines.append(f"(assert (> {lie_sum} 0.0))")
    lines.append("(check-sat)")
    return "\n".join(lines)


def call_dreal(smt2_str: str, delta=0.01, timeout=30, dreal_bin=DREAL_BIN) -> str:
    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.smt2', delete=False, dir='/tmp') as f:
        f.write(smt2_str)
        tmp = f.name
    try:
        result = subprocess.run(
            [dreal_bin, "--precision", str(delta), tmp],
            capture_output=True, text=True, timeout=timeout
        )
        out = (result.stdout + result.stderr).lower()
        if "unsat" in out:
            return "SAFE"
        elif "sat" in out:
            return "UNSAFE"
        return "ERROR"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except FileNotFoundError:
        raise RuntimeError(f"dReal not found at {dreal_bin}")
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def verify_cell_dreal(vertices, sv_i, p_i, layer_W, layer_b,
                      boundary_H, boundary_b, symbols, f_sym,
                      barrier_model, TH, delta=DREAL_DELTA,
                      timeout=DREAL_TIMEOUT):
    """Returns (label, runtime_s)."""
    model_dtype   = next(barrier_model.parameters()).dtype
    np_dtype      = np.float64 if model_dtype == torch.float64 else np.float32
    total_neurons = sum(W.shape[0] for W in layer_W)
    v0 = vertices[0]
    with torch.no_grad():
        B_v0 = float(barrier_model(
            torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
        ).numpy().ravel()[0])
    b_i   = B_v0 - float(p_i @ v0)
    H_all, b_all = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b)
    smt2  = build_smt2(p_i, b_i, H_all, b_all, sv_i, total_neurons,
                       TH, symbols, f_sym, delta=delta)
    t0    = time.perf_counter()
    label = call_dreal(smt2, delta=delta, timeout=timeout)
    return label, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════
# Instrumented refinement — records (depth_level, r_i, M_i) per sub-cell
# ═══════════════════════════════════════════════════════════════════════════

def _refine_instrumented(
    x_stars, worst_cond, remainder, M_i_parent, p_i,
    barrier_model, dyn, hb, cell_idx, continuous_time, _use_wide,
    max_depth, vertices, sv_i, layer_W, layer_b,
    boundary_H, boundary_b, TH, model_dtype,
):
    """
    Mirror of _refine_barrier_adaptive that additionally records
    (depth_used, r_i, M_i) for every sub-cell evaluated by _two_step_label.

    Returns
    -------
    label, M_i, r_i, remainder, ce  — same as the original
    records : list of (depth_used, r_i, M_i)
    """
    np_dtype = np.float64 if model_dtype == torch.float64 else np.float32
    n        = x_stars.shape[1]

    queue = deque()
    queue.append((x_stars, vertices, worst_cond, remainder, max_depth,
                  boundary_H, boundary_b))

    worst_M          = 0.0
    worst_r          = 0.0
    worst_rem        = 0.0
    any_inconclusive = False
    records          = []   # (depth_used, r_i, M_i)

    while queue:
        xs, verts, wc, _rem, depth, bH, bb = queue.popleft()  # noqa: F841

        depth_used = max_depth - depth  # 0 = first split, 1 = second, …

        if depth == 0 or len(xs) < 2:
            any_inconclusive = True
            continue

        idx_a, idx_b, max_dist = _find_farthest_pair_nb(
            np.ascontiguousarray(xs, dtype=np.float64))
        if max_dist < 1e-12:
            any_inconclusive = True
            continue

        normal = (xs[idx_b] - xs[idx_a]) / max_dist
        n_refs = 5

        base_proj = float(xs[idx_a] @ normal)
        step_proj = float((xs[idx_b] - xs[idx_a]) @ normal) / n_refs
        cuts      = [base_proj + k * step_proj for k in range(1, n_refs)]

        H_refine = np.tile(normal, (n_refs - 1, 1)).astype(np.float64)
        b_refine = np.array([-c for c in cuts], dtype=np.float64)

        H_cell, b_cell = get_cell_hyperplanes_input_space(
            sv_i, layer_W, layer_b, bH, bb)
        use_wide_cell = (len(H_cell) + len(H_refine)) > 64
        try:
            sub_cells = Enumerator_rapid(
                H_refine, b_refine,
                [np.asarray(verts, dtype=np.float64)],
                TH, [H_cell.tolist()], [b_cell.tolist()],
                False, None, 0, use_wide_cell,
            )
        except Exception:
            sub_cells = None

        if sub_cells is None or len(sub_cells) <= 1:
            any_inconclusive = True
            continue

        bH_child = np.vstack([bH, H_refine])
        bb_child = np.concatenate([bb, b_refine])

        zls_tol = 1e-3
        for sub_verts in sub_cells:
            sub_verts = np.asarray(sub_verts, dtype=np.float64)
            if len(sub_verts) < n + 1:
                continue

            with torch.no_grad():
                B_sub = barrier_model(
                    torch.tensor(sub_verts.astype(np_dtype), dtype=model_dtype)
                ).numpy().ravel().astype(np.float64)

            xs_k = sub_verts[np.abs(B_sub) < zls_tol]
            if len(xs_k) == 0:
                any_inconclusive = True
                continue

            m_override = M_i_parent if n_refs <= 1 else None
            label, M_i, r_i, rem_i, wc_i, ce = _two_step_label(
                xs_k, barrier_model, dyn, hb, p_i, cell_idx, continuous_time,
                M_i_override=m_override,
            )
            records.append((depth_used, r_i, M_i))
            worst_M   = max(worst_M,   M_i)
            worst_r   = max(worst_r,   r_i)
            worst_rem = max(worst_rem, rem_i)

            if label == Label.UNSAFE:
                return Label.UNSAFE, M_i, r_i, rem_i, ce, records

            if label == Label.INCONCLUSIVE:
                if len(xs_k) >= 2:
                    queue.append((xs_k, sub_verts, wc_i, rem_i, depth - 1,
                                  bH_child, bb_child))
                else:
                    any_inconclusive = True

    final_label = Label.INCONCLUSIVE if any_inconclusive else Label.SAFE_REFINEMENT
    return final_label, worst_M, worst_r, worst_rem, None, records


# ═══════════════════════════════════════════════════════════════════════════
# Per-cell verification with timing
# ═══════════════════════════════════════════════════════════════════════════

def run_per_cell_timed(
    BC, sv, layer_W, layer_b, W_out, boundary_H, boundary_b,
    barrier_model, dyn, hb, TH, model_dtype, use_wide,
    refinement_max_depth=8, continuous_time=True,
):
    """
    Run the same per-cell logic as verify_barrier but record wall-clock
    time for each cell and which path it took.

    Returns
    -------
    results : list of dict with keys:
        cell_idx, label, time_s, M_i, r_i, remainder
        needed_refinement (bool)
    """
    np_dtype = np.float64 if model_dtype == torch.float64 else np.float32
    results  = []

    for i, vertices in enumerate(BC):
        vertices = np.asarray(vertices, dtype=float)
        sv_i     = sv[i].ravel()
        p_i      = compute_local_gradient(sv_i, layer_W, W_out)

        t0 = time.perf_counter()

        with torch.no_grad():
            B_vals = barrier_model(
                torch.tensor(vertices.astype(np_dtype), dtype=model_dtype)
            ).numpy().ravel()

        x_stars, _x_masks, verts_B_neg, verts_B_pos = _get_zero_level_set_crossings(
            vertices, sv_i, B_vals,
            layer_W, layer_b, boundary_H, boundary_b, use_wide,
            barrier_model=barrier_model,
            model_dtype=model_dtype,
        )

        if len(x_stars) == 0:
            elapsed = time.perf_counter() - t0
            results.append(dict(cell_idx=i, label=Label.SAFE_TAYLOR,
                                time_s=elapsed, M_i=0.0, r_i=0.0,
                                remainder=0.0, needed_refinement=False))
            continue

        label, M_i, r_i, remainder, worst_cond, _ce = _two_step_label(
            x_stars, barrier_model, dyn, hb, p_i, i, continuous_time)

        needed_refinement = False
        if label == Label.INCONCLUSIVE:
            needed_refinement = True
            v0 = vertices[0]
            with torch.no_grad():
                B_v0 = float(barrier_model(
                    torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
                ).numpy().ravel()[0])
            q_i = B_v0 - float(p_i @ v0)
            zls_boundary_H = np.vstack([boundary_H, p_i[None, :]])
            zls_boundary_b = np.append(boundary_b, q_i)
            sub_verts_for_refine = (
                verts_B_neg if len(verts_B_neg) <= len(verts_B_pos) else verts_B_pos)
            label, M_i, r_i, remainder, _ce2 = _refine_barrier_adaptive(
                x_stars, worst_cond, remainder, M_i, p_i,
                barrier_model, dyn, hb, i, continuous_time, use_wide,
                refinement_max_depth, sub_verts_for_refine, sv_i,
                layer_W, layer_b, zls_boundary_H, zls_boundary_b, TH, model_dtype,
            )

        elapsed = time.perf_counter() - t0
        results.append(dict(cell_idx=i, label=label,
                            time_s=elapsed, M_i=M_i, r_i=r_i,
                            remainder=remainder,
                            needed_refinement=needed_refinement))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _show(fig, name):
    fig.canvas.manager.set_window_title(name)
    plt.show(block=False)
    plt.pause(0.1)


def plot_time_comparison(ref_times, dreal_times, _cell_indices, out_name="fig_time_comparison.png"):
    """
    Left  — dot plot: refinement time vs dReal time per cell (plot markers only).
    Right — empirical CDF using plot() only.
    All drawn with ax.plot() / ax.fill_between() — safe on mpl 3.8 + numpy 2.x.
    """
    ref = np.asarray(ref_times,   dtype=float)
    dr  = np.asarray(dreal_times, dtype=float)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: dot plot (plot with marker, no PathCollection) ─────────────
    ax.plot(ref.tolist(), dr.tolist(),
            linestyle="none", marker="o", markersize=4,
            markerfacecolor="steelblue", markeredgewidth=0, alpha=0.6,
            label="cell")
    lim = max(float(ref.max()), float(dr.max())) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=0.9,
            label="equal time")
    ax.set_xlabel("Refinement time (s)", fontsize=11)
    ax.set_ylabel("dReal time (s)", fontsize=11)
    ax.set_title("Per-cell time: Refinement vs dReal", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # ── Right: empirical CDF with plot() + staircase interpolation ───────
    for vals, color, lbl in [
        (ref, "steelblue", "Refinement"),
        (dr,  "tomato",    "dReal"),
    ]:
        sv  = np.sort(vals)
        cdf = np.arange(1, len(sv) + 1, dtype=float) / len(sv)
        # staircase: repeat each x value so the CDF looks like steps
        xs = np.repeat(sv, 2)[1:]
        ys = np.repeat(cdf, 2)[:-1]
        ax2.plot(xs.tolist(), ys.tolist(), color=color, linewidth=2, label=lbl)
    ax2.set_xlabel("Wall-clock time (s)", fontsize=11)
    ax2.set_ylabel("Empirical CDF", fontsize=11)
    ax2.set_title("CDF of per-cell time", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 1.02)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.13, top=0.92, wspace=0.32)
    _show(fig, out_name)

    print(f"\n  Time summary (cells where dReal was called: {len(ref)})")
    print(f"    Refinement — mean={ref.mean():.4f}s  median={np.median(ref):.4f}s  max={ref.max():.4f}s")
    print(f"    dReal      — mean={dr.mean():.4f}s  median={np.median(dr):.4f}s  max={dr.max():.4f}s")
    faster = int((dr > ref).sum())
    print(f"    Refinement faster than dReal in {faster}/{len(ref)} cells ({100*faster/len(ref):.1f}%)")


def plot_radius_Mi_reduction(records_original, records_by_depth,
                             original_r, original_M,
                             out_name="fig_radius_Mi_reduction.png"):
    """
    Two-panel figure showing r_i and M_i across refinement depth levels.

    Parameters
    ----------
    records_original : list of (depth, r_i, M_i) from _refine_instrumented
    records_by_depth : dict depth → list of (r_i, M_i) — pre-aggregated
    original_r, original_M : values for the parent cell (depth = -1)
    """
    depths_present = sorted(records_by_depth.keys())
    depth_labels   = ["parent"] + [f"depth {d}" for d in depths_present]

    r_mean = [original_r] + [np.mean([v[0] for v in records_by_depth[d]])
                              for d in depths_present]
    r_std  = [0.0] + [np.std([v[0] for v in records_by_depth[d]])
                      for d in depths_present]
    M_mean = [original_M] + [np.mean([v[1] for v in records_by_depth[d]])
                              for d in depths_present]
    M_std  = [0.0] + [np.std([v[1] for v in records_by_depth[d]])
                      for d in depths_present]

    x      = list(range(len(depth_labels)))
    r_mean = [float(v) for v in r_mean]
    r_std  = [float(v) for v in r_std]
    M_mean = [float(v) for v in M_mean]
    M_std  = [float(v) for v in M_std]
    r_lo   = [r_mean[i] - r_std[i] for i in range(len(x))]
    r_hi   = [r_mean[i] + r_std[i] for i in range(len(x))]
    M_lo   = [M_mean[i] - M_std[i] for i in range(len(x))]
    M_hi   = [M_mean[i] + M_std[i] for i in range(len(x))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── r_i ──────────────────────────────────────────────────────────────
    ax1.fill_between(x, r_lo, r_hi, color="steelblue", alpha=0.25)
    ax1.plot(x, r_mean, color="steelblue", linewidth=2,
             linestyle="-", marker="o", markersize=7, label="mean +/- std")
    # individual subcell dots via plot markers
    pts_x = [float(v[0]) + 1.0 for v in records_original]   # +1: parent is x=0
    pts_r = [float(v[1]) for v in records_original]
    pts_M = [float(v[2]) for v in records_original]
    ax1.plot(pts_x, pts_r, linestyle="none", marker="o", markersize=4,
             markerfacecolor="steelblue", markeredgewidth=0, alpha=0.35,
             label="subcells")
    ax1.set_xticks(x)
    ax1.set_xticklabels(depth_labels, fontsize=9)
    ax1.set_ylabel("r_i  (cell radius)", fontsize=11)
    ax1.set_title("Cell radius r_i across refinement", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_ylim(bottom=0)

    # ── M_i ──────────────────────────────────────────────────────────────
    ax2.fill_between(x, M_lo, M_hi, color="tomato", alpha=0.25)
    ax2.plot(x, M_mean, color="tomato", linewidth=2,
             linestyle="-", marker="o", markersize=7, label="mean +/- std")
    ax2.plot(pts_x, pts_M, linestyle="none", marker="o", markersize=4,
             markerfacecolor="tomato", markeredgewidth=0, alpha=0.35,
             label="subcells")
    ax2.set_xticks(x)
    ax2.set_xticklabels(depth_labels, fontsize=9)
    ax2.set_ylabel("M_i  (Hessian bound)", fontsize=11)
    ax2.set_title("Hessian bound M_i across refinement", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_ylim(bottom=0)

    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.13, top=0.92, wspace=0.35)
    _show(fig, out_name)

    print(f"\n  r_i   parent={original_r:.6f}  →  depth-{depths_present[-1]} mean={r_mean[-1]:.6f}"
          f"  (reduction {100*(1-r_mean[-1]/original_r):.1f}%)")
    print(f"  M_i   parent={original_M:.6f}  →  depth-{depths_present[-1]} mean={M_mean[-1]:.6f}"
          f"  (reduction {100*(1-M_mean[-1]/max(original_M,1e-30)):.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("Loading model ...")
    model   = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    params  = [p.detach().numpy() for _, p in model.named_parameters()]
    layer_W = [params[0], params[2]]
    layer_b = [params[1], params[3]]
    W_out   = params[4]
    n       = layer_W[0].shape[1]
    model_dtype = next(model.parameters()).dtype
    np_dtype    = np.float64 if model_dtype == torch.float64 else np.float32

    boundary_H = np.vstack((np.eye(n), -np.eye(n)))
    boundary_b = np.array(TH * 2, dtype=np.float64)

    total_neurons = sum(W.shape[0] for W in layer_W)
    use_wide      = (total_neurons + len(boundary_H)) > 64

    print("Loading boundary cells ...")
    with h5py.File(BOUNDARY_H5, "r") as f:
        offsets = f["offsets"][:]
        verts   = f["vertices"][:]
        sv      = f["activation_patterns"][:]
    BC = [verts[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
    BC = BC[:MAX_CELLS]
    sv = sv[:MAX_CELLS]
    print(f"  {len(BC)} cells, {n}D")

    print("Loading dynamics ...")
    symbols, f_sym = load_dynamics(DYNAMICS)
    dyn = DynamicsEvaluator(symbols, f_sym)

    # Dynamics linearity check (same as in verify_barrier)
    _is_linear = all(
        sp.diff(fi, xj, xk) == 0
        for fi in f_sym
        for xj in symbols
        for xk in symbols
    )

    hb = None if _is_linear else HessianBounder(symbols, f_sym)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1 — Per-cell refinement with timing
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Per-cell refinement (timed)")
    print("="*60)
    # Keep a fresh copy of BC since run_per_cell_timed does NOT free them
    BC_copy = [np.asarray(verts[offsets[i]:offsets[i+1]]) for i in range(len(offsets)-1)]
    BC_copy = BC_copy[:MAX_CELLS]

    ref_results = run_per_cell_timed(
        BC_copy, sv, layer_W, layer_b, W_out,
        boundary_H, boundary_b, model, dyn, hb, TH,
        model_dtype, use_wide,
        refinement_max_depth=8, continuous_time=True,
    )
    ref_time_map  = {r["cell_idx"]: r["time_s"] for r in ref_results}
    ref_label_map = {r["cell_idx"]: r["label"]  for r in ref_results}
    refinement_cells = [r["cell_idx"] for r in ref_results if r["needed_refinement"]]
    print(f"  Cells needing refinement : {len(refinement_cells)}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2 — Taylor + dReal (for cells that are inconclusive from Taylor)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — Taylor pre-filter then dReal on inconclusive cells")
    print("="*60)

    BC_fresh = [np.asarray(verts[offsets[i]:offsets[i+1]]) for i in range(len(offsets)-1)]
    BC_fresh = BC_fresh[:MAX_CELLS]

    hb_dr = None if _is_linear else HessianBounder(symbols, f_sym)
    dreal_time_map   = {}
    dreal_label_map  = {}
    inconclusive_set = set()

    for i, vertices_i in enumerate(BC_fresh):
        vertices_i = np.asarray(vertices_i, dtype=float)
        sv_i       = sv[i].ravel()
        p_i        = compute_local_gradient(sv_i, layer_W, W_out)

        with torch.no_grad():
            B_vals = model(
                torch.tensor(vertices_i.astype(np_dtype), dtype=model_dtype)
            ).numpy().ravel()

        x_stars, _x_masks, verts_B_neg, verts_B_pos = _get_zero_level_set_crossings(
            vertices_i, sv_i, B_vals,
            layer_W, layer_b, boundary_H, boundary_b, use_wide,
            barrier_model=model, model_dtype=model_dtype,
        )
        if len(x_stars) == 0:
            dreal_label_map[i] = "SAFE"
            dreal_time_map[i]  = 0.0
            continue

        t_label, M_i, r_i, remainder, worst_cond, ce = _two_step_label(
            x_stars, model, dyn, hb_dr, p_i, i, continuous_time=True)

        if t_label == Label.SAFE_TAYLOR:
            dreal_label_map[i] = "SAFE"
            dreal_time_map[i]  = 0.0
        elif t_label == Label.UNSAFE:
            dreal_label_map[i] = "UNSAFE"
            dreal_time_map[i]  = 0.0
        else:
            inconclusive_set.add(i)
            dr_label, dr_rt = verify_cell_dreal(
                vertices_i, sv_i, p_i, layer_W, layer_b,
                boundary_H, boundary_b, symbols, f_sym, model, TH=TH,
            )
            dreal_label_map[i] = dr_label
            dreal_time_map[i]  = dr_rt
            print(f"  cell {i:4d}  dReal={dr_label:<8s}  time={dr_rt:.3f}s  "
                  f"ref={ref_label_map.get(i,'?')}")

    print(f"\n  Taylor-inconclusive cells sent to dReal: {len(inconclusive_set)}")

    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 1 — Time comparison for cells where dReal was actually called
    # ─────────────────────────────────────────────────────────────────────
    shared_inconclusive = sorted(
        inconclusive_set & set(ref_time_map.keys())
    )
    if shared_inconclusive:
        print("\n" + "="*60)
        print("FIGURE 1 — Time comparison")
        print("="*60)
        ref_t  = [ref_time_map[i] for i in shared_inconclusive]
        dr_t   = [dreal_time_map[i] for i in shared_inconclusive]
        plot_time_comparison(ref_t, dr_t, shared_inconclusive,
                             out_name="fig_time_comparison.png")
    else:
        print("\n  [Figure 1 skipped] No cells needed dReal.")

    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 2 — r_i / M_i reduction for one refinement cell
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FIGURE 2 — r_i / M_i reduction over refinement")
    print("="*60)

    # Pick the first cell that (a) needed refinement and (b) was resolved
    # as SAFE_REFINEMENT (not just INCONCLUSIVE).
    target_cell = None
    for cidx in refinement_cells:
        if ref_label_map.get(cidx) == Label.SAFE_REFINEMENT:
            target_cell = cidx
            break

    if target_cell is None and refinement_cells:
        target_cell = refinement_cells[0]
        print(f"  Warning: no SAFE_REFINEMENT cell found; using first refinement cell {target_cell}")

    if target_cell is None:
        print("  [Figure 2 skipped] No cell needed refinement.")
    else:
        print(f"  Instrumenting cell {target_cell} …")
        vertices_t = np.asarray(
            verts[offsets[target_cell]:offsets[target_cell+1]], dtype=float)
        sv_t = sv[target_cell].ravel()
        p_t  = compute_local_gradient(sv_t, layer_W, W_out)

        with torch.no_grad():
            B_t = model(
                torch.tensor(vertices_t.astype(np_dtype), dtype=model_dtype)
            ).numpy().ravel()

        x_stars_t, _, verts_neg, verts_pos = _get_zero_level_set_crossings(
            vertices_t, sv_t, B_t,
            layer_W, layer_b, boundary_H, boundary_b, use_wide,
            barrier_model=model, model_dtype=model_dtype,
        )

        _, M_parent, r_parent, _, _, _ = _two_step_label(
            x_stars_t, model, dyn, hb, p_t, target_cell, True)

        print(f"    Parent cell  r_i={r_parent:.6f}  M_i={M_parent:.6f}")

        v0 = vertices_t[0]
        with torch.no_grad():
            B_v0 = float(model(
                torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
            ).numpy().ravel()[0])
        q_t              = B_v0 - float(p_t @ v0)
        zls_bH           = np.vstack([boundary_H, p_t[None, :]])
        zls_bb           = np.append(boundary_b, q_t)
        sub_verts_refine = verts_neg if len(verts_neg) <= len(verts_pos) else verts_pos

        # Placeholder worst_cond — same as in verify_barrier
        _, _, _, remainder_t, worst_cond_t, _ = _two_step_label(
            x_stars_t, model, dyn, hb, p_t, target_cell, True)

        _, _, _, _, _, records = _refine_instrumented(
            x_stars_t, worst_cond_t, remainder_t, M_parent, p_t,
            model, dyn, hb, target_cell, True, use_wide, 8,
            sub_verts_refine, sv_t, layer_W, layer_b,
            zls_bH, zls_bb, TH, model_dtype,
        )

        print(f"    Subcell records collected: {len(records)}")
        if records:
            records_by_depth = defaultdict(list)
            for depth_used, r_i, M_i in records:
                records_by_depth[depth_used].append((r_i, M_i))
            plot_radius_Mi_reduction(
                records, records_by_depth,
                r_parent, M_parent,
                out_name="fig_radius_Mi_reduction.png",
            )
        else:
            print("  [Figure 2 skipped] Instrumented refinement returned no subcell records.")

    print("\nDone. Close the figure windows to exit.")
    plt.show()
