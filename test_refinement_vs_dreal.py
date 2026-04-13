"""
test_refinement_vs_dreal.py
===========================
Validates the adaptive refinement against dReal as ground truth on Decay 6D
boundary cells.

dReal is called via WSL subprocess — no Python binding needed.
It checks per cell: does there exist x in X(C) with B(x)=0 and p_i.f(x) > 0?
  - UNSAT     -> SAFE   (no violation exists)
  - delta-SAT -> UNSAFE (violation found within precision delta)

Convention for get_cell_hyperplanes_input_space output:
  Network rows : H[row]@x + b[row] >= 0  if sv_i[row]=1 (active)
                 H[row]@x + b[row] <= 0  if sv_i[row]=0 (inactive)
  Domain rows  : H[row]@x + b[row] >= 0  always

Usage
-----
    python test_refinement_vs_dreal.py

Requirements
------------
    dReal installed at /opt/dreal/4.21.06.2/bin/dreal  (WSL Ubuntu 22.04)
"""

from __future__ import annotations

import sys, os, subprocess, tempfile, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import h5py
import sympy as sp

from relu_region_enumerator.verify_certificate_new import (
    verify_barrier as verify_ref,
    Label as LabelRef,
    DynamicsEvaluator,
)
from relu_region_enumerator.hessian_bound import compute_local_gradient
from relu_region_enumerator.Dynamics import load_dynamics
from relu_region_enumerator.bitwise_utils import get_cell_hyperplanes_input_space

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
# SMT2 helpers
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


def build_smt2(
    p_i          : np.ndarray,
    b_i          : float,
    H_all        : np.ndarray,
    b_all        : np.ndarray,
    sv_i         : np.ndarray,
    total_neurons: int,
    TH           : list,
    symbols      : list,
    f_sym        : list,
    delta        : float = 0.01,
) -> str:
    """
    Build SMT2 for dReal:
        exists x: B(x)=0  AND  x in X(C)  AND  p_i.f(x) > 0

    Uses full cell polytope (activation constraints + domain bounds).
    UNSAT -> SAFE, delta-SAT -> UNSAFE.
    """
    n         = len(symbols)
    var_names = {s: f"x{k+1}" for k, s in enumerate(symbols)}
    lines     = []

    lines.append("(set-logic QF_NRA)")
    lines.append(f"(set-info :precision {delta})")

    # Variables with full domain bounds
    for k in range(n):
        lines.append(f"(declare-fun x{k+1} () Real)")
        lines.append(f"(assert (>= x{k+1} {-float(TH[k]):.10f}))")
        lines.append(f"(assert (<= x{k+1} {float(TH[k]):.10f}))")

    # ZLS equality: p_i . x + b_i = 0
    zls_terms = [f"(* {float(p_i[k]):.10f} x{k+1})"
                 for k in range(n) if abs(p_i[k]) > 1e-15]
    zls_sum = ("(+ " + " ".join(zls_terms) + ")"
               if len(zls_terms) > 1 else zls_terms[0] if zls_terms else "0.0")
    lines.append(f"(assert (= (+ {zls_sum} {float(b_i):.10f}) 0.0))")

    # Network activation constraints
    # Active  (sv=1): H[row]@x + b[row] >= 0
    # Inactive(sv=0): H[row]@x + b[row] <= 0
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

    # Domain boundary constraints: H@x + b >= 0
    for row in range(total_neurons, len(b_all)):
        terms = [f"(* {float(H_all[row,k]):.10f} x{k+1})"
                 for k in range(n) if abs(H_all[row, k]) > 1e-15]
        if not terms:
            continue
        lhs  = "(+ " + " ".join(terms) + ")" if len(terms) > 1 else terms[0]
        expr = f"(+ {lhs} {float(b_all[row]):.10f})"
        lines.append(f"(assert (>= {expr} 0.0))")

    # Violation: p_i . f(x) > 0
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


# ═══════════════════════════════════════════════════════════════════════════
# dReal caller
# ═══════════════════════════════════════════════════════════════════════════

def call_dreal(smt2_str: str, delta: float = 0.01,
               timeout: int = 30, dreal_bin: str = DREAL_BIN) -> str:
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.smt2', delete=False, dir='/tmp'
    ) as f:
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
        else:
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


# ═══════════════════════════════════════════════════════════════════════════
# Per-cell dReal verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_cell_dreal(
    vertices      : np.ndarray,
    sv_i          : np.ndarray,
    p_i           : np.ndarray,
    layer_W       : list,
    layer_b       : list,
    boundary_H    : np.ndarray,
    boundary_b    : np.ndarray,
    symbols       : list,
    f_sym         : list,
    barrier_model,
    TH            : list,
    delta         : float = DREAL_DELTA,
    timeout       : int   = DREAL_TIMEOUT,
) -> tuple:
    """Returns (label, runtime_s). label: SAFE/UNSAFE/TIMEOUT/ERROR"""
    model_dtype   = next(barrier_model.parameters()).dtype
    np_dtype      = np.float64 if model_dtype == torch.float64 else np.float32
    total_neurons = sum(W.shape[0] for W in layer_W)

    # b_i from first vertex
    v0 = vertices[0]
    with torch.no_grad():
        B_v0 = float(barrier_model(
            torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
        ).numpy().ravel()[0])
    b_i = B_v0 - float(p_i @ v0)

    # Full cell hyperplanes
    H_all, b_all = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b
    )

    smt2 = build_smt2(
        p_i, b_i, H_all, b_all, sv_i,
        total_neurons, TH,
        symbols, f_sym, delta=delta
    )

    t0    = time.perf_counter()
    label = call_dreal(smt2, delta=delta, timeout=timeout)
    rt    = time.perf_counter() - t0
    return label, rt


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

    boundary_H = np.vstack((np.eye(n), -np.eye(n)))
    boundary_b = np.array(TH * 2, dtype=np.float64)

    print("Loading boundary cells ...")
    with h5py.File(BOUNDARY_H5, "r") as f:
        offsets = f["offsets"][:]
        verts   = f["vertices"][:]
        sv      = f["activation_patterns"][:]
    BC = [verts[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
    BC = BC[:MAX_CELLS]
    sv = sv[:MAX_CELLS]
    print(f"  {len(BC)} cells, {n}D\n")

    print("Loading dynamics ...")
    symbols, f_sym = load_dynamics(DYNAMICS)
    dyn = DynamicsEvaluator(symbols, f_sym)

    model_dtype = next(model.parameters()).dtype
    np_dtype    = np.float64 if model_dtype == torch.float64 else np.float32

    # ── Run refinement ────────────────────────────────────────────────────
    print("=" * 60)
    print("RUN 1 — Adaptive refinement")
    print("=" * 60)
    summary_ref = verify_ref(
        BC, sv, layer_W, layer_b, W_out,
        boundary_H, boundary_b, model,
        dynamics_name=DYNAMICS,
        continuous_time=True,
        early_exit=False,
        refinement_max_depth=6,
        TH=TH,
    )

    # ── Run dReal per cell ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RUN 2 — dReal ground truth (full polytope constraints)")
    print("=" * 60)

    ref_map       = {r.cell_idx: r.label for r in summary_ref.results}
    dreal_map     = {}
    dreal_time_map= {}
    n_safe_dr     = 0
    n_unsafe_dr   = 0
    n_timeout_dr  = 0
    n_error_dr    = 0
    total_dr_time = 0.0

    for i, vertices in enumerate(BC):
        vertices = np.asarray(vertices, dtype=float)
        sv_i     = sv[i].ravel()
        p_i      = compute_local_gradient(sv_i, layer_W, W_out)

        # Check if cell crosses B=0
        with torch.no_grad():
            B_vals = model(
                torch.tensor(vertices.astype(np_dtype), dtype=model_dtype)
            ).numpy().ravel()

        if B_vals.min() >= -1e-9 or B_vals.max() <= 1e-9:
            dreal_map[i]      = "SAFE"
            dreal_time_map[i] = 0.0
            n_safe_dr        += 1
            continue

        dr_label, dr_rt = verify_cell_dreal(
            vertices, sv_i, p_i,
            layer_W, layer_b,
            boundary_H, boundary_b,
            symbols, f_sym, model,
            TH=TH,
            delta=DREAL_DELTA,
            timeout=DREAL_TIMEOUT,
        )

        dreal_map[i]      = dr_label
        dreal_time_map[i] = dr_rt
        total_dr_time    += dr_rt

        if   dr_label == "SAFE":    n_safe_dr    += 1
        elif dr_label == "UNSAFE":  n_unsafe_dr  += 1
        elif dr_label == "TIMEOUT": n_timeout_dr += 1
        else:                       n_error_dr   += 1

        print(f"  cell {i:4d}  dReal={dr_label:<8s}  "
              f"ref={ref_map.get(i,'?').value:<22s}  "
              f"time={dr_rt:.3f}s")

    print(f"\n  dReal: SAFE={n_safe_dr}  UNSAFE={n_unsafe_dr}  "
          f"TIMEOUT={n_timeout_dr}  ERROR={n_error_dr}")
    print(f"  Total dReal time: {total_dr_time:.1f}s")

    # ── Comparison ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON: Refinement vs dReal")
    print("=" * 60)

    def coarse(label):
        v = label.value
        if "SAFE"   in v: return "SAFE"
        if "UNSAFE" in v: return "UNSAFE"
        return "INCONCLUSIVE"

    conflicts     = []
    agreements    = []
    disagreements = []
    skipped       = []

    for idx in sorted(set(ref_map) & set(dreal_map)):
        lr = ref_map[idx]
        ld = dreal_map[idx]
        if ld in ("TIMEOUT", "ERROR"):
            skipped.append((idx, lr.value, ld))
            continue
        cr = coarse(lr)
        if (cr == "SAFE" and ld == "UNSAFE") or (cr == "UNSAFE" and ld == "SAFE"):
            conflicts.append((idx, lr.value, ld))
        elif cr == ld:
            agreements.append((idx, lr.value, ld))
        else:
            disagreements.append((idx, lr.value, ld))

    print(f"\n  Cells compared        : {len(agreements)+len(disagreements)+len(conflicts)}")
    print(f"  Full agreements       : {len(agreements)}")
    print(f"  Disagreements         : {len(disagreements)}")
    print(f"  CONFLICTS (soundness) : {len(conflicts)}  <- must be 0")
    print(f"  TIMEOUT/ERROR         : {len(skipped)}")

    if conflicts:
        print("\n  !! CONFLICTS:")
        for idx, lr, ld in conflicts:
            print(f"    cell {idx:5d}  ref={lr:<25s}  dReal={ld}")
    else:
        print("\n  No conflicts — refinement is sound with respect to dReal.")

    if disagreements:
        print(f"\n  Disagreements (ref found violation, dReal missed):")
        for idx, lr, ld in disagreements[:10]:
            print(f"    cell {idx:5d}  ref={lr:<25s}  dReal={ld}")

    print(f"\n{'='*60}")
    print(f"{'Metric':<40} {'Refinement':>12} {'dReal':>8}")
    print(f"{'='*60}")
    print(f"{'SAFE (Taylor)':<40} {summary_ref.n_safe_taylor:>12} {'—':>8}")
    print(f"{'SAFE (Refinement)':<40} {summary_ref.n_safe_refinement:>12} {n_safe_dr:>8}")
    print(f"{'SAFE total':<40} {summary_ref.n_safe:>12} {n_safe_dr:>8}")
    print(f"{'UNSAFE':<40} {summary_ref.n_unsafe:>12} {n_unsafe_dr:>8}")
    print(f"{'INCONCLUSIVE':<40} {summary_ref.n_inconclusive:>12} {'—':>8}")
    print(f"{'Runtime (s)':<40} {summary_ref.runtime_s:>12.2f} {total_dr_time:>8.2f}")
    print(f"{'='*60}")

    print()
    if len(conflicts) == 0:
        print("CONCLUSION: Adaptive refinement is SOUND with respect to dReal.")
        if len(disagreements) > 0:
            print(f"  {len(disagreements)} cells: refinement found violation, "
                  f"dReal missed — refinement more sensitive for falsification.")
    else:
        print(f"CONCLUSION: {len(conflicts)} SOUNDNESS VIOLATIONS FOUND.")