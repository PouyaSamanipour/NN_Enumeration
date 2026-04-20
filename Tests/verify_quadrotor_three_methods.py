"""
verify_quadrotor_three_methods.py
==================================
Verify the quadrotor barrier certificate B(x) = 0.3 - V(x) using three methods:

  Method 1 — Taylor + adaptive refinement  (full run, no early exit)
  Method 2 — Taylor first; dReal only for INCONCLUSIVE cells
  Method 3 — Pure dReal  (no Taylor pre-filter)

Prints a comparison table at the end.

Requirements
------------
  - NN_files/model_quadrotor_lyapunov_B03.pt   (shifted barrier model)
  - quadrotor_boundary_cells.h5                (from enumeration run)
  - dReal at DREAL_BIN path
"""

from __future__ import annotations

import sys, os, subprocess, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import h5py
import sympy as sp

from relu_region_enumerator.verify_certificate_new import (
    verify_barrier as verify_ref,
    Label,
    DynamicsEvaluator,
    _get_zero_level_set_crossings,
    _two_step_label,
)
from relu_region_enumerator.hessian_bound import HessianBounder, compute_local_gradient
from relu_region_enumerator.Dynamics import load_dynamics
from relu_region_enumerator.bitwise_utils import get_cell_hyperplanes_input_space

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PATH    = "NN_files/model_quadrotor_lyapunov_ct.pt"
BOUNDARY_H5   = "quadrotor_boundary_cells.h5"
TH            = [1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
DYNAMICS      = "quadrotor"
DREAL_BIN     = "/opt/dreal/4.21.06.2/bin/dreal"
DREAL_DELTA        = 0.001   # fine delta — SAT results remain sound counterexamples
DREAL_DELTA_COARSE = 0.01    # fast first-pass (UNSAT only is trusted)
DREAL_TIMEOUT      = 300     # 5 minutes per cell
DREAL_TIMEOUT_FAST = 60      # first-pass timeout for coarse delta
MAX_CELLS     = None   # set to int to limit (None = all cells)


# ═══════════════════════════════════════════════════════════════════════════
# SMT2 helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sympy_to_smt2(expr, sym_names: dict) -> str:
    if expr.is_Number:
        return repr(float(expr))
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
        if exp == -1:                            # division: 1/base
            return f"(/ 1.0 {b})"
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
    raise ValueError(f"Cannot convert sympy expr to SMT2: {expr}")


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
    Check:  exists x  s.t.  B(x)=0  AND  x in X(C)  AND  p_i.f(x) > 0
    UNSAT     -> SAFE
    delta-SAT -> UNSAFE
    """
    n         = len(symbols)
    var_names = {s: f"x{k+1}" for k, s in enumerate(symbols)}
    lines     = ["(set-logic QF_NRA)", f"(set-info :precision {delta})"]

    for k in range(n):
        lines.append(f"(declare-fun x{k+1} () Real)")
        lines.append(f"(assert (>= x{k+1} {-float(TH[k]):.10f}))")
        lines.append(f"(assert (<= x{k+1}  {float(TH[k]):.10f}))")

    # ZLS equality:  p_i . x + b_i = 0
    zls_terms = [f"(* {float(p_i[k]):.10f} x{k+1})"
                 for k in range(n) if abs(p_i[k]) > 1e-15]
    zls_sum   = ("(+ " + " ".join(zls_terms) + ")"
                 if len(zls_terms) > 1 else (zls_terms[0] if zls_terms else "0.0"))
    lines.append(f"(assert (= (+ {zls_sum} {float(b_i):.10f}) 0.0))")

    # Activation constraints
    for row in range(total_neurons):
        terms = [f"(* {float(H_all[row,k]):.10f} x{k+1})"
                 for k in range(n) if abs(H_all[row, k]) > 1e-15]
        if not terms:
            continue
        lhs  = "(+ " + " ".join(terms) + ")" if len(terms) > 1 else terms[0]
        expr = f"(+ {lhs} {float(b_all[row]):.10f})"
        op   = ">=" if sv_i[row] == 1 else "<="
        lines.append(f"(assert ({op} {expr} 0.0))")

    # Domain boundary constraints
    for row in range(total_neurons, len(b_all)):
        terms = [f"(* {float(H_all[row,k]):.10f} x{k+1})"
                 for k in range(n) if abs(H_all[row, k]) > 1e-15]
        if not terms:
            continue
        lhs  = "(+ " + " ".join(terms) + ")" if len(terms) > 1 else terms[0]
        expr = f"(+ {lhs} {float(b_all[row]):.10f})"
        lines.append(f"(assert (>= {expr} 0.0))")

    # Violation:  p_i . f(x) > 0
    lie_terms = []
    for k, fk in enumerate(f_sym):
        coeff = float(p_i[k])
        if abs(coeff) < 1e-15:
            continue
        lie_terms.append(f"(* {coeff:.10f} {_sympy_to_smt2(fk, var_names)})")
    lie_sum = ("(+ " + " ".join(lie_terms) + ")"
               if len(lie_terms) > 1 else (lie_terms[0] if lie_terms else "0.0"))
    lines.append(f"(assert (> {lie_sum} 0.0))")
    lines.append("(check-sat)")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# dReal runner
# ═══════════════════════════════════════════════════════════════════════════

def _run_dreal_once(smt2_str: str, delta: float, timeout: int) -> str:
    """Single dReal invocation. Returns SAFE / UNSAFE / TIMEOUT / ERROR."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.smt2', delete=False, dir='/tmp'
    ) as f:
        f.write(smt2_str)
        tmp = f.name
    try:
        result = subprocess.run(
            [DREAL_BIN, "--precision", str(delta), tmp],
            capture_output=True, text=True, timeout=timeout
        )
        out = (result.stdout + result.stderr).lower()
        if   "unsat" in out: return "SAFE"
        elif "sat"   in out: return "UNSAFE"
        else:                return "ERROR"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except FileNotFoundError:
        raise RuntimeError(f"dReal not found at {DREAL_BIN}")
    finally:
        try: os.unlink(tmp)
        except Exception: pass


def call_dreal(smt2_str: str, delta: float = DREAL_DELTA,
               timeout: int = DREAL_TIMEOUT) -> str:
    """Two-pass dReal call.

    Pass 1 — coarse delta (fast): if UNSAT, return SAFE immediately (sound).
             SAT with coarse delta is NOT trusted (delta-witness ≠ true CE).
    Pass 2 — fine delta (slow):   both SAT and UNSAT are meaningful.
    """
    # Pass 1: quick check with coarse delta — only trust UNSAT
    quick = _run_dreal_once(smt2_str, DREAL_DELTA_COARSE, DREAL_TIMEOUT_FAST)
    if quick == "SAFE":
        return "SAFE"

    # Pass 2: fine delta with full timeout
    return _run_dreal_once(smt2_str, delta, timeout)


def verify_cell_dreal(vertices, sv_i, p_i, layer_W, layer_b,
                      boundary_H, boundary_b, symbols, f_sym,
                      barrier_model, TH) -> tuple[str, float]:
    """Returns (label, runtime_s).  label in {SAFE, UNSAFE, TIMEOUT, ERROR}."""
    model_dtype   = next(barrier_model.parameters()).dtype
    np_dtype      = np.float64 if model_dtype == torch.float64 else np.float32
    total_neurons = sum(W.shape[0] for W in layer_W)

    v0 = vertices[0]
    with torch.no_grad():
        B_v0 = float(barrier_model(
            torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
        ).numpy().ravel()[0])
    b_i = B_v0 - float(p_i @ v0)

    H_all, b_all = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b
    )
    smt2  = build_smt2(p_i, b_i, H_all, b_all, sv_i,
                       total_neurons, TH, symbols, f_sym, delta=DREAL_DELTA)
    t0    = time.perf_counter()
    label = call_dreal(smt2)
    return label, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def coarse(label: Label) -> str:
    v = label.value
    if "UNSAFE" in v: return "UNSAFE"
    if "SAFE"   in v: return "SAFE"
    return "INCONCLUSIVE"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading barrier model ...")
    model  = torch.jit.load(MODEL_PATH, map_location="cpu"); model.eval()
    params = [p.detach().numpy() for _, p in model.named_parameters()]
    layer_W = [params[0], params[2]]
    layer_b = [params[1], params[3]]
    W_out   = params[4]
    n       = layer_W[0].shape[1]
    print(f"  Input dim = {n},  hidden layers = {len(layer_W)}")

    boundary_H = np.vstack((np.eye(n), -np.eye(n)))
    boundary_b = np.array(TH * 2, dtype=np.float64)

    total_neurons = sum(W.shape[0] for W in layer_W)
    use_wide      = (total_neurons + len(boundary_H) > 64)
    model_dtype   = next(model.parameters()).dtype
    np_dtype      = np.float64 if model_dtype == torch.float64 else np.float32

    # ── Load boundary cells (lazy generator — no big verts array in RAM) ────────
    print("Loading boundary cells ...")
    with h5py.File(BOUNDARY_H5, "r") as f:
        offsets = f["offsets"][:]
        sv_all  = f["activation_patterns"][:]
    n_cells = len(offsets) - 1
    if MAX_CELLS is not None:
        n_cells = min(n_cells, MAX_CELLS)
        sv_all  = sv_all[:n_cells]

    def _bc_generator():
        """Yield one cell's vertex array at a time, reading from HDF5."""
        with h5py.File(BOUNDARY_H5, "r") as f:
            for i in range(n_cells):
                yield f["vertices"][offsets[i]:offsets[i+1]].copy()

    BC = _bc_generator()
    print(f"  {n_cells} boundary cells (streaming from HDF5)\n")

    # ── Load dynamics ────────────────────────────────────────────────────────
    print("Loading dynamics ...")
    symbols, f_sym = load_dynamics(DYNAMICS)
    dyn = DynamicsEvaluator(symbols, f_sym)
    hb  = HessianBounder(symbols, f_sym)
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # METHOD 1 — Taylor + adaptive refinement  (no early exit)
    # ═════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("METHOD 1 — Taylor + adaptive refinement  (full, no early exit)")
    print("=" * 65)

    t_m1 = time.perf_counter()
    summary_m1 = verify_ref(
        _bc_generator(), sv_all, layer_W, layer_b, W_out,
        boundary_H, boundary_b, model,
        dynamics_name=DYNAMICS,
        continuous_time=True,
        early_exit=False,
        refinement_max_depth=8,
        TH=TH,
    )
    t_m1 = time.perf_counter() - t_m1

    # ═════════════════════════════════════════════════════════════════════════
    # METHOD 2 — Taylor first; dReal fallback for INCONCLUSIVE
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("METHOD 2 — Taylor first; dReal fallback for INCONCLUSIVE cells")
    print("=" * 65)

    m2_labels   : dict[int, str]   = {}
    m2_times    : dict[int, float] = {}
    m2_n_safe_t = m2_n_unsafe_t = 0        # decided by Taylor alone
    m2_n_safe_d = m2_n_unsafe_d = 0        # decided by dReal fallback
    m2_n_to     = m2_n_err      = 0
    m2_dr_time  = 0.0

    t_m2_start = time.perf_counter()

    for i, vertices in enumerate(_bc_generator()):
        vertices = np.asarray(vertices, dtype=float)
        sv_i     = sv_all[i].ravel()
        p_i      = compute_local_gradient(sv_i, layer_W, W_out)

        with torch.no_grad():
            B_vals = model(
                torch.tensor(vertices.astype(np_dtype), dtype=model_dtype)
            ).numpy().ravel()

        x_stars = _get_zero_level_set_crossings(
            vertices, sv_i, B_vals,
            layer_W, layer_b, boundary_H, boundary_b, use_wide,
            barrier_model=model, model_dtype=model_dtype,
        )

        if len(x_stars) == 0:
            m2_labels[i] = "SAFE"; m2_times[i] = 0.0
            m2_n_safe_t += 1
            continue

        t_label, *_ = _two_step_label(
            x_stars, model, dyn, hb, p_i, i, continuous_time=True
        )

        if t_label == Label.SAFE_TAYLOR:
            m2_labels[i] = "SAFE"; m2_times[i] = 0.0
            m2_n_safe_t += 1

        elif t_label == Label.UNSAFE:
            m2_labels[i] = "UNSAFE"; m2_times[i] = 0.0
            m2_n_unsafe_t += 1

        else:   # INCONCLUSIVE → dReal
            lbl, rt = verify_cell_dreal(
                vertices, sv_i, p_i, layer_W, layer_b,
                boundary_H, boundary_b, symbols, f_sym, model, TH
            )
            m2_labels[i] = lbl; m2_times[i] = rt; m2_dr_time += rt
            if   lbl == "SAFE":    m2_n_safe_d  += 1
            elif lbl == "UNSAFE":  m2_n_unsafe_d += 1
            elif lbl == "TIMEOUT": m2_n_to       += 1
            else:                  m2_n_err      += 1
            print(f"  cell {i:4d}  Taylor=INCONCLUSIVE  dReal={lbl:<8s}  t={rt:.2f}s")

    t_m2 = time.perf_counter() - t_m2_start

    # ═════════════════════════════════════════════════════════════════════════
    # METHOD 3 — Pure dReal  (no Taylor pre-filter)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("METHOD 3 — Pure dReal  (every boundary cell)")
    print("=" * 65)

    m3_labels  : dict[int, str]   = {}
    m3_times   : dict[int, float] = {}
    m3_n_safe = m3_n_unsafe = m3_n_to = m3_n_err = 0
    m3_dr_time = 0.0

    t_m3_start = time.perf_counter()

    for i, vertices in enumerate(_bc_generator()):
        vertices = np.asarray(vertices, dtype=float)
        sv_i     = sv_all[i].ravel()
        p_i      = compute_local_gradient(sv_i, layer_W, W_out)

        lbl, rt  = verify_cell_dreal(
            vertices, sv_i, p_i, layer_W, layer_b,
            boundary_H, boundary_b, symbols, f_sym, model, TH
        )
        m3_labels[i] = lbl; m3_times[i] = rt; m3_dr_time += rt

        if   lbl == "SAFE":    m3_n_safe   += 1
        elif lbl == "UNSAFE":  m3_n_unsafe += 1
        elif lbl == "TIMEOUT": m3_n_to     += 1
        else:                  m3_n_err    += 1

        if (i + 1) % 50 == 0 or (i + 1) == n_cells:
            print(f"  [{i+1}/{n_cells}]  SAFE={m3_n_safe}  UNSAFE={m3_n_unsafe}  "
                  f"TIMEOUT={m3_n_to}  ERROR={m3_n_err}  total_t={m3_dr_time:.1f}s")

    t_m3 = time.perf_counter() - t_m3_start

    # ═════════════════════════════════════════════════════════════════════════
    # Comparison
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("COMPARISON — all three methods")
    print("=" * 65)

    m1_map = {r.cell_idx: r.label for r in summary_m1.results}

    # Cross-check M1 vs M2 and M1 vs M3
    def compare(map_a, map_b, name_a, name_b):
        conflicts = disagreements = agreements = skipped = 0
        for idx in sorted(set(map_a) & set(map_b)):
            la = map_a[idx]
            lb = map_b[idx] if isinstance(map_b[idx], str) else map_b[idx].value
            if lb in ("TIMEOUT", "ERROR"):
                skipped += 1; continue
            ca = coarse(la) if isinstance(la, Label) else la
            cb = lb
            if (ca == "SAFE" and cb == "UNSAFE") or (ca == "UNSAFE" and cb == "SAFE"):
                conflicts += 1
            elif ca == cb:
                agreements += 1
            else:
                disagreements += 1
        print(f"\n  {name_a} vs {name_b}:")
        total = agreements + disagreements + conflicts
        print(f"    Agreements   : {agreements}/{total}")
        print(f"    Disagreements: {disagreements}/{total}")
        print(f"    CONFLICTS    : {conflicts}/{total}  <- must be 0")
        print(f"    Skipped      : {skipped}  (TIMEOUT/ERROR)")
        return conflicts

    c12 = compare(m1_map, m2_labels, "M1-Refinement", "M2-Taylor+dReal")
    c13 = compare(m1_map, m3_labels, "M1-Refinement", "M3-dReal only")
    c23 = compare(m2_labels, m3_labels, "M2-Taylor+dReal", "M3-dReal only")

    # Summary table
    m2_safe  = m2_n_safe_t  + m2_n_safe_d
    m2_unsafe= m2_n_unsafe_t + m2_n_unsafe_d
    m2_dr_calls = m2_n_safe_d + m2_n_unsafe_d + m2_n_to + m2_n_err

    W = 22
    print(f"\n{'='*75}")
    print(f"{'Metric':<38} {'M1-Refinement':>{W}} {'M2-Tay+dReal':>{W}} {'M3-dReal':>{W}}")
    print(f"{'='*75}")
    print(f"{'SAFE (Taylor certified)':<38} {summary_m1.n_safe_taylor:>{W}} {m2_n_safe_t:>{W}} {'—':>{W}}")
    print(f"{'SAFE (refinement / dReal fallback)':<38} {summary_m1.n_safe_refinement:>{W}} {m2_n_safe_d:>{W}} {m3_n_safe:>{W}}")
    print(f"{'SAFE total':<38} {summary_m1.n_safe:>{W}} {m2_safe:>{W}} {m3_n_safe:>{W}}")
    print(f"{'UNSAFE total':<38} {summary_m1.n_unsafe:>{W}} {m2_unsafe:>{W}} {m3_n_unsafe:>{W}}")
    print(f"{'INCONCLUSIVE / TIMEOUT+ERROR':<38} {summary_m1.n_inconclusive:>{W}} {m2_n_to+m2_n_err:>{W}} {m3_n_to+m3_n_err:>{W}}")
    print(f"{'dReal calls made':<38} {'0':>{W}} {m2_dr_calls:>{W}} {len(BC):>{W}}")
    print(f"{'Total runtime (s)':<38} {t_m1:>{W}.1f} {t_m2:>{W}.1f} {t_m3:>{W}.1f}")
    print(f"{'dReal total time (s)':<38} {'0':>{W}} {m2_dr_time:>{W}.1f} {m3_dr_time:>{W}.1f}")
    print(f"{'='*75}")

    print()
    total_conflicts = c12 + c13 + c23
    if total_conflicts == 0:
        print("CONCLUSION: No soundness conflicts across all three methods.")
    else:
        print(f"CONCLUSION: {total_conflicts} soundness conflict(s) detected — investigate!")
