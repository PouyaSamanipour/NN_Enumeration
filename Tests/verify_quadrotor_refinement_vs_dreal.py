"""
verify_quadrotor_refinement_vs_dreal.py
========================================
Compare adaptive refinement vs NLP+dReal on MAX_CELLS quadrotor boundary cells.

Method 1 — Taylor + adaptive refinement (no early exit)
Method 2 — NLP counterexample search first; dReal fallback for formal SAFE verdict

Barrier certificate:  B(x) = V(x) - LEVEL,  LEVEL = -0.01697
Continuous-time condition:  nabla B(x) . f(x) <= 0  on  {B(x) = 0}

Run from the project root:
    python Tests/verify_quadrotor_refinement_vs_dreal.py
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import h5py
import sympy as sp

from relu_region_enumerator.verify_certificate_face import verify_barrier
from relu_region_enumerator.verify_certificate_new import Label, DynamicsEvaluator
from relu_region_enumerator.hessian_bound import HessianBounder, compute_local_gradient
from relu_region_enumerator.Dynamics import load_dynamics
from relu_region_enumerator.bitwise_utils import get_cell_hyperplanes_input_space


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

BARRIER_PATH  = "NN_files/model_quadrotor_ct_B.pt"
BOUNDARY_H5   = "quadrotor_boundary_cells.h5"
TH            = [1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
DYNAMICS      = "quadrotor"

DREAL_BIN          = "/opt/dreal/4.21.06.2/bin/dreal"
DREAL_DELTA        = 0.001   # fine delta — SAT is a sound counterexample
DREAL_DELTA_COARSE = 0.01    # coarse first-pass (only UNSAT trusted)
DREAL_TIMEOUT_FAST = 10      # seconds per cell (coarse pass)
DREAL_TIMEOUT      = 50     # seconds per cell (fine pass; coarse+fine ≤ 300s)

CELL_TIMEOUT_S = 300         # per-cell budget for both methods

NLP_N_STARTS = 15   # random convex-combination starts for NLP counterexample search

MAX_CELLS = 10


# ═══════════════════════════════════════════════════════════════════════════
# Load model and boundary cells
# ═══════════════════════════════════════════════════════════════════════════

print("Loading barrier model ...")
model  = torch.jit.load(BARRIER_PATH, map_location="cpu"); model.eval()
params = [p.detach().numpy() for _, p in model.named_parameters()]
layer_W = [params[0], params[2]]
layer_b = [params[1], params[3]]
W_out   = params[4]
n       = layer_W[0].shape[1]
print(f"  Input dim={n},  hidden layers={len(layer_W)},  "
      f"neurons={[W.shape[0] for W in layer_W]}")

boundary_H    = np.vstack((np.eye(n), -np.eye(n)))
boundary_b    = np.array(TH * 2, dtype=np.float64)
total_neurons = sum(W.shape[0] for W in layer_W)
model_dtype   = next(model.parameters()).dtype
np_dtype      = np.float64 if model_dtype == torch.float64 else np.float32

RANDOM_SEED = None   # set to None for a different sample each run

print(f"\nLoading boundary cells from {BOUNDARY_H5} ...")
with h5py.File(BOUNDARY_H5, "r") as f:
    offsets     = f["offsets"][:]
    sv_all_full = f["activation_patterns"][:]
    n_total     = len(offsets) - 1
    n_cells     = min(n_total, MAX_CELLS)
    rng_cells   = np.random.default_rng(RANDOM_SEED)
    cell_indices = np.sort(rng_cells.choice(n_total, size=n_cells, replace=False))
    sv_all      = sv_all_full[cell_indices]
    BC          = [f["vertices"][offsets[i]:offsets[i + 1]][:] for i in cell_indices]
print(f"  Sampled {n_cells} / {n_total} boundary cells  "
      f"(seed={RANDOM_SEED}, indices={cell_indices.tolist()})\n")


# ═══════════════════════════════════════════════════════════════════════════
# Load dynamics
# ═══════════════════════════════════════════════════════════════════════════

print("Loading dynamics ...")
symbols, f_sym = load_dynamics(DYNAMICS)
_dyn = DynamicsEvaluator(symbols, f_sym)
print()


# ═══════════════════════════════════════════════════════════════════════════
# dReal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sympy_to_smt2(expr, var_names: dict) -> str:
    if expr.is_Number:
        return repr(float(expr))
    if expr.is_Symbol:
        return var_names[expr]
    if expr.is_Add:
        return "(+ " + " ".join(_sympy_to_smt2(a, var_names) for a in expr.args) + ")"
    if expr.is_Mul:
        return "(* " + " ".join(_sympy_to_smt2(a, var_names) for a in expr.args) + ")"
    if expr.is_Pow:
        b_e, e_e = expr.args
        b_s = _sympy_to_smt2(b_e, var_names)
        if e_e == 2:  return f"(* {b_s} {b_s})"
        if e_e == 3:  return f"(* {b_s} (* {b_s} {b_s}))"
        if e_e == -1: return f"(/ 1.0 {b_s})"
        if e_e == sp.Rational(1, 2): return f"(sqrt {b_s})"
        return f"(^ {b_s} {_sympy_to_smt2(e_e, var_names)})"
    if expr.is_negative:
        return f"(- {_sympy_to_smt2(-expr, var_names)})"
    _fmap = {sp.sin:"sin", sp.cos:"cos", sp.tan:"tan", sp.exp:"exp",
             sp.log:"log", sp.sqrt:"sqrt", sp.tanh:"tanh",
             sp.sinh:"sinh", sp.cosh:"cosh", sp.Abs:"abs"}
    if expr.func in _fmap and len(expr.args) == 1:
        return f"({_fmap[expr.func]} {_sympy_to_smt2(expr.args[0], var_names)})"
    raise ValueError(f"Cannot convert to SMT2: {expr}")


def _build_smt2(p_i: np.ndarray, b_i: float,
                H_all: np.ndarray, b_all: np.ndarray,
                sv_i: np.ndarray, delta: float) -> str:
    """SMT2 formula: exists x in cell s.t. B(x)=0 AND p_i.f(x) > 0"""
    var_names = {s: f"x{k+1}" for k, s in enumerate(symbols)}
    lines = ["(set-logic QF_NRA)", f"(set-info :precision {delta})"]

    for k in range(n):
        lines.append(f"(declare-fun x{k+1} () Real)")
        lines.append(f"(assert (>= x{k+1} {-float(TH[k]):.10f}))")
        lines.append(f"(assert (<= x{k+1}  {float(TH[k]):.10f}))")

    # ZLS equality: p_i . x + b_i = 0
    zls_terms = [f"(* {float(p_i[k]):.10f} x{k+1})"
                 for k in range(n) if abs(p_i[k]) > 1e-15]
    zls_sum = ("(+ " + " ".join(zls_terms) + ")"
               if len(zls_terms) > 1 else (zls_terms[0] if zls_terms else "0.0"))
    lines.append(f"(assert (= (+ {zls_sum} {float(b_i):.10f}) 0.0))")

    # Activation constraints (rows 0..total_neurons-1) + domain (rest)
    for row in range(len(b_all)):
        terms = [f"(* {float(H_all[row, k]):.10f} x{k+1})"
                 for k in range(n) if abs(H_all[row, k]) > 1e-15]
        if not terms:
            continue
        lhs  = "(+ " + " ".join(terms) + ")" if len(terms) > 1 else terms[0]
        expr = f"(+ {lhs} {float(b_all[row]):.10f})"
        op   = (">=" if sv_i[row] == 1 else "<=") if row < total_neurons else ">="
        lines.append(f"(assert ({op} {expr} 0.0))")

    # Violation: p_i . f(x) > 0
    lie_terms = [f"(* {float(p_i[k]):.10f} {_sympy_to_smt2(fk, var_names)})"
                 for k, fk in enumerate(f_sym) if abs(float(p_i[k])) > 1e-15]
    lie_sum = ("(+ " + " ".join(lie_terms) + ")"
               if len(lie_terms) > 1 else (lie_terms[0] if lie_terms else "0.0"))
    lines.append(f"(assert (> {lie_sum} 0.0))")
    lines.append("(check-sat)")
    return "\n".join(lines)


def _run_dreal_once(smt2: str, delta: float, timeout: int) -> str:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False, dir='/tmp') as f:
        f.write(smt2); tmp = f.name
    try:
        res = subprocess.run(
            [DREAL_BIN, "--precision", str(delta), tmp],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (res.stdout + res.stderr).lower()
        if "unsat" in out: return "SAFE"
        if "sat"   in out: return "UNSAFE"
        return "ERROR"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except FileNotFoundError:
        raise RuntimeError(f"dReal not found at {DREAL_BIN}")
    finally:
        try: os.unlink(tmp)
        except Exception: pass


def verify_cell_dreal(vertices: np.ndarray, sv_i: np.ndarray,
                      p_i: np.ndarray) -> tuple[str, float]:
    """Two-pass dReal: coarse delta first (only UNSAT trusted), then fine delta."""
    v0 = vertices[0]
    with torch.no_grad():
        B_v0 = float(model(
            torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
        ).numpy().ravel()[0])
    b_i = B_v0 - float(p_i @ v0)

    H_all, b_all = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b
    )
    smt2 = _build_smt2(p_i, b_i, H_all, b_all, sv_i, delta=DREAL_DELTA)

    t0 = time.perf_counter()
    lbl = _run_dreal_once(smt2, DREAL_DELTA_COARSE, DREAL_TIMEOUT_FAST)
    if lbl != "SAFE":
        lbl = _run_dreal_once(smt2, DREAL_DELTA, DREAL_TIMEOUT)
    return lbl, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════
# NLP + dReal helpers
# ═══════════════════════════════════════════════════════════════════════════

def verify_cell_nlp_dreal(
    vertices: np.ndarray,
    sv_i    : np.ndarray,
    p_i     : np.ndarray,
) -> tuple[str, float, bool]:
    """
    NLP counterexample search followed by dReal formal verification.

    Step 1 — NLP: maximize p_i · f(x) subject to
        p_i · x + b_i = 0   (zero level set of B, exact within the linear cell)
        H_all · x + b_all >= 0  (cell activation + domain constraints)
        -TH <= x <= TH

    If the NLP optimum is > 0 the constraint is violated → UNSAFE (no dReal needed).

    Step 2 — dReal: only reached when NLP finds no violation.
        Two-pass dReal (coarse delta then fine delta) for a formal SAFE/UNSAFE verdict.

    Returns (label, elapsed_s, nlp_found).
      nlp_found=True  means the counterexample was found by NLP (dReal skipped).
    """
    from scipy.optimize import minimize, LinearConstraint

    t0 = time.perf_counter()

    v0 = vertices[0]
    with torch.no_grad():
        B_v0 = float(model(
            torch.tensor(v0[None].astype(np_dtype), dtype=model_dtype)
        ).numpy().ravel()[0])
    b_i = B_v0 - float(p_i @ v0)

    H_all, b_all = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b
    )

    # Sign-correct the neuron rows: active neurons need H@x+b >= 0,
    # inactive neurons need H@x+b <= 0  →  negate their rows so all become >= 0.
    _sign = np.ones(len(b_all))
    _sign[:total_neurons] = np.where(sv_i.ravel()[:total_neurons] == 1, 1.0, -1.0)
    H_signed = H_all * _sign[:, None]
    b_signed = b_all * _sign

    # NLP objective: minimize -(p_i · f(x)), return (val, grad) for jac=True
    def _obj(x):
        f_val, f_jac = _dyn.eval(np.asarray(x, dtype=float))
        return -(p_i @ f_val), -(f_jac.T @ p_i)

    bounds   = [(-TH[k], TH[k]) for k in range(n)]
    lin_ineq = LinearConstraint(H_signed, -b_signed, np.full(len(b_signed), np.inf))
    eq_con   = {'type': 'eq',
                'fun': lambda x: float(p_i @ x + b_i),
                'jac': lambda x: p_i}

    rng    = np.random.default_rng(42)
    V      = len(vertices)
    starts = [vertices.mean(axis=0)]
    for _ in range(NLP_N_STARTS - 1):
        w = rng.dirichlet(np.ones(V))
        starts.append(w @ vertices)

    nlp_found = False
    for x0 in starts:
        try:
            res = minimize(
                _obj, x0, jac=True, method='SLSQP',
                bounds=bounds, constraints=[lin_ineq, eq_con],
                options={'maxiter': 300, 'ftol': 1e-9},
            )
            if -res.fun > 1e-6:
                nlp_found = True
                break
        except Exception:
            pass

    if nlp_found:
        return "UNSAFE", time.perf_counter() - t0, True

    # NLP found no violation — dReal for a formal verdict
    smt2 = _build_smt2(p_i, b_i, H_all, b_all, sv_i, delta=DREAL_DELTA)
    lbl  = _run_dreal_once(smt2, DREAL_DELTA_COARSE, DREAL_TIMEOUT_FAST)
    if lbl != "SAFE":
        lbl = _run_dreal_once(smt2, DREAL_DELTA, DREAL_TIMEOUT)
    return lbl, time.perf_counter() - t0, False


# ═══════════════════════════════════════════════════════════════════════════
# Method 1 — Adaptive refinement (no early exit)
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("METHOD 1 — ZLS face refinement  (verify_certificate_face, no early exit)")
print("=" * 65)

t_m1 = time.perf_counter()
summary_m1 = verify_barrier(
    list(BC), sv_all, layer_W, layer_b, W_out,
    boundary_H, boundary_b, model,
    dynamics_name        = DYNAMICS,
    continuous_time      = True,
    early_exit           = False,
    refinement_max_depth = 15,
    TH                   = TH,
    cell_timeout_s       = CELL_TIMEOUT_S,
)
t_m1 = time.perf_counter() - t_m1
print(f"  Wall time: {t_m1:.2f} s")

# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic: check candidate point against cell 1
# ═══════════════════════════════════════════════════════════════════════════

_diag_x    = np.array([
    -0.034499, -0.23202,  0.0513,   0.3,      -0.3,     -0.046119,
     0.061846,  1.0,     -0.390885, 1.0,      -1.0,      0.111671,
], dtype=np.float64)
_diag_cell = 1

print("\n" + "=" * 65)
print(f"DIAGNOSTIC — candidate point vs cell {_diag_cell}")
print("=" * 65)

_d_verts = np.asarray(BC[_diag_cell], dtype=float)
_d_sv    = sv_all[_diag_cell].ravel()
_d_p     = compute_local_gradient(_d_sv, layer_W, W_out)

with torch.no_grad():
    _d_Bx = float(model(
        torch.tensor(_diag_x[None].astype(np_dtype), dtype=model_dtype)
    ).numpy().ravel()[0])
    _d_Bv0 = float(model(
        torch.tensor(_d_verts[0][None].astype(np_dtype), dtype=model_dtype)
    ).numpy().ravel()[0])

_d_bi      = _d_Bv0 - float(_d_p @ _d_verts[0])
_d_zls_res = float(_d_p @ _diag_x + _d_bi)   # ≈0 if point is on the ZLS
_d_fx, _   = _dyn.eval(_diag_x)
_d_lie     = float(_d_p @ _d_fx)              # violation if > 0

_d_H, _d_b = get_cell_hyperplanes_input_space(
    _d_sv, layer_W, layer_b, boundary_H, boundary_b
)
# Apply sign correction: inactive neurons need H@x+b <= 0, negate their rows
_d_sign          = np.ones(len(_d_b))
_d_sign[:total_neurons] = np.where(_d_sv[:total_neurons] == 1, 1.0, -1.0)
_d_slack         = (_d_H * _d_sign[:, None]) @ _diag_x + _d_b * _d_sign
_d_n_violated    = int((_d_slack < -1e-6).sum())

print(f"  B(x)           = {_d_Bx:.6e}  (should be ≈ 0 for a ZLS point)")
print(f"  ZLS residual   = {_d_zls_res:.6e}  (p_i·x + b_i)")
print(f"  p_i · f(x)     = {_d_lie:.6e}  ({'VIOLATION' if _d_lie > 1e-6 else 'ok'})")
print(f"  Cell slack min = {_d_slack.min():.6e}  ({_d_n_violated} constraints violated)")
print()

# ═══════════════════════════════════════════════════════════════════════════
# Method 2 — NLP + dReal
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("METHOD 2 — NLP + dReal fallback  (NLP first, dReal if no CEX found)")
print("=" * 65)

m2_labels     : dict[int, str]  = {}
m2_times      : dict[int, float] = {}
m2_n_safe = m2_n_unsafe = m2_n_timeout = m2_n_error = 0
m2_n_nlp_cex  = 0   # cells where NLP found a counterexample (dReal skipped)
m2_dr_time    = 0.0

t_m2 = time.perf_counter()

for i, vertices in enumerate(BC):
    vertices = np.asarray(vertices, dtype=float)
    sv_i     = sv_all[i].ravel()
    p_i      = compute_local_gradient(sv_i, layer_W, W_out)

    lbl, rt, nlp_hit = verify_cell_nlp_dreal(vertices, sv_i, p_i)
    m2_labels[i] = lbl
    m2_times[i]  = rt
    m2_dr_time  += rt

    if nlp_hit:
        m2_n_nlp_cex += 1
    if   lbl == "SAFE":    m2_n_safe    += 1
    elif lbl == "UNSAFE":  m2_n_unsafe  += 1
    elif lbl == "TIMEOUT": m2_n_timeout += 1
    else:                  m2_n_error   += 1

    print(f"  [{i+1:3d}/{n_cells}]  lbl={lbl:<8s}  nlp={'CEX' if nlp_hit else '—':>3s}  "
          f"t={rt:.2f}s  "
          f"(SAFE={m2_n_safe}  UNSAFE={m2_n_unsafe}  "
          f"TIMEOUT={m2_n_timeout}  NLP_CEX={m2_n_nlp_cex})")

t_m2 = time.perf_counter() - t_m2
print(f"\n  NLP+dReal total wall time : {t_m2:.2f} s")
print(f"  NLP counterexamples found : {m2_n_nlp_cex}  (dReal skipped for these)")





# ═══════════════════════════════════════════════════════════════════════════
# Comparison — per-cell table + summary
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 95)
print("COMPARISON — ZLS Face Refinement vs NLP+dReal  (per-cell)")
print("=" * 95)

m1_map      = {r.cell_idx: r for r in summary_m1.results}

def _coarse(label) -> str:
    v = label.value if isinstance(label, Label) else label
    if "UNSAFE" in v: return "UNSAFE"
    if "SAFE"   in v: return "SAFE"
    return "INCONCLUSIVE"

HDR = (f"{'#':>4}  {'CellID':>7}  {'Face-Refine Label':<24}  {'t_ref (s)':>10}  "
       f"{'dReal Label':<10}  {'t_drl (s)':>10}  {'Status':<12}")
SEP = "-" * len(HDR)
print(f"\n{HDR}")
print(SEP)

n_agree = n_conflict = n_disagree = n_skip = 0

for idx in range(n_cells):
    cell_id = int(cell_indices[idx])
    r1    = m1_map.get(idx)
    lbl_r = r1.label.value if r1 is not None else "MISSING"
    t_r   = r1.time_s      if r1 is not None else float("nan")
    lbl_d = m2_labels.get(idx, "MISSING")
    t_d   = m2_times.get(idx, float("nan"))

    if lbl_d in ("TIMEOUT", "ERROR", "MISSING"):
        status = "SKIP"
        n_skip += 1
    else:
        cr, cd = _coarse(lbl_r), _coarse(lbl_d)
        if (cr == "SAFE" and cd == "UNSAFE") or (cr == "UNSAFE" and cd == "SAFE"):
            status = "!! CONFLICT"
            n_conflict += 1
        elif cr == cd:
            status = "agree"
            n_agree += 1
        else:
            status = "disagree"
            n_disagree += 1

    print(f"  {idx:>3}  {cell_id:>7}  {lbl_r:<24}  {t_r:>10.2f}  "
          f"{lbl_d:<10}  {t_d:>10.2f}  {status:<12}")

print(SEP)
print(f"  {'':4}  {'TOTAL':>7}  {'':24}  {t_m1:>10.2f}  "
      f"{'':10}  {t_m2:>10.2f}")

W = 18
print(f"\n{'='*75}")
print(f"{'Metric':<42} {'Face Refinement':>{W}} {'NLP+dReal':>{W}}")
print(f"{'='*75}")
print(f"{'SAFE (Taylor certified)':<42} {summary_m1.n_safe_taylor:>{W}} {'—':>{W}}")
print(f"{'SAFE (after refinement)':<42} {summary_m1.n_safe_refinement:>{W}} {'—':>{W}}")
print(f"{'SAFE total':<42} {summary_m1.n_safe:>{W}} {m2_n_safe:>{W}}")
print(f"{'UNSAFE total':<42} {summary_m1.n_unsafe:>{W}} {m2_n_unsafe:>{W}}")
print(f"{'  of which NLP CEX (no dReal)':<42} {'—':>{W}} {m2_n_nlp_cex:>{W}}")
print(f"{'INCONCLUSIVE / TIMEOUT+ERROR':<42} {summary_m1.n_inconclusive:>{W}} {m2_n_timeout+m2_n_error:>{W}}")
print(f"{'Wall time (s)':<42} {t_m1:>{W}.2f} {t_m2:>{W}.2f}")
print(f"{'='*75}")
print(f"\n  Agreements   : {n_agree}")
print(f"  Disagreements: {n_disagree}  (face-refine more sensitive; dReal may miss at coarse delta)")
print(f"  CONFLICTS    : {n_conflict}  <- must be 0")
print(f"  Skipped      : {n_skip}  (dReal TIMEOUT/ERROR)")

print()
if n_conflict == 0:
    print("CONCLUSION: No soundness conflicts — face refinement and NLP+dReal agree on all decidable cells.")
    if n_disagree:
        print(f"  {n_disagree} cell(s): face refinement found a violation that NLP+dReal "
              "reported as SAFE (delta-precision artefact or NLP local optimum).")
else:
    print(f"CONCLUSION: {n_conflict} soundness conflict(s) found — investigate!")
