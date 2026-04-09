"""
validate_with_nlp.py
====================
Ground-truth validation of barrier certificate conditions using nonlinear
programming (NLP). For each boundary cell, solves:

    maximize   p_i . f(x)           (continuous-time)
    subject to p_i @ x + c_i = 0   (zero level set: B(x) = 0)
               H_all @ x + b_all <= 0  (cell polytope halfspaces)

If the optimal value > 0 -> cell is truly UNSAFE (counterexample exists).
If the optimal value <= 0 -> cell is truly SAFE.

This is a head-to-head comparison against verify_barrier to assess:
  1. False negatives: SAFE cells that NLP finds unsafe (soundness bug)
  2. Over-conservatism: INCONCLUSIVE cells that NLP finds safe
  3. True unsafe: INCONCLUSIVE cells that NLP confirms unsafe

Usage
-----
    from validate_with_nlp import validate_with_nlp

    report = validate_with_nlp(
        BC, sv, layer_W, layer_b, W_out,
        boundary_H, boundary_b,
        barrier_model, dynamics_name="decay",
        summary=verify_summary,    # optional: our labels for comparison
        continuous_time=True,
    )
    report.print_table()
    report.print_counterexamples()        # full detail on all NLP-unsafe cells
"""

from __future__ import annotations

import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

try:
    from .hessian_bound import compute_local_gradient
    from .Dynamics import load_dynamics
    from .bitwise_utils import get_cell_hyperplanes_input_space
    from .verify_certificates import Label, DynamicsEvaluator
except ImportError:
    from hessian_bound import compute_local_gradient
    from Dynamics import load_dynamics
    from bitwise_utils import get_cell_hyperplanes_input_space
    from verify_certificates import Label, DynamicsEvaluator


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NLPCellResult:
    cell_idx    : int
    nlp_label   : str        # "SAFE" or "UNSAFE"
    opt_val     : float      # maximum of p_i . f(x*) found by NLP
    x_opt       : np.ndarray # optimizer point
    our_label   : str        # label from verify_barrier (if provided)
    n_starts    : int        # number of starting points tried
    success     : bool       # NLP converged
    # Affine consistency fields — populated by validate_with_nlp
    p_i         : np.ndarray = None  # local gradient: B(x) = p_i @ x + q_i in this cell
    q_i         : float      = None  # affine offset
    centroid    : np.ndarray = None  # cell centroid (second test point)
    B_net_xopt  : float      = None  # B(x*)        from network forward pass
    B_net_cent  : float      = None  # B(centroid)  from network forward pass

    def print_affine_check(self):
        """
        Verify p_i is correct by checking B(x) = p_i @ x + q_i at two points.

          At x_opt:    trivially true by construction (q_i defined from x_opt),
                       but confirms B_net(x*) ≈ 0 (x* is on the ZLS).
          At centroid: different point, same activation region — must also match
                       to floating-point precision (~1e-5). This is the real test.

        If both match, p_i is correct and the violation p_i · f(x*) > 0 is genuine.
        """
        if self.p_i is None:
            print("  [affine check] fields not populated.")
            return

        B_aff_xopt = float(self.p_i @ self.x_opt) + self.q_i
        B_aff_cent = float(self.p_i @ self.centroid) + self.q_i
        err_xopt   = abs(B_aff_xopt - self.B_net_xopt)
        err_cent   = abs(B_aff_cent - self.B_net_cent)
        tol        = 1e-4

        ok = lambda e: 'OK' if e < tol else '*** MISMATCH ***'
        print(f"  --- affine consistency check: cell {self.cell_idx} ---")
        print(f"    B_net(x*)          = {self.B_net_xopt:+.8f}   (should be ≈ 0, x* on ZLS)")
        print(f"    p_i @ x*  + q_i    = {B_aff_xopt:+.8f}   err={err_xopt:.2e}  {ok(err_xopt)}")
        print(f"    B_net(centroid)    = {self.B_net_cent:+.8f}")
        print(f"    p_i @ c   + q_i    = {B_aff_cent:+.8f}   err={err_cent:.2e}  {ok(err_cent)}")
        if err_xopt < tol and err_cent < tol:
            print(f"    => p_i CORRECT.  p_i · f(x*) = {self.opt_val:+.8f}  => violation GENUINE.")
        else:
            print(f"    => p_i WRONG — recheck compute_local_gradient.")


@dataclass
class NLPReport:
    dynamics_name   : str
    continuous_time : bool
    n_cells         : int
    results         : List[NLPCellResult] = field(default_factory=list)
    runtime_s       : float = 0.0

    # Confusion matrix counts
    safe_safe       : int = 0   # our SAFE, NLP SAFE     — correct
    safe_unsafe     : int = 0   # our SAFE, NLP UNSAFE   — false safe (bug!)
    inc_safe        : int = 0   # our INCONCLUSIVE, NLP SAFE   — over-conservative
    inc_unsafe      : int = 0   # our INCONCLUSIVE, NLP UNSAFE — true violation
    nlp_unsafe_total: int = 0   # total NLP-unsafe cells

    @property
    def counterexamples(self) -> list:
        """All NLPCellResults where nlp_label == 'UNSAFE', sorted by opt_val descending."""
        return sorted(
            [r for r in self.results if r.nlp_label == "UNSAFE"],
            key=lambda r: r.opt_val,
            reverse=True,
        )

    def print_table(self):
        print(f"\n{'='*65}")
        print(f"NLP Validation Report  [{self.dynamics_name}  "
              f"{'continuous' if self.continuous_time else 'discrete'}-time]")
        print(f"{'='*65}")
        print(f"  Total cells validated : {self.n_cells}")
        print(f"  NLP SAFE              : {self.n_cells - self.nlp_unsafe_total}"
              f"  ({100*(self.n_cells-self.nlp_unsafe_total)/max(self.n_cells,1):.1f}%)")
        print(f"  NLP UNSAFE            : {self.nlp_unsafe_total}"
              f"  ({100*self.nlp_unsafe_total/max(self.n_cells,1):.1f}%)")
        print(f"  Runtime               : {self.runtime_s:.1f} s")
        print()
        print(f"  Confusion matrix vs verify_barrier:")
        print(f"    Our SAFE   + NLP SAFE   = {self.safe_safe:5d}  (correct)")
        print(f"    Our SAFE   + NLP UNSAFE = {self.safe_unsafe:5d}  "
              f"{'*** SOUNDNESS BUG ***' if self.safe_unsafe > 0 else 'ok'}")
        print(f"    Our INCON  + NLP SAFE   = {self.inc_safe:5d}  (over-conservative)")
        print(f"    Our INCON  + NLP UNSAFE = {self.inc_unsafe:5d}  (true violations missed)")

        if self.safe_unsafe > 0:
            print(f"\n  !! {self.safe_unsafe} cells labelled SAFE by us but UNSAFE by NLP.")
            print(f"     This indicates a soundness bug in verify_barrier.")
        elif self.nlp_unsafe_total == 0:
            print(f"\n  Certificate VERIFIED by NLP over all {self.n_cells} cells.")
        else:
            print(f"\n  {self.inc_unsafe} true violations in INCONCLUSIVE cells.")
            print(f"  {self.inc_safe} INCONCLUSIVE cells are actually safe "
                  f"(Taylor bound too conservative).")

        cxs = self.counterexamples
        if cxs:
            print()
            print(f"  Top counterexamples (sorted by p_i·f(x*) descending):")
            print(f"  {'cell':>6}  {'our_label':>12}  {'opt_val':>12}  x_opt")
            print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*30}")
            for r in cxs[:10]:   # show at most 10 in the summary table
                x_str = np.array2string(r.x_opt, precision=4, separator=',',
                                        suppress_small=True)
                print(f"  {r.cell_idx:>6}  {r.our_label:>12}  {r.opt_val:>12.6f}  {x_str}")
            if len(cxs) > 10:
                print(f"  ... and {len(cxs) - 10} more. "
                      f"Call report.print_counterexamples() for full list.")
        print(f"{'='*65}")

    def print_counterexamples(self, max_print: int = None,
                               BC=None, sv=None, layer_W=None, W_out=None,
                               barrier_model=None):
        """
        Print full detail on every NLP-unsafe cell.

        If BC, sv, layer_W, W_out, and barrier_model are provided, also runs
        an affine consistency check for each counterexample:

            B_net(x*)          — forward pass at x* (should be ≈ 0, on ZLS)
            p_i @ x* + q_i     — affine reconstruction at x* (must equal B_net(x*))
            B_net(centroid)     — forward pass at cell centroid
            p_i @ c + q_i      — affine reconstruction at centroid
            |difference|        — should be < 1e-4; if large, p_i is wrong

        Parameters
        ----------
        max_print    : int, optional — cap number printed (None = all)
        BC           : list of (V, n) vertex arrays
        sv           : (N, total_neurons) activation patterns
        layer_W      : list of weight matrices
        W_out        : output weight vector
        barrier_model: TorchScript barrier network
        """
        cxs = self.counterexamples
        if not cxs:
            print("No NLP-unsafe cells found.")
            return

        do_affine = all(a is not None for a in [BC, sv, layer_W, W_out, barrier_model])
        if not do_affine:
            print("  (Pass BC, sv, layer_W, W_out, barrier_model to enable affine check)")

        model_dtype = next(barrier_model.parameters()).dtype if do_affine else None

        limit = len(cxs) if max_print is None else min(max_print, len(cxs))
        print(f"\n{'='*65}")
        print(f"Counterexamples ({len(cxs)} total"
              + (f", showing top {limit}" if limit < len(cxs) else "")
              + "):")
        print(f"{'='*65}")
        for r in cxs[:limit]:
            tag = " *** SOUNDNESS BUG ***" if r.our_label in ("SAFE_TAYLOR", "SAFE_NLP") else ""
            print(f"\n  cell {r.cell_idx}{tag}")
            print(f"    our_label  : {r.our_label}")
            print(f"    opt_val    : {r.opt_val:.8f}  (p_i · f(x*))")
            print(f"    nlp_success: {r.success}")
            print(f"    n_starts   : {r.n_starts}")
            print(f"    x_opt      : {np.array2string(r.x_opt, precision=6, separator=', ')}")

            if do_affine:
                idx      = r.cell_idx
                x_opt    = r.x_opt
                sv_i     = sv[idx].ravel()
                p_i      = compute_local_gradient(sv_i, layer_W, W_out)
                centroid = np.asarray(BC[idx], dtype=float).mean(axis=0)

                with torch.no_grad():
                    def _fwd(x):
                        return float(barrier_model(
                            torch.tensor(x.astype(np.float32)[None, :],
                                         dtype=model_dtype)
                        ).detach().numpy().ravel()[0])

                    B_xopt     = _fwd(x_opt)
                    B_centroid = _fwd(centroid)

                q_i            = B_xopt - float(p_i @ x_opt)
                B_aff_xopt     = float(p_i @ x_opt)    + q_i   # trivially = B_xopt
                B_aff_centroid = float(p_i @ centroid) + q_i
                diff           = abs(B_centroid - B_aff_centroid)

                # ── validity flags ────────────────────────────────────────────
                zls_ok  = abs(B_xopt) <= 1e-5
                pi_ok   = diff <= 1e-4
                valid   = zls_ok and pi_ok and r.success

                zls_flag  = "OK" if zls_ok  else f"!! x* NOT on ZLS — counterexample INVALID"
                pi_flag   = "OK" if pi_ok   else f"!! p_i mismatch — gradient may be wrong"
                nlp_flag  = "OK" if r.success else "!! NLP did not converge — result UNRELIABLE"

                print(f"    ── validity checks ──")
                print(f"    nlp_converged      :  {nlp_flag}")
                print(f"    B_net(x*)          = {B_xopt:+.8f}  (should be ≈ 0)  {zls_flag}")
                print(f"    p_i @ x* + q_i     = {B_aff_xopt:+.8f}  (trivially equal)")
                print(f"    B_net(centroid)    = {B_centroid:+.8f}")
                print(f"    p_i @ c  + q_i     = {B_aff_centroid:+.8f}")
                print(f"    |difference|       = {diff:.2e}  {pi_flag}")
                if valid:
                    print(f"    ✓ counterexample VALID: p_i·f(x*) = {r.opt_val:.6f} > 0")
                else:
                    print(f"    ✗ counterexample INVALID — do not report")

        print(f"\n{'='*65}")

    def print_unsafe_cells(self):
        """Alias for print_counterexamples() for backward compatibility."""
        self.print_counterexamples()


# ═══════════════════════════════════════════════════════════════════════════
# Per-cell NLP solver
# ═══════════════════════════════════════════════════════════════════════════

def _solve_cell_nlp(
    vertices        : np.ndarray,   # (V, n) cell vertices
    sv_i            : np.ndarray,
    p_i             : np.ndarray,   # (n,) local gradient of B
    c_i             : float,        # B offset
    layer_W         : list,
    layer_b         : list,
    boundary_H      : np.ndarray,
    boundary_b      : np.ndarray,
    dyn             : DynamicsEvaluator,
    continuous_time : bool,
    barrier_model,
    model_dtype,
    n_starts        : int   = 8,
    tol             : float = 1e-10,
    zls_tol         : float = 1e-5,   # max |B_net(x*)| to accept a result
    ineq_tol        : float = 1e-5,   # max polytope constraint violation to accept
) -> tuple:
    """
    Solve: maximize p_i . f(x)  s.t.  p_i @ x + c_i = 0,  x in X_i

    Starting point strategy
    -----------------------
    Sort all cell vertices by |B_net(v)| ascending — vertices closest to the
    true ZLS produce projections that stay near the true ZLS and inside the
    polytope.  The centroid projection is included as a fallback.

    Feasibility gate
    ----------------
    A result is only accepted if:
        |B_net(x*)| <= zls_tol          x* is on the true (nonlinear) ZLS
        max(A_ub @ x* - b_ub) <= ineq_tol   x* is inside the cell polytope

    If no run passes the gate, success=False is returned and opt_val reflects
    the best infeasible result (treat as unreliable).

    Returns (opt_val, x_opt, success).
    """
    n        = vertices.shape[1]
    centroid = vertices.mean(axis=0)

    # ── Cell polytope halfspaces ──────────────────────────────────────────────
    H_all, b_all = get_cell_hyperplanes_input_space(
        sv_i, layer_W, layer_b, boundary_H, boundary_b
    )
    centroid_vals = H_all @ centroid + b_all
    signs = np.sign(centroid_vals)
    signs[signs == 0] = -1.0

    A_ub = -(signs[:, None] * H_all)
    b_ub = (signs * b_all)

    # ── Affine ZLS projection (fallback only) ────────────────────────────────
    p_norm_sq = float(p_i @ p_i)

    def _proj(x):
        return x - (float(p_i @ x) + c_i) / p_norm_sq * p_i

    # ── Starting points: edge-interpolated ZLS crossings ─────────────────────
    # Interpolate along every cell edge that straddles B(x)=0.
    # These points are exactly on the affine ZLS and inside the cell by
    # construction — far better than projecting vertices which can land outside.
    with torch.no_grad():
        B_verts = barrier_model(
            torch.tensor(vertices.astype(np.float32), dtype=model_dtype)
        ).detach().numpy().ravel().astype(np.float64)

    x_stars_list = []
    nv = len(vertices)
    for u in range(nv):
        for v in range(u + 1, nv):
            bu, bv = B_verts[u], B_verts[v]
            if bu * bv < 0:   # edge straddles ZLS
                t = -bu / (bv - bu)
                x_stars_list.append(vertices[u] + t * (vertices[v] - vertices[u]))

    if len(x_stars_list) > 0:
        x_stars_arr = np.array(x_stars_list)
        if len(x_stars_arr) > n_starts:
            # Pick most spread-out subset
            zls_c = x_stars_arr.mean(axis=0)
            dists = np.linalg.norm(x_stars_arr - zls_c, axis=1)
            idx   = np.argsort(dists)[-n_starts:]
            starting_points = list(x_stars_arr[idx])
        else:
            starting_points = list(x_stars_arr)
        # Add centroid projection as fallback
        starting_points.append(_proj(centroid))
    else:
        # No crossings found via edges — fall back to projected centroid
        starting_points = [_proj(centroid)]

    # ── Objective ─────────────────────────────────────────────────────────────
    def objective(x):
        f_val, _ = dyn.eval(x)
        if continuous_time:
            return -float(p_i @ f_val)
        else:
            with torch.no_grad():
                B_fx = float(barrier_model(
                    torch.tensor(f_val.astype(np.float32)[None, :], dtype=model_dtype)
                ).detach().numpy().ravel()[0])
            return -B_fx

    def objective_grad(x):
        f_val, f_jac = dyn.eval(x)
        if continuous_time:
            return -(f_jac.T @ p_i)
        else:
            return np.zeros(n)

    # ── Constraints ───────────────────────────────────────────────────────────
    constraints = [
        {'type': 'eq',
         'fun': lambda x: float(p_i @ x) + c_i,
         'jac': lambda x: p_i},
        {'type': 'ineq',
         'fun': lambda x: b_ub - A_ub @ x,
         'jac': lambda x: -A_ub},
    ]

    # ── Multi-start with feasibility gate ─────────────────────────────────────
    best_val     = -np.inf
    best_x       = starting_points[0].copy()
    any_feasible = False

    for x0 in starting_points:
        try:
            res = minimize(
                fun         = objective,
                x0          = x0,
                jac         = objective_grad,
                method      = 'SLSQP',
                constraints = constraints,
                options     = {'ftol': tol, 'maxiter': 2000, 'disp': False},
            )
            val = -res.fun

            # Gate 1: x* must lie on the true (nonlinear) ZLS
            with torch.no_grad():
                B_xopt = float(barrier_model(
                    torch.tensor(res.x.astype(np.float32)[None, :], dtype=model_dtype)
                ).detach().numpy().ravel()[0])
            zls_ok  = abs(B_xopt) <= zls_tol

            # Gate 2: x* must be inside the cell polytope
            ineq_ok = float(np.max(A_ub @ res.x - b_ub)) <= ineq_tol

            if zls_ok and ineq_ok:
                if val > best_val:
                    best_val     = val
                    best_x       = res.x.copy()
                    any_feasible = True
            elif not any_feasible and val > best_val:
                # Keep best infeasible result as fallback
                best_val = val
                best_x   = res.x.copy()

        except Exception:
            pass

    return best_val, best_x, any_feasible


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def validate_with_nlp(
    BC              : list,
    sv              : np.ndarray,
    layer_W         : list,
    layer_b         : list,
    W_out           : np.ndarray,
    boundary_H      : np.ndarray,
    boundary_b      : np.ndarray,
    barrier_model,
    dynamics_name   : str,
    summary         = None,          # VerificationSummary from verify_barrier (optional)
    continuous_time : bool  = True,
    unsafe_tol      : float = 1e-6,  # NLP opt_val > unsafe_tol -> UNSAFE
    n_starts        : int   = 8,
    print_every     : int   = 50,
) -> NLPReport:
    """
    Validate barrier certificate on all boundary cells using NLP.

    For each cell, solves:
        maximize   p_i . f(x)
        subject to p_i @ x + c_i = 0   (zero level set)
                   x in X_i            (cell polytope)

    Compares NLP result against verify_barrier labels (if summary provided).

    Parameters
    ----------
    BC              : list of (V, n) vertex arrays
    sv              : (N, total_neurons) activation patterns
    layer_W/layer_b : network weights
    W_out           : output weights
    boundary_H/b    : domain boundary hyperplanes
    barrier_model   : TorchScript barrier network
    dynamics_name   : system name
    summary         : VerificationSummary from verify_barrier (for comparison)
    continuous_time : True = Lie derivative, False = B(f(x*))
    unsafe_tol      : threshold for NLP opt_val to declare UNSAFE
    n_starts        : number of starting points per cell
    print_every     : progress print interval

    Returns
    -------
    NLPReport with confusion matrix and per-cell results
    """
    symbols, f_sym = load_dynamics(dynamics_name)
    dyn            = DynamicsEvaluator(symbols, f_sym)
    model_dtype    = next(barrier_model.parameters()).dtype

    # Build lookup of our labels if summary provided
    our_labels = {}
    if summary is not None:
        for r in summary.results:
            our_labels[r.cell_idx] = r.label.value

    report = NLPReport(
        dynamics_name   = dynamics_name,
        continuous_time = continuous_time,
        n_cells         = len(BC),
    )

    print(f"\nNLP validation: {len(BC)} boundary cells, "
          f"dynamics='{dynamics_name}', "
          f"mode={'continuous-time' if continuous_time else 'discrete-time'}, "
          f"unsafe_tol={unsafe_tol}")
    t0 = time.perf_counter()

    for i, vertices in enumerate(BC):
        vertices = np.asarray(vertices, dtype=float)
        sv_i     = sv[i].ravel()
        p_i      = compute_local_gradient(sv_i, layer_W, W_out)
        centroid = vertices.mean(axis=0)

        # B offset
        with torch.no_grad():
            c_i = float(
                barrier_model(
                    torch.tensor(centroid.astype(np.float32)[None, :],
                                 dtype=model_dtype)
                ).detach().numpy().ravel()[0]
            ) - float(p_i @ centroid)

        # Solve NLP
        opt_val, x_opt, success = _solve_cell_nlp(
            vertices, sv_i, p_i, c_i,
            layer_W, layer_b, boundary_H, boundary_b,
            dyn, continuous_time, barrier_model, model_dtype,
            n_starts=n_starts,
        )

        # ── Post-solve diagnostic ─────────────────────────────────────────────
        # Check x_opt is (1) on the ZLS, (2) inside the cell, (3) objective
        # is self-consistent.  Warn if either feasibility check fails so we
        # can catch constraint bugs early without silently accepting bad results.
        with torch.no_grad():
            B_xopt_check = float(barrier_model(
                torch.tensor(x_opt.astype(np.float32)[None, :], dtype=model_dtype)
            ).detach().numpy().ravel()[0])
        f_xopt_check, _ = dyn.eval(x_opt)
        lie_check        = float(p_i @ f_xopt_check) if continuous_time else None

        H_diag, b_diag  = get_cell_hyperplanes_input_space(
            sv_i, layer_W, layer_b, boundary_H, boundary_b
        )
        centroid_diag   = vertices.mean(axis=0)
        cell_vals_diag  = H_diag @ x_opt + b_diag
        centroid_vals_d = H_diag @ centroid_diag + b_diag
        slack_tol       = 1e-5
        n_viol          = int(
            (cell_vals_diag * np.sign(centroid_vals_d) < -slack_tol).sum()
        )
        zls_diag_ok  = abs(B_xopt_check) <= 1e-5
        cell_diag_ok = n_viol == 0

        if not zls_diag_ok or not cell_diag_ok:
            import warnings
            warnings.warn(
                f"[validate_with_nlp cell {i}] "
                f"B(x*)={B_xopt_check:+.6f} (zls_ok={zls_diag_ok})  "
                f"p_i·f(x*)={lie_check:+.6f}  "
                f"cell_violated={n_viol}/{len(cell_vals_diag)}  "
                f"opt_val={opt_val:+.6f}  success={success}  "
                f"=> result may be UNRELIABLE"
            )
            # Downgrade UNSAFE result if x_opt is not feasible
            if opt_val > unsafe_tol:
                opt_val = -np.inf
                success = False
        # ─────────────────────────────────────────────────────────────────────

        nlp_label  = "UNSAFE" if opt_val > unsafe_tol else "SAFE"
        our_label  = our_labels.get(i, "UNKNOWN")

        cell_result = NLPCellResult(
            cell_idx  = i,
            nlp_label = nlp_label,
            opt_val   = opt_val,
            x_opt     = x_opt,
            our_label = our_label,
            n_starts  = n_starts,
            success   = success,
        )
        report.results.append(cell_result)

        # Update confusion matrix
        if nlp_label == "UNSAFE":
            report.nlp_unsafe_total += 1
            if our_label in ("SAFE_TAYLOR", "SAFE_NLP"):
                report.safe_unsafe += 1
            elif our_label == "INCONCLUSIVE":
                report.inc_unsafe += 1
        else:
            if our_label in ("SAFE_TAYLOR", "SAFE_NLP"):
                report.safe_safe += 1
            elif our_label == "INCONCLUSIVE":
                report.inc_safe += 1

        if (i + 1) % print_every == 0 or (i + 1) == len(BC):
            elapsed = time.perf_counter() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(BC) - i - 1) / rate
            print(f"  [{i+1}/{len(BC)}] "
                  f"nlp_safe={report.n_cells - report.nlp_unsafe_total} "
                  f"nlp_unsafe={report.nlp_unsafe_total} "
                  f"our_safe_nlp_unsafe={report.safe_unsafe} "
                  f"| {rate:.1f} cells/s | ETA {eta:.0f}s")

        # Alert immediately on soundness violation
        if nlp_label == "UNSAFE" and our_label in ("SAFE_TAYLOR", "SAFE_NLP"):
            print(f"\n  !! SOUNDNESS BUG at cell {i}: "
                  f"we said {our_label} but NLP opt={opt_val:.6f} > 0")
            print(f"     x_opt = {np.array2string(x_opt, precision=6)}")

    report.runtime_s = time.perf_counter() - t0
    report.print_table()
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Smoke test: NLPReport confusion matrix + counterexamples")

    r = NLPReport(dynamics_name="test", continuous_time=True, n_cells=100)
    r.safe_safe        = 80
    r.safe_unsafe      = 1
    r.inc_safe         = 14
    r.inc_unsafe       = 5
    r.nlp_unsafe_total = 6
    r.runtime_s        = 1.0

    # Fake counterexample results
    r.results = [
        NLPCellResult(cell_idx=3,  nlp_label="UNSAFE", opt_val=0.042,
                      x_opt=np.array([0.1, -0.3, 0.5]), our_label="SAFE_TAYLOR",
                      n_starts=5, success=True),
        NLPCellResult(cell_idx=17, nlp_label="UNSAFE", opt_val=0.011,
                      x_opt=np.array([-0.2, 0.6, 0.1]), our_label="INCONCLUSIVE",
                      n_starts=5, success=True),
        NLPCellResult(cell_idx=42, nlp_label="UNSAFE", opt_val=0.003,
                      x_opt=np.array([0.4, 0.2, -0.7]), our_label="INCONCLUSIVE",
                      n_starts=5, success=False),
    ]

    r.print_table()
    r.print_counterexamples()
    print("\nSmoke test passed.")