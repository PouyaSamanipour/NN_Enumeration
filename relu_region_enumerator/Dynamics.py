"""
dynamics.py
===========
All system dynamics in SymPy, plus a registry to load them by name.

Each system is defined as a function get_dynamics_<name>() returning:
    symbols : tuple of sympy symbols, length n
    f       : list of n sympy expressions representing f(x)

Usage
-----
    from dynamics import load_dynamics
    symbols, f = load_dynamics("arch3")
"""

import sympy as sp
import numpy as np
from scipy.linalg import solve_discrete_are


# ═══════════════════════════════════════════════════════════════════════════
# Arch3  (2D polynomial)
# ═══════════════════════════════════════════════════════════════════════════
def get_dynamics_arch3():
    """
    2D nonlinear polynomial system.
    Domain: [-3, 3]^2
    Reference: Ren et al. 2026 (Arch3 benchmark)
    """
    x1, x2 = sp.symbols('x1 x2')
    symbols = (x1, x2)
    f = [
        x1 - x1**3 + x2 - x1*x2**2,
        -x1 + x2 - x1**2*x2 - x2**3,
    ]
    return symbols, f


# ═══════════════════════════════════════════════════════════════════════════
# Complex  (3D non-polynomial)
# ═══════════════════════════════════════════════════════════════════════════
def get_dynamics_complex():
    """
    3D nonlinear non-polynomial system.
    Domain: [-3, 3]^3
    Reference: Ren et al. 2026 (Complex benchmark)
    """
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    symbols = (x1, x2, x3)
    f = [
        -x1 * (1 + sp.sin(x2)**2 + sp.exp(-x3**2)),
        -x2 * (1 + sp.cos(x3)**2 + sp.tanh(x1**2)),
        -x3 * (1 + sp.log(1 + x1**2 + x2**2)),
    ]
    return symbols, f


# ═══════════════════════════════════════════════════════════════════════════
# Quadrotor  (12D, Drake RPY parameterization)
# ═══════════════════════════════════════════════════════════════════════════

# ── Physical parameters (Drake QuadrotorPlant defaults, matches Quad.py) ──
_m   = 0.775
_g   = 9.81
_l   = 0.15
_kF  = 1.0
_kM  = 0.0245
_I   = np.diag([0.0015, 0.0025, 0.006])
_I_inv = np.linalg.inv(_I)
_dt  = 0.05

# ── LQR gain (recomputed to match Quad.py exactly) ────────────────────────
_A_c = np.zeros((12, 12))
_A_c[0,6]=1; _A_c[1,7]=1; _A_c[2,8]=1
_A_c[3,9]=1; _A_c[4,10]=1; _A_c[5,11]=1
_A_c[6,4]=_g; _A_c[7,3]=-_g

_B_c = np.zeros((12, 4))
_B_c[8,:]  = _kF / _m
_B_c[9,1]  =  _kF * _l / _I[0,0];  _B_c[9,3]  = -_kF * _l / _I[0,0]
_B_c[10,2] =  _kF * _l / _I[1,1];  _B_c[10,0] = -_kF * _l / _I[1,1]
_B_c[11,0] =  _kM / _I[2,2];       _B_c[11,1] = -_kM / _I[2,2]
_B_c[11,2] =  _kM / _I[2,2];       _B_c[11,3] = -_kM / _I[2,2]

_A_d = np.eye(12) + _dt * _A_c
_B_d = _dt * _B_c
_Q   = np.diag([1,1,1,0.1,0.1,0.1,0.1,0.1,0.1,0.01,0.01,0.01])
_R   = np.diag([0.1,0.1,0.1,0.1])
_P   = solve_discrete_are(_A_d, _B_d, _Q, _R)
_K   = np.linalg.inv(_R + _B_d.T @ _P @ _B_d) @ (_B_d.T @ _P @ _A_d)
_u_eq = np.array([_m * _g / (4 * _kF)] * 4)


def get_dynamics_quadrotor():
    """
    12D quadrotor with LQR controller (unclipped, valid near hover).
    State: [px, py, pz, roll, pitch, yaw, px_dot, py_dot, pz_dot,
            omega_x, omega_y, omega_z]
    Domain: [-1,1]^3 x [-1,1]^3 x [-0.3,0.3]^3 x [-1,1]^3
    Reference: Dai et al. CDC 2024 (quadrotor benchmark)

    Control: u = u_eq - K @ x  (linearized LQR, clipping inactive near hover)
    """
    syms = sp.symbols('x1:13')   # x1 ... x12
    px, py, pz, roll, pitch, yaw, pxd, pyd, pzd, ox, oy, oz = syms

    # Trig shorthands
    cr  = sp.cos(roll);   sr  = sp.sin(roll)
    cp  = sp.cos(pitch);  sp_ = sp.sin(pitch)
    cy  = sp.cos(yaw);    sy  = sp.sin(yaw)

    # Control: u = u_eq - K @ x
    K_sym    = sp.Matrix(_K.tolist())
    x_vec    = sp.Matrix(list(syms))
    u_eq_sym = sp.Matrix(_u_eq.tolist())
    u        = u_eq_sym - K_sym * x_vec       # (4,1)

    # Thrust and torques
    uF      = _kF * u
    F_total = sum(uF)
    Mx      = _l * (uF[1] - uF[3])
    My      = _l * (uF[2] - uF[0])
    Mz      = _kM * (u[0] - u[1] + u[2] - u[3])
    tau     = sp.Matrix([Mx, My, Mz])

    # Translational acceleration (thrust direction = third col of R_NB)
    pxdd = (_kF * F_total * sp_)          / _m
    pydd = (_kF * F_total * (-sr * cp))   / _m
    pzdd = (_kF * F_total * ( cr * cp))   / _m - _g

    # RPY rates from body angular velocity (Drake CalcRpyDtFromAngularVelocityInChild)
    one_over_cp = 1 / cp
    rolld  = ox + sr * one_over_cp * sp_ * oy + cr * one_over_cp * sp_ * oz
    pitchd =      cr * oy                     - sr * oz
    yawd   =      sr * one_over_cp * oy       + cr * one_over_cp * oz

    # Angular acceleration: I^-1 (tau - omega x I*omega)
    I_sym    = sp.Matrix(_I.tolist())
    I_inv_sym = sp.Matrix(_I_inv.tolist())
    omega    = sp.Matrix([ox, oy, oz])
    Iomega   = I_sym * omega
    wIw      = omega.cross(Iomega)
    alpha    = I_inv_sym * (tau - wIw)

    f = [
        pxd, pyd, pzd,                   # pos_dot      (linear)
        rolld, pitchd, yawd,             # rpy_dot      (trig + 1/cos)
        pxdd, pydd, pzdd,                # pos_ddot     (trig)
        alpha[0], alpha[1], alpha[2],    # omega_dot    (quadratic in omega)
    ]

    return syms, f


# ═══════════════════════════════════════════════════════════════════════════
# Decay  (6D polynomial)
# ═══════════════════════════════════════════════════════════════════════════
def get_dynamics_decay():
    """
    6D nonlinear polynomial system — each component decays toward origin.
    Domain: [-3, 3]^6
    Reference: Ren et al. 2026 (Decay benchmark), Zhang et al. NeurIPS 2023

    Dynamics:
        x_dot_i = -x_i * (1 + x1^2 + x2^2 + x3^2 + x4^2 + x5^2 + x6^2)
    for i = 1, ..., 6.

    The common factor r^2 = sum_i x_i^2 makes this a radially symmetric
    contracting system. The origin is globally asymptotically stable.
    """
    syms = sp.symbols('x1:7')   # x1, x2, x3, x4, x5, x6
    x1, x2, x3, x4, x5, x6 = syms

    r_sq = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
    factor = 1 + r_sq

    f = [
        -x1 * factor,
        -x2 * factor,
        -x3 * factor,
        -x4 * factor,
        -x5 * factor,
        -x6 * factor,
    ]
    return syms, f

# ═══════════════════════════════════════════════════════════════════════════
# Decay-x1  (6D, adds cubic dissipation per component)
# ═══════════════════════════════════════════════════════════════════════════
def get_dynamics_decay_x1():
    """
    6D decay system augmented with per-component cubic dissipation.
    Dynamics: x_dot_i = -x_i*(1 + ||x||²) - x_i³
    The added -x_i³ term is odd and dissipative; the barrier certificate
    valid for Decay remains valid here (SAFE verdict is preserved).
    """
    syms = sp.symbols('x1:7')
    x1, x2, x3, x4, x5, x6 = syms
    r_sq = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
    factor = 1 + r_sq
    f = [
        -x1 * factor - x1,
        -x2 * factor - x2,
        -x3 * factor - x3,
        -x4 * factor - x4,
        -x5 * factor - x5,
        -x6 * factor - x6,
    ]
    return syms, f
# ═══════════════════════════════════════════════════════════════════════════
# Decay-x3  (6D, adds cubic dissipation per component)
# ═══════════════════════════════════════════════════════════════════════════
def get_dynamics_decay_x3():
    """
    6D decay system augmented with per-component cubic dissipation.
    Dynamics: x_dot_i = -x_i*(1 + ||x||²) - x_i³
    The added -x_i³ term is odd and dissipative; the barrier certificate
    valid for Decay remains valid here (SAFE verdict is preserved).
    """
    syms = sp.symbols('x1:7')
    x1, x2, x3, x4, x5, x6 = syms
    r_sq = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
    factor = 1 + r_sq
    f = [
        -x1 * factor -x1+ x1**3,
        -x2 * factor -x2+ x2**3,
        -x3 * factor -x3+ x3**3,
        -x4 * factor -x4+ x4**3,
        -x5 * factor -x5+ x5**3,
        -x6 * factor -x6+ x6**3,
    ]
    return syms, f


# ═══════════════════════════════════════════════════════════════════════════
# Decay-x5  (6D, adds cubic + quintic dissipation per component)
# ═══════════════════════════════════════════════════════════════════════════
def get_dynamics_decay_x5():
    """
    6D decay system augmented with per-component cubic and quintic dissipation.
    Dynamics: x_dot_i = -x_i*(1 + ||x||²) - x_i³ - x_i⁵
    Both added terms are odd and dissipative; SAFE verdict is preserved.
    The quintic term raises the Hessian degree further, inflating M_i bounds.
    """
    syms = sp.symbols('x1:7')
    x1, x2, x3, x4, x5, x6 = syms
    r_sq = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
    factor = 1 + r_sq
    f = [
        -x1 * factor -x1+ x1**3 - x1**5,
        -x2 * factor -x2+ x2**3 - x2**5,
        -x3 * factor -x3+ x3**3 - x3**5,
        -x4 * factor -x4+ x4**3 - x4**5,
        -x5 * factor -x5+ x5**3 - x5**5,
        -x6 * factor -x6+ x6**3 - x6**5,
    ]
    return syms, f

def get_dynamics_decay_x7():
    """
    6D decay system augmented with per-component cubic and quintic dissipation.
    Dynamics: x_dot_i = -x_i*(1 + ||x||²) - x_i³ - x_i⁵
    Both added terms are odd and dissipative; SAFE verdict is preserved.
    The quintic term raises the Hessian degree further, inflating M_i bounds.
    """
    syms = sp.symbols('x1:7')
    x1, x2, x3, x4, x5, x6 = syms
    r_sq = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
    factor = 1 + r_sq
    f = [
        -x1 * factor -x1+ x1**3 - x1**5+ x1**7,
        -x2 * factor -x2+ x2**3 - x2**5+ x2**7,
        -x3 * factor -x3+ x3**3 - x3**5+ x3**7,
        -x4 * factor -x4+ x4**3 - x4**5+ x4**7,
        -x5 * factor -x5+ x5**3 - x5**5+ x5**7,
        -x6 * factor -x6+ x6**3 - x6**5+ x6**7,
    ]
    return syms, f
def get_dynamics_decay_sin():
    """
    6D decay system augmented with per-component cubic and quintic dissipation.
    Dynamics: x_dot_i = -x_i*(1 + ||x||²) - x_i³ - x_i⁵
    Both added terms are odd and dissipative; SAFE verdict is preserved.
    The quintic term raises the Hessian degree further, inflating M_i bounds.
    """
    syms = sp.symbols('x1:7')
    x1, x2, x3, x4, x5, x6 = syms
    r_sq = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
    factor = 1 + r_sq
    f = [
        -x1 * factor - sp.sin(x1),
        -x2 * factor - sp.sin(x2),
        -x3 * factor - sp.sin(x3),
        -x4 * factor - sp.sin(x4),
        -x5 * factor - sp.sin(x5),
        -x6 * factor - sp.sin(x6),
    ]
    return syms, f


# ═══════════════════════════════════════════════════════════════════════════
# Hi-Ord8  (8D linear, companion form)
# ═══════════════════════════════════════════════════════════════════════════
def get_dynamics_hiord8():
    """
    8th-order linear ODE in companion (state-space) form.
    Reference: SEEV benchmark (hi-ord8).

    Scalar ODE:
        x^(8) + 20x^(7) + 170x^(6) + 800x^(5) + 2273x^(4)
              + 3980x^(3) + 4180x^(2) + 2400x^(1) + 576x = 0

    State: [x1, x2, ..., x8] = [x, x^(1), ..., x^(7)]

    Dynamics:
        x1_dot = x2
        x2_dot = x3
        x3_dot = x4
        x4_dot = x5
        x5_dot = x6
        x6_dot = x7
        x7_dot = x8
        x8_dot = -576*x1 - 2400*x2 - 4180*x3 - 3980*x4
                 - 2273*x5 - 800*x6 - 170*x7 - 20*x8

    Domain:  x1^2 + ... + x8^2 <= 4  (sphere radius 2)
    Initial: (x1-1)^2 + ... + (x8-1)^2 <= 1
    Safe:    (x1+2)^2 + ... + (x8+2)^2 >= 3

    Note: the dynamics are linear so the Hessian is identically zero —
    the Taylor remainder bound is always 0 and every cell is decided by
    the exact check alone (no refinement needed).
    """
    syms = sp.symbols('x1:9')   # x1, ..., x8
    x1, x2, x3, x4, x5, x6, x7, x8 = syms

    f = [
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        -576*x1 - 2400*x2 - 4180*x3 - 3980*x4
        - 2273*x5 - 800*x6 - 170*x7 - 20*x8,
    ]
    return syms, f


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

_REGISTRY = {
    "arch3"     : get_dynamics_arch3,
    "complex"   : get_dynamics_complex,
    "quadrotor" : get_dynamics_quadrotor,
    "decay"     : get_dynamics_decay,
    "decay_x1"  : get_dynamics_decay_x1,
    "decay_x3"  : get_dynamics_decay_x3,
    "decay_x5"  : get_dynamics_decay_x5,
    "decay_x7"  : get_dynamics_decay_x7,
    "decay_sin" : get_dynamics_decay_sin, 
    "hiord8"    : get_dynamics_hiord8,
}


def load_dynamics(name: str):
    """
    Return (symbols, f) for the named system.

    Parameters
    ----------
    name : str
        One of: 'arch3', 'complex', 'quadrotor'

    Returns
    -------
    symbols : tuple of sympy.Symbol
    f       : list of sympy expressions
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dynamics '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()


def list_systems():
    """Return list of registered system names."""
    return list(_REGISTRY.keys())


# ── Smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for name in list_systems():
        syms, f = load_dynamics(name)
        n = len(syms)
        print(f"\n{'='*50}")
        print(f"System  : {name}")
        print(f"Dim     : n={n}, len(f)={len(f)}")
        # Spot-check: second derivative of first component wrt first state
        H00 = sp.diff(f[0], syms[0], syms[0])
        print(f"d²f₀/dx₁² = {H00}")
        # Confirm all components are valid sympy expressions
        assert all(isinstance(fi, sp.Basic) for fi in f), "Non-sympy entry in f"
        print(f"All {n} components are valid sympy expressions.")
    print("\nAll systems loaded successfully.")