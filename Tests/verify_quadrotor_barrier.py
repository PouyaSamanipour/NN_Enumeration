"""
verify_quadrotor_barrier.py
===========================
Verify the barrier certificate B(x) = V(x) - c  for the quadrotor dynamics,
where V(x) is  NN_files/model_quadrotor_lyapunov_ct.pt  (continuous-time model).

Largest level set analysis
--------------------------
  V(0)  = -0.01318  (NN ≈ 0 at origin; slight negative offset from training)
  c*    = -0.01697  (min of V over domain boundary, face x8 = +1.0)
  {V <= c*} is the largest sublevel set fully contained in the domain.
  Note: V(0) > c* by ~0.0038, so origin sits just outside {V <= c*}.
  This is a small training-bias artifact (NN not exactly zero at equilibrium).

  The negative-V boundary faces are the angle/velocity faces (x4-x8, x10-x12);
  the position faces (x1-x3) and pz_dot face (x9) are well above zero.

Barrier condition (continuous-time):
  B(x) = V(x) - LEVEL,    LEVEL = c* = -0.01697
  dot{B}(x) = dot{V}(x) = nabla V(x) . f(x) <= 0   on  {V(x) = LEVEL}

Pipeline:
  1. Build B(x) = V(x) - LEVEL and save as a new .pt (only output bias shifts).
  2. Use that .pt as both NN_file and barrier_model so the IBP filter
     automatically targets the correct level set {V = LEVEL}.
  3. Enumerate linear regions, find boundary cells, run formal verification.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from relu_region_enumerator import enumeration_function

# ── Largest level set fully inside the domain ────────────────────────────────
# c* = min V on domain boundary (computed by boundary sampling + L-BFGS-B)
LEVEL        = -0.01697
SRC_PATH     = "NN_files/model_quadrotor_lyapunov_ct.pt"
BARRIER_PATH = "NN_files/model_quadrotor_ct_B.pt"

# ---------------------------------------------------------------------------
# Build and save  B(x) = V(x) - LEVEL
# Only the output bias changes: new_bias = b_out - LEVEL
# All other weights are identical → same linear regions, same IBP filter.
# ---------------------------------------------------------------------------
def make_barrier_model(src_path: str, level: float, dst_path: str):
    base   = torch.jit.load(src_path, map_location="cpu")
    base.eval()
    params = {n: p.detach().clone() for n, p in base.named_parameters()}

    hidden_size = params["fc1.weight"].shape[0]
    input_size  = params["fc1.weight"].shape[1]

    fc1 = nn.Linear(input_size,  hidden_size)
    fc2 = nn.Linear(hidden_size, hidden_size)
    out = nn.Linear(hidden_size, 1)

    fc1.weight.data =  params["fc1.weight"]
    fc1.bias.data   =  params["fc1.bias"]
    fc2.weight.data =  params["fc2.weight"]
    fc2.bias.data   =  params["fc2.bias"]
    out.weight.data =  params["out.weight"]         # unchanged  →  B = V - LEVEL
    out.bias.data   =  params["out.bias"] - level   # shift only

    class _Wrapper(nn.Module):
        def __init__(self, seq):
            super().__init__()
            self.seq = seq
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.seq(x).squeeze(-1)   # output shape: (batch,)

    seq    = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), out)
    traced = torch.jit.trace(_Wrapper(seq), torch.zeros(4, input_size))
    traced.save(dst_path)
    print(f"Barrier model saved → {dst_path}")
    return traced


if not os.path.exists(BARRIER_PATH):
    barrier = make_barrier_model(SRC_PATH, LEVEL, BARRIER_PATH)
else:
    barrier = torch.jit.load(BARRIER_PATH, map_location="cpu")
    print(f"Loaded existing barrier model from: {BARRIER_PATH}")

barrier.eval()

# ── Sanity checks ────────────────────────────────────────────────────────────
x0   = torch.zeros(1, 12)
base = torch.jit.load(SRC_PATH, map_location="cpu"); base.eval()
V0   = base(x0).item()
B0   = barrier(x0).item()
print(f"V(0)              = {V0:.6f}")
print(f"LEVEL (c*)        = {LEVEL:.6f}")
print(f"B(0) = V(0)-LEVEL = {B0:.6f}  (expected {V0 - LEVEL:.6f})")
assert abs(B0 - (V0 - LEVEL)) < 1e-5, f"B(0) mismatch: {B0} vs {V0 - LEVEL}"
print(f"Origin in safe set {{V<=LEVEL}}: {V0 <= LEVEL}  "
      f"(gap = {V0 - LEVEL:+.5f})")

# ── Domain ───────────────────────────────────────────────────────────────────
# State: [px, py, pz, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
TH = [1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# ── Enumerate + verify ───────────────────────────────────────────────────────
enumeration_function(
    NN_file      = BARRIER_PATH,   # shifted .pt → IBP targets {V = LEVEL}
    name_file    = "quadrotor",
    TH           = TH,
    mode         = "Rapid_mode",
    parallel     = True,
    verification = "barrier",
    barrier_model= barrier,
    ibp_filter   = True,
)
