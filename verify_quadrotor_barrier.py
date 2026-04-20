"""
verify_quadrotor_barrier.py
===========================
Verify the barrier certificate B(x) = V(x) - 0.3 for the quadrotor dynamics,
where V(x) is NN_files/model_quadrotor_lyapunov.pt.

The safe set is  S = { x : V(x) <= 0.3 }
which we showed is fully contained in the domain
  x in [-1,1]^3 x [-0.3,0.3]^3 x [-1,1]^3 x [-1,1]^3  (c* = 0.318).

Barrier condition to verify (continuous-time):
  dot{B}(x) = nabla V(x) . f(x) <= 0  for all x on {V(x) = 0.3}

Pipeline:
  1. Enumerate all linear regions of V(x) in the domain.
  2. Find boundary cells  (cells straddling {V = 0.3}).
  3. Formally verify the barrier condition on each boundary cell via
     exact zero-level-set crossings + Taylor remainder + adaptive refinement.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from relu_region_enumerator import enumeration_function


# ---------------------------------------------------------------------------
# Shifted model:  B(x) = V(x) - LEVEL
# ---------------------------------------------------------------------------
LEVEL = 0.3   # the barrier level set

class ShiftedModel(nn.Module):
    """Outputs V(x) - LEVEL so that {B(x) = 0} == {V(x) = LEVEL}."""
    def __init__(self, base_model, level: float):
        super().__init__()
        self.base  = base_model
        self.level = level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) - self.level


# ---------------------------------------------------------------------------
# Load model and build the shifted barrier
# ---------------------------------------------------------------------------
base_model = torch.jit.load(
    "NN_files/model_quadrotor_lyapunov.pt", map_location="cpu"
)
base_model.eval()

barrier_model = ShiftedModel(base_model, LEVEL)
barrier_model.eval()

# Quick sanity check
x0 = torch.zeros(1, 12)
print(f"V(0)    = {base_model(x0).item():.6f}")
print(f"B(0)    = {barrier_model(x0).item():.6f}  (expected V(0) - {LEVEL})")

# ---------------------------------------------------------------------------
# Domain  TH[i]:  x_i in [-TH[i], +TH[i]]
# State:  [px, py, pz, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
# ---------------------------------------------------------------------------
TH = [1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# ---------------------------------------------------------------------------
# Run enumeration + barrier verification
# ---------------------------------------------------------------------------
# ibp_filter=False: the IBP filter assumes barrier output = raw NN output
# (i.e. zero level at V=0).  Since our barrier is V - 0.3, the filter would
# wrongly prune boundary cells with V in (0, 0.3).  Disabling it is safe
# (just slower) and ensures no boundary cells are missed.

enumeration_function(
    NN_file      = "NN_files/model_quadrotor_lyapunov.pt",
    name_file    = "quadrotor",
    TH           = TH,
    mode         = "Rapid_mode",
    parallel     = True,
    verification = "barrier",
    barrier_model= barrier_model,
    ibp_filter   = False,
)
