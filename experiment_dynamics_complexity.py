"""
experiment_dynamics_complexity.py
==================================
Dynamics Complexity vs. Verification Difficulty ablation.

Setup
-----
- Fixed pretrained 6-10-σ-10-σ-1 Decay barrier network.
- Fixed domain TH = [2.0]*6, same boundary cells for all three runs.
- Only the dynamics change:
    decay    : x_dot_i = -x_i * (1 + Σ x_j²)
    decay_x3 : x_dot_i = -x_i * (1 + Σ x_j²) - x_i³
    decay_x5 : x_dot_i = -x_i * (1 + Σ x_j²) - x_i³ - x_i⁵

The negative sign on each added term preserves the SAFE verdict by
construction; any cost increase is unambiguously due to wider Hessian
bounds M_i driven by higher-degree dynamics.

Usage
-----
    python experiment_dynamics_complexity.py
"""

import copy
import os
import time
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt

from relu_region_enumerator.verify_certificate_face import verify_barrier

# ── Configuration ────────────────────────────────────────────────────────────
NN_FILE     = "NN_files/model_decay_2_10_ren.pt"
BC_H5       = "decay_boundary_cells.h5"
TH          = [2.0] * 6
DYNAMICS    = ["decay", "decay_x1", "decay_x3", "decay_x5","decay_x7", "decay_sin"]

# ── Load barrier model and extract network weights ────────────────────────────
barrier_model = torch.jit.load(NN_FILE, map_location="cpu")
barrier_model.eval()

params = [p.detach().numpy() for _, p in barrier_model.named_parameters()]
# params layout: [W1, b1, W2, b2, ..., W_out, b_out]
num_hidden = (len(params) - 2) // 2
layer_W = [params[2 * i]     for i in range(num_hidden)]
layer_b = [params[2 * i + 1] for i in range(num_hidden)]
W_out   = params[-2]

n = layer_W[0].shape[1]   # input dimension = 6

# ── Build domain boundary hyperplanes ────────────────────────────────────────
#  [I; -I] @ x <= TH  defines the hypercube [-TH, TH]^n
boundary_H = np.vstack((np.eye(n), -np.eye(n)))
boundary_b = TH + TH   # length 2n

# ── Load boundary cells from HDF5 ────────────────────────────────────────────
print(f"Loading boundary cells from: {BC_H5}")
with h5py.File(BC_H5, "r") as f:
    offsets  = f["offsets"][:]          # (N_cells+1,)
    vertices = f["vertices"][:]         # (total_verts, n)
    sv_all   = f["activation_patterns"][:]  # (N_cells, total_neurons)

# N_cells = len(offsets) - 1
N_cells = 500
BC_master = [
    vertices[offsets[i]:offsets[i + 1]].astype(np.float64)
    for i in range(N_cells)
]
print(f"  Loaded {N_cells} boundary cells, activation pattern shape: {sv_all.shape}")

# ── Run verification for each dynamics variant ────────────────────────────────
results = {}

for dyn_name in DYNAMICS:
    print(f"\n{'='*60}")
    print(f"  Dynamics: {dyn_name}")
    print(f"{'='*60}")

    # Deep-copy BC so each run starts from the same vertex data.
    # verify_barrier nulls entries in-place to free memory.
    BC_copy = [cell.copy() for cell in BC_master]
    sv_copy = sv_all.copy()

    t0      = time.perf_counter()
    summary = verify_barrier(
        BC              = BC_copy,
        sv              = sv_copy,
        layer_W         = layer_W,
        layer_b         = layer_b,
        W_out           = W_out,
        boundary_H      = boundary_H,
        boundary_b      = boundary_b,
        barrier_model   = barrier_model,
        dynamics_name   = dyn_name,
        continuous_time = True,
        early_exit      = False,
        refinement_max_depth = 20,
        TH              = TH,
        n_refs          = 10,
    )
    elapsed = time.perf_counter() - t0

    results[dyn_name] = dict(
        n_safe_taylor     = int(summary.n_safe_taylor),
        n_safe_refinement = int(summary.n_safe_refinement),
        n_unsafe          = int(summary.n_unsafe),
        n_inconclusive    = int(summary.n_inconclusive),
        runtime_s         = float(summary.runtime_s),
        # Aggregate M_i and refinement depth from per-cell results
        M_vals   = np.array([r.M_i      for r in summary.results], dtype=np.float64),
        r_vals   = np.array([r.r_i      for r in summary.results], dtype=np.float64),
        rem_vals = np.array([r.remainder for r in summary.results], dtype=np.float64),
    )

    print("  Run summary: "
          f"SAFE_TAYLOR={results[dyn_name]['n_safe_taylor']} "
          f"SAFE_REFINEMENT={results[dyn_name]['n_safe_refinement']} "
          f"UNSAFE={results[dyn_name]['n_unsafe']} "
          f"INCONCLUSIVE={results[dyn_name]['n_inconclusive']} "
          f"runtime={results[dyn_name]['runtime_s']:.2f}s")

# ── Print comparison table ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DYNAMICS COMPLEXITY vs. VERIFICATION DIFFICULTY — COMPARISON")
print("=" * 70)

header = f"{'Metric':<28}" + "".join(f"{d:>14}" for d in DYNAMICS)
print(header)
print("-" * 70)

def _safe_scalar(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        if len(v) == 0:
            return float("nan")
        return v[0] if len(v) == 1 else len(v)
    return v

def _row(label, key, fmt=".0f"):
    vals = []
    for d in DYNAMICS:
        v = results[d][key]
        if isinstance(v, (list, tuple, np.ndarray)):
            if key == "M_vals" or key == "rem_vals":
                v = float(np.mean(v))
            else:
                v = len(v)
        vals.append(v)
    spec = f">14{fmt}"
    row  = f"{label:<28}" + "".join(format(float(v), spec) for v in vals)
    print(row)

def plot_results(results, out_png="experiment_dynamics_complexity_plot.png"):
    fig, axs = plt.subplots(3, 1, figsize=(10, 14), constrained_layout=True)
    m_data = [results[d]["M_vals"] for d in DYNAMICS]
    r_data = [results[d]["r_vals"] for d in DYNAMICS]

    axs[0].boxplot(m_data, labels=DYNAMICS, showfliers=False, patch_artist=True)
    axs[0].set_title("M_i distribution by dynamics")
    axs[0].set_ylabel(r"$M_i$")
    axs[0].set_yscale("log")
    axs[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    axs[1].boxplot(r_data, labels=DYNAMICS, showfliers=False, patch_artist=True)
    axs[1].set_title("r_i distribution by dynamics")
    axs[1].set_ylabel(r"$r_i$")
    axs[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    for i, d in enumerate(DYNAMICS):
        axs[2].scatter(
            results[d]["r_vals"],
            results[d]["M_vals"],
            s=10,
            alpha=0.35,
            label=d,
        )
    axs[2].set_title("M_i vs r_i by dynamics")
    axs[2].set_xlabel(r"$r_i$")
    axs[2].set_ylabel(r"$M_i$")
    axs[2].set_yscale("log")
    axs[2].legend()
    axs[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Dynamics complexity ablation: $M_i$ and $r_i$ comparison", fontsize=16)
    fig.savefig(out_png, dpi=200)
    print(f"Saved comparison plot to {out_png}")


def _row_stat(label, key, stat_fn, fmt=".4f"):
    vals = []
    for d in DYNAMICS:
        v = results[d][key]
        if isinstance(v, (list, tuple, np.ndarray)):
            v = np.array(v, dtype=np.float64)
            v = stat_fn(v) if v.size else float("nan")
        else:
            v = stat_fn(v)
        vals.append(v)
    spec = f">14{fmt}"
    row  = f"{label:<28}" + "".join(format(float(v), spec) for v in vals)
    print(row)

_row("# boundary cells",       "n_safe_taylor",     fmt=".0f")   # just reuse to print N
# Reprint actual cell count:
total_row = f"{'# boundary cells':<28}" + "".join(f"{N_cells:>14}" for _ in DYNAMICS)
print(total_row)

_row("safe (Taylor)",          "n_safe_taylor",     fmt=".0f")
_row("safe (refinement)",      "n_safe_refinement", fmt=".0f")
_row("unsafe",                 "n_unsafe",          fmt=".0f")
_row("inconclusive",           "n_inconclusive",    fmt=".0f")
_row("runtime (s)",            "runtime_s",         fmt=".2f")

print("-" * 70)
_row_stat("mean M_i",    "M_vals",   np.mean,   fmt=".4f")
_row_stat("max  M_i",    "M_vals",   np.max,    fmt=".4f")
_row_stat("mean rem",    "rem_vals", np.mean,   fmt=".6f")
_row_stat("max  rem",    "rem_vals", np.max,    fmt=".6f")

print("=" * 70)

# Normalised cost relative to Decay baseline
baseline_rt = results["decay"]["runtime_s"]
print("\nRuntime normalised to Decay baseline:")
for d in DYNAMICS:
    ratio = results[d]["runtime_s"] / baseline_rt
    print(f"  {d:<14}: {ratio:.2f}x")

pct_inconclusive = {d: 100 * results[d]["n_inconclusive"] / N_cells for d in DYNAMICS}
print("\nInconclusive fraction (%):")
for d in DYNAMICS:
    print(f"  {d:<14}: {pct_inconclusive[d]:.1f}%")

save_path = "experiment_dynamics_complexity_results.npz"
save_kwargs = {}
for d in DYNAMICS:
    save_kwargs[f"{d}_M_vals"] = results[d]["M_vals"]
    save_kwargs[f"{d}_r_vals"] = results[d]["r_vals"]
    save_kwargs[f"{d}_runtime_s"] = np.array([results[d]["runtime_s"]], dtype=np.float64)
    save_kwargs[f"{d}_n_safe_taylor"] = np.array([results[d]["n_safe_taylor"]], dtype=np.int32)
    save_kwargs[f"{d}_n_safe_refinement"] = np.array([results[d]["n_safe_refinement"]], dtype=np.int32)
    save_kwargs[f"{d}_n_unsafe"] = np.array([results[d]["n_unsafe"]], dtype=np.int32)
    save_kwargs[f"{d}_n_inconclusive"] = np.array([results[d]["n_inconclusive"]], dtype=np.int32)
np.savez(save_path, **save_kwargs)
print(f"Saved experiment data to {save_path}")

plot_results(results)

print("\nDone.")
