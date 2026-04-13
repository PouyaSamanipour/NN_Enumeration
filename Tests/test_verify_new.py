"""
test_verify_new.py
==================
Visualise the adaptive refinement on arch3 boundary cells.

For each boundary cell that has ZLS crossings, we show:
  - The 2D polytope (cell boundary)
  - The ZLS line (B(x) = 0 crossing points, shown as a segment)
  - The refinement hyperplanes that would be inserted if the cell
    were INCONCLUSIVE (n_refs cuts parallel, normal = direction of
    farthest pair of ZLS crossing points)

This illustrates the algorithm geometrically regardless of whether
Taylor alone certifies the cell.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from scipy.spatial import ConvexHull

# ── internal imports ──────────────────────────────────────────────────────
from relu_region_enumerator.hessian_bound      import HessianBounder, compute_local_gradient
from relu_region_enumerator.Dynamics           import load_dynamics
from relu_region_enumerator.verify_certificate_new import (
    DynamicsEvaluator, _get_zero_level_set_crossings, _two_step_label, Label
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Setup
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PATH  = "NN_files/Arch3_2_96.pt"
BOUNDARY_H5 = "arch_boundary_cells.h5"
TH          = [3.0, 3.0]
DYNAMICS    = "arch3"

print("Loading model …")
barrier_model = torch.jit.load(MODEL_PATH, map_location="cpu")
barrier_model.eval()
model_dtype = next(barrier_model.parameters()).dtype   # float64

params  = [p.detach().numpy() for _, p in barrier_model.named_parameters()]
layer_W = [params[0], params[2]]
layer_b = [params[1], params[3]]
W_out   = params[4]

n = layer_W[0].shape[1]   # 2
boundary_H = np.vstack((np.eye(n), -np.eye(n)))
boundary_b = np.array(TH + TH, dtype=np.float64)
total_neurons = sum(W.shape[0] for W in layer_W)
use_wide = (total_neurons + len(boundary_H) > 64)

print("Loading boundary cells …")
with h5py.File(BOUNDARY_H5, "r") as f:
    offsets  = f["offsets"][:]
    vertices = f["vertices"][:]
    sv_all   = f["activation_patterns"][:]

BC = [vertices[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
print(f"  {len(BC)} boundary cells")

print("Building dynamics …")
symbols, f_sym = load_dynamics(DYNAMICS)
hb  = HessianBounder(symbols, f_sym)
dyn = DynamicsEvaluator(symbols, f_sym)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Collect cells with ZLS crossings + their two-step results
# ═══════════════════════════════════════════════════════════════════════════

CellInfo = []   # (cell_idx, verts, x_stars, label, remainder, worst_cond, n_refs, normal, cuts)

for i, verts in enumerate(BC):
    verts  = np.asarray(verts, dtype=float)
    sv_i   = sv_all[i].ravel()
    p_i    = compute_local_gradient(sv_i, layer_W, W_out)
    np_dt  = np.float64 if model_dtype == torch.float64 else np.float32

    with torch.no_grad():
        B_vals = barrier_model(
            torch.tensor(verts.astype(np_dt), dtype=model_dtype)
        ).numpy().ravel()

    x_stars, x_masks = _get_zero_level_set_crossings(
        verts, sv_i, B_vals,
        layer_W, layer_b, boundary_H, boundary_b, use_wide,
        barrier_model=barrier_model, model_dtype=model_dtype,
    )

    if len(x_stars) < 2:
        continue

    label, M_i, r_i, remainder, worst_cond, ce = _two_step_label(
        x_stars, barrier_model, dyn, hb, p_i, i, continuous_time=True
    )

    # ── Real n_refs ───────────────────────────────────────────────────────
    abs_f_lin   = max(abs(worst_cond), 1e-10)
    n_refs_real = max(2, int(np.ceil(remainder / abs_f_lin)))
    n_refs_real = min(n_refs_real, 32)

    # Farthest pair
    E = len(x_stars)
    dists = np.linalg.norm(x_stars[:, None] - x_stars[None, :], axis=2)
    flat  = int(np.argmax(dists))
    ia, ib = divmod(flat, E)
    max_dist = float(dists[ia, ib])

    if max_dist < 1e-12:
        continue

    normal = (x_stars[ib] - x_stars[ia]) / max_dist

    CellInfo.append(dict(
        idx=i, verts=verts, x_stars=x_stars,
        label=label, remainder=remainder, worst_cond=worst_cond,
        n_refs_real=n_refs_real, max_dist=max_dist,
        normal=normal,
        p_i=p_i, ia=ia, ib=ib,
    ))

print(f"\nCells with ZLS crossings (>=2 pts): {len(CellInfo)}")
label_counts = {}
for c in CellInfo:
    k = c['label'].value
    label_counts[k] = label_counts.get(k, 0) + 1
for k, v in label_counts.items():
    print(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Select 6 cells and assign a DIFFERENT n_refs to each subplot
# ═══════════════════════════════════════════════════════════════════════════

# Each subplot gets a distinct viz n_refs so the viewer can compare them.
VIZ_N_REFS_LIST = [2, 3, 4, 5, 6, 8]

# Sort by ZLS spread (max_dist) to pick geometrically varied cells.
sorted_by_spread = sorted(CellInfo, key=lambda c: c['max_dist'], reverse=True)

# Sample 6 cells roughly evenly across the spread range.
N_CELLS = len(sorted_by_spread)
indices = [int(i * (N_CELLS - 1) / (len(VIZ_N_REFS_LIST) - 1))
           for i in range(len(VIZ_N_REFS_LIST))]
to_draw = [sorted_by_spread[j] for j in indices]

# Attach the assigned n_refs and recompute cuts for each cell.
for cell, n_refs_viz in zip(to_draw, VIZ_N_REFS_LIST):
    cell['n_refs'] = n_refs_viz
    base  = float(cell['x_stars'][cell['ia']] @ cell['normal'])
    step  = float((cell['x_stars'][cell['ib']] - cell['x_stars'][cell['ia']]) @ cell['normal']) / n_refs_viz
    cell['cuts'] = [base + k * step for k in range(1, n_refs_viz)]

print(f"\nVisualising {len(to_draw)} cells:")
for c in to_draw:
    print(f"  cell {c['idx']:4d}  label={c['label'].value:<20s}  "
          f"n_refs(real)={c['n_refs_real']}  n_refs(viz)={c['n_refs']}  "
          f"max_dist={c['max_dist']:.4f}  remainder={c['remainder']:.4e}")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Per-cell visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

LABEL_COLOR = {
    Label.SAFE_TAYLOR     : "#2ecc71",
    Label.SAFE_REFINEMENT : "#1abc9c",
    Label.UNSAFE          : "#e74c3c",
    Label.INCONCLUSIVE    : "#f39c12",
}


def hull_polygon(verts):
    """Ordered convex hull vertices for plotting."""
    verts = np.asarray(verts)
    if len(verts) < 3:
        return verts
    try:
        h = ConvexHull(verts)
        return verts[h.vertices]
    except Exception:
        return verts


def clip_line_to_bbox(point_on_line, direction, bbox):
    """Clip an infinite 2D line d^T x = d^T p to a bounding box [xlo,xhi]x[ylo,yhi].
    Returns two endpoints (or None if no intersection)."""
    xlo, xhi, ylo, yhi = bbox
    # Parametric: x = p + t * perp,  where perp is perpendicular to direction
    # The line is  normal[0]*x + normal[1]*y = normal @ p
    nx, ny = direction
    c = float(direction @ point_on_line)

    pts = []
    # Intersect with x = xlo
    if abs(nx) > 1e-12:
        t = (c - nx * xlo) / ny if abs(ny) > 1e-12 else None
        if t is not None and ylo - 1e-9 <= t <= yhi + 1e-9:
            pts.append((xlo, t))
        elif abs(ny) < 1e-12:
            # horizontal line: y = c/ny — skip (handled below)
            pass
    if len(pts) < 2 and abs(nx) > 1e-12:
        t = (c - nx * xhi) / ny if abs(ny) > 1e-12 else None
        if t is not None and ylo - 1e-9 <= t <= yhi + 1e-9:
            pts.append((xhi, t))

    if len(pts) < 2 and abs(ny) > 1e-12:
        x_at_ylo = (c - ny * ylo) / nx if abs(nx) > 1e-12 else None
        if x_at_ylo is not None and xlo - 1e-9 <= x_at_ylo <= xhi + 1e-9:
            pts.append((x_at_ylo, ylo))

    if len(pts) < 2 and abs(ny) > 1e-12:
        x_at_yhi = (c - ny * yhi) / nx if abs(nx) > 1e-12 else None
        if x_at_yhi is not None and xlo - 1e-9 <= x_at_yhi <= xhi + 1e-9:
            pts.append((x_at_yhi, yhi))

    if len(pts) >= 2:
        return np.array(pts[0]), np.array(pts[1])
    return None, None


def draw_cell(ax, info, show_title=True):
    """Draw one boundary cell with ZLS and refinement hyperplanes."""
    verts   = info['verts']
    x_stars = info['x_stars']
    normal  = info['normal']
    cuts    = info['cuts']
    n_refs  = info['n_refs']
    ia, ib  = info['ia'], info['ib']

    # ── bounding box of cell (with margin) ───────────────────────────────
    margin = 0.05 * (verts.max(axis=0) - verts.min(axis=0)).max()
    margin = max(margin, 0.02)
    xlo    = verts[:, 0].min() - margin
    xhi    = verts[:, 0].max() + margin
    ylo    = verts[:, 1].min() - margin
    yhi    = verts[:, 1].max() + margin
    bbox   = (xlo, xhi, ylo, yhi)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")

    # ── cell polygon ──────────────────────────────────────────────────────
    poly = hull_polygon(verts)
    cell_patch = plt.Polygon(poly, closed=True,
                             facecolor=LABEL_COLOR[info['label']], alpha=0.35,
                             edgecolor="k", linewidth=1.2)
    ax.add_patch(cell_patch)

    # ── ZLS crossing points (x_stars) ────────────────────────────────────
    ax.scatter(x_stars[:, 0], x_stars[:, 1],
               c="navy", s=25, zorder=5, label="ZLS crossings")

    # ── ZLS segment (connect in order along the principal direction) ──────
    proj_ord = np.argsort(x_stars @ normal)
    xs_ord   = x_stars[proj_ord]
    ax.plot(xs_ord[:, 0], xs_ord[:, 1],
            color="navy", lw=2.0, zorder=4, label="ZLS (B=0)")

    # ── Farthest pair arrow ───────────────────────────────────────────────
    ax.annotate("", xy=x_stars[ib], xytext=x_stars[ia],
                arrowprops=dict(arrowstyle="<->", color="purple", lw=1.2))

    # ── Refinement hyperplanes (clipped to cell bbox) ─────────────────────
    cmap = plt.cm.Reds
    for k, cut_off in enumerate(cuts):
        # The hyperplane is: normal @ x = cut_off
        # A point on this line:
        p_on = normal * cut_off   # normal^T p = cut_off if ||normal||=1
        p1, p2 = clip_line_to_bbox(p_on, normal, bbox)
        if p1 is None:
            continue
        color  = cmap(0.3 + 0.6 * k / max(len(cuts) - 1, 1))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, lw=1.4, ls="--", zorder=3,
                label=f"cut {k+1}" if k < 3 else None)

    # ── Label ──────────────────────────────────────────────────────────────
    if show_title:
        ax.set_title(
            f"Cell {info['idx']}  [{info['label'].value}]\n"
            f"n_refs(real)={info['n_refs_real']}  n_refs(viz)={n_refs}\n"
            f"rem={info['remainder']:.2e}   |f_lin|={abs(info['worst_cond']):.2e}",
            fontsize=7.5
        )
    ax.set_xlabel("$x_1$", fontsize=7)
    ax.set_ylabel("$x_2$", fontsize=7)
    ax.tick_params(labelsize=6)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Figure: one subplot per selected cell
# ═══════════════════════════════════════════════════════════════════════════

ncols = min(3, len(to_draw))
nrows = (len(to_draw) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(4.5 * ncols, 4.5 * nrows))
if len(to_draw) == 1:
    axes = np.array([[axes]])
elif nrows == 1:
    axes = axes[np.newaxis, :]
elif ncols == 1:
    axes = axes[:, np.newaxis]

for idx_plot, info in enumerate(to_draw):
    r, c = divmod(idx_plot, ncols)
    draw_cell(axes[r, c], info)

# hide unused subplots
for idx_plot in range(len(to_draw), nrows * ncols):
    r, c = divmod(idx_plot, ncols)
    axes[r, c].set_visible(False)

# legend handles
handles = [
    mpatches.Patch(color=LABEL_COLOR[Label.SAFE_TAYLOR],     label="SAFE (Taylor)"),
    mpatches.Patch(color=LABEL_COLOR[Label.SAFE_REFINEMENT], label="SAFE (Refinement)"),
    mpatches.Patch(color=LABEL_COLOR[Label.UNSAFE],          label="UNSAFE"),
    mpatches.Patch(color=LABEL_COLOR[Label.INCONCLUSIVE],    label="INCONCLUSIVE"),
    plt.Line2D([0],[0], color="navy", lw=2,       label="ZLS (B=0)"),
    plt.Line2D([0],[0], color="navy", lw=1, marker="o", ls="none", label="ZLS crossings"),
    plt.Line2D([0],[0], color="crimson", lw=1.4, ls="--", label="Refinement cuts"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
           bbox_to_anchor=(0.5, 0.0))

plt.suptitle(
    "Arch3 — ZLS crossing points & adaptive refinement hyperplanes\n"
    "(polytope shown per cell; cuts = parallel hyperplanes along farthest-pair direction)",
    fontsize=9
)
plt.tight_layout(rect=[0, 0.07, 1, 1])
out = "verify_refinement_cells_arch3.png"
plt.savefig(out, dpi=160)
print(f"\nFigure saved: {out}")
