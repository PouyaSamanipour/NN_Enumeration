"""
plot_refinement_results.py
==========================
Standalone visualization for refinement_results.csv and refinement_depth_data.csv.
Only requires numpy and matplotlib (no torch, sympy, h5py, etc.).

Usage:
    python Tests/plot_refinement_results.py
    python Tests/plot_refinement_results.py --results my_results.csv --depth my_depth.csv
"""

import csv
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RESULTS = "refinement_results.csv"
DEFAULT_DEPTH   = "refinement_depth_data.csv"

LABEL_COLORS = {
    "SAFE_TAYLOR":      "#2196F3",
    "SAFE_REFINEMENT":  "#4CAF50",
    "INCONCLUSIVE":     "#FF9800",
    "UNSAFE":           "#F44336",
    "NO_ZLS":           "#9E9E9E",
}


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_results(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "cell_idx":    int(r["cell_idx"]),
                "label":       r["label"],
                "ref_time_s":  float(r["ref_time_s"]),
                "r_i":         float(r["r_i"]),
                "M_i":         float(r["M_i"]),
                "remainder":   float(r["remainder"]),
                "dreal_called": r["dreal_called"].strip().lower() == "true",
                "dreal_label": r["dreal_label"],
                "dreal_time_s":float(r["dreal_time_s"]),
            })
    return rows


def load_depth(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "depth":      int(r["depth"]),
                "n_subcells": int(r["n_subcells"]),
                "r_mean":     float(r["r_mean"]),
                "r_std":      float(r["r_std"]),
                "M_mean":     float(r["M_mean"]),
                "M_std":      float(r["M_std"]),
            })
    return sorted(rows, key=lambda x: x["depth"])


# ── Individual figure helpers ─────────────────────────────────────────────────

def fig_label_distribution(rows):
    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]
    colors = [LABEL_COLORS.get(l, "#607D8B") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("Verification label distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of cells")
    ax.set_xlabel("Label")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_r_M_by_label(rows):
    from collections import defaultdict
    by_label = defaultdict(lambda: {"r": [], "M": []})
    for r in rows:
        if r["label"] == "NO_ZLS":
            continue
        by_label[r["label"]]["r"].append(r["r_i"])
        by_label[r["label"]]["M"].append(r["M_i"])

    labels = sorted(by_label.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, key, title, ylabel in zip(
        axes,
        ["r", "M"],
        ["Ratio $r_i = \\|\\nabla B\\cdot f\\| / \\|\\nabla B\\| \\cdot \\|f\\|$",
         "Lipschitz bound $M_i$"],
        ["$r_i$", "$M_i$"],
    ):
        data   = [by_label[l][key] for l in labels]
        colors = [LABEL_COLORS.get(l, "#607D8B") for l in labels]
        bps    = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops=dict(color="black", linewidth=1.5),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8),
                            flierprops=dict(marker=".", markersize=3, alpha=0.4))
        for patch, c in zip(bps["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$r_i$ and $M_i$ by verification label", fontsize=12)
    fig.tight_layout()
    return fig


def fig_dreal_vs_refinement(rows):
    """Three-panel comparison: agreement matrix, timing scatter, per-cell time breakdown."""
    dual = [r for r in rows if r["dreal_called"]]
    if not dual:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No dReal calls in data",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        return fig

    dreal_t = np.array([r["dreal_time_s"] for r in dual])
    ref_t   = np.array([r["ref_time_s"]   for r in dual])
    dreal_faster = dreal_t < ref_t

    order  = np.arange(len(dual))
    xs     = np.arange(len(dual))
    d_cols = np.where(dreal_faster[order], "#4CAF50", "#FF9800")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                             gridspec_kw={"height_ratios": [2, 1]})

    # ── top: stacked bars, dReal coloured by who won ──────────────────────────
    ax = axes[0]
    ax.bar(xs, ref_t[order],   color="#2196F3", width=1.0, linewidth=0, label="Refinement")
    for x, rt, dt, c in zip(xs, ref_t[order], dreal_t[order], d_cols):
        ax.bar(x, dt, bottom=rt, color=c, width=1.0, linewidth=0)
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2196F3", label="Refinement"),
        Patch(color="#4CAF50", label="dReal (faster)"),
        Patch(color="#FF9800", label="dReal (slower)"),
    ], fontsize=8, loc="upper left")
    ax.set_ylabel("Wall-clock time (s)", fontsize=9)
    ax.set_title(f"dReal vs Refinement — time breakdown ({len(dual)} cells, "
                 f"dReal faster in {int(dreal_faster.sum())} / {len(dual)})",
                 fontweight="bold")
    ax.set_xticks([])
    ax.spines[["top", "right"]].set_visible(False)

    # ── bottom: dReal/refinement ratio, log scale ─────────────────────────────
    ax = axes[1]
    ratio = dreal_t[order] / np.where(ref_t[order] > 0, ref_t[order], np.nan)
    ax.bar(xs, ratio, color=d_cols, width=1.0, linewidth=0, alpha=0.85)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="equal time")
    ax.set_yscale("log")
    ax.set_ylabel("dReal / Refinement time", fontsize=9)
    ax.set_xlabel("Cell (sorted by total time)", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


def fig_depth_convergence(depth_rows):
    if not depth_rows:
        return None

    depths = np.array([d["depth"] for d in depth_rows])
    r_mean = np.array([d["r_mean"] for d in depth_rows])
    r_std  = np.array([d["r_std"]  for d in depth_rows])
    M_mean = np.array([d["M_mean"] for d in depth_rows])
    M_std  = np.array([d["M_std"]  for d in depth_rows])
    n_sub  = np.array([d["n_subcells"] for d in depth_rows])

    # x-axis labels: depth -1 → "root", 0,1,2,... → level 0,1,...
    x_labels = ["root" if d == -1 else str(d) for d in depths]
    xs = np.arange(len(depths))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # r_i convergence
    ax = axes[0]
    ax.fill_between(xs, r_mean - r_std, r_mean + r_std, alpha=0.25, color="#2196F3")
    ax.plot(xs, r_mean, "o-", color="#2196F3", linewidth=1.8, markersize=5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="$r_i=1$")
    ax.set_xticks(xs); ax.set_xticklabels(x_labels)
    ax.set_title("$r_i$ vs refinement depth", fontweight="bold")
    ax.set_xlabel("Depth (−1 = root cell)")
    ax.set_ylabel("$r_i = \\|\\nabla B \\cdot f\\| / \\|\\nabla B\\|\\|f\\|$")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # M_i convergence
    ax = axes[1]
    ax.fill_between(xs, M_mean - M_std, M_mean + M_std, alpha=0.25, color="#4CAF50")
    ax.plot(xs, M_mean, "o-", color="#4CAF50", linewidth=1.8, markersize=5)
    ax.set_xticks(xs); ax.set_xticklabels(x_labels)
    ax.set_title("$M_i$ vs refinement depth", fontweight="bold")
    ax.set_xlabel("Depth (−1 = root cell)")
    ax.set_ylabel("$M_i$ (Lipschitz bound)")
    ax.spines[["top", "right"]].set_visible(False)

    # Sub-cell count
    ax = axes[2]
    ax.bar(xs, n_sub, color="#FF9800", edgecolor="white", linewidth=0.6, alpha=0.85)
    ax.set_xticks(xs); ax.set_xticklabels(x_labels)
    ax.set_title("Sub-cells per depth level", fontweight="bold")
    ax.set_xlabel("Depth (−1 = root cell)")
    ax.set_ylabel("Number of sub-cells")
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(ax.containers[0], padding=2, fontsize=8)

    fig.suptitle("Adaptive refinement convergence (hardest cell)", fontsize=12)
    fig.tight_layout()
    return fig


def fig_remainder_vs_r(rows):
    r_vals  = np.array([r["r_i"]      for r in rows if r["label"] != "NO_ZLS"])
    rem_vals= np.array([r["remainder"] for r in rows if r["label"] != "NO_ZLS"])
    labels  = [r["label"] for r in rows if r["label"] != "NO_ZLS"]
    colors  = [LABEL_COLORS.get(l, "#607D8B") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(r_vals, rem_vals, c=colors, s=12, alpha=0.6, linewidths=0)
    ax.set_xlabel("$r_i$")
    ax.set_ylabel("Remainder (Taylor bound)")
    ax.set_title("Remainder vs $r_i$ across all cells", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    unique_labels = sorted(set(labels))
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=LABEL_COLORS.get(l, "#607D8B"),
                          markersize=7, label=l)
               for l in unique_labels]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="Path to refinement_results.csv")
    parser.add_argument("--depth",   default=DEFAULT_DEPTH,
                        help="Path to refinement_depth_data.csv")
    parser.add_argument("--save",    action="store_true",
                        help="Save figures as PNG instead of showing interactively")
    parser.add_argument("--out_dir", default=".",
                        help="Directory to save figures (used with --save)")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        raise FileNotFoundError(f"Results CSV not found: {args.results}")

    print(f"Loading {args.results} ...")
    rows = load_results(args.results)
    print(f"  {len(rows)} cells loaded")

    depth_rows = []
    if os.path.exists(args.depth):
        print(f"Loading {args.depth} ...")
        depth_rows = load_depth(args.depth)
        print(f"  {len(depth_rows)} depth levels")
    else:
        print(f"  {args.depth} not found — skipping depth plots")

    figs = [
        ("label_distribution",    fig_label_distribution(rows)),
        ("r_M_by_label",          fig_r_M_by_label(rows)),
        ("dreal_vs_refinement",   fig_dreal_vs_refinement(rows)),
        ("remainder_vs_r",        fig_remainder_vs_r(rows)),
    ]
    if depth_rows:
        figs.append(("depth_convergence", fig_depth_convergence(depth_rows)))

    if args.save:
        os.makedirs(args.out_dir, exist_ok=True)
        for name, fig in figs:
            if fig is None:
                continue
            path = os.path.join(args.out_dir, f"{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
