"""
visualization.py
================

Plotting utilities for ReLU region enumeration results.

Functions
---------
plot_polytope              -- 2-D filled partition plot (main output visualization).
plot_polytope_3d           -- 3-D polytope plot with barrier certificate values on z-axis.
plot_polytope_2D           -- 2-D streamplot of a TorchScript vector field model.
plot_hyperplanes_and_vertices -- Debug plot of hyperplanes and vertices (2-D or 3-D).
plotting_results           -- Lyapunov/flow overlay plot with level sets (2-D systems).
"""

import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Primary output visualization
# ---------------------------------------------------------------------------

def plot_polytope(enumerate_poly, name):
    """Plot the 2-D linear region partition as filled convex polygons.

    Each region is filled with a distinct random colour.  Intended for
    2-D input networks only.

    Parameters
    ----------
    enumerate_poly : list of (V_k, 2) arrays -- enumerated polytope vertices.
    name           : str -- figure title (currently unused; reserved for saving).
    """
    for region in enumerate_poly:
        hull  = ConvexHull(region)
        color = np.random.rand(3,)
        points = np.array(region)
        plt.fill(
            points[hull.vertices, 0],
            points[hull.vertices, 1],
            color=color,
        )

    plt.xlabel("$X_1$", fontsize=18)
    plt.ylabel("$X_2$", fontsize=18)
    plt.show()


def plot_polytope_3d(enumerate_poly, model):
    """3-D visualization of enumerated polytopes with barrier certificate height.

    For each 2-D region, the barrier function b(x) is evaluated at every vertex
    and used as the z-coordinate.  Regions with positive mean barrier value are
    coloured blue (safe); negative mean is coloured red (unsafe).  The b(x) = 0
    level plane is drawn in translucent green.

    Parameters
    ----------
    enumerate_poly : list of (V_k, 2) arrays -- enumerated polytope vertices.
    model          : torch.nn.Module -- TorchScript barrier certificate network,
                     mapping R^2 -> R.
    """
    import torch

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection="3d")
    model.eval()

    for region_vertices in enumerate_poly:
        vertices = np.array(region_vertices)
        if len(vertices) < 3:
            continue

        # Evaluate barrier at each vertex.
        b_vals = []
        for v in vertices:
            with torch.no_grad():
                b_vals.append(model(torch.tensor(v, dtype=torch.float64)).item())
        b_vals = np.array(b_vals)

        try:
            hull = ConvexHull(vertices)
        except Exception:
            continue

        hull_verts = hull.vertices
        poly_x = vertices[hull_verts, 0]
        poly_y = vertices[hull_verts, 1]
        poly_z = b_vals[hull_verts]

        # Blue = safe (b >= 0), red = unsafe (b < 0).
        mean_b    = np.mean(poly_z)
        facecolor = [0.2, 0.4, 0.8, 0.5] if mean_b >= 0 else [0.8, 0.2, 0.2, 0.5]

        poly = Poly3DCollection([list(zip(poly_x, poly_y, poly_z))], alpha=0.5)
        poly.set_facecolor(facecolor)
        poly.set_edgecolor([0.2, 0.2, 0.2, 0.2])
        ax.add_collection3d(poly)

    # Draw the b(x) = 0 plane for reference.
    x_range = np.linspace(-3, 3, 30)
    y_range = np.linspace(-3, 3, 30)
    xx, yy  = np.meshgrid(x_range, y_range)
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color="green", zorder=0)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$b(x)$")
    ax.set_title("Linear Regions with Barrier Certificate")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Vector-field / flow visualization
# ---------------------------------------------------------------------------

def plot_polytope_2D(NN_model, TH):
    """Overlay a 2-D streamplot of the dynamics encoded by a TorchScript model.

    Evaluates the network on a 200×200 grid over [-TH, TH]^2 and plots the
    resulting vector field as a streamplot.

    Parameters
    ----------
    NN_model : str   -- path to TorchScript (.pt) model file.
    TH       : float -- scalar domain half-width (same for both dimensions).
    """
    import torch

    model = torch.jit.load(NN_model)
    model = model.to("cpu")
    model.eval()

    x1 = np.linspace(-TH, TH, 200)
    x2 = np.linspace(-TH, TH, 200)
    X1, X2 = np.meshgrid(x1, x2)
    data = np.array(list(zip(np.ravel(X1), np.ravel(X2))))

    with torch.no_grad():
        output = model(torch.FloatTensor(data))

    plt.streamplot(
        X1, X2,
        np.array(output[:, 0]).reshape(len(X1), len(X2)),
        np.array(output[:, 1]).reshape(len(X1), len(X2)),
        color="blue",
    )


# ---------------------------------------------------------------------------
# Lyapunov level-set and flow overlay
# ---------------------------------------------------------------------------

def plotting_results(TH, all_hyperplanes, all_bias, c_v, W_v, W, c, enumerate_poly, name):
    """Overlay Lyapunov level sets and system flows on the 2-D input plane.

    Evaluates the Lyapunov function V(x) and the vector field f(x) on a dense
    grid, then plots contour level sets of V and a streamplot of f.  The figure
    is saved to ``Figures/<name>_<k>.png`` with an auto-incremented index to
    avoid overwriting existing files.

    Parameters
    ----------
    TH              : float       -- scalar domain half-width.
    all_hyperplanes : (H, n) array -- stacked weight matrix for the network.
    all_bias        : (H,)   array -- stacked bias vector.
    c_v             : array        -- Lyapunov output bias.
    W_v             : array        -- Lyapunov output weight.
    W               : (2, H) array -- dynamics output weights.
    c               : (2,)   array -- dynamics output bias.
    enumerate_poly  : list         -- enumerated polytopes (unused here; kept for API consistency).
    name            : str          -- base name for the saved figure.
    """
    x1 = np.linspace(-TH, TH, 200)
    x2 = np.linspace(-TH, TH, 200)
    X1, X2 = np.meshgrid(x1, x2)
    data = tuple(zip(np.ravel(X1), np.ravel(X2)))

    hidden = np.maximum(np.dot(all_hyperplanes, np.array(data).T) + all_bias, 0)
    LV   = np.dot(W_v, hidden) + c_v
    dX1, dX2 = np.dot(W, hidden) + np.reshape(c, (2, 1))

    Z   = LV.reshape(len(X1), len(X2))
    dX1 = dX1.reshape(len(X1), len(X2))
    dX2 = dX2.reshape(len(X1), len(X2))

    levels = np.linspace(np.min(LV), 0.4 * np.max(LV), 5)
    plt.contour(X1, X2, Z, levels, colors="red")
    plt.streamplot(
        X1, X2, dX1, dX2,
        color="blue", linewidth=1,
        density=1, arrowstyle="-|>", arrowsize=1.5,
    )

    plt.xlabel("$X_1$", fontweight="bold", fontsize=20, style="italic")
    plt.ylabel("$X_2$", fontweight="bold", fontsize=20, style="italic")

    red_line = mlines.Line2D([], [], color="red", marker="_", label="Level sets", markersize=15)
    arrow    = plt.scatter(0, 0, c="blue", marker=r"$\longrightarrow$", s=40, label="Flows")
    plt.legend(handles=[red_line, arrow], loc="upper right")
    plt.title(name)

    # Auto-increment filename to avoid overwriting.
    cntr     = 0
    name_new = f"{name}_{cntr}.png"
    figures_dir = os.path.join(os.getcwd(), "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    while os.path.exists(os.path.join(figures_dir, name_new)):
        cntr    += 1
        name_new = f"{name}_{cntr}.png"
    plt.savefig(os.path.join(figures_dir, name_new))

    # 3-D surface of the Lyapunov function.
    fig3d = plt.figure("3D")
    ax    = fig3d.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    ax.set_zlabel("Lyapunov")
    ax.set_title("Lyapunov Surface")
    plt.show()


# ---------------------------------------------------------------------------
# Debug utility
# ---------------------------------------------------------------------------

def plot_hyperplanes_and_vertices(hyperplanes, vertices):
    """Plot hyperplanes and vertices for debugging (2-D or 3-D).

    Parameters
    ----------
    hyperplanes : list of arrays -- each row is [a, b, c] (2-D) or [a, b, c, d] (3-D),
                  representing a*x + b*y (+ c*z) >= d.
    vertices    : list of arrays -- points to overlay as scatter markers.
    """
    dim = hyperplanes[0][:-1].size

    if dim == 2:
        for E in hyperplanes:
            x = np.linspace(-10, 10, 5)
            if E[1] != 0:
                y = (-E[2] - E[0] * x) / E[1]
            else:
                x = np.array([-E[2] / E[0]] * 5)
                y = np.zeros(5)
            plt.plot(x, y, label=f"{E[0]}x + {E[1]}y >= {E[2]}")

        for vertex in vertices:
            plt.scatter(*vertex, color="red")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.grid(True)
        plt.xlim(-0.9, 0.9)
        plt.ylim(-0.9, 0.9)
        plt.show()

    elif dim == 3:
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection="3d")

        for E in hyperplanes:
            x, y = np.meshgrid(np.linspace(-10, 10, 5), np.linspace(-10, 10, 5))
            z = (E[3] - E[0] * x - E[1] * y) / E[2]
            ax.plot_surface(x, y, z, alpha=0.5)

        vertices = np.array(vertices)
        ax.scatter(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            c="red", marker="o", s=100, label="Vertices",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()
