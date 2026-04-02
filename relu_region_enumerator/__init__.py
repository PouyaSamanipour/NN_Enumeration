"""
relu_region_enumerator
======================

Exact and complete enumeration of polytopic linear regions of ReLU neural
networks via a vertex-based algorithm with bitwise adjacency testing.

Public API
----------
enumeration_function : Layer-by-layer region enumeration entry point.
generate_hypercube    : Generate all vertices of an axis-aligned hypercube.
Finding_cell_id       : Compute activation-pattern IDs for enumerated cells.
"""

from .core import enumeration_function, generate_hypercube, Finding_cell_id
from .visualization import (
    plot_polytope,
    plot_polytope_3d,
    plot_polytope_2D,
    plot_hyperplanes_and_vertices,
    plotting_results,
)
from .bitwise_utils import (
    Enumerator_rapid,
    generate_mask,
    slice_polytope_with_hyperplane,
    slice_polytope_with_hyperplane_jit,
    slice_polytope_parallel,
    Polytope_formation_hd,
    RaggedPolytopeStorage,
    HybridRaggedStorage,
)
from .Dynamics import load_dynamics, list_systems
from .hessian_bound import (
    HessianBounder,
    compute_local_gradient
)
__all__ = [
    "enumeration_function",
    "generate_hypercube",
    "Finding_cell_id",
    "plot_polytope",
    "plot_polytope_3d",
    "plot_polytope_2D",
    "plot_hyperplanes_and_vertices",
    "plotting_results",
    "Enumerator_rapid",
    "generate_mask",
    "slice_polytope_with_hyperplane",
    "slice_polytope_with_hyperplane_jit",
    "slice_polytope_parallel",
    "Polytope_formation_hd",
    "RaggedPolytopeStorage",
    "HybridRaggedStorage",
    "load_dynamics",
    "list_systems",
    "compute_local_gradient",
    "HessianBounder",
]
