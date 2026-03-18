"""
run.py
======

Command-line entry point for running the ReLU region enumerator.

Usage
-----
    python run.py

Edit the ``NN_file`` and ``TH`` variables below to target a different network
or input domain.  Profiling statistics are saved to ``profiling.prof`` and can
be inspected with::

    python -m pstats profiling.prof

or visualised with snakeviz::

    snakeviz profiling.prof
"""

import cProfile
import pstats

from relu_region_enumerator import enumeration_function

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to a TorchScript-saved (.pt) ReLU network.
NN_FILE = "NN_files/Arch3_2_96.pt"

# Per-dimension half-width of the input hypercube.
# The domain for dimension i will be [-TH[i], +TH[i]].
# Length must match the network input dimension.
TH = [3.0, 3.0]

# Output file base name (no extension).  Results are written to
# ``<OUTPUT_NAME>_polytope.h5``.
OUTPUT_NAME = "arch3_result"

# Enumeration mode: "Rapid_mode" (recommended) or "Low_Ram" (CSV-based).
MODE = "Rapid_mode"

# Whether to use the parallel JIT slicer for large polytopes.
PARALLEL = True

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        enumeration_function(NN_FILE, OUTPUT_NAME, TH, MODE, PARALLEL)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="profiling.prof")
    print("\nProfiling data written to profiling.prof")
