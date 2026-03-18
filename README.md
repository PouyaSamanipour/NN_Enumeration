# ReLU Region Enumerator

[![Python](https://img.shields.io/badge/python-3.8--3.11-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-PouyaSamanipour%2FNN__Enumeration-lightgrey?logo=github)](https://github.com/PouyaSamanipour/NN_Enumeration)

Exact and complete enumeration of the polytopic linear regions of feedforward ReLU neural networks, using a vertex-based algorithm with bitwise edge-adjacency testing.

## Overview

A ReLU network partitions its input domain into a finite number of convex polytopes, on each of which the network is an affine function. Enumerating these regions exactly is useful for Lyapunov stability verification, barrier certificate computation, and formal safety analysis of neural-network control policies.

This package replaces LP-feasibility adjacency checks — which cost O(n³) per vertex pair — with a single bitwise AND plus a popcount, reducing per-edge cost to O(1). Each polytope vertex carries a bitmask encoding which accumulated boundary hyperplanes it lies on; two vertices share an edge if and only if their AND has at least (n−1) bits set, as guaranteed by the combinatorial structure of simple polytopes (Ziegler, *Lectures on Polytopes*, 1995).

## Installation

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/PouyaSamanipour/NN_Enumeration.git
cd NN_Enumeration
```

**Step 2 — Create and activate the conda environment:**
```bash
conda env create -f environment.yml
conda activate relu_enum
```

**Step 3 — Install the package in editable mode:**
```bash
pip install -e .
```

> **Why conda?** Numba requires a specific LLVM version that conda resolves automatically. Using plain pip on Windows can produce LLVM linkage errors.

Compatible with Python 3.8–3.11.

## Quick start

```python
from relu_region_enumerator import enumeration_function

enumeration_function(
    NN_file="NN_files/Arch3_2_96.pt",   # TorchScript model
    name_file="arch3_result",            # output file base name
    TH=[3.0, 3.0],                       # per-dimension domain half-widths
    mode="Rapid_mode",
    parallel=True,
)
```

Or run the provided script:

```bash
python run.py
```

Profiling statistics are written to `profiling.prof`.

## Output

- **`<name_file>_polytope.h5`** — HDF5 file with datasets:
  - `offsets` — integer array of shape `(N+1,)`; region k occupies rows `offsets[k]:offsets[k+1]`.
  - `vertices` — float64 array of shape `(total_vertices, n)`.

## Model format

The model must be saved with `torch.jit.save` (TorchScript). The architecture must be a fully-connected ReLU network: alternating `nn.Linear` and `nn.ReLU` layers, with a final linear output layer (no trailing ReLU).

## High-dimensional mode

For input dimension n > 5, intermediate per-layer polytope lists are streamed through HDF5 scratch files in `layer_tmp/` to stay within RAM. The scratch files are deleted after each layer.

## Repository structure

```
relu_region_enumerator/
    __init__.py          — public API exports
    bitwise_utils.py     — bitmask generation, slicing, storage primitives
    core.py              — top-level enumeration pipeline
    visualization.py     — partition plots, barrier certificate 3-D view, Lyapunov overlays
legacy/                  — original scripts prior to packaging (preserved for reference)
run.py                   — command-line entry point with profiling
requirements.txt         — runtime dependencies
requirements-dev.txt     — development dependencies
pyproject.toml           — package metadata
LICENSE
CHANGELOG.md
README.md
```

## Citation

If you use this code in academic work, please cite:

> P. Shahvali, "Exact Enumeration of Polytopic Linear Regions of ReLU Neural Networks via Bitwise Vertex-Adjacency Testing," *Proc. IEEE Conference on Decision and Control (CDC)*, 2025.

## License

MIT © Pouya Samanipour
