# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2025

### Added
- Initial public release.
- Vertex-based exact enumeration of ReLU linear regions (`Enumerator_rapid`).
- Bitwise edge-adjacency test replacing LP feasibility checks (`generate_mask`, `slice_polytope_with_hyperplane`).
- Two-pass parallel slicer for large polytopes (`slice_polytope_parallel`).
- HDF5 streaming for high-dimensional inputs (n > 5).
- Visualization utilities: 2-D partition plot, 3-D barrier certificate view, Lyapunov overlays (`visualization.py`).
- `RaggedPolytopeStorage` and `HybridRaggedStorage` in-memory/disk storage helpers.
- Command-line entry point with cProfile integration (`run.py`).
