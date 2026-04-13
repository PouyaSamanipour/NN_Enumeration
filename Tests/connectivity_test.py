"""
Connectivity test for boundary cells.
Tests whether the 20 cells found by our method but not Ren et al.
are disconnected from Ren's cells in the boundary adjacency graph.

Two cells are adjacent if their sign vectors differ in exactly one neuron.

Usage:
    python connectivity_test.py decay__boundary_cells.h5 verification_patterns.npy
"""

import numpy as np
import h5py
import sys
from collections import defaultdict, deque
import time

# ─────────────────────────────────────────────
# Load functions (same as compare_cells.py)
# ─────────────────────────────────────────────
def load_our_cells(h5_path):
    with h5py.File(h5_path, 'r') as f:
        ap = f['activation_patterns']
        cells = ap[:] if isinstance(ap, h5py.Dataset) else np.array([ap[k][:] for k in ap.keys()])
    print(f"[Ours] {len(cells)} boundary cells, shape: {cells.shape}")
    return cells

def load_ren_cells(npy_path):
    patterns = np.load(npy_path, allow_pickle=True)
    print(f"[Ren]  {len(patterns)} verification patterns")
    return patterns

def ren_to_tuple(pattern):
    sign_vec = []
    hidden_keys = sorted(k for k in pattern.keys() if k != max(pattern.keys()))
    for k in hidden_keys:
        for v in pattern[k]:
            sign_vec.append(1 if v == 1 else 0)
    return tuple(sign_vec)

def our_to_tuple(cell):
    return tuple(int(round(v)) for v in cell)

# ─────────────────────────────────────────────
# Adjacency: differ in exactly 1 neuron
# ─────────────────────────────────────────────
def are_adjacent(c1, c2):
    diff = 0
    for a, b in zip(c1, c2):
        if a != b:
            diff += 1
            if diff > 1:
                return False
    return diff == 1

# ─────────────────────────────────────────────
# Build adjacency list efficiently using numpy
# ─────────────────────────────────────────────
def build_adjacency(cell_list):
    n = len(cell_list)
    arr = np.array(cell_list, dtype=np.int8)
    adj = defaultdict(list)
    
    print(f"Building adjacency graph for {n} cells...")
    t0 = time.time()
    
    for i in range(n):
        # compute hamming distance from cell i to all j > i at once
        diffs = np.sum(arr[i+1:] != arr[i], axis=1)
        neighbors = np.where(diffs == 1)[0] + (i + 1)
        for j in neighbors:
            adj[i].append(j)
            adj[j].append(i)
    
    t1 = time.time()
    print(f"Adjacency graph built in {t1-t0:.2f}s — {sum(len(v) for v in adj.values())//2} edges")
    return adj

# ─────────────────────────────────────────────
# Find connected components via BFS
# ─────────────────────────────────────────────
def find_components(n, adj):
    visited = np.zeros(n, dtype=bool)
    components = []
    
    for start in range(n):
        if visited[start]:
            continue
        component = []
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    queue.append(nb)
        components.append(component)
    
    return components

# ─────────────────────────────────────────────
# Main connectivity test
# ─────────────────────────────────────────────
def connectivity_test(h5_path, npy_path):
    # Load data
    our_raw = load_our_cells(h5_path)
    ren_raw = load_ren_cells(npy_path)
    
    # Convert to tuples
    our_set = set(our_to_tuple(c) for c in our_raw)
    ren_set = set(ren_to_tuple(p) for p in ren_raw)
    
    only_ours = our_set - ren_set
    shared    = our_set & ren_set
    
    print(f"\nOur unique cells:      {len(our_set)}")
    print(f"Ren unique cells:      {len(ren_set)}")
    print(f"Shared:                {len(shared)}")
    print(f"Only in ours:          {len(only_ours)}")
    
    if not only_ours:
        print("\n[INFO] No missing cells — connectivity test not needed.")
        return
    
    # Build index mapping
    cell_list = list(our_set)
    cell_to_idx = {c: i for i, c in enumerate(cell_list)}
    
    # Tag each cell
    ren_indices     = set(cell_to_idx[c] for c in shared)
    missing_indices = set(cell_to_idx[c] for c in only_ours)
    
    # Build adjacency and find components
    adj = build_adjacency(cell_list)
    components = find_components(len(cell_list), adj)
    
    print(f"\n{'='*55}")
    print(f"Connected components in our boundary cell graph: {len(components)}")
    for i, comp in enumerate(components):
        comp_set = set(comp)
        n_ren     = len(comp_set & ren_indices)
        n_missing = len(comp_set & missing_indices)
        print(f"  Component {i:2d}: {len(comp):4d} cells "
              f"| {n_ren:4d} shared with Ren "
              f"| {n_missing:3d} only in ours")
    print(f"{'='*55}")
    
    # For each missing cell, report which component it's in
    # and whether that component contains any Ren cells
    print(f"\nDetailed report for {len(only_ours)} cells only in ours:")
    print("-"*55)
    
    # Build cell -> component map
    cell_to_comp = {}
    for i, comp in enumerate(components):
        for node in comp:
            cell_to_comp[node] = i
    
    disconnected = []
    connected_but_missed = []
    
    for cell in only_ours:
        idx  = cell_to_idx[cell]
        comp_id = cell_to_comp[idx]
        comp_set = set(components[comp_id])
        has_ren = bool(comp_set & ren_indices)
        
        if has_ren:
            connected_but_missed.append((cell, comp_id))
        else:
            disconnected.append((cell, comp_id))
    
    print(f"\n[DISCONNECTED from Ren's cells]: {len(disconnected)}")
    if disconnected:
        print("  These cells are in components with NO Ren cells.")
        print("  Ren's BFS cannot reach them by construction.")
        for cell, comp_id in disconnected[:3]:
            print(f"    Component {comp_id}: {cell[:10]}...")
    
    print(f"\n[CONNECTED but missed by Ren]:   {len(connected_but_missed)}")
    if connected_but_missed:
        print("  These cells share a component with Ren's cells.")
        print("  Ren's BFS should have found them — likely a TestOne bug or LP tolerance issue.")
        for cell, comp_id in connected_but_missed[:3]:
            print(f"    Component {comp_id}: {cell[:10]}...")
    
    print(f"\n{'='*55}")
    print("SUMMARY:")
    print(f"  Total missing from Ren:         {len(only_ours)}")
    print(f"  Disconnected (BFS unreachable): {len(disconnected)}")
    print(f"  Connected but missed (bug):     {len(connected_but_missed)}")
    print(f"{'='*55}")

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    h5_path  = sys.argv[1] if len(sys.argv) > 1 else "decay_boundary_cells.h5"
    npy_path = sys.argv[2] if len(sys.argv) > 2 else "verification_patterns.npy"
    connectivity_test(h5_path, npy_path)
