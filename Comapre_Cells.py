"""
Compare boundary cells between:
- Our method: decay__boundary_cells.h5 (sign vectors as numpy arrays, 0/1, float)
- Ren et al.: verification_patterns.npy (activation dicts, 1/-1 convention)

Ren format: {0: [1,-1,...], 1: [1,-1,...], 2: [0]}  <- key 2 is output layer, skip
Our format: numpy array of shape (N, 20), values 0.0/1.0
"""

import numpy as np
import h5py
import sys

def load_our_cells(h5_path):
    with h5py.File(h5_path, 'r') as f:
        ap = f['activation_patterns']
        if isinstance(ap, h5py.Dataset):
            cells = ap[:]
        else:
            cells = np.array([ap[k][:] for k in ap.keys()])
    print(f"[Ours] {len(cells)} boundary cells, shape: {cells.shape}")
    print(f"[Ours] Sample: {cells[0]}")
    return cells

def load_ren_cells(npy_path):
    patterns = np.load(npy_path, allow_pickle=True)
    print(f"[Ren]  {len(patterns)} verification patterns")
    print(f"[Ren]  Sample: {patterns[0]}")
    return patterns

def ren_to_tuple(pattern):
    """
    {0:[1,-1..], 1:[1,-1..], 2:[0]} -> flat tuple of 0/1
    Skip output layer (max key). Map 1->1, -1->0.
    """
    sign_vec = []
    hidden_keys = sorted(k for k in pattern.keys() if k != max(pattern.keys()))
    for k in hidden_keys:
        for v in pattern[k]:
            if v == 1:
                sign_vec.append(1)
            elif v == -1:
                sign_vec.append(0)
            else:
                print(f"  [WARN] Unstable neuron (value={v}) in layer {k}")
                sign_vec.append(v)
    return tuple(sign_vec)

def our_to_tuple(cell):
    return tuple(int(round(v)) for v in cell)

def compare(h5_path, npy_path):
    our_raw = load_our_cells(h5_path)
    ren_raw = load_ren_cells(npy_path)

    patterns = np.load(npy_path, allow_pickle=True)
    print(type(patterns[0][0]))  # check if list or numpy array
    print(patterns[0] == patterns[1])  # see what equality returns
    ren_tuples_list = [ren_to_tuple(p) for p in ren_raw]
    from collections import Counter
    counts = Counter(ren_tuples_list)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    print(f"Duplicate tuples: {len(duplicates)}")
    print(f"Total duplicate entries: {sum(v-1 for v in duplicates.values())}")
    print("Sample duplicate:", list(duplicates.items())[0])
    our_set = set(our_to_tuple(c) for c in our_raw)
    ren_set = set(ren_to_tuple(p) for p in ren_raw)

    our_lens = set(len(c) for c in our_set)
    ren_lens  = set(len(c) for c in ren_set)
    print(f"\n[Sanity] Our sign vector lengths:  {our_lens}")
    print(f"[Sanity] Ren sign vector lengths:  {ren_lens}")
    if our_lens != ren_lens:
        print("[ERROR] Length mismatch — conversion may be wrong!")
        return None, None, None

    shared    = our_set & ren_set
    only_ours = our_set - ren_set
    only_ren  = ren_set - our_set

    print("\n" + "="*55)
    print(f"  Our method unique cells :  {len(our_set)}")
    print(f"  Ren et al. unique cells :  {len(ren_set)}")
    print(f"  Shared                  :  {len(shared)}")
    print(f"  Only in ours            :  {len(only_ours)}")
    print(f"  Only in Ren et al.      :  {len(only_ren)}")
    print("="*55)

    if only_ren:
        print(f"\n[WARNING] {len(only_ren)} cells in Ren but NOT in ours:")
        for c in list(only_ren)[:5]:
            print(" ", c)

    if only_ours:
        print(f"\n[INFO] {len(only_ours)} cells in ours but NOT in Ren:")
        for c in list(only_ours)[:5]:
            print(" ", c)

    if not only_ren and not only_ours:
        print("\n[PERFECT MATCH] Both methods found exactly the same boundary cells.")

    return shared, only_ours, only_ren

if __name__ == "__main__":
    h5_path  = sys.argv[1] if len(sys.argv) > 1 else "complex_boundary_cells.h5"
    npy_path = sys.argv[2] if len(sys.argv) > 2 else "facet_enumeration_result.npy"
    compare(h5_path, npy_path)