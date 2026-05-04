"""
bitwise_utils_efficient.py
==========================

HDF5-backed extensions for the vertex-based ReLU region enumerator.

This module provides a streaming variant of :func:`Enumerator_rapid` that
writes the final enumerated polytopes directly to an HDF5 file to avoid
holding the last-layer output entirely in RAM.
"""

import os
import time

import h5py
import numpy as np

from .bitwise_utils import (
    Enumerator_rapid,
    generate_mask,
    generate_mask_serial,
    generate_mask_wide,
    slice_polytope_with_hyperplane,
    slice_polytope_with_hyperplane_jit,
    slice_polytope_parallel,
    slice_polytope_wide,
    _dedup_verts,
    finding_side_polytope,
    finding_deep_hype,
)
from .ibp_filter import vertex_ibp_filter


def _remove_file_with_retry(path, retries=40, delay_s=0.05, strict=False):
    for attempt in range(retries):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if attempt == retries - 1:
                if strict:
                    raise
                return False
            time.sleep(delay_s)
    return False


def _open_h5_lock_tolerant(path, mode, retries=5, delay_s=0.05, **kwargs):
    try:
        return h5py.File(path, mode, **kwargs)
    except OSError as exc:
        msg = str(exc).lower()
        if os.name != "nt" or "unable to lock file" not in msg:
            raise

    if any(flag in mode for flag in ("w", "a", "x", "+")) and os.path.exists(path):
        _remove_file_with_retry(path, retries=40, delay_s=delay_s, strict=False)

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    last_exc = None
    for _ in range(retries):
        try:
            return h5py.File(path, mode, **kwargs)
        except OSError as exc:
            last_exc = exc
            if "unable to lock file" not in str(exc).lower():
                raise
            time.sleep(delay_s)
    raise last_exc


def _flush_h5_write_buffer(ds, write_buffer, write_buffer_offsets, offsets):
    if not write_buffer:
        return
    batch = np.vstack(write_buffer)
    ds.resize(ds.shape[0] + len(batch), axis=0)
    ds[-len(batch):] = batch
    for sz in write_buffer_offsets:
        offsets.append(offsets[-1] + sz)
    write_buffer.clear()
    write_buffer_offsets.clear()


def Enumerator_rapid_h5(
    hyperplanes,
    b,
    original_polytope_test,
    TH,
    boundary_hyperplanes,
    border_bias,
    parallel,
    D,
    m,
    out_h5_path,
    use_wide=False,
    W_pos_ibp=None,
    W_neg_ibp=None,
    b_arr_ibp=None,
    is_last_layer=False,
    mask_tolerance=1e-10,
    write_buffer_size=5000,
):
    """Enumerate all linear regions for one hidden layer and stream output.

    Parameters are identical to :func:`relu_region_enumerator.bitwise_utils.Enumerator_rapid`.
    The additional ``out_h5_path`` argument writes the final output polytopes
    directly to an HDF5 file with datasets ``vertices`` and ``offsets``.

    Returns
    -------
    offsets : np.ndarray int64
        Cumulative vertex offsets for each enumerated region.
    n_regions : int
        Number of regions written to ``out_h5_path``.
    """

    if out_h5_path is None:
        raise ValueError("out_h5_path must be provided for HDF5 streaming.")

    os.makedirs(os.path.dirname(out_h5_path) or ".", exist_ok=True)
    enumerate_poly = list(original_polytope_test)

    n = len(enumerate_poly[0][0])
    if os.path.exists(out_h5_path):
        h5f = _open_h5_lock_tolerant(out_h5_path, "a", rdcc_nbytes=0)
        ds = h5f["vertices"]
        offsets = list(h5f["offsets"][:])
    else:
        h5f = _open_h5_lock_tolerant(out_h5_path, "w", rdcc_nbytes=0)
        ds = h5f.create_dataset(
            "vertices",
            shape=(0, n),
            maxshape=(None, n),
            dtype=np.float64,
            chunks=(512, n),
        )
        offsets = [0]
    write_buffer = []
    write_buffer_offsets = []

    for i in range(len(hyperplanes)):
        # print(f"Processing hyperplane {i+1}/{len(hyperplanes)} with {len(enumerate_poly)} polytopes...")
        if not enumerate_poly:
            break

        is_final_hyperplane = i == len(hyperplanes) - 1
        intact_poly = []
        poly_dummy = []
        n = len(enumerate_poly[0][0])

        sgn_var = []
        for k in enumerate_poly:
            dum = np.dot(k, hyperplanes[i].T) + b[i]
            if np.min(dum) < -1e-5 and np.max(dum) > 1e-5:
                sgn_var.append(np.max(dum) * np.min(dum))
            else:
                sgn_var.append(0.0)

        global_bit_index = i + len(boundary_hyperplanes[0])
        bh = np.array(boundary_hyperplanes[0])
        bb = np.array(border_bias[0])

        for j in range(len(enumerate_poly)):
            if sgn_var[j] < -1e-9:
                hyperplane_val = np.dot(enumerate_poly[j], hyperplanes[i].T) + b[i]
                verts = np.array(enumerate_poly[j])
                bh_n, bb_n = finding_side_polytope(bh, verts, bb)

                if len(bh_n) <= 64:
                    use_wide = False

                if use_wide:
                    masks_w = generate_mask_wide(verts, bh_n, bb_n, tolerance=mask_tolerance)
                    polytops_test, _, created_verts = slice_polytope_wide(
                        verts,
                        np.array(hyperplane_val),
                        masks_w,
                        global_bit_index,
                        n,
                    )
                else:
                    if len(verts) < 200:
                        masks = generate_mask_serial(verts, bh_n, bb_n, tolerance=mask_tolerance)
                    else:
                        masks = generate_mask(verts, bh_n, bb_n, tolerance=mask_tolerance)
                    if len(masks) < 1000:
                        polytops_test, _, created_verts = slice_polytope_with_hyperplane(
                            verts,
                            np.array(hyperplane_val),
                            masks,
                            global_bit_index,
                            n,
                        )
                    else:
                        polytops_test, _, created_verts = slice_polytope_parallel(
                            verts,
                            np.array(hyperplane_val),
                            masks,
                            global_bit_index,
                            n,
                        )

                    if len(created_verts) <= n - 1:
                        polytops_test, _, created_verts = slice_polytope_with_hyperplane_jit(
                            verts,
                            np.array(hyperplane_val),
                            masks,
                            global_bit_index,
                            n,
                        )

                if len(created_verts) > n - 1:
                    fv_in = _dedup_verts(np.asarray(polytops_test[0], dtype=np.float64))
                    fv_out = _dedup_verts(np.asarray(polytops_test[1], dtype=np.float64))
                    result = [fv_in, fv_out]
                else:
                    result = [enumerate_poly[j]]
                    print(
                        "Warning: Slicer returned fewer than (n-1) intersection points; "
                        "this may lead to incorrect enumeration results for this hyperplane."
                    )

                if is_final_hyperplane:
                    for poly in result:
                        write_buffer.append(np.asarray(poly, dtype=np.float64))
                        write_buffer_offsets.append(len(poly))
                        if len(write_buffer) >= write_buffer_size:
                            _flush_h5_write_buffer(ds, write_buffer, write_buffer_offsets, offsets)
                else:
                    poly_dummy.extend(result)
            else:
                if is_final_hyperplane:
                    write_buffer.append(np.asarray(enumerate_poly[j], dtype=np.float64))
                    write_buffer_offsets.append(len(enumerate_poly[j]))
                    if len(write_buffer) >= write_buffer_size:
                        _flush_h5_write_buffer(ds, write_buffer, write_buffer_offsets, offsets)
                else:
                    intact_poly.append(enumerate_poly[j])

        if is_final_hyperplane:
            if write_buffer:
                _flush_h5_write_buffer(ds, write_buffer, write_buffer_offsets, offsets)
            h5f.create_dataset("offsets", data=np.array(offsets, dtype=np.int64))
            h5f.flush()
            try:
                _sync_fd = os.open(out_h5_path, os.O_RDONLY)
                os.fsync(_sync_fd)
                os.close(_sync_fd)
            except OSError:
                pass
            h5f.close()
            return np.array(offsets, dtype=np.int64), len(offsets) - 1

        if W_pos_ibp is not None and len(poly_dummy) > 1:
            is_final_layer = (i == len(hyperplanes) - 1)
            poly_dummy, _ = vertex_ibp_filter(
                poly_dummy, W_pos_ibp, W_neg_ibp, b_arr_ibp,
                is_last_layer=(is_last_layer and is_final_layer),
                n_known=m + 1 if is_final_layer else m,
                verbose=False,
            )

        intact_poly.extend(poly_dummy)
        enumerate_poly = intact_poly

    if out_h5_path is not None:
        if write_buffer:
            _flush_h5_write_buffer(ds, write_buffer, write_buffer_offsets, offsets)
    if "offsets" in h5f:
        del h5f["offsets"]
    h5f.create_dataset("offsets", data=np.array(offsets, dtype=np.int64))
    h5f.flush()
    try:
        _sync_fd = os.open(out_h5_path, os.O_RDONLY)
        os.fsync(_sync_fd)
        os.close(_sync_fd)
    except OSError:
        pass
    h5f.close()
    return np.array(offsets, dtype=np.int64), len(offsets) - 1
