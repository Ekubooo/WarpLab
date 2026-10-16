"""Full-domain CUDA NanoVDB baking; receives validated arrays, never an OBJ path."""

import math
import os
from pathlib import Path
import tempfile

import numpy as np
import warp as wp

from .config import BakeOptions


@wp.kernel
def _fill(mesh: wp.uint64, grid: wp.uint64, tiles: wp.array(dtype=wp.vec3i),
          search_radius: float, errors: wp.array(dtype=wp.int32)):
    thread = wp.tid()
    tile = tiles[thread // 512]
    local = thread % 512
    cell = tile + wp.vec3i(local // 64, (local // 8) % 8, local % 8)
    position = wp.volume_index_to_world(grid, wp.vec3(float(cell[0]), float(cell[1]), float(cell[2])))
    hit = wp.mesh_query_point_sign_winding_number(mesh, position, search_radius)
    if hit.result:
        nearest = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
        distance = hit.sign * wp.length(position - nearest)
        if wp.isfinite(distance):
            wp.volume_store_f(grid, cell[0], cell[1], cell[2], distance)
        else:
            wp.atomic_add(errors, 0, 1)
    else:
        wp.atomic_add(errors, 0, 1)


def check_options(options):
    if not math.isfinite(options.voxel_size) or options.voxel_size <= 0:
        raise ValueError("Voxel size must be finite and positive")
    if (type(options.padding_voxels) is not int or options.padding_voxels < 1):
        raise ValueError("Padding must be a positive integer number of voxels")


def bake(mesh, *, options=BakeOptions(), device="cuda:0"):
    check_options(options)
    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise ValueError("NanoVDB baking requires a CUDA device")
    spacing = options.voxel_size
    extent = mesh.bounds[1] - mesh.bounds[0]
    dimensions = np.ceil(extent / spacing) + 2 * options.padding_voxels
    if not np.isfinite(dimensions).all() or np.any(dimensions > np.iinfo(np.int32).max - 8):
        raise ValueError("Grid exceeds NanoVDB int32 coordinate limits")
    upper = dimensions.astype(np.int64)
    background = float(np.linalg.norm(extent) + spacing)
    volume = wp.Volume.allocate([0, 0, 0], upper.tolist(), spacing, bg_value=background,
        translation=(mesh.bounds[0] - options.padding_voxels * spacing).tolist(), device=selected)
    tiles = wp.empty(volume.get_tile_count(), dtype=wp.vec3i, device=selected)
    volume.get_tiles(out=tiles)
    gpu_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=selected),
        indices=wp.array(mesh.faces.reshape(-1), dtype=wp.int32, device=selected),
        support_winding_number=True,
    )
    failures = wp.zeros(1, dtype=wp.int32, device=selected)
    radius = float(np.linalg.norm(((upper // 8 + 1) * 8 - 1) * spacing) + spacing)
    wp.launch(_fill, dim=512 * volume.get_tile_count(),
              inputs=[gpu_mesh.id, volume.id, tiles, radius, failures], device=selected)
    wp.synchronize_device(selected)
    if int(failures.numpy()[0]):
        raise RuntimeError(f"Distance query failed at {int(failures.numpy()[0])} voxels")
    return volume


def save_volume(volume, path):
    """Write beside the destination, replacing an existing file only on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".nvdb.tmp", dir=path.parent)
    os.close(descriptor)
    try:
        volume.save_to_nvdb(temporary, codec="none")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
