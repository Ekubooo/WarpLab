"""Bake a validated static container into a full-domain CUDA NanoVDB SDF.

Negative values denote solid material; the fluid cavity and exterior are positive.
The source coordinates are preserved. Existing same-stem caches are trusted without
checking source contents or bake parameters; use force_rebuild after changing them.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import warp as wp


@dataclass(frozen=True)
class SDFOptions:
    voxel_size: float = 0.0075
    padding_voxels: int = 3
    force_rebuild: bool = False


@dataclass
class BakedSDF:
    # Keep the owning Volume alive, rather than retaining only its integer id.
    volume: wp.Volume
    source_path: Path
    cache_path: Path
    cache_hit: bool


@wp.kernel
def _bake_tiles(
    mesh: wp.uint64,
    volume: wp.uint64,
    tiles: wp.array(dtype=wp.vec3i),
    max_distance: float,
    failures: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    tile = tiles[tid // 512]
    offset = tid % 512
    i = tile[0] + offset // 64
    j = tile[1] + (offset // 8) % 8
    k = tile[2] + offset % 8
    p = wp.volume_index_to_world(volume, wp.vec3(float(i), float(j), float(k)))
    query = wp.mesh_query_point_sign_winding_number(mesh, p, max_distance)
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        distance = query.sign * wp.length(p - closest)
        if wp.isfinite(distance):
            wp.volume_store_f(volume, i, j, k, distance)
        else:
            wp.atomic_add(failures, 0, 1)
    else:
        wp.atomic_add(failures, 0, 1)


def _cuda_device(device):
    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise ValueError("SDF baking/loading requires a CUDA device; CPU is unsupported")
    return selected


def _check_options(options):
    if not math.isfinite(options.voxel_size) or options.voxel_size <= 0:
        raise ValueError("SDF voxel_size must be finite and greater than zero")
    if (isinstance(options.padding_voxels, bool)
            or not isinstance(options.padding_voxels, int)
            or options.padding_voxels < 1):
        raise ValueError("SDF padding_voxels must be a positive integer")


def _bake_validated_mesh(validated, options, device):
    h = options.voxel_size
    bounds = np.asarray(validated.bounds, dtype=np.float64)
    origin = bounds[0] - options.padding_voxels * h
    index_max = np.ceil((bounds[1] - bounds[0]) / h).astype(np.int64)
    index_max += 2 * options.padding_voxels
    if np.any(index_max > np.iinfo(np.int32).max - 8):
        raise ValueError("SDF dimensions exceed NanoVDB integer index limits")

    # allocate rounds out to complete 8^3 tiles. Fill every allocated voxel,
    # including the rounded padding, rather than leaving zeros in partial tiles.
    volume = wp.Volume.allocate(
        [0, 0, 0], index_max.tolist(), h, bg_value=float(np.linalg.norm(bounds[1] - bounds[0]) + h),
        translation=origin.tolist(), device=device,
    )
    tiles = wp.empty(volume.get_tile_count(), dtype=wp.vec3i, device=device)
    volume.get_tiles(out=tiles)
    rounded_extent = ((index_max // 8 + 1) * 8 - 1) * h
    max_distance = float(np.linalg.norm(rounded_extent) + h)
    points = wp.array(np.asarray(validated.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    indices = wp.array(np.asarray(validated.faces, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device)
    # Whole-mesh orientation is essential: the inner cavity shell has opposite
    # orientation to the outer shell and must not be independently reoriented.
    mesh = wp.Mesh(points=points, indices=indices, support_winding_number=True)
    failures = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _bake_tiles, dim=volume.get_tile_count() * 512,
        inputs=[mesh.id, volume.id, tiles, max_distance, failures], device=device,
    )
    wp.synchronize_device(device)
    failure_count = int(failures.numpy()[0])
    if failure_count:
        raise RuntimeError(f"SDF mesh distance query failed at {failure_count} voxels; cache was not saved")
    # No Mesh/BVH, tile-coordinate, or failure buffer is retained by BakedSDF.
    return volume


def _save_atomic(volume, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=cache_path.stem + ".", suffix=".nvdb.tmp", dir=cache_path.parent)
    os.close(fd)
    try:
        volume.save_to_nvdb(temporary, codec="none")
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_or_bake_sdf(obj_path, *, cache_dir, options=None, device=None):
    """Load a trusted existing cache, or validate, bake and atomically save it.

    The cache is ``cache_dir/<OBJ stem>.nvdb``. Existence alone selects a cache;
    no source hash, parameter comparison, or sidecar is used. An unreadable cache
    raises an error requiring explicit force_rebuild, rather than silently baking.
    """
    options = options if options is not None else SDFOptions()
    source_path = Path(obj_path).expanduser().resolve()
    cache_path = Path(cache_dir).expanduser().resolve() / (source_path.stem + ".nvdb")
    selected = _cuda_device(device)
    if cache_path.exists() and not options.force_rebuild:
        try:
            with cache_path.open("rb") as stream:
                volume = wp.Volume.load_from_nvdb(stream, device=selected)
            if volume.dtype != wp.float32 or volume.is_index:
                raise ValueError("cache must contain a float32 scalar distance grid")
            wp.synchronize_device(selected)
        except Exception as error:
            raise RuntimeError(
                f"Cannot load SDF cache {cache_path}; explicitly force rebuild to replace it: {error}"
            ) from error
        return BakedSDF(volume, source_path, cache_path, True)

    _check_options(options)
    # Import only on a cache miss; even mesh parsing/validation is skipped on hits.
    if __package__:
        from .mesh_validation import load_validate_container_obj
    else:
        from mesh_validation import load_validate_container_obj
    validated = load_validate_container_obj(source_path)
    volume = _bake_validated_mesh(validated, options, selected)
    _save_atomic(volume, cache_path)
    return BakedSDF(volume, source_path, cache_path, False)
