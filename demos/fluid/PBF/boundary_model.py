"""Load static NanoVDB SDFs and precompute their boundary volume maps."""

from contextlib import contextmanager
import math
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp


ASSET_DIR = Path(__file__).resolve().parent / "Assets"


@contextmanager
def validate_boundary(
    mode, *, filename=None, path=None, device=None, sdf=None,
    support_radius=None, quadrature_points=None,
):
    """Check a function's inputs and results through one context-manager call.

    Filename mode yields the name; precompute yields the allocated tile count.
    Load mode yields a state with the resolved device: assign its ``sdf`` in
    the body. Quadrature mode yields a state: assign ``offsets`` and
    ``volume_weights`` in the body. These results are checked on normal exit,
    including when the body returns. Loading errors retain their path context.
    """
    if mode == "filename":
        if (not isinstance(filename, str) or not filename.strip()
                or any(character in filename for character in "/\\:\x00")
                or Path(filename).name != filename
                or Path(filename).suffix.lower() != ".nvdb"):
            raise ValueError("SDF must be a bare .nvdb filename without a directory or drive")
        yield filename
        return

    if mode == "load":
        if path is None:
            raise ValueError("load validation requires a path")
        if not path.is_file():
            raise FileNotFoundError(f"Static SDF not found: {path}; generate it with python -m demos.sim_Ready first")
        selected = wp.get_device(device)
        if not selected.is_cuda:
            raise ValueError(f"Static NanoVDB SDF loading requires a CUDA device: {path}")
        state = SimpleNamespace(device=selected, sdf=None)
        try:
            yield state
            if state.sdf is None:
                raise TypeError("sdf must be a wp.Volume")
            if state.sdf.dtype != wp.float32 or state.sdf.is_index:
                raise ValueError("Static SDF must be a float32 scalar distance grid")
            wp.synchronize_device(selected)
        except Exception as error:
            raise RuntimeError(f"Cannot load static SDF {path}: {error}") from error
        return

    if mode in ("quadrature", "precompute"):
        if support_radius is None or not math.isfinite(support_radius) or support_radius <= 0:
            raise ValueError("support_radius must be finite and positive")
        if (isinstance(quadrature_points, bool)
                or not isinstance(quadrature_points, Integral) or quadrature_points < 1):
            raise ValueError("quadrature_points must be a positive integer")

    if mode == "quadrature":
        state = SimpleNamespace(offsets=None, volume_weights=None)
        yield state
        if state.offsets is None or state.volume_weights is None:
            raise ValueError("quadrature validation requires offsets and volume_weights")
        if (not np.isfinite(state.offsets).all() or not np.isfinite(state.volume_weights).all()
                or np.any(state.volume_weights <= 0)):
            raise ValueError("Quadrature offsets and weights must be representable as positive finite float32 weights")
        return

    if mode != "precompute":
        raise ValueError(f"Unknown boundary validation mode: {mode}")
    if not isinstance(sdf, wp.Volume):
        raise TypeError("sdf must be a wp.Volume")
    if not sdf.device.is_cuda:
        raise ValueError("Volume-map precomputation requires a CUDA SDF")
    if sdf.dtype != wp.float32 or sdf.is_index:
        raise ValueError("SDF must be a float32 scalar distance grid")
    tile_count = sdf.get_tile_count()
    if tile_count == 0:
        raise ValueError("SDF must have at least one allocated tile")
    yield tile_count


def resolve_sdf_path(filename):
    with validate_boundary("filename", filename=filename) as name:
        return (Path(ASSET_DIR) / name).resolve()


def resolve_scene_path(filename):
    """The renderer consumes the OBJ paired with the selected NanoVDB stem."""
    return resolve_sdf_path(filename).with_suffix(".obj")


@wp.struct
class SDFDeviceData:
    volume_id: wp.uint64
    inverse_voxel_size: wp.vec3


@wp.func
def sample_boundary_sdf(boundary: SDFDeviceData, position: wp.vec3):
    """Return distance and world-space gradient within the baked grid domain.

    The baker uses axis-aligned, uniformly spaced voxels. Outside allocated
    voxels NanoVDB returns its background value, not a mesh distance query.
    """
    index = wp.volume_world_to_index(boundary.volume_id, position)
    gradient = wp.vec3(0.0)
    distance = wp.volume_sample_grad_f(boundary.volume_id, index, wp.Volume.LINEAR, gradient)
    world_gradient = wp.cw_mul(gradient, boundary.inverse_voxel_size)
    return distance, world_gradient


def load_static_sdf(filename, *, device=None) -> wp.Volume:
    """Read an offline SDF from ASSET_DIR; the caller must retain the Volume.

    This startup operation neither reads an OBJ nor invokes the offline baker.
    """
    path = resolve_sdf_path(filename)
    with validate_boundary("load", path=path, device=device) as checked:
        with path.open("rb") as stream:
            volume = wp.Volume.load_from_nvdb(stream, device=checked.device)
        checked.sdf = volume
    print(f"Static SDF loaded: {path} (voxel size {volume.get_voxel_size()[0]:g} m)")

    return volume


def _build_volume_quadrature(support_radius, quadrature_points):
    """Return ball-filtered Gauss offsets and physical volume weights.

    ``quadrature_points`` is the number of one-dimensional Gauss points,
    not the polynomial order. The cube-to-ball cutoff is deliberately not
    renormalized, matching SPlisHSPlasH's volume-map quadrature.
    """
    with validate_boundary(
        "quadrature", support_radius=support_radius, quadrature_points=quadrature_points,
    ) as checked:
        nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_points))
        points = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
        products = (weights[:, None, None] * weights[None, :, None] * weights[None, None, :]).reshape(-1)
        inside = np.sum(points * points, axis=1) <= 1.0
        checked.offsets = np.asarray(points[inside] * support_radius, dtype=np.float32)
        checked.volume_weights = np.asarray(products[inside] * support_radius**3, dtype=np.float32)
        return checked.offsets, checked.volume_weights


@wp.func
def _volume_occupancy(distance: float, support_radius: float):
    """Cubic spline extension gamma*: one inside, zero beyond the radius."""
    if distance <= 0.0:
        return 1.0
    q = distance / support_radius
    if q <= 0.5:
        return 1.0 - 6.0 * q * q + 6.0 * q * q * q
    if q < 1.0:
        t = 1.0 - q
        return 2.0 * t * t * t
    return 0.0


@wp.kernel
def precompute_VM(
    sdf_id: wp.uint64,
    volume_map_id: wp.uint64,
    tiles: wp.array(dtype=wp.vec3i),
    offsets: wp.array(dtype=wp.vec3),
    volume_weights: wp.array(dtype=float),
    support_radius: float,
):
    tid = wp.tid()
    local = tid % 512
    cell = tiles[tid // 512] + wp.vec3i(local // 64, (local // 8) % 8, local % 8)
    cell_f = wp.vec3(float(cell[0]), float(cell[1]), float(cell[2]))
    x = wp.volume_index_to_world(volume_map_id, cell_f)
    volume = float(0.0)
    for sample in range(offsets.shape[0]):
        index = wp.volume_world_to_index(sdf_id, x + offsets[sample])
        distance = wp.volume_sample_f(sdf_id, index, wp.Volume.LINEAR)
        volume += volume_weights[sample] * _volume_occupancy(distance, support_radius)
    wp.volume_store_f(volume_map_id, cell[0], cell[1], cell[2], volume)


def precompute_volumeMap(
    sdf: wp.Volume,
    support_radius: float,
    *,
    quadrature_points: int = 16,
) -> wp.Volume:
    """Integrate gamma*(SDF) over each node's support ball, in cubic meters.

    Return a completed, independent float32 CUDA volume with the input's
    allocated tiles and coordinate transform, but a zero background. The input
    is unchanged. Changing the support radius requires recomputing the map.

    Sampling outside allocated SDF voxels uses the input's NanoVDB background.
    Accurate geometric integrals require the SDF to cover the entire support
    ball plus the trilinear interpolation stencil at each queried map node.
    No mesh reconstruction, padding, file IO, or solver integration is done.
    """
    with validate_boundary(
        "precompute", sdf=sdf, support_radius=support_radius, quadrature_points=quadrature_points,
    ) as tile_count:
        offsets, volume_weights = _build_volume_quadrature(support_radius, quadrature_points)

        device = sdf.device
        tiles = wp.empty(tile_count, dtype=wp.vec3i, device=device)
        sdf.get_tiles(out=tiles)
        info = sdf.get_grid_info()
        volume_map = wp.Volume.allocate_by_tiles(
            tiles, transform=info.transform_matrix, translation=info.translation,
            bg_value=0.0, device=device,
        )
        gpu_offsets = wp.array(offsets, dtype=wp.vec3, device=device)
        gpu_weights = wp.array(volume_weights, dtype=float, device=device)
        wp.launch(
            precompute_VM,
            dim=tile_count * 512,
            inputs=[sdf.id, volume_map.id, tiles, gpu_offsets, gpu_weights, float(support_radius)],
            device=device,
        )
        wp.synchronize_device(device)
        return volume_map

