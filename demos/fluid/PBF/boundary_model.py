"""Load shared boundary assets and manage static/rigid instances.

Volume-map generation and file caching are independent of instance counts and
poses. This module owns boundary resources, not rigid-body dynamics or rendering.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp


# =============================================================================
# 1. Default scene configuration
# Configure asset paths, the static container and rigid counts. The volume-map
# algorithm version determines cache validity.
# =============================================================================

ASSET_DIR = Path(__file__).resolve().parent / "Assets"
# Default PBF scene. Rigid keys are filename stems; zero disables an asset.
SDF_FILENAME = "stairs_scene.nvdb"
DEFAULT_RIGID_COUNT = 1
RIGID_COUNTS = {}
VOLUME_MAP_ALGORITHM_VERSION = 1


# =============================================================================
# 2. Host configuration and asset records
# Describe scene selection, loading options, shared Volumes and instance
# relationships without loading resources or solving dynamics.
# =============================================================================

@dataclass(frozen=True)
class BoundaryScene:
    """Static filenames and per-asset instance counts under Assets/Rigid."""

    static_assets: tuple[str, ...] = ()
    default_rigid_count: int = 1
    rigid_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryLoadOptions:
    support_radius: float
    device: str | wp.Device | None = None
    quadrature_points: int = 16


@dataclass(frozen=True)
class BoundaryAsset:
    sdf_path: Path
    mesh_path: Path
    sdf: wp.Volume
    volume_map: wp.Volume


@dataclass(frozen=True)
class BoundaryInstance:
    asset_id: int
    # Internal rigid index, not a Newton body_id. Static instances use -1.
    rigid_id: int = -1

    @property
    def is_dynamic(self):
        return self.rigid_id >= 0


# =============================================================================
# 3. GPU data and boundary sampling
# Define kernel data layouts and use instance poses to transform world-space
# query points into asset-local coordinates before sampling.
# =============================================================================

@wp.struct
class SDFDeviceData:
    volume_id: wp.uint64
    inverse_voxel_size: wp.vec3


@wp.struct
class BoundaryDeviceData:
    """Stable device buffers; resource ownership remains in BoundarySystem."""

    sdf_ids: wp.array(dtype=wp.uint64)
    volume_map_ids: wp.array(dtype=wp.uint64)
    inverse_voxel_sizes: wp.array(dtype=wp.vec3)
    asset_ids: wp.array(dtype=int)
    rigid_ids: wp.array(dtype=int)
    transforms: wp.array(dtype=wp.transform)
    rigid_instance_ids: wp.array(dtype=int)
    count: int
    rigid_count: int


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


@wp.func
def sample_instance_sdf(data: BoundaryDeviceData, instance_id: int, position: wp.vec3):
    """Sample in asset coordinates; return distance and world-space gradient."""
    transform = data.transforms[instance_id]
    local = wp.transform_point(wp.transform_inverse(transform), position)
    asset_id = data.asset_ids[instance_id]
    boundary = SDFDeviceData()
    boundary.volume_id = data.sdf_ids[asset_id]
    boundary.inverse_voxel_size = data.inverse_voxel_sizes[asset_id]
    distance, gradient = sample_boundary_sdf(boundary, local)
    return distance, wp.transform_vector(transform, gradient)


@wp.func
def sample_instance_volume(data: BoundaryDeviceData, instance_id: int, position: wp.vec3):
    """Return the cached boundary volume [m^3] at a world-space position."""
    local = wp.transform_point(wp.transform_inverse(data.transforms[instance_id]), position)
    volume_id = data.volume_map_ids[data.asset_ids[instance_id]]
    index = wp.volume_world_to_index(volume_id, local)
    return wp.volume_sample_f(volume_id, index, wp.Volume.LINEAR)


# =============================================================================
# 4. Instance management and pose updates
# BoundarySystem owns resources and fixed-size device buffers, updates rigid
# poses and resets transforms in place.
# =============================================================================

@wp.kernel
def _set_rigid_poses(poses: wp.array(dtype=wp.transform), instance_ids: wp.array(dtype=int),
                     transforms: wp.array(dtype=wp.transform)):
    i = wp.tid()
    transforms[instance_ids[i]] = poses[i]


class BoundarySystem:
    """Shared Volumes plus a fixed-size instance table on one CUDA device.

    Mesh paths are recorded, not read. Transforms map asset-local coordinates
    to world coordinates and initially equal identity. GPU pose inputs must
    contain finite translations and normalized XYZW quaternions.
    """

    def __init__(self, assets, instances, device):
        self.assets = tuple(assets)
        self.instances = tuple(instances)
        self.device = wp.get_device(device)
        rigid_indices = [i for i, instance in enumerate(self.instances) if instance.is_dynamic]
        self.rigid_count = len(rigid_indices)
        self.static_count = len(self.instances) - self.rigid_count
        data = BoundaryDeviceData()
        data.sdf_ids = wp.array([a.sdf.id for a in self.assets], dtype=wp.uint64, device=self.device)
        data.volume_map_ids = wp.array([a.volume_map.id for a in self.assets], dtype=wp.uint64, device=self.device)
        spacing = np.array([a.sdf.get_voxel_size() for a in self.assets], dtype=np.float32).reshape(-1, 3)
        data.inverse_voxel_sizes = wp.array(1.0 / spacing, dtype=wp.vec3, device=self.device)
        data.asset_ids = wp.array([i.asset_id for i in self.instances], dtype=int, device=self.device)
        data.rigid_ids = wp.array([i.rigid_id for i in self.instances], dtype=int, device=self.device)
        data.rigid_instance_ids = wp.array(rigid_indices, dtype=int, device=self.device)
        data.transforms = wp.empty(len(self.instances), dtype=wp.transform, device=self.device)
        data.count = len(self.instances)
        data.rigid_count = self.rigid_count
        self.device_data = data
        self.reset()

    def set_rigid_poses(self, poses):
        """Update by internal rigid order, preserving every static transform.

        CPU inputs are N x 7 arrays (XYZ, XYZW); quaternions are normalized on
        a copy. Warp inputs are one-dimensional transform arrays and are copied
        on device, without downloading data or reallocating instance buffers.
        """
        if isinstance(poses, wp.array):
            if poses.dtype != wp.transform or poses.shape != (self.rigid_count,):
                raise ValueError(f"Expected {self.rigid_count} Warp transforms")
            source = poses.to(self.device)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.array(poses, dtype=np.float32, copy=True)
            if values.shape != (self.rigid_count, 7) or not np.isfinite(values).all():
                raise ValueError(f"Expected finite rigid poses with shape ({self.rigid_count}, 7)")
            norms = np.linalg.norm(values[:, 3:].astype(np.float64), axis=1)
            if np.any(norms == 0):
                raise ValueError("Rigid pose quaternions must be nonzero")
            values[:, 3:] /= norms[:, None]
            source = wp.array(values, dtype=wp.transform, device=self.device)
        if self.rigid_count:
            wp.launch(_set_rigid_poses, self.rigid_count,
                inputs=[source, self.device_data.rigid_instance_ids],
                outputs=[self.device_data.transforms], device=self.device)

    def reset(self):
        """Restore identity poses in place, retaining all loaded resources."""
        if self.device_data.count:
            self.device_data.transforms.fill_(wp.transform_identity())


# =============================================================================
# 5. Input validation, path resolution and asset discovery
# Validate loading and quadrature inputs, resolve static paths and rigid
# OBJ/NVDB pairs, and determine instance counts before loading.
# =============================================================================

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


@dataclass(frozen=True)
class _BoundaryRequest:
    path: Path
    mesh_path: Path
    count: int
    dynamic: bool


def _instance_count(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _resolve_boundary_requests(scene):
    """Preflight discovery/counts before loading or rewriting any NVDB cache."""
    default = _instance_count(scene.default_rigid_count, "default_rigid_count")
    if not isinstance(scene.rigid_counts, Mapping) or any(not isinstance(k, str) for k in scene.rigid_counts):
        raise ValueError("rigid_counts must map asset stems to nonnegative integers")
    counts = {name: _instance_count(value, f"rigid_counts[{name!r}]")
              for name, value in scene.rigid_counts.items()}
    if isinstance(scene.static_assets, (str, Path)):
        raise ValueError("static_assets must be a sequence of NVDB filenames")
    requests = []
    for filename in scene.static_assets:
        path = resolve_sdf_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Static SDF not found: {path}")
        requests.append(_BoundaryRequest(path, path.with_suffix(".obj"), 1, False))
    directory = Path(ASSET_DIR) / "Rigid"
    meshes = {}
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"Rigid asset directory is not a directory: {directory}")
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if path.is_file() and path.suffix.lower() == ".obj":
                meshes[path.stem] = path
    unknown = counts.keys() - meshes.keys()
    if unknown:
        raise ValueError(f"Unknown rigid assets in rigid_counts: {sorted(unknown)}")
    for name, mesh in meshes.items():
        count = counts.get(name, default)
        if count == 0:
            continue
        path = mesh.with_suffix(".nvdb").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing paired NVDB for rigid mesh {mesh}: {path}")
        requests.append(_BoundaryRequest(path, mesh.resolve(), count, True))
    if sum(request.count for request in requests) > np.iinfo(np.int32).max:
        raise ValueError("Boundary instance count exceeds int32 indexing")
    return requests


# =============================================================================
# 6. Volume-map quadrature and GPU precomputation
# Build support-ball quadrature points and compute the volume map on the SDF's
# existing sparse grid. File caching is handled separately below.
# =============================================================================

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


# =============================================================================
# 7. NanoVDB loading and in-file caching
# Read and validate SDF/volume-map pairs. Rebuild stale caches, atomically save
# them to the same NVDB file and reload the persisted grids.
# =============================================================================

def _nvdb_io():
    # Direct PBF.py execution puts only its own directory on sys.path.
    if not __package__:
        import sys
        root = str(Path(__file__).resolve().parents[3])
        if root not in sys.path:
            sys.path.insert(0, root)
    from demos.sim_Ready import nvdb_io
    return nvdb_io


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


def _volume_map_name(support_radius, quadrature_points):
    return (f"PBF_VMAP|v={VOLUME_MAP_ALGORITHM_VERSION}"
            f"|h={float(support_radius).hex()}|q={int(quadrature_points)}")


def _check_sdf_support(sdf, support_radius):
    name = sdf.get_grid_info().name
    if not name.startswith("PBF_SDF|"):
        return  # Legacy full-domain assets have no positive-side truncation.
    try:
        fields = dict(item.split("=", 1) for item in name.split("|")[1:])
        maximum = float.fromhex(fields["hmax"])
        if fields["v"] != "1" or not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("Unsupported SDF metadata")
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid sparse SDF metadata: {name}") from error
    if support_radius > maximum:
        raise ValueError(f"SDF supports h <= {maximum:g} m, requested {support_radius:g} m; "
                         "regenerate the asset with a larger max support radius")


def _read_boundary_volumes(path, device):
    count = _nvdb_io().grid_count(path)
    with path.open("rb") as stream:
        sdf = wp.Volume.load_from_nvdb(stream, device=device)
    if sdf.dtype != wp.float32 or sdf.is_index:
        raise ValueError("Static SDF must be a float32 scalar distance grid")
    info = sdf.get_grid_info()
    if info.grid_index != 0 or info.grid_count != count:
        raise ValueError("NanoVDB file and first grid headers disagree on grid count or index")
    if info.name.startswith("PBF_VMAP|"):
        raise ValueError("The first NanoVDB grid must be the SDF, not a volume map")
    volume_map = sdf.load_next_grid()
    if (count == 2) != (volume_map is not None):
        raise ValueError("NanoVDB file and grid headers disagree on grid count")
    if volume_map is not None:
        info = volume_map.get_grid_info()
        if info.grid_index != 1 or info.grid_count != count:
            raise ValueError("NanoVDB second grid header has an invalid count or index")
        if volume_map.dtype != wp.float32 or volume_map.is_index:
            raise ValueError("Volume map must be a float32 scalar grid")
        if volume_map.load_next_grid() is not None:
            raise ValueError("Unexpected third NanoVDB grid")
    wp.synchronize_device(device)
    return sdf, volume_map


def _check_volume_map_layout(sdf, volume_map):
    a, b = sdf.get_grid_info(), volume_map.get_grid_info()
    if (not np.array_equal(a.transform_matrix, b.transform_matrix)
            or not np.array_equal(a.translation, b.translation)
            or sdf.get_tile_count() != volume_map.get_tile_count()
            or not np.array_equal(sdf.get_tiles().numpy(), volume_map.get_tiles().numpy())):
        raise ValueError("Volume map coordinate transform or allocated tiles differ from SDF")


def load_boundary_volumes(path, support_radius, *, quadrature_points=16, device=None):
    """Load a matched grid pair, or compute, atomically cache and reload it.

    Only the selected NVDB is updated. Geometry baking and asset publication
    remain the responsibility of sim_Ready's standalone pipeline.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Static SDF not found: {path}; generate it with python -m demos.sim_Ready first")
    if not math.isfinite(support_radius) or support_radius <= 0:
        raise ValueError("support_radius must be finite and positive")
    if (isinstance(quadrature_points, bool) or not isinstance(quadrature_points, Integral)
            or quadrature_points < 1):
        raise ValueError("quadrature_points must be a positive integer")
    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise ValueError(f"Static NanoVDB boundary loading requires a CUDA device: {path}")
    expected = _volume_map_name(support_radius, quadrature_points)
    try:
        sdf, volume_map = _read_boundary_volumes(path, selected)
        _check_sdf_support(sdf, support_radius)
        if volume_map is not None and volume_map.get_grid_info().name == expected:
            _check_volume_map_layout(sdf, volume_map)
            print(f"Boundary cache hit: {path} (h={support_radius:g} m, q={quadrature_points})")
            return sdf, volume_map
        print(f"Boundary cache rebuild: {path} (h={support_radius:g} m, q={quadrature_points})")
        volume_map = None
        volume_map = precompute_volumeMap(sdf, support_radius, quadrature_points=quadrature_points)
        _nvdb_io().save_volumes(
            (sdf, volume_map), path, grid_names=(sdf.get_grid_info().name, expected),
        )
        # Reload the persisted pair through the same reader, without retaining
        # the old GPU allocations (including any stale map aliased by sdf).
        sdf = volume_map = None
        sdf, volume_map = _read_boundary_volumes(path, selected)
        if volume_map is None or volume_map.get_grid_info().name != expected:
            raise ValueError("Saved boundary cache metadata does not match requested parameters")
        _check_sdf_support(sdf, support_radius)
        _check_volume_map_layout(sdf, volume_map)
        return sdf, volume_map
    except Exception as error:
        raise RuntimeError(f"Cannot load boundary volumes {path}: {error}") from error


# =============================================================================
# 8. Unified scene loading entry points
# Load each asset once and build the instance table. The default PBF entry
# accepts only the smoothing length and uses the current Warp device.
# =============================================================================

def load_boundaries(scene: BoundaryScene, options: BoundaryLoadOptions) -> BoundarySystem:
    """Load a scene once per asset, returning owned resources and device data.

    Rigid OBJ/NVDB pairs are discovered only under Assets/Rigid. A missing or
    empty rigid directory is valid. Geometry is not loaded or baked here.
    """
    requests = _resolve_boundary_requests(scene)
    if not math.isfinite(options.support_radius) or options.support_radius <= 0:
        raise ValueError("support_radius must be finite and positive")
    if (isinstance(options.quadrature_points, bool) or not isinstance(options.quadrature_points, Integral)
            or options.quadrature_points < 1):
        raise ValueError("quadrature_points must be a positive integer")
    device = wp.get_device(options.device)
    if not device.is_cuda:
        raise ValueError("Boundary Volume loading requires a CUDA device")
    assets, instances, asset_ids = [], [], {}
    rigid_id = 0
    for request in requests:
        if request.path not in asset_ids:
            sdf, volume_map = load_boundary_volumes(
                request.path, options.support_radius,
                quadrature_points=options.quadrature_points, device=device,
            )
            asset_ids[request.path] = len(assets)
            assets.append(BoundaryAsset(request.path, request.mesh_path, sdf, volume_map))
        asset_id = asset_ids[request.path]
        for _ in range(request.count):
            instances.append(BoundaryInstance(asset_id, rigid_id if request.dynamic else -1))
            if request.dynamic:
                rigid_id += 1
    return BoundarySystem(assets, instances, device)


def load_default_boundaries(support_radius: float) -> BoundarySystem:
    """Load the configured PBF scene on the current Warp device.

    Asset selection and loading defaults stay here; PBF supplies only its
    smoothing length. Use load_boundaries for an explicitly configured scene.
    """
    scene = BoundaryScene(
        static_assets=(SDF_FILENAME,),
        default_rigid_count=DEFAULT_RIGID_COUNT,
        rigid_counts=RIGID_COUNTS,
    )
    return load_boundaries(scene, BoundaryLoadOptions(support_radius))
