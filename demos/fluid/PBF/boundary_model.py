"""Own static boundary resources; keep baking and solver configuration separate."""

from dataclasses import dataclass
from pathlib import Path
import sys

import warp as wp


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
CONTAINER_DIR = ASSET_DIR / "Container"
BOUNDARY_DIR = ASSET_DIR / "Boundary"
DEFAULT_VOXEL_SIZE = 0.0075


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


@dataclass
class StaticBoundary:
    # Keep the Volume owner alive for as long as kernels may use its id.
    volume: wp.Volume
    source_path: Path
    cache_path: Path
    cache_hit: bool
    device_data: SDFDeviceData


def resolve_container_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute() and path.parent == Path("."):
        path = CONTAINER_DIR / path
    return path.resolve()


def load_static_boundary(
    boundary_obj=None, *, sdf_voxel_size=DEFAULT_VOXEL_SIZE,
    sdf_force_rebuild=False, device=None,
):
    """Load/bake only when requested; no SDF collision response is applied."""
    if boundary_obj is None:
        return None

    # Direct-script and package execution share the same canonical baker module.
    if not __package__:
        root = str(Path(__file__).resolve().parents[3])
        if root not in sys.path:
            sys.path.insert(0, root)
    from demos.fluid.boundary.sdf_baker import SDFOptions, load_or_bake_sdf

    result = load_or_bake_sdf(
        resolve_container_path(boundary_obj), cache_dir=BOUNDARY_DIR,
        options=SDFOptions(voxel_size=sdf_voxel_size, force_rebuild=sdf_force_rebuild),
        device=device,
    )
    data = SDFDeviceData()
    data.volume_id = result.volume.id
    data.inverse_voxel_size = wp.vec3(*(1.0 / value for value in result.volume.get_voxel_size()))
    boundary = StaticBoundary(
        volume=result.volume, source_path=result.source_path,
        cache_path=result.cache_path, cache_hit=result.cache_hit, device_data=data,
    )
    action = "cache loaded" if boundary.cache_hit else "baked and cached"
    print(f"SDF {action}: {boundary.cache_path} (voxel size {result.volume.get_voxel_size()[0]:g} m)")
    return boundary


def add_sdf_arguments(parser):
    group = parser.add_argument_group("Static boundary SDF (load only; no collision response yet)")
    group.add_argument(
        "--boundary-obj", metavar="PATH",
        help="Container OBJ; a bare filename is resolved under PBF/Assets/Container.",
    )
    group.add_argument(
        "--sdf-voxel-size", type=float, default=DEFAULT_VOXEL_SIZE, metavar="METERS",
        help="Bake spacing; existing same-name cache takes precedence (default: 0.0075).",
    )
    group.add_argument(
        "--sdf-force-rebuild", action="store_true",
        help="Validate the OBJ and replace its cache in PBF/Assets/Boundary.",
    )


def boundary_arguments(args):
    return dict(
        boundary_obj=args.boundary_obj, sdf_voxel_size=args.sdf_voxel_size,
        sdf_force_rebuild=args.sdf_force_rebuild,
    )
