"""Load the configured static GPU SDF independently of the PBF solver."""

from pathlib import Path

import warp as wp


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
BOUNDARY_DIR = ASSET_DIR / "Boundary"
STATIC_SDF_PATH = BOUNDARY_DIR / "stairs_container_bool_simplified.nvdb"


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


def load_static_sdf(*, device=None) -> wp.Volume:
    """Read STATIC_SDF_PATH into GPU memory; the caller must retain the Volume.

    This startup operation neither reads an OBJ nor invokes the offline baker.
    """
    path = Path(STATIC_SDF_PATH).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Static SDF not found: {path}; bake it first or update STATIC_SDF_PATH")
    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise ValueError("Static NanoVDB SDF loading requires a CUDA device")
    try:
        with path.open("rb") as stream:
            volume = wp.Volume.load_from_nvdb(stream, device=selected)
        if volume.dtype != wp.float32 or volume.is_index:
            raise ValueError("Static SDF must be a float32 scalar distance grid")
        wp.synchronize_device(selected)
    except Exception as error:
        raise RuntimeError(f"Cannot load static SDF {path}: {error}") from error
    print(f"Static SDF loaded: {path} (voxel size {volume.get_voxel_size()[0]:g} m)")
    return volume
