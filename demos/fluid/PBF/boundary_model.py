"""Resolve, validate and load offline NanoVDB assets selected by PBF."""

from pathlib import Path

import warp as wp


ASSET_DIR = Path(__file__).resolve().parent / "Assets"


def validate_sdf_filename(filename):
    """Accept a bare NanoVDB filename, never a path or Windows drive prefix."""
    if (not isinstance(filename, str) or not filename.strip()
            or any(character in filename for character in "/\\:\x00")
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != ".nvdb"):
        raise ValueError("SDF must be a bare .nvdb filename without a directory or drive")
    return filename


def resolve_sdf_path(filename):
    name = validate_sdf_filename(filename)
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
    if not path.is_file():
        raise FileNotFoundError(f"Static SDF not found: {path}; generate it with python -m demos.sim_Ready first")
    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise ValueError(f"Static NanoVDB SDF loading requires a CUDA device: {path}")
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
