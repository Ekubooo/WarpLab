"""Read published assets without importing mesh processing or asset builders."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
ASSET_VERSION = 1


@dataclass
class ConvexHull:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class PublishedAsset:
    mesh_path: Path
    sdf_path: Path
    metadata: dict
    hulls: tuple = ()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sdf_digest(volume):
    """Identify SDF contents independently of the mutable second NVDB grid.

    Saving a new grid pair changes the checksum, flags, index and grid count.
    Exclude these header fields; retain the SDF values, topology and transform.
    """
    import warp as wp
    size = volume.get_grid_info().size_in_bytes
    raw = wp.array(ptr=volume.array().ptr, dtype=wp.uint8, shape=size, device=volume.device).numpy()
    digest = hashlib.sha256()
    # Hash slices rather than copying a full grid or modifying a CPU-backed view.
    for chunk in (raw[:8], bytes(8), raw[16:20], bytes(12), raw[32:]):
        digest.update(chunk)
    return digest.hexdigest()


def load_asset(stem, expected_geometry, *, command):
    """Read a complete published set; never repair or produce missing assets."""
    stem = Path(stem)
    try:
        metadata = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata["version"] != ASSET_VERSION or metadata["geometry"] != expected_geometry:
            raise ValueError("Published geometry or asset version differs from the scene")
        mesh, sdf = stem.with_suffix(".obj"), stem.with_suffix(".nvdb")
        if not sdf.is_file() or file_digest(mesh) != metadata["mesh_sha256"]:
            raise ValueError("Published OBJ/SDF is missing or the mesh has changed")
        if len(metadata["sdf_sha256"]) != 64:
            raise ValueError("Missing SDF identity")
        if expected_geometry.get("kind") == "spot" and (
                not metadata.get("mass") or not metadata.get("collision_sha256")):
            raise ValueError("Missing published Spot mass or convex collision data")
        mass = metadata.get("mass")
        if mass is not None:
            com = np.asarray(mass["com"], dtype=float)
            inertia = np.asarray(mass["inertia_unit_density"], dtype=float)
            if (not np.isfinite(mass["volume"]) or mass["volume"] <= 0
                    or com.shape != (3,) or not np.isfinite(com).all()
                    or inertia.shape != (3, 3) or not np.isfinite(inertia).all()
                    or not np.allclose(inertia, inertia.T, atol=1e-12)
                    or np.any(np.linalg.eigvalsh(inertia) <= 0)):
                raise ValueError("Invalid published mass properties")
        hulls = []
        if "collision_sha256" in metadata:
            path = stem.with_suffix(".collision.npz")
            if file_digest(path) != metadata["collision_sha256"]:
                raise ValueError("Published collision data has changed")
            with np.load(path, allow_pickle=False) as saved:
                count = int(saved["count"])
                if not 1 <= count <= 32:
                    raise ValueError("Invalid convex piece count")
                for i in range(count):
                    vertices, faces = saved[f"vertices_{i}"], saved[f"faces_{i}"]
                    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not 4 <= len(vertices) <= 64
                            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
                            or len(faces) < 4 or faces.dtype.kind not in "iu"
                            or faces.min() < 0 or faces.max() >= len(vertices)):
                        raise ValueError("Invalid published collision arrays")
                    hulls.append(ConvexHull(vertices, faces))
        return PublishedAsset(mesh, sdf, metadata, tuple(hulls))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"Cannot load published asset {stem}: {error}. Build it offline:\n{command}") from error


def load_asset_volumes(asset, support_radius, device):
    from demos.PBF_Rigid.solver.boundary_model import load_boundary_volumes
    def validate_sdf(sdf):
        if sdf_digest(sdf) != asset.metadata["sdf_sha256"]:
            raise ValueError(f"Published SDF content has changed: {asset.sdf_path}; rebuild its offline asset set")
    return load_boundary_volumes(asset.sdf_path, support_radius, device=device, validate_sdf=validate_sdf)


def build_command(scene, config):
    from dataclasses import replace
    from .config import PBFConfig
    from .scene_data import tank_box_config
    base = tank_box_config() if scene == "tank-box" else PBFConfig()
    try:
        replace(base, particle_radius=config.particle_radius, container_size=config.container_size)
    except ValueError:
        return (f"python -m demos.PBF_Rigid.sim_ready.pbf_assets --scene {scene} --config SAME_FLUID_CONFIG.json\n"
                "Use the same full fluid JSON; runtime CLI overrides can be saved with --export-config.")
    size = " ".join(str(value) for value in config.container_size)
    return (f"python -m demos.PBF_Rigid.sim_ready.pbf_assets --scene {scene} "
            f"--particle-radius {config.particle_radius} --container-size {size}")
