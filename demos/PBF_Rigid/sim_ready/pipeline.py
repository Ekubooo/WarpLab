"""Archive destinations and grouped publication on top of shared asset tools."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

from demos.sim_Ready import pipeline as _shared
from demos.sim_Ready.pipeline import AssetPaths, _temporary, save_mesh_obj, validate_name
from . import config


@dataclass(frozen=True)
class BuildResult:
    landscape: Path
    sdf: Path
    destinations: tuple[Path, ...]


def resolve_obj(value):
    path = Path(value).expanduser()
    if not path.is_absolute() and "/" not in str(value) and "\\" not in str(value):
        path = config.ASSETS / path
    return _shared.resolve_obj(path.resolve())


def resolve_asset_paths(obj_path, name, *, output=None, destinations=None):
    return _shared.resolve_asset_paths(
        resolve_obj(obj_path), name,
        output=config.OUTPUT if output is None else output,
        destinations=config.OUTPUT_DIRECTORIES if destinations is None else destinations,
    )


def publish_pair(sources, directories):
    """Prepare every copy and backup first; roll back replacements on failure.

    There is no cross-file atomic visibility guarantee. Run publication with
    consumers stopped; rollback protects previous assets on handled I/O failures.
    """
    publish_files([(source, directory / source.name) for directory in directories for source in sources])


def publish_files(destinations):
    """Publish explicit (source, target) pairs as one rollback-protected group.

    Supports OBJ/NVDB plus physical/collision attachments across subdirectories.
    Like publish_pair, consumers must be stopped during publication.
    """
    prepared, replaced, recovery_files, rollback_errors = [], [], set(), []
    try:
        for source, destination in destinations:
            temporary = _temporary(destination)
            entry = [destination, temporary, None]
            prepared.append(entry)
            shutil.copyfile(source, temporary)
            if destination.exists():
                entry[2] = _temporary(destination)
                shutil.copyfile(destination, entry[2])
        for destination, temporary, backup in prepared:
            os.replace(temporary, destination)
            replaced.append((destination, backup))
    except BaseException as error:
        for destination, backup in reversed(replaced):
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except OSError as rollback_error:
                rollback_errors.append(f"{destination}: {rollback_error}")
                if backup is not None:
                    recovery_files.add(backup)
        if rollback_errors:
            raise RuntimeError(f"Publication rollback could not restore all files: {rollback_errors}; "
                               f"recovery copies: {sorted(str(p) for p in recovery_files)}") from error
        raise
    finally:
        for _, temporary, backup in prepared:
            temporary.unlink(missing_ok=True)
            if backup is not None and backup not in recovery_files:
                backup.unlink(missing_ok=True)


def build_assets(obj_path="bunny_flat_bottom.obj", *, name=config.DEFAULT_NAME,
                 geometry_options=config.GeometryOptions(), bake_options=config.BakeOptions(),
                 obj_up_axis="Y", device="cuda:0"):
    """Always rebuild. The local OBJ is available before collision baking starts."""
    paths = resolve_asset_paths(obj_path, name)
    # Lazy imports keep help/name/path handling independent of CUDA and mesh libraries.
    from . import geometry, sdf, validation
    sdf.check_options(bake_options)
    stage = "scene preparation"
    try:
        scene = geometry.prepare_scene(paths.source, options=geometry_options, obj_up_axis=obj_up_axis)
        print(f"Model scale: {scene.scale:.9g}", flush=True)
        stage = "landscape boolean and simplification"
        visible = geometry.landscape(scene, geometry_options)
        obj = paths.mesh
        save_mesh_obj(visible, obj)
        print(f"Landscape: {obj} ({len(visible.faces)} triangles)", flush=True)
        stage = "container boolean and simplification"
        collision, bounds = geometry.container(scene, geometry_options)
        stage = "container validation"
        checked = validation.validate_container(collision, bounds, tolerance=geometry_options.simplify_tolerance)
        print(f"Cavity bounds: {checked.cavity_bounds.tolist()}", flush=True)
        stage = "SDF baking"
        volume = sdf.bake(checked, options=bake_options, device=device)
        nvdb = paths.sdf
        sdf.save_volume(volume, nvdb)
        stage = "asset publication"
        publish_pair((obj, nvdb), [p for p in paths.destinations if p != obj.parent])
        for directory in paths.destinations:
            print(f"Assets: {directory / name}.obj / .nvdb", flush=True)
        return BuildResult(obj, nvdb, paths.destinations)
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"{stage} failed: {error}") from error
