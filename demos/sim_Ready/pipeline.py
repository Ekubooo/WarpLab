"""Stage orchestration and publication. Consumers are filesystem destinations only."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import tempfile

from . import config


@dataclass(frozen=True)
class BuildResult:
    landscape: Path
    sdf: Path
    destinations: tuple[Path, ...]


def validate_name(value):
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(c in value for c in '/\\:.<>"|?*')
            or any(ord(c) < 32 for c in value)
            or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", value)):
        raise ValueError("Output name must be a bare filename stem without an extension or reserved characters")
    return value


def resolve_obj(value):
    text = str(value)
    path = Path(value).expanduser()
    if not path.is_absolute() and "/" not in text and "\\" not in text:
        path = config.ASSETS / path
    path = path.resolve()
    if path.suffix.lower() != ".obj":
        raise ValueError(f"Expected an OBJ input: {path}")
    return path


def _protect_input(source, paths):
    for path in paths:
        if path.resolve() == source or (path.exists() and source.exists() and path.samefile(source)):
            raise ValueError(f"Output would overwrite the input model: {path}")


def _temporary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=path.stem + ".", suffix=".asset.tmp", dir=path.parent)
    os.close(descriptor)
    return Path(filename)


def _save_landscape(mesh, path):
    temporary = _temporary(path)
    try:
        temporary.write_text(mesh.export(file_type="obj"), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_pair(sources, directories):
    """Prepare every copy and backup first; roll back replacements on failure.

    There is no cross-file atomic visibility guarantee. Run publication with
    consumers stopped; rollback protects previous assets on handled I/O failures.
    """
    prepared, replaced, recovery_files, rollback_errors = [], [], set(), []
    try:
        for directory in directories:
            for source in sources:
                destination = directory / source.name
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
    name = validate_name(name)
    source = resolve_obj(obj_path)
    directories = tuple(dict.fromkeys(Path(p).resolve() for p in config.OUTPUT_DIRECTORIES))
    output = config.OUTPUT.resolve()
    if output not in directories:
        raise ValueError("Output destinations must include the primary output directory")
    paths = [directory / (name + extension) for directory in directories for extension in (".obj", ".nvdb")]
    _protect_input(source, paths)
    if not source.is_file():
        raise FileNotFoundError(f"Input OBJ not found: {source}")
    # Lazy imports keep help/name/path handling independent of CUDA and mesh libraries.
    from . import geometry, sdf, validation
    sdf.check_options(bake_options)
    stage = "scene preparation"
    try:
        scene = geometry.prepare_scene(source, options=geometry_options, obj_up_axis=obj_up_axis)
        print(f"Model scale: {scene.scale:.9g}", flush=True)
        stage = "landscape boolean and simplification"
        visible = geometry.landscape(scene, geometry_options)
        obj = output / (name + ".obj")
        _save_landscape(visible, obj)
        print(f"Landscape: {obj} ({len(visible.faces)} triangles)", flush=True)
        stage = "container boolean and simplification"
        collision, bounds = geometry.container(scene, geometry_options)
        stage = "container validation"
        checked = validation.validate_container(collision, bounds, tolerance=geometry_options.simplify_tolerance)
        print(f"Cavity bounds: {checked.cavity_bounds.tolist()}", flush=True)
        stage = "SDF baking"
        volume = sdf.bake(checked, options=bake_options, device=device)
        nvdb = output / (name + ".nvdb")
        sdf.save_volume(volume, nvdb)
        stage = "asset publication"
        publish_pair((obj, nvdb), [p for p in directories if p != output])
        for directory in directories:
            print(f"Assets: {directory / name}.obj / .nvdb", flush=True)
        return BuildResult(obj, nvdb, directories)
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"{stage} failed: {error}") from error
