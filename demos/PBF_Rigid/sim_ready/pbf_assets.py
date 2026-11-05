"""Offline preparation and publication of the three PBF scene recipes.

Consumers only read published files. This tool owns geometric validation,
booleans, SDF baking, convex decomposition and imported-solid mass integration.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from . import config as asset_config, pipeline
from demos.PBF_Rigid.config import PBFConfig
from demos.PBF_Rigid import config_io
from demos.PBF_Rigid.scene_data import (
    BOX_HALF_EXTENTS, BOX_SIZE, SPHERE_RADIUS, SPOT_LENGTH, tank_box_config,
    tank_walls, asset_geometry,
)
from demos.PBF_Rigid.asset_io import ASSET_VERSION, file_digest, sdf_digest


CACHE_DIR = asset_config.ROOT.parents[2] / ".cache" / "pbf"


def _box_mesh(center, half):
    import trimesh
    mesh = trimesh.creation.box(extents=[2. * value for value in half])
    mesh.apply_translation(center)
    return mesh


def asset_cache_path(config, *, cache_dir=None, voxel_size=.0075):
    """Retain the original tank cache key to migrate validated baked assets."""
    import math
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("Voxel size must be finite and positive")
    recipe = dict(version=1, walls=tank_walls(config), box_half=BOX_HALF_EXTENTS,
                  voxel_size=voxel_size, support_radius=4. * config.particle_radius)
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:20]
    return Path(CACHE_DIR / "tank-box" if cache_dir is None else cache_dir) / key


def _stage_asset(mesh_path, sdf_path, stem, geometry_recipe, device, *, solid=None,
                 hulls=(), static_boxes=None, source_paths=(), build=None):
    """Stage an entire set before any consumer file is replaced."""
    import numpy as np
    import warp as wp
    from .rigid_assets import _validate_hulls

    stem.parent.mkdir(parents=True, exist_ok=True)
    mesh_target, sdf_target = stem.with_suffix(".obj"), stem.with_suffix(".nvdb")
    shutil.copy2(mesh_path, mesh_target)
    shutil.copy2(sdf_path, sdf_target)
    with sdf_target.open("rb") as stream:
        volume = wp.Volume.load_from_nvdb(stream, device=device)
    metadata = dict(version=ASSET_VERSION, geometry=geometry_recipe,
        mesh_sha256=file_digest(mesh_target), sdf_sha256=sdf_digest(volume),
        local_frame="asset", sources={Path(p).name: file_digest(p) for p in source_paths},
        build=build or {})
    if solid is not None:
        metadata["mass"] = dict(volume=float(solid.volume), com=solid.center_mass.tolist(),
                                inertia_unit_density=solid.moment_inertia.tolist())
    if static_boxes is not None:
        metadata["static_boxes"] = static_boxes
    paths = [mesh_target, sdf_target]
    if hulls:
        _validate_hulls(hulls)
        arrays = dict(count=np.asarray(len(hulls), dtype=np.int32))
        for i, hull in enumerate(hulls):
            arrays[f"vertices_{i}"] = np.asarray(hull.vertices, dtype=np.float64)
            arrays[f"faces_{i}"] = np.asarray(hull.faces, dtype=np.int32)
        path = stem.with_suffix(".collision.npz")
        np.savez_compressed(path, **arrays)
        metadata["collision_sha256"] = file_digest(path)
        paths.append(path)
    path = stem.with_suffix(".json")
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    paths.append(path)
    return paths


def _prepare_stairs(config, root, stage, device, cache_dir, voxel_size):
    import numpy as np
    import trimesh
    from . import geometry, validation
    from .rigid_assets import prepare_boundary, prepare_hulls, COACD_OPTIONS, COACD_VERSION

    cache = Path(CACHE_DIR / "stairs-rigids" if cache_dir is None else cache_dir)
    options = asset_config.GeometryOptions()
    bunny_source = asset_config.ASSETS / "bunny_flat_bottom.obj"
    spot_source = asset_config.ASSETS / "spot_triangulated_good.obj"
    scene = geometry.prepare_scene(bunny_source, options=options)
    boxes = geometry.terrain_boxes(options)
    walls, _ = geometry.container_parts(scene.terrain, options)
    static_boxes = [(m.bounds.mean(axis=0).tolist(), (m.extents / 2.).tolist()) for m in [*boxes, *walls]]
    # Keep the archived full-domain SDF exactly; a clean install can explicitly
    # build it with the existing standalone container tool first.
    static_mesh, static_sdf = root / "stairs_scene.obj", root / "stairs_scene.nvdb"
    if not static_mesh.is_file() or not static_sdf.is_file():
        raise FileNotFoundError("Prepare stairs_scene.obj/.nvdb with the reference synchronization "
                                "tool or python -m demos.PBF_Rigid.sim_ready before publishing this scene")
    visible = geometry.landscape(scene, options)
    stored = trimesh.load(static_mesh, force="mesh", process=False)
    if (not np.array_equal(np.round(visible.vertices, 8), stored.vertices)
            or not np.array_equal(visible.faces, stored.faces)):
        raise ValueError("Reference stairs mesh differs from the collision geometry recipe")
    decomposition = dict(backend=COACD_VERSION, options=COACD_OPTIONS)
    bunny_hulls = prepare_hulls(scene.model, "bunny", cache)
    published = _stage_asset(static_mesh, static_sdf, stage / "stairs_scene",
        asset_geometry("stairs_scene", config), device, hulls=bunny_hulls, static_boxes=static_boxes,
        source_paths=(bunny_source,), build=dict(geometry=asdict(options), decomposition=decomposition))

    source = geometry.prepare_rigid(spot_source)
    source.apply_scale(SPOT_LENGTH / source.extents.max())
    spot = geometry.union(validation.validate_rigid_parts(source))
    validation.validate_solid(spot)
    spot_hulls = prepare_hulls(spot, "spot", cache)
    meshes = dict(box=trimesh.creation.box(extents=(BOX_SIZE,) * 3),
                  sphere=trimesh.creation.icosphere(subdivisions=3, radius=SPHERE_RADIUS), spot=spot)
    for name, mesh in meshes.items():
        asset = prepare_boundary(mesh, name, cache, support_radius=max(.05, 4. * config.particle_radius),
                                  voxel_size=voxel_size, device=device)
        published += _stage_asset(asset.mesh_path, asset.sdf_path, stage / "Rigid" / f"stairs_{name}",
            asset_geometry(f"stairs_{name}", config), device, solid=spot if name == "spot" else None,
            hulls=spot_hulls if name == "spot" else (), source_paths=(spot_source,) if name == "spot" else (),
            build=dict(voxel_size=voxel_size, max_support_radius=max(.05, 4. * config.particle_radius),
                       decomposition=decomposition if name == "spot" else None))
    return published


def _prepare_tank(config, stage, device, cache_dir, voxel_size):
    import numpy as np
    import trimesh
    from . import geometry, sdf, validation
    from .config import BakeOptions

    directory = asset_cache_path(config, cache_dir=cache_dir, voxel_size=voxel_size)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "Rigid").mkdir(exist_ok=True)
    options = BakeOptions(voxel_size=voxel_size, max_support_radius=4. * config.particle_radius)
    published = []
    for old_name, name, relative in (("tank", "tank_scene", "tank_scene"),
                                    ("Rigid/box", "tank_box", "Rigid/tank_box")):
        mesh = (geometry.union([_box_mesh(c, h) for c, h in tank_walls(config)])
                if name == "tank_scene" else _box_mesh((0., 0., 0.), BOX_HALF_EXTENTS))
        checked = validation.validate_solid(mesh)
        mesh_path, sdf_path = directory / f"{old_name}.obj", directory / f"{old_name}.nvdb"
        if not mesh_path.is_file() or not sdf_path.is_file():
            volume = sdf.bake_rigid(checked, options=options, device=device)
            pipeline.save_mesh_obj(mesh, mesh_path, digits=17)
            sdf.save_volume(volume, sdf_path)
        else:
            cached = trimesh.load(mesh_path, force="mesh", process=False)
            if (cached.vertices.shape != mesh.vertices.shape or cached.faces.shape != mesh.faces.shape
                    or not np.allclose(cached.vertices, mesh.vertices, rtol=0., atol=1e-15)
                    or not np.array_equal(cached.faces, mesh.faces)):
                raise ValueError(f"Cached tank geometry differs from its recipe: {mesh_path}")
        published += _stage_asset(mesh_path, sdf_path, stage / relative,
            asset_geometry(name, config), device, build=asdict(options))
    return published


def build_scene_assets(scene, config=None, *, device="cuda:0", asset_dir=None,
                       cache_dir=None, voxel_size=None):
    """Prepare all files, then publish the set with rollback on handled failures."""
    import warp as wp
    from demos.PBF_Rigid.solver.boundary_model import load_boundary_volumes

    if scene not in ("static", "tank-box", "stairs-rigids"):
        raise ValueError(f"Unknown PBF scene: {scene}")
    config = config or (tank_box_config() if scene == "tank-box" else PBFConfig())
    root = Path(asset_config.PBF_ASSETS if asset_dir is None else asset_dir)
    output = asset_config.OUTPUT / "pbf_assets" / scene
    output.mkdir(parents=True, exist_ok=True)
    with wp.ScopedDevice(device), tempfile.TemporaryDirectory(prefix="publish-", dir=output) as directory:
        stage = Path(directory)
        if scene == "stairs-rigids":
            paths = _prepare_stairs(config, root, stage, device, cache_dir,
                                    .005 if voxel_size is None else voxel_size)
        elif scene == "tank-box":
            paths = _prepare_tank(config, stage, device, cache_dir,
                                  .0075 if voxel_size is None else voxel_size)
        else:
            mesh, sdf = root / "stairs_scene.obj", root / "stairs_scene.nvdb"
            if not mesh.is_file() or not sdf.is_file():
                raise FileNotFoundError("Prepare the reference stairs OBJ/SDF before publishing static assets")
            paths = _stage_asset(mesh, sdf, stage / "stairs_scene", asset_geometry("stairs_scene", config), device)
            # Preserve richer static collision metadata if already published.
            metadata_path = root / "stairs_scene.json"
            if metadata_path.is_file():
                old = json.loads(metadata_path.read_text(encoding="utf-8"))
                fresh = json.loads((stage / "stairs_scene.json").read_text(encoding="utf-8"))
                if all(old.get(key) == fresh[key] for key in ("version", "geometry", "mesh_sha256", "sdf_sha256")):
                    collision = root / "stairs_scene.collision.npz"
                    if old.get("collision_sha256") and file_digest(collision) == old["collision_sha256"]:
                        shutil.copy2(collision, stage / collision.name)
                        paths.append(stage / collision.name)
                        (stage / "stairs_scene.json").write_text(json.dumps(old, indent=2) + "\n", encoding="utf-8")
        # Prime default volume maps offline, without changing the stored SDF identity.
        for path in paths:
            if path.suffix == ".nvdb":
                load_boundary_volumes(path, 4. * config.particle_radius, device=device)
        # One transaction for the entire scene, including assets in Rigid/.
        destinations = [(path, root / path.relative_to(stage)) for path in paths]
        destinations += [(path, output / path.relative_to(stage)) for path in paths]
        pipeline.publish_files(destinations)
    result = [destination for _, destination in destinations if destination.is_relative_to(root)]
    print(f"Published {scene}: {root}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("static", "tank-box", "stairs-rigids"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--particle-radius", type=float)
    parser.add_argument("--container-size", type=float, nargs=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.config is not None and (args.particle_radius is not None or args.container_size is not None):
        parser.error("--config cannot be combined with PBF configuration arguments")
    try:
        if args.config is not None:
            args.config = config_io.load_config(args.config)
        else:
            config = tank_box_config() if args.scene == "tank-box" else PBFConfig()
            changes = {}
            if args.particle_radius is not None:
                changes["particle_radius"] = args.particle_radius
            if args.container_size is not None:
                changes["container_size"] = tuple(args.container_size)
            args.config = replace(config, **changes)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        build_scene_assets(args.scene, args.config, device=args.device)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Cannot publish PBF assets: {error}") from error


if __name__ == "__main__":
    main()
