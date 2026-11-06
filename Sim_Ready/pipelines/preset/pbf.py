"""Build and publish the current PBF scenes, independently of runtime assembly."""

import argparse
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

from ...asset_io import file_digest, sdf_digest
from ...config import GeometryOptions, BakeOptions
from ...publish import publish_files


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
SOURCE = ROOT / "Assets" / "bunny_flat_bottom.obj"
OUTPUT = PROJECT / "PBF" / "Assets"
CACHE = PROJECT / ".cache" / "sim-ready" / "pbf"
BUILD_VERSION = 1


def asset_names(scene):
    if scene not in ("static", "stairs-rigids"):
        raise ValueError(f"Unknown PBF scene: {scene}")
    return ("stairs_scene",) if scene == "static" else (
        "stairs_scene", "Rigid/stairs_box", "Rigid/stairs_sphere")


def build_recipe(scene, config):
    from PBF.scene_data import BOX_SIZE, SPHERE_RADIUS, SPHERE_SUBDIVISIONS
    from ...collision import COACD_OPTIONS, COACD_VERSION

    support = max(.05, 4. * config.particle_radius)
    geometry = {"stairs_scene": dict(kind="stairs", version=1),
                "Rigid/stairs_box": dict(kind="box", size=[BOX_SIZE] * 3),
                "Rigid/stairs_sphere": dict(kind="sphere", radius=SPHERE_RADIUS,
                                            subdivisions=SPHERE_SUBDIVISIONS)}
    recipe = dict(version=BUILD_VERSION, source=file_digest(SOURCE),
        geometry={name: geometry[name] for name in asset_names(scene)},
        terrain=asdict(GeometryOptions()),
        static_bake=asdict(BakeOptions(max_support_radius=support)),
        rigid_bake=asdict(BakeOptions(voxel_size=.005, max_support_radius=support)),
        decomposition=dict(backend=COACD_VERSION, options=COACD_OPTIONS),
        dependencies={name: version(name) for name in
                      ("newton", "warp-lang", "trimesh", "manifold3d", "coacd")})
    return json.loads(json.dumps(recipe))


def prepare_geometry(recipe):
    import numpy as np
    from .. import container, rigid

    scene, visible, static_boxes = container.prepare(SOURCE, options=GeometryOptions(**recipe["terrain"]))
    meshes = {"stairs_scene": visible}
    for name, description in recipe["geometry"].items():
        if name != "stairs_scene":
            meshes[name] = rigid.prepare(description)
    if np.asarray(static_boxes).shape != (24, 2, 3):
        raise ValueError("The current stairs recipe requires 24 collision boxes")
    return scene, meshes, static_boxes


def validate_asset(root, name, description, device):
    """Offline content checks; the mutable VM is excluded from SDF identity."""
    import warp as wp
    import numpy as np
    import trimesh
    from ...collision import _validate_hulls
    from PBF.boundary_model import _read_boundary_volumes, _check_volume_map_layout

    stem = root / name
    metadata = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if (metadata["version"] != 1 or metadata["geometry"] != description
            or metadata["local_frame"] != "asset"):
        raise ValueError(f"Asset recipe/frame differs: {stem}")
    if file_digest(stem.with_suffix(".obj")) != metadata["mesh_sha256"]:
        raise ValueError(f"Mesh digest differs: {stem}")
    mesh = trimesh.load(stem.with_suffix(".obj"), force="mesh", process=False)
    if not np.isfinite(mesh.vertices).all() or mesh.is_empty:
        raise ValueError(f"Invalid published mesh: {stem}")
    sdf, vm = _read_boundary_volumes(stem.with_suffix(".nvdb"), wp.get_device(device))
    if sdf_digest(sdf) != metadata["sdf_sha256"]:
        raise ValueError(f"SDF digest differs: {stem}")
    if vm is not None:
        _check_volume_map_layout(sdf, vm)
        if not np.array_equal(sdf.get_tiles().numpy(), vm.get_tiles().numpy()):
            raise ValueError(f"VM tile layout differs: {stem}")
    if name == "stairs_scene":
        path = stem.with_suffix(".collision.npz")
        if file_digest(path) != metadata["collision_sha256"]:
            raise ValueError(f"Collision digest differs: {stem}")
        with np.load(path, allow_pickle=False) as data:
            hulls = [trimesh.Trimesh(data[f"vertices_{i}"], data[f"faces_{i}"], process=False)
                     for i in range(int(data["count"]))]
        _validate_hulls(hulls)
        boxes = np.asarray(metadata["static_boxes"])
        if boxes.shape != (24, 2, 3) or not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= 0):
            raise ValueError(f"Invalid static collision boxes: {stem}")
    return metadata, mesh, sdf


def asset_files(root, names):
    return [root / (name + suffix) for name in names
            for suffix in ((".obj", ".nvdb", ".json", ".collision.npz")
                           if name == "stairs_scene" else (".obj", ".nvdb", ".json"))]


def copy_assets(source, destination, names):
    for path in asset_files(source, names):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def reuse_published(root, stage, recipe, meshes, static_boxes, device):
    """Adopt a complete verified set; no source code or output from demos is used."""
    import numpy as np
    from PBF.boundary_model import _check_sdf_support

    names = tuple(recipe["geometry"])
    if not all(p.is_file() for p in asset_files(root, names)):
        return False
    for name, description in recipe["geometry"].items():
        metadata, stored, sdf = validate_asset(root, name, description, device)
        generated = meshes[name]
        precision = 8 if name == "stairs_scene" else 17
        if (generated.vertices.shape != stored.vertices.shape
                or not np.allclose(np.round(generated.vertices, precision), stored.vertices,
                                   rtol=0., atol=0. if name == "stairs_scene" else 1e-15)
                or not np.array_equal(generated.faces, stored.faces)):
            return False
        if metadata["build"].get("dependencies", recipe["dependencies"]) != recipe["dependencies"]:
            return False
        if metadata["build"].get("producer_version", 1) != BUILD_VERSION:
            return False
        bake = recipe["static_bake" if name == "stairs_scene" else "rigid_bake"]
        if not np.allclose(sdf.get_voxel_size(), bake["voxel_size"], rtol=1e-6, atol=0.):
            return False
        if name == "stairs_scene":
            if (metadata["sources"] != {SOURCE.name: recipe["source"]}
                    or metadata["build"]["geometry"] != recipe["terrain"]
                    or metadata["build"]["decomposition"] != recipe["decomposition"]
                    or not np.array_equal(metadata["static_boxes"], static_boxes)):
                return False
        elif metadata["build"].get("voxel_size") != bake["voxel_size"]:
            return False
        try:
            _check_sdf_support(sdf, recipe["static_bake"]["max_support_radius"])
        except ValueError:
            return False
    copy_assets(root, stage, names)
    print(f"Reusing verified published assets: {root}", flush=True)
    return True


def generate_assets(stage, recipe, scene, meshes, static_boxes, device):
    import numpy as np
    import warp as wp
    from .. import container, rigid

    products = {"stairs_scene": container.build(scene, meshes["stairs_scene"], stage / "stairs_scene",
        geometry_options=GeometryOptions(**recipe["terrain"]), bake_options=BakeOptions(**recipe["static_bake"]),
        collision_cache=stage / "collision-cache", collision_name="bunny", device=device)}
    for name, mesh in meshes.items():
        if name == "stairs_scene":
            continue
        products[name] = rigid.build(mesh, stage / name, bake_options=BakeOptions(**recipe["rigid_bake"]),
                                     device=device)
    # The preset owns the published attachment format and the scene-wide transaction.
    for name, (mesh_path, sdf_path, hulls) in products.items():
        stem = stage / name
        is_static = name == "stairs_scene"
        bake = BakeOptions(**recipe["static_bake" if is_static else "rigid_bake"])
        with sdf_path.open("rb") as stream:
            volume = wp.Volume.load_from_nvdb(stream, device=device)
        metadata = dict(version=1, geometry=recipe["geometry"][name], local_frame="asset",
            mesh_sha256=file_digest(mesh_path), sdf_sha256=sdf_digest(volume),
            sources={SOURCE.name: recipe["source"]} if is_static else {},
            build=dict(geometry=recipe["terrain"], decomposition=recipe["decomposition"],
                       bake=asdict(bake), dependencies=recipe["dependencies"]) if is_static else
                  dict(asdict(bake), decomposition=None, dependencies=recipe["dependencies"]))
        if is_static:
            arrays = dict(count=np.asarray(len(hulls), dtype=np.int32))
            for i, hull in enumerate(hulls):
                arrays[f"vertices_{i}"] = np.asarray(hull.vertices, dtype=np.float64)
                arrays[f"faces_{i}"] = np.asarray(hull.faces, dtype=np.int32)
            np.savez_compressed(stem.with_suffix(".collision.npz"), **arrays)
            metadata.update(static_boxes=static_boxes,
                            collision_sha256=file_digest(stem.with_suffix(".collision.npz")))
        metadata["build"]["producer_version"] = BUILD_VERSION
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def build_scene_assets(scene, config=None, *, device="cuda:0", output_dir=None,
                       cache_dir=None, rebuild=False):
    """Prepare and validate all attachments before replacing any published file."""
    import warp as wp
    from PBF.pbf_helper import PBFConfig
    from PBF.boundary_model import load_boundary_volumes
    from ...collision import cache_directory

    config = config or PBFConfig()
    recipe = build_recipe(scene, config)
    root = Path(OUTPUT if output_dir is None else output_dir).resolve()
    cache_root = Path(CACHE if cache_dir is None else cache_dir).resolve()
    for target in (root, cache_root):
        if target.is_relative_to(PROJECT / "demos") or target.is_relative_to(ROOT / "Assets"):
            raise ValueError("Outputs must not overwrite demos, the archive or source assets")
    cache = cache_directory(cache_root, scene, recipe)
    if root == cache or root.is_relative_to(cache) or cache.is_relative_to(root):
        raise ValueError("Published assets and the build cache must be separate directories")
    names = tuple(recipe["geometry"])
    with wp.ScopedDevice(device), tempfile.TemporaryDirectory(prefix="build-", dir=cache.parent) as temporary:
        stage = Path(temporary)
        if not rebuild and (cache / "recipe.json").is_file():
            if json.loads((cache / "recipe.json").read_text(encoding="utf-8")) != recipe:
                raise ValueError("Cached recipe differs; use --rebuild")
            for name, description in recipe["geometry"].items():
                validate_asset(cache, name, description, device)
            copy_assets(cache, stage, names)
            print(f"Build cache hit: {cache}", flush=True)
        else:
            prepared, meshes, boxes = prepare_geometry(recipe)
            reused = False
            if not rebuild:
                for candidate in dict.fromkeys((root, OUTPUT)):
                    if reuse_published(candidate, stage, recipe, meshes, boxes, device):
                        reused = True
                        break
            if not reused:
                generate_assets(stage, recipe, prepared, meshes, boxes, device)
        for name, description in recipe["geometry"].items():
            # VM precomputation is the same implementation as startup's cache repair.
            load_boundary_volumes((stage / name).with_suffix(".nvdb"),
                                  4. * config.particle_radius, device=device)
            validate_asset(stage, name, description, device)
        recipe_path = stage / "recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        files = asset_files(stage, names)
        targets = [(path, cache / path.relative_to(stage)) for path in files]
        targets += [(recipe_path, cache / "recipe.json")]
        targets += [(path, root / path.relative_to(stage)) for path in files]
        # Retain new decomposition diagnostics in the offline cache only.
        for path in (stage / "collision-cache").rglob("*.json"):
            targets.append((path, cache / path.relative_to(stage)))
        publish_files(targets)
    print(f"Published {scene}: {root}", flush=True)
    return asset_files(root, names)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("static", "stairs-rigids"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--particle-radius", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--rebuild", action="store_true", help="Regenerate geometry, SDF and collision, ignoring reuse")
    args = parser.parse_args(argv)
    if args.config is not None and args.particle_radius is not None:
        parser.error("--config cannot be combined with --particle-radius")
    from PBF.pbf_helper import PBFConfig
    from PBF.config_io import load_config
    try:
        config = PBFConfig() if args.config is None else load_config(args.config)
        args.config = config if args.particle_radius is None else replace(config, particle_radius=args.particle_radius)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    del args.particle_radius
    return args


def main(argv=None):
    args = parse_args(argv)
    build_scene_assets(**vars(args))


if __name__ == "__main__":
    main()
