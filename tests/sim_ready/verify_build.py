"""Reproduce migration and optional clean-generation checks in private directories."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import trimesh
import warp as wp

from Sim_Ready.asset_io import file_digest, sdf_digest
from Sim_Ready.pipelines.preset import pbf


@wp.kernel(enable_backward=False)
def sample(grid: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=float)):
    i = wp.tid()
    values[i] = wp.volume_sample_f(grid, wp.volume_world_to_index(grid, points[i]), wp.Volume.LINEAR)


def compare_generated(reference, generated, scene="stairs-rigids"):
    results = {}
    rng = np.random.default_rng(0)
    for name in pbf.asset_names(scene):
        stem, fresh = reference / name, generated / name
        old_meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        new_meta, new_mesh, new_sdf = pbf.validate_asset(generated, name, old_meta["geometry"], "cuda:0")
        old_mesh = trimesh.load(stem.with_suffix(".obj"), force="mesh", process=False)
        np.testing.assert_array_equal(old_mesh.vertices, new_mesh.vertices)
        np.testing.assert_array_equal(old_mesh.faces, new_mesh.faces)
        with stem.with_suffix(".nvdb").open("rb") as stream:
            old_sdf = wp.Volume.load_from_nvdb(stream)
        # Compare in the fluid-accessible scene and around all primitive surfaces.
        bounds = np.array([[0., 0., 0.], [3.1, 4.5, 1.6]])
        if name != "stairs_scene":
            bounds = old_mesh.bounds + np.array([[-.06] * 3, [.06] * 3])
        points = wp.array(rng.uniform(bounds[0], bounds[1], (20000, 3)), dtype=wp.vec3)
        values = []
        for volume in (old_sdf, new_sdf):
            result = wp.empty(len(points), dtype=float)
            wp.launch(sample, len(points), inputs=[volume.id, points, result])
            values.append(result.numpy())
        a, b = values
        assert np.isfinite(a).all() and np.isfinite(b).all()
        band = (np.abs(a) <= .05) | (np.abs(b) <= .05)
        error = np.abs(a[band] - b[band])
        voxel = float(new_sdf.get_voxel_size()[0])
        # Rebuilt grids need not be bitwise identical. Bound error in the PBF query band.
        assert error.size and error.max() <= np.sqrt(3.) * voxel, (name, error.max())
        results[name] = dict(mesh_arrays_equal=True, sdf_identical=sdf_digest(old_sdf) == sdf_digest(new_sdf),
            samples=len(points), query_band_samples=int(band.sum()),
            query_band_max_error_m=float(error.max()), query_band_mean_error_m=float(error.mean()),
            tolerance_m=float(np.sqrt(3.) * voxel))
        if name == "stairs_scene":
            np.testing.assert_array_equal(old_meta["static_boxes"], new_meta["static_boxes"])
            with np.load(stem.with_suffix(".collision.npz")) as old, np.load(fresh.with_suffix(".collision.npz")) as new:
                results[name]["collision_arrays_equal"] = old.files == new.files and all(
                    np.array_equal(old[key], new[key]) for key in old.files)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/pbf-newton/sim-ready"))
    parser.add_argument("--rebuild", action="store_true", help="Also run expensive geometry, CoACD and SDF generation")
    parser.add_argument("--compare-rebuilt", type=Path, help="Compare an already regenerated set")
    parser.add_argument("--compare-static", type=Path, help="Compare an already regenerated static set")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(migration={}, seconds={})
    for scene in ("static", "stairs-rigids"):
        destination = args.output / f"verify-{scene}"
        for run in range(2):
            start = time.perf_counter()
            paths = pbf.build_scene_assets(scene, output_dir=destination, cache_dir=args.output / "verify-cache")
            report["seconds"][f"{scene}-{run}"] = time.perf_counter() - start
        report["migration"][scene] = {str(path.relative_to(destination)): file_digest(path) ==
            file_digest(pbf.OUTPUT / path.relative_to(destination)) for path in paths}
        assert all(report["migration"][scene].values())
    generated = args.compare_rebuilt
    generated_static = args.compare_static
    if args.rebuild:
        for scene in ("static", "stairs-rigids"):
            destination = args.output / f"clean-{scene}"
            start = time.perf_counter()
            pbf.build_scene_assets(scene, output_dir=destination,
                                   cache_dir=args.output / "clean-cache", rebuild=True)
            report["seconds"][f"clean-{scene}"] = time.perf_counter() - start
        generated_static = args.output / "clean-static"
        generated = args.output / "clean-stairs-rigids"
    if generated_static:
        report["regenerated_static"] = compare_generated(pbf.OUTPUT, generated_static, "static")
    if generated:
        report["regenerated"] = compare_generated(pbf.OUTPUT, generated)
    (args.output / "asset-verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
