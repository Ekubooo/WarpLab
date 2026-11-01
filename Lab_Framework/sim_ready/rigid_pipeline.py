"""Build an independent rigid OBJ and a single-grid SDF NanoVDB asset."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Lab_Framework.sim_ready import config, pipeline


@dataclass(frozen=True)
class RigidBuildResult:
    mesh: Path
    sdf: Path
    destinations: tuple[Path, ...]


def build_rigid_asset(obj_path, *, name, scale=1.0, obj_up_axis=None,
                      bake_options=config.BakeOptions(), device="cuda:0"):
    """Validate, union, bake and publish one rigid asset in its local frame.

    A new OBJ is saved before baking, as in the container pipeline. Consumers
    receive a matching pair only after successful baking. PBF owns volume-map
    generation and its cache; this pipeline writes just the SDF grid.
    """
    paths = pipeline.resolve_asset_paths(
        obj_path, name, output=config.RIGID_OUTPUT,
        destinations=config.RIGID_OUTPUT_DIRECTORIES,
    )
    # CLI help and argument validation must not import Warp or initialize CUDA.
    from Lab_Framework.sim_ready import geometry, sdf, validation

    sdf.check_options(bake_options)
    stage = "rigid geometry preparation"
    try:
        source = geometry.prepare_rigid(paths.source, scale=scale, obj_up_axis=obj_up_axis)
        stage = "rigid input validation"
        parts = validation.validate_rigid_parts(source)
        stage = "rigid boolean union"
        solid = geometry.union(parts)
        stage = "rigid solid validation"
        checked = validation.validate_solid(solid)
        stage = "rigid OBJ saving"
        # Enough digits to preserve the coordinates checked for float32 upload.
        pipeline.save_mesh_obj(solid, paths.mesh, digits=17)
        print(f"Rigid mesh: {paths.mesh} ({len(solid.faces)} triangles)", flush=True)
        stage = "rigid SDF baking"
        volume = sdf.bake_rigid(checked, options=bake_options, device=device)
        stage = "rigid SDF saving"
        sdf.save_volume(volume, paths.sdf)
        stage = "rigid asset publication"
        consumers = [p for p in paths.destinations if p != paths.mesh.parent]
        pipeline.publish_pair((paths.mesh, paths.sdf), consumers)
        for directory in paths.destinations:
            print(f"Assets: {directory / name}.obj / .nvdb", flush=True)
        return RigidBuildResult(paths.mesh, paths.sdf, paths.destinations)
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"{stage} failed: {error}") from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True, metavar="PATH",
                        help="Input OBJ; bare names resolve in sim_Ready/Assets")
    parser.add_argument("--name", required=True, type=pipeline.validate_name, metavar="NAME",
                        help="Output filename stem, without a path or extension")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Explicit positive unit/size multiplier (default: 1)")
    parser.add_argument("--obj-up-axis", choices=("Y", "Z"), default=None,
                        help="Explicit conversion to Y-up; omit to preserve local axes")
    parser.add_argument("--sdf-voxel-size", type=float, default=config.BakeOptions.voxel_size)
    parser.add_argument("--sdf-max-support-radius", type=float,
                        default=config.BakeOptions.max_support_radius)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    for name in ("scale", "sdf_voxel_size", "sdf_max_support_radius"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        return build_rigid_asset(
            args.obj, name=args.name, scale=args.scale, obj_up_axis=args.obj_up_axis,
            bake_options=config.BakeOptions(voxel_size=args.sdf_voxel_size,
                max_support_radius=args.sdf_max_support_radius), device=args.device,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Cannot build rigid asset: {error}") from error


if __name__ == "__main__":
    main()
