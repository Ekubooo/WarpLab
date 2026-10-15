"""Bake or load a container SDF independently of the PBF simulation."""

import argparse
from pathlib import Path
import sys


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
CONTAINER_DIR = ASSET_DIR / "Container"
BOUNDARY_DIR = ASSET_DIR / "Boundary"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-obj", required=True, metavar="PATH",
                        help="Container OBJ; bare filenames resolve in PBF/Assets/Container")
    parser.add_argument("--sdf-voxel-size", type=float, default=0.0075,
                        help="Voxel spacing in meters (default: 0.0075; used when baking)")
    parser.add_argument("--sdf-force-rebuild", action="store_true",
                        help="Validate and bake again, replacing the cache only on success")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    return parser.parse_args(argv)


def resolve_container_path(path):
    source = Path(path).expanduser()
    # Preserve an explicit directory, including ./, rather than normalizing it away.
    if not source.is_absolute() and "/" not in str(path) and "\\" not in str(path):
        source = CONTAINER_DIR / source
    return source.resolve()


def main(argv=None):
    args = parse_args(argv)
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    # Keep help and argument errors independent of Warp and simulation imports.
    from demos.fluid.boundary.sdf_baker import SDFOptions, load_or_bake_sdf

    try:
        result = load_or_bake_sdf(
            resolve_container_path(args.boundary_obj),
            cache_dir=BOUNDARY_DIR,
            options=SDFOptions(voxel_size=args.sdf_voxel_size,
                               force_rebuild=args.sdf_force_rebuild),
            device=args.device,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Cannot bake/load SDF: {error}") from error
    action = "SDF cache loaded" if result.cache_hit else "SDF baked and cached"
    print(f"{action} (voxel size {result.volume.get_voxel_size()[0]:g} m)")
    print(result.cache_path)
    return result.cache_path


if __name__ == "__main__":
    main()
