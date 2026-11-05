"""Build a flat-shaded landscape OBJ and matching closed-container NanoVDB."""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demos.PBF_Rigid.sim_ready import config
from demos.PBF_Rigid.sim_ready.pipeline import validate_name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", default="bunny_flat_bottom.obj", metavar="PATH",
                        help="Input OBJ; bare names resolve in sim_Ready/Assets")
    parser.add_argument("--name", default=config.DEFAULT_NAME, type=validate_name, metavar="NAME",
                        help="Output filename stem (default: stairs_scene); no path or extension")
    parser.add_argument("--obj-up-axis", choices=("Y", "Z"), default="Y")
    parser.add_argument("--sdf-voxel-size", type=float, default=config.BakeOptions.voxel_size)
    parser.add_argument("--sdf-max-support-radius", type=float,
                        default=config.BakeOptions.max_support_radius,
                        help="Maximum supported fluid smoothing radius, in meters (default: 0.05)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not 0 < args.sdf_voxel_size < float("inf"):
        parser.error("--sdf-voxel-size must be finite and positive")
    if not 0 < args.sdf_max_support_radius < float("inf"):
        parser.error("--sdf-max-support-radius must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    from demos.PBF_Rigid.sim_ready.pipeline import build_assets
    try:
        return build_assets(args.obj, name=args.name, device=args.device,
                            obj_up_axis=args.obj_up_axis,
                            bake_options=config.BakeOptions(voxel_size=args.sdf_voxel_size,
                                max_support_radius=args.sdf_max_support_radius))
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Cannot build assets: {error}") from error


if __name__ == "__main__":
    main()
