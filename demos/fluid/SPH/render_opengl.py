"""Executable OpenGL frontend for the Warp SPH simulation."""

import argparse
import sys
from pathlib import Path

import warp as wp

try:
    from demos.fluid.common.billboard_renderer import (
        HIGH_SPEED_COLOR,
        LOW_SPEED_COLOR,
        MID_SPEED_COLOR,
        BILLBOARD_FRAGMENT_SHADER,
        BILLBOARD_VERTEX_SHADER,
        BillboardRenderer,
        configure_nvidia_prime_render_offload,
        create_billboard_renderer,
        update_billboard_transforms,
    )
except ModuleNotFoundError:
    # Support ``python demos/fluid/SPH/render_opengl.py`` from the repository
    # root, where Python initially puts only the SPH directory on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from demos.fluid.common.billboard_renderer import (
        HIGH_SPEED_COLOR,
        LOW_SPEED_COLOR,
        MID_SPEED_COLOR,
        BILLBOARD_FRAGMENT_SHADER,
        BILLBOARD_VERTEX_SHADER,
        BillboardRenderer,
        configure_nvidia_prime_render_offload,
        create_billboard_renderer,
        update_billboard_transforms,
    )


__all__ = [
    "HIGH_SPEED_COLOR",
    "BILLBOARD_FRAGMENT_SHADER",
    "BILLBOARD_VERTEX_SHADER",
    "LOW_SPEED_COLOR",
    "MID_SPEED_COLOR",
    "BillboardRenderer",
    "configure_nvidia_prime_render_offload",
    "create_sph_renderer",
    "update_billboard_transforms",
]

try:
    from .simulation import create_sph_simulation
except ImportError:
    from simulation import create_sph_simulation


def create_sph_renderer(device=None):
    """Create the SPH billboard renderer with the established SPH settings."""
    return create_billboard_renderer(device=device, title="Warp SPH")


def nonnegative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point value for argparse."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--num-frames",
        type=nonnegative_int,
        default=0,
        help="Maximum rendered frames; zero runs until the window is closed.",
    )
    parser.add_argument(
        "--speed-color-mid",
        type=positive_float,
        default=2.25,
        help="Particle speed mapped to the pale-blue middle of the color map.",
    )
    parser.add_argument(
        "--speed-color-max",
        type=positive_float,
        default=3.5,
        help="Particle speed mapped to the white end of the billboard color gradient.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print additional per-kernel timing information.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.speed_color_max <= args.speed_color_mid:
        raise SystemExit("--speed-color-max must be greater than --speed-color-mid")

    with wp.ScopedDevice(args.device):
        simulation = create_sph_simulation(verbose=args.verbose)
        renderer = create_sph_renderer(device=wp.get_device())
        billboard_radius = simulation.smoothing_length
        frame = 0

        try:
            while renderer.is_running() and (args.num_frames == 0 or frame < args.num_frames):
                # Render first so initialized particles are visible before the
                # potentially expensive simulation step.
                renderer.begin_frame(simulation.sim_time)
                renderer.render_billboards(
                    name="points",
                    points=simulation.x,
                    velocities=simulation.v,
                    radius=billboard_radius,
                    speed_color_mid=args.speed_color_mid,
                    speed_color_max=args.speed_color_max,
                )
                renderer.end_frame()

                simulation.step()
                frame += 1
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
