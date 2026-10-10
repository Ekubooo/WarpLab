"""Executable OpenGL billboard frontend for the Warp PBF simulation."""

import argparse
import math
import sys
import time
from pathlib import Path

import warp as wp

try:
    from demos.fluid.common.billboard_renderer import (
        BillboardRenderer, configure_nvidia_prime_render_offload,
    )
except ModuleNotFoundError:
    # Support direct execution from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from demos.fluid.common.billboard_renderer import (
        BillboardRenderer, configure_nvidia_prime_render_offload,
    )

try:
    from .pbf_helper import PBFConfig
    from .simulation import create_pbf_simulation
except ImportError:
    from pbf_helper import PBFConfig
    from simulation import create_pbf_simulation


class PBFRenderer(BillboardRenderer):
    """Let the PBF main loop handle pause and deferred reset requests."""

    def begin_frame(self, t=None):
        super().begin_frame(t)
        # Warp treats zero as an omitted time; reset must display simulation time zero.
        if t is not None:
            self.time = t

    def end_frame(self):
        self._last_end_frame_time = time.time()
        if self._add_shape_instances:
            self.allocate_shape_instances()
        if self._update_shape_instances:
            self.update_shape_instances()
        self.update()


def create_pbf_renderer(
    device=None, title="Warp PBF", *, scaling=0.05,
    camera_pos=(2.0, 3.0, 10.0), camera_front=(-0.1, -0.1, -1.0),
):
    """Keep the shared frontend's view settings with PBF-specific pause handling."""
    configure_nvidia_prime_render_offload()
    return PBFRenderer(
        title=title, scaling=scaling, fps=120, up_axis="Y",
        screen_width=1280, screen_height=720, near_plane=0.1, far_plane=100.0,
        camera_pos=camera_pos, camera_front=camera_front, camera_up=(0.0, 1.0, 0.0),
        background_color=(0.0, 0.0, 0.0), vsync=True, device=device,
    )


def register_keyboard_controls(renderer, *, on_reset, on_reverse_gravity):
    """Consume PBF shortcuts before Warp applies its default G binding."""
    import pyglet

    def on_key_press(symbol, _modifiers):
        if symbol == pyglet.window.key.G:
            on_reverse_gravity()
        elif symbol == pyglet.window.key.R:
            renderer.paused = True
            on_reset()
        elif symbol == pyglet.window.key.P:
            renderer.draw_grid = not renderer.draw_grid
        else:
            return None
        return pyglet.event.EVENT_HANDLED

    renderer.register_key_press_callback(on_key_press)
    return on_key_press


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


def config_from_args(args):
    """Use config defaults for every field that was not explicitly overridden."""
    overrides = {}
    for name in (
        "particle_radius", "rest_density", "frame_dt", "lambda_regularization",
        "substeps", "pressure_iterations", "container_size", "block_start", "block_end",
        "clamp_negative_pressure",
    ):
        if hasattr(args, name):
            value = getattr(args, name)
            if name in ("container_size", "block_start", "block_end"):
                value = tuple(value)
            overrides[name] = value
    return PBFConfig(**overrides) if overrides else None


def parse_args(argv=None):
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
        default=2.0,
        help="Particle speed mapped to the pale-blue middle of the color map.",
    )
    parser.add_argument(
        "--speed-color-max",
        type=positive_float,
        default=PBFConfig.max_speed,
        help="Particle speed mapped to the white end of the billboard color gradient.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print additional per-kernel timing information.")

    config_options = parser.add_argument_group("PBF configuration")
    config_options.add_argument(
        "--clamp-negative-pressure", action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help=f"Clamp negative density constraints to zero (default: {PBFConfig.clamp_negative_pressure}).",
    )
    config_options.add_argument(
        "--particle-radius", type=float, default=argparse.SUPPRESS,
        help=f"Particle radius in meters (default: {PBFConfig.particle_radius}).",
    )
    config_options.add_argument(
        "--rest-density", type=float, default=argparse.SUPPRESS,
        help=(
            "Rest density for particle mass; cancels in normalized equal-mass fluid constraints "
            f"(default: {PBFConfig.rest_density})."
        ),
    )
    config_options.add_argument(
        "--frame-dt", type=float, default=argparse.SUPPRESS,
        help=f"Physical time per frame in seconds (default: {PBFConfig.frame_dt}).",
    )
    config_options.add_argument(
        "--lambda-regularization", type=float, default=argparse.SUPPRESS,
        help=(
            "Dimensionless solver regularization alpha; epsilon = alpha / h^2 "
            f"(default: {PBFConfig.lambda_regularization})."
        ),
    )
    config_options.add_argument(
        "--substeps", type=int, default=argparse.SUPPRESS,
        help=f"Physical substeps per frame (default: {PBFConfig.substeps}).",
    )
    config_options.add_argument(
        "--pressure-iterations", type=int, default=argparse.SUPPRESS,
        help=f"Pressure iterations per substep (default: {PBFConfig.pressure_iterations}).",
    )
    config_options.add_argument(
        "--container-size", type=float, nargs=3, metavar=("X", "Y", "Z"),
        default=argparse.SUPPRESS,
        help=f"Container dimensions in meters (default: {PBFConfig.container_size}).",
    )
    config_options.add_argument(
        "--block-start", type=float, nargs=3, metavar=("X", "Y", "Z"),
        default=argparse.SUPPRESS,
        help=f"Fluid block minimum coordinates (default: {PBFConfig.block_start}).",
    )
    config_options.add_argument(
        "--block-end", type=float, nargs=3, metavar=("X", "Y", "Z"),
        default=argparse.SUPPRESS,
        help=f"Fluid block maximum coordinates (default: {PBFConfig.block_end}).",
    )

    args = parser.parse_args(argv)
    try:
        args.config = config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()

    if args.speed_color_max <= args.speed_color_mid:
        raise SystemExit("--speed-color-max must be greater than --speed-color-mid")

    with wp.ScopedDevice(args.device):
        simulation = create_pbf_simulation(verbose=args.verbose, config=args.config)
        # Focus on the fluid near the floor; distance follows the container footprint.
        camera_azimuth_deg = 35.0
        camera_elevation_deg = 25.0
        camera_distance_scale = 1.75
        camera_target = wp.vec3(0.5 * simulation.width, 0.25 * simulation.width, 0.25 * simulation.length)
        # camera_target = wp.vec3(0.0, 0.0, 0.0)
        azimuth = math.radians(camera_azimuth_deg)
        elevation = math.radians(camera_elevation_deg)
        camera_offset = wp.vec3(
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.cos(azimuth),
        )
        camera_distance = camera_distance_scale * max(simulation.width, simulation.length)
        camera_pos = camera_target + camera_distance * camera_offset
        renderer = create_pbf_renderer(
            device=wp.get_device(),
            title="Warp PBF",
            scaling=1.0,
            camera_pos=tuple(camera_pos),
            camera_front=tuple(-camera_offset),
        )
        billboard_radius = simulation.particle_radius
        frame = 0
        reset_requested = False

        def request_reset():
            nonlocal reset_requested
            reset_requested = True

        register_keyboard_controls(
            renderer, on_reset=request_reset,
            on_reverse_gravity=simulation.reverse_gravity,
        )

        try:
            while renderer.is_running() and (args.num_frames == 0 or frame < args.num_frames):
                if reset_requested:
                    simulation.reset()
                    reset_requested = False
                    renderer.paused = True
                # Render first so the initialized PBF particles are visible.
                renderer.begin_frame(simulation.sim_time)
                renderer.render_billboards(
                    name="points",
                    points=simulation.pos,
                    velocities=simulation.v,
                    radius=billboard_radius,
                    speed_color_mid=args.speed_color_mid,
                    speed_color_max=args.speed_color_max,
                )
                renderer.end_frame()

                if not renderer.paused and not reset_requested and renderer.is_running():
                    simulation.step()
                frame += 1
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
