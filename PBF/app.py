"""Standalone PBF application: configuration, playback and user commands."""

import argparse
import math
import sys
from pathlib import Path

import warp as wp

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PBF.pbf_helper import PBFConfig
from PBF import config_io
from PBF.simulation import Simulation, create_pbf_simulation, reset_simulation, reverse_gravity


def register_keyboard_controls(simulation, *, on_reset):
    """Bind demo commands; the renderer translates and dispatches native keys."""
    renderer = simulation.renderer
    slow_motion = False
    last_direction = None
    gravity_strength = math.hypot(*simulation.config.gravity)

    def flip_gravity():
        nonlocal last_direction
        reverse_gravity(simulation)
        last_direction = None

    def select_direction(direction):
        nonlocal last_direction
        if last_direction == direction:
            simulation.gravity = wp.vec3(0.0, -gravity_strength, 0.0)
            last_direction = None
        else:
            simulation.gravity = wp.vec3(direction * gravity_strength, 0.0, 0.0)
            last_direction = direction

    def toggle_slow_motion():
        nonlocal slow_motion
        slow_motion = not slow_motion
        simulation.frame_dt = simulation.config.frame_dt * (0.5 if slow_motion else 1.0)

    def request_reset():
        nonlocal slow_motion, last_direction
        slow_motion = False
        last_direction = None
        simulation.frame_dt = simulation.config.frame_dt
        renderer.paused = True
        on_reset()

    def toggle_grid():
        renderer.draw_grid = not renderer.draw_grid

    simulation.register_keyboard_callback("G", flip_gravity, repeat=False)
    simulation.register_keyboard_callback("Q", lambda: select_direction(-1.0), repeat=False)
    simulation.register_keyboard_callback("E", lambda: select_direction(1.0), repeat=False)
    simulation.register_keyboard_callback("X", toggle_slow_motion, repeat=False)
    simulation.register_keyboard_callback("R", request_reset)
    simulation.register_keyboard_callback("P", toggle_grid)


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
    """Apply either JSON or explicit CLI overrides to config defaults."""
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
    if args.config_path is not None:
        if overrides:
            raise ValueError("--config cannot be combined with PBF configuration arguments")
        return config_io.load_config(args.config_path)
    return PBFConfig(**overrides) if overrides else None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene", choices=("static", "stairs-rigids"), default="static",
                        help="Original static stairs or pressure-coupled free rigid bodies.")
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
    parser.add_argument("--cuda-graph", action="store_true", help="Capture GPU frames, or fluid substeps with CPU rigid bodies.")
    parser.add_argument("--verbose", action="store_true", help="Print additional per-kernel timing information.")
    parser.add_argument(
        "--config", dest="config_path", metavar="PATH",
        help="Load JSON config instead of PBF configuration arguments.",
    )
    parser.add_argument(
        "--export-config", metavar="NAME", nargs="?", const=config_io.DEFAULT_CONFIG_NAME,
        help=f"Save config to the demo's config/NAME and exit (default name: {config_io.DEFAULT_CONFIG_NAME}).",
    )

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

    if args.export_config is not None:
        try:
            print(config_io.save_config(args.config or PBFConfig(), args.export_config))
        except (OSError, ValueError) as error:
            raise SystemExit(f"Cannot export config: {error}") from error
        return

    if args.speed_color_max <= args.speed_color_mid:
        raise SystemExit("--speed-color-max must be greater than --speed-color-mid")

    from PBF.renderers.opengl import create_pbf_renderer

    with wp.ScopedDevice(args.device):
        simulation = create_pbf_simulation(
            verbose=args.verbose, config=args.config, scene=args.scene,
        )
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
            particle_positions=simulation.pos,
            speed_color_mid=args.speed_color_mid, speed_color_max=args.speed_color_max,
        )
        frame = 0
        reset_requested = False

        def request_reset():
            nonlocal reset_requested
            reset_requested = True

        try:
            simulation.set_renderer(renderer)
            register_keyboard_controls(simulation, on_reset=request_reset)
            while renderer.is_running() and (args.num_frames == 0 or frame < args.num_frames):
                if reset_requested:
                    reset_simulation(simulation)
                    reset_requested = False
                    renderer.paused = True
                # Render first so the initialized PBF particles are visible.
                simulation.render()

                if not renderer.paused and not reset_requested and renderer.is_running():
                    if args.cuda_graph:
                        simulation.launch_graph()
                    else:
                        simulation.step()
                frame += 1
        finally:
            simulation.close()


def parse_usd_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--scene", choices=("static", "stairs-rigids"), default="static",
                        help="Keep the original static scene or add three coupled rigid bodies.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_pbf.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=480, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument(
        "--config", dest="config_path", metavar="PATH",
        help="Load JSON config instead of PBF configuration arguments.",
    )
    parser.add_argument(
        "--export-config", metavar="NAME", nargs="?", const=config_io.DEFAULT_CONFIG_NAME,
        help=f"Save config to the demo's config/NAME and exit (default name: {config_io.DEFAULT_CONFIG_NAME}).",
    )
    parser.add_argument(
        "--clamp-negative-pressure",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help=f"Clamp negative density constraints to zero (default: {PBFConfig.clamp_negative_pressure}).",
    )

    args = parser.parse_args(argv)
    try:
        if args.config_path is not None:
            if hasattr(args, "clamp_negative_pressure"):
                raise ValueError("--config cannot be combined with PBF configuration arguments")
            args.config = config_io.load_config(args.config_path)
        else:
            overrides = (
                {"clamp_negative_pressure": args.clamp_negative_pressure}
                if hasattr(args, "clamp_negative_pressure") else {}
            )
            args.config = PBFConfig(**overrides)
    except ValueError as error:
        parser.error(str(error))
    return args


def main_usd():
    args = parse_usd_args()
    if args.export_config is not None:
        try:
            print(config_io.save_config(args.config, args.export_config))
        except (OSError, ValueError) as error:
            raise SystemExit(f"Cannot export config: {error}") from error
        return

    with wp.ScopedDevice(args.device):
        example = Simulation(
            stage_path=args.stage_path, verbose=args.verbose,
            config=args.config, scene=args.scene,
        )

        try:
            for _ in range(args.num_frames):
                example.render()
                example.step()
        finally:
            example.close()


if __name__ == "__main__":
    main()


