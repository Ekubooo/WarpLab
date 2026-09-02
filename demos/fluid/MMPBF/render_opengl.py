"""Executable OpenGL billboard frontend for the Warp MMPBF simulation."""

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import warp as wp

try:
    from demos.fluid.common.billboard_renderer import create_billboard_renderer
except ModuleNotFoundError:
    # Support direct execution from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from demos.fluid.common.billboard_renderer import create_billboard_renderer

try:
    from .simulation import MMPBFConfig, create_mmpbf_simulation
except ImportError:
    from simulation import MMPBFConfig, create_mmpbf_simulation


# The shared renderer was originally tuned for the old 80-unit PBF scene and
# scales all geometry by 0.05.  The official FluidDemo container is only four
# units wide, so MMPBF must render at world scale.
MMPBF_SCENE_SCALING = 1.0
MMPBF_CAMERA_POS = (2.5, 2.0, 7.0)
MMPBF_CAMERA_FRONT = (-0.332, -0.159, -0.930)
MMPBF_CAMERA_UP = (0.0, 1.0, 0.0)
MMPBF_GRAVITY_ROTATION_STEP = 10.0


class PlaybackScheduler:
    """Map wall-clock elapsed time to complete adaptive PBF substeps."""

    def __init__(
        self,
        playback_speed: float = 0.5,
        max_substeps_per_frame: int = 8,
        max_wall_delta: float = 0.1,
    ):
        if playback_speed <= 0.0:
            raise ValueError("playback_speed must be positive")
        if max_substeps_per_frame < 1:
            raise ValueError("max_substeps_per_frame must be positive")
        if max_wall_delta <= 0.0:
            raise ValueError("max_wall_delta must be positive")
        self.playback_speed = playback_speed
        self.max_substeps_per_frame = max_substeps_per_frame
        self.max_wall_delta = max_wall_delta
        self.accumulator = 0.0

    def advance(self, simulation, wall_delta: float) -> int:
        """Advance for one rendered frame and return the executed step count."""
        clamped_wall_delta = min(max(wall_delta, 0.0), self.max_wall_delta)
        self.accumulator += clamped_wall_delta * self.playback_speed

        substeps = 0
        while (
            substeps < self.max_substeps_per_frame
            and self.accumulator >= simulation.current_dt
        ):
            simulation.step()
            self.accumulator -= simulation.last_dt
            substeps += 1

        # If the frame exhausted its compute budget, discard all whole pending
        # steps but retain the fractional phase for smooth subsequent pacing.
        if (
            substeps == self.max_substeps_per_frame
            and self.accumulator >= simulation.current_dt
        ):
            self.accumulator = math.fmod(self.accumulator, simulation.current_dt)

        return substeps


def configure_mmpbf_view(renderer):
    """Frame the official 4 x 4 x 0.8 FluidDemo container."""
    renderer.scaling = MMPBF_SCENE_SCALING
    renderer.update_view_matrix(
        cam_pos=MMPBF_CAMERA_POS,
        cam_front=MMPBF_CAMERA_FRONT,
        cam_up=MMPBF_CAMERA_UP,
    )


def reverse_gravity(simulation):
    """Reverse the current effective gravity without mutating its frozen config."""
    gravity = simulation.config.gravity
    simulation.config = replace(
        simulation.config,
        gravity=tuple(-component for component in gravity),
    )


def rotate_gravity(simulation, angle_degrees):
    """Rotate effective gravity around the positive Z axis."""
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    gravity_x, gravity_y, gravity_z = simulation.config.gravity
    simulation.config = replace(
        simulation.config,
        gravity=(
            cosine * gravity_x - sine * gravity_y,
            sine * gravity_x + cosine * gravity_y,
            gravity_z,
        ),
    )


def register_keyboard_controls(
    renderer,
    *,
    on_reset,
    on_reverse_gravity,
    on_rotate_gravity,
    extra_actions=None,
):
    """Register MMPBF controls plus optional key-symbol callbacks."""
    import pyglet

    def reset():
        # Release end_frame() when R is pressed while already paused.
        renderer.paused = False
        on_reset()

    actions = {
        pyglet.window.key.SPACE: lambda: setattr(renderer, "paused", not renderer.paused),
        pyglet.window.key.R: reset,
        pyglet.window.key.G: on_reverse_gravity,
        pyglet.window.key.Q: lambda: on_rotate_gravity(MMPBF_GRAVITY_ROTATION_STEP),
        pyglet.window.key.E: lambda: on_rotate_gravity(-MMPBF_GRAVITY_ROTATION_STEP),
    }
    if extra_actions:
        actions.update(extra_actions)

    def on_key_press(symbol, _modifiers):
        action = actions.get(symbol)
        if action is None:
            return None
        action()
        return pyglet.event.EVENT_HANDLED

    renderer.register_key_press_callback(on_key_press)
    return on_key_press


def create_rendered_simulation(*, config, verbose, device, renderer, speed_color_mid, speed_color_max):
    """Create a simulation and attach the existing OpenGL frontend state."""
    simulation = create_mmpbf_simulation(
        verbose=verbose,
        config=config,
        device=device,
    )
    simulation.renderer = renderer
    simulation.speed_color_mid = speed_color_mid
    simulation.speed_color_max = speed_color_max
    return simulation


def nonnegative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point value for argparse."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


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
        "--preset",
        choices=("high-resolution", "official"),
        default="high-resolution",
        help="Particle-resolution preset for the unchanged official scene.",
    )
    parser.add_argument(
        "--playback-speed",
        type=positive_float,
        default=0.5,
        help="Simulated seconds advanced per wall-clock second.",
    )
    parser.add_argument(
        "--max-substeps-per-frame",
        type=positive_int,
        default=8,
        help="Maximum complete physics substeps computed for one rendered frame.",
    )
    parser.add_argument(
        "--speed-color-mid",
        type=positive_float,
        default=2,
        help="Particle speed mapped to the pale-blue middle of the color map.",
    )
    parser.add_argument(
        "--speed-color-max",
        type=positive_float,
        default=6,
        help="Particle speed mapped to the white end of the billboard color gradient.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print additional per-kernel timing information.")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    if args.speed_color_max <= args.speed_color_mid:
        raise SystemExit("--speed-color-max must be greater than --speed-color-mid")

    with wp.ScopedDevice(args.device):
        device = wp.get_device()
        config = (
            MMPBFConfig.high_resolution()
            if args.preset == "high-resolution"
            else MMPBFConfig()
        )
        renderer = create_billboard_renderer(device=device, title="Warp MMPBF")
        configure_mmpbf_view(renderer)
        simulation = create_rendered_simulation(
            config=config,
            verbose=args.verbose,
            device=device,
            renderer=renderer,
            speed_color_mid=args.speed_color_mid,
            speed_color_max=args.speed_color_max,
        )
        scheduler = PlaybackScheduler(
            playback_speed=args.playback_speed,
            max_substeps_per_frame=args.max_substeps_per_frame,
        )
        previous_wall_time = time.perf_counter()
        frame = 0
        reset_requested = False

        def request_reset():
            nonlocal reset_requested
            reset_requested = True

        register_keyboard_controls(
            renderer,
            on_reset=request_reset,
            on_reverse_gravity=lambda: reverse_gravity(simulation),
            on_rotate_gravity=lambda angle: rotate_gravity(simulation, angle),
        )

        try:
            while renderer.is_running():
                if reset_requested:
                    reset_requested = False
                    simulation = create_rendered_simulation(
                        config=config,
                        verbose=args.verbose,
                        device=device,
                        renderer=renderer,
                        speed_color_mid=args.speed_color_mid,
                        speed_color_max=args.speed_color_max,
                    )
                    scheduler = PlaybackScheduler(
                        playback_speed=args.playback_speed,
                        max_substeps_per_frame=args.max_substeps_per_frame,
                    )
                    renderer.paused = True
                    simulation.render()
                    previous_wall_time = time.perf_counter()
                    continue

                if args.num_frames != 0 and frame >= args.num_frames:
                    break

                current_wall_time = time.perf_counter()
                scheduler.advance(simulation, current_wall_time - previous_wall_time)
                previous_wall_time = current_wall_time

                simulation.render()

                frame += 1
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
