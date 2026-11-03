"""Executable OpenGL billboard frontend for the Warp PBF simulation."""

import argparse
from dataclasses import replace
import math
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Lab_Framework.pbf.billboard_renderer import BillboardRenderer, configure_nvidia_prime_render_offload
from Lab_Framework.pbf.config import PBFConfig
from Lab_Framework.pbf import config_io
from Lab_Framework.pbf.simulation import Simulation, create_pbf_simulation, reset_simulation, reverse_gravity


def load_static_scene(boundary_system):
    """Load the mesh paired with the system's first static boundary asset."""
    instance = next((item for item in boundary_system.instances if not item.is_dynamic), None)
    if instance is None:
        raise ValueError("Static rendering requires a static boundary instance")
    path = boundary_system.assets[instance.asset_id].mesh_path
    return load_boundary_mesh(path)


def load_boundary_mesh(path):
    """Load geometry once; dynamic instance poses are updated independently."""
    if not path.is_file():
        raise FileNotFoundError(f"Static scene not found: {path}; generate the container assets first")
    try:
        import trimesh

        mesh = trimesh.load(path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty or not np.isfinite(mesh.vertices).all():
            raise ValueError("Static scene must contain a nonempty, finite triangle mesh")
    except Exception as error:
        raise RuntimeError(f"Cannot load static scene {path}: {error}") from error
    return mesh.vertices, mesh.faces


def load_rigid_scene(simulation):
    """Host geometry and fixed local transforms, never per-frame GPU downloads."""
    boundaries = simulation.BoundarySys
    if not boundaries.rigid_count:
        return []
    data = boundaries.device_data
    body_ids = data.body_ids.numpy()
    local_transforms = data.local_transforms.numpy()
    meshes = {}
    instances = []
    for index, instance in enumerate(boundaries.instances):
        if instance.is_dynamic:
            if instance.asset_id not in meshes:
                meshes[instance.asset_id] = load_boundary_mesh(boundaries.assets[instance.asset_id].mesh_path)
            instances.append((f"rigid_{index}", int(body_ids[instance.rigid_id]),
                              local_transforms[instance.rigid_id], meshes[instance.asset_id]))
    return instances


def register_rigid_meshes(renderer, simulation):
    """Upload each rigid asset once, then attach independent body instances."""
    bodies = [renderer.register_body(f"body_{body}") for body in range(simulation.model.body_count)]
    shapes = {}
    color = (.85, .45, .15)
    for name, body, local, mesh in load_rigid_scene(simulation):
        # load_rigid_scene shares the same host mesh tuple for each asset.
        # Repeating Warp's render_mesh on a flat-shaded shape updates its
        # expanded GPU vertices with the unexpanded source vertex array.
        key = id(mesh)
        if key not in shapes:
            shapes[key] = renderer.render_mesh(name, *mesh, colors=color,
                pos=tuple(local[:3]), rot=tuple(local[3:]), parent_body=f"body_{body}",
                smooth_shading=False)
        else:
            renderer.add_shape_instance(name, shapes[key], bodies[body],
                tuple(local[:3]), tuple(local[3:]), color1=color, color2=color)


class PBFRenderer(BillboardRenderer):
    """Let the PBF main loop handle pause and deferred reset requests."""

    def __init__(self, *args, particle_positions: wp.array | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.body_transforms = None
        # Snapshot the initialized array's host metadata without retaining its buffer.
        self._particle_count = None if particle_positions is None else len(particle_positions)
        self._particle_count_label = None
        if self._particle_count is not None:
            import pyglet

            self._particle_count_label = pyglet.text.Label(
                f"Particles: {self._particle_count:,}",
                font_name=self._info_label.font_name, font_size=self._info_label.font_size,
                color=self._info_label.color, x=self._info_label.x,
                anchor_x="left", anchor_y="top",
            )
            self.render_2d_callbacks.append(self._draw_particle_count)

        # Keep Warp's info overlay, but publish FPS only once per time window.
        self._fps_alpha = 0.0
        self._fps_refresh_interval = 0.5
        self._fps_window_start = None
        self._fps_last_update_time = None
        self._fps_frame_start = None
        self._fps_pending_frame_duration = None
        self._fps_update_samples = 0
        self._fps_update_duration = 0.0
        self._fps_render_samples = 0
        self._fps_render_duration = 0.0

    def _draw_particle_count(self):
        if not self.show_info:
            return
        label = self._particle_count_label
        position = (self._info_label.x, self._info_label.y - self._info_label.content_height - 4, label.z)
        if label.position != position:
            label.position = position
        gl = self.gl
        # Warp enables depth testing after its info overlay, before this first 2D callback.
        gl.glDisable(gl.GL_DEPTH_TEST)
        try:
            label.draw()
        finally:
            gl.glEnable(gl.GL_DEPTH_TEST)

    def register_shape(self, geo_hash, vertices, indices, color1=None, color2=None):
        # Warp 1.15 flat-shaded meshes use np.arange(), which is int64 in NumPy 2.
        # Its OpenGL draw calls require GL_UNSIGNED_INT indices (32 bits).
        return super().register_shape(geo_hash, vertices, np.asarray(indices, dtype=np.uint32), color1, color2)

    def begin_frame(self, t=None):
        super().begin_frame(t)
        self._fps_frame_start = time.perf_counter()
        # Warp treats zero as an omitted time; reset must display simulation time zero.
        if t is not None:
            self.time = t

    def end_frame(self):
        self._last_end_frame_time = time.time()
        if self._fps_frame_start is not None:
            self._fps_pending_frame_duration = time.perf_counter() - self._fps_frame_start
            self._fps_frame_start = None
        if self._add_shape_instances:
            self.allocate_shape_instances()
        if self._update_shape_instances:
            self.update_shape_instances()
        if self.body_transforms is not None:
            self.update_body_transforms(self.body_transforms)
        self.update()

    def update(self):
        now = time.perf_counter()
        if self._fps_window_start is None:
            self._fps_window_start = now

        frame_duration = self._fps_pending_frame_duration
        self._fps_pending_frame_duration = None
        if not self.skip_rendering:
            if frame_duration is not None and frame_duration > 0.0:
                self._fps_update_samples += 1
                self._fps_update_duration += frame_duration
            if self._fps_last_update_time is not None:
                render_duration = now - self._fps_last_update_time
                if render_duration > 0.0:
                    self._fps_render_samples += 1
                    self._fps_render_duration += render_duration
        self._fps_last_update_time = now

        if now - self._fps_window_start >= self._fps_refresh_interval:
            # Invert the mean duration instead of averaging instantaneous FPS.
            if self._fps_update_samples:
                self._fps_update = self._fps_update_samples / self._fps_update_duration
            if self._fps_render_samples:
                self._fps_render = self._fps_render_samples / self._fps_render_duration
            self._fps_window_start = now
            self._fps_update_samples = 0
            self._fps_update_duration = 0.0
            self._fps_render_samples = 0
            self._fps_render_duration = 0.0

        super().update()


def create_pbf_renderer(
    device=None, title="Warp PBF", *, scaling=0.05,
    camera_pos=(2.0, 3.0, 10.0), camera_front=(-0.1, -0.1, -1.0),
    particle_positions: wp.array | None = None,
):
    """Keep the shared frontend's view settings with PBF-specific pause handling."""
    configure_nvidia_prime_render_offload()
    return PBFRenderer(
        title=title, scaling=scaling, fps=120, up_axis="Y",
        screen_width=1280, screen_height=720, near_plane=0.1, far_plane=100.0,
        camera_pos=camera_pos, camera_front=camera_front, camera_up=(0.0, 1.0, 0.0),
        background_color=(0.0, 0.0, 0.0), vsync=True, device=device,
        particle_positions=particle_positions,
    )


def register_keyboard_controls(renderer, simulation: Simulation, *, on_reset):
    """Handle PBF playback, gravity and reset controls in one place."""
    import pyglet

    slow_motion = False
    last_direction_key = None
    gravity_strength = math.hypot(*simulation.config.gravity)
    pressed_keys = set()

    def on_key_press(symbol, _modifiers):
        nonlocal slow_motion, last_direction_key

        if symbol in (pyglet.window.key.G, pyglet.window.key.X,
                      pyglet.window.key.Q, pyglet.window.key.E):
            if symbol in pressed_keys:
                return pyglet.event.EVENT_HANDLED
            pressed_keys.add(symbol)

        if symbol == pyglet.window.key.G:
            reverse_gravity(simulation)
            last_direction_key = None
        elif symbol in (pyglet.window.key.Q, pyglet.window.key.E):
            # Q/E select world -X/+X; repeating that selection restores world -Y.
            if last_direction_key == symbol:
                simulation.gravity = wp.vec3(0.0, -gravity_strength, 0.0)
                last_direction_key = None
            else:
                direction = -1.0 if symbol == pyglet.window.key.Q else 1.0
                simulation.gravity = wp.vec3(direction * gravity_strength, 0.0, 0.0)
                last_direction_key = symbol
        elif symbol == pyglet.window.key.X:
            slow_motion = not slow_motion
            simulation.frame_dt = simulation.config.frame_dt * (0.5 if slow_motion else 1.0)
        elif symbol == pyglet.window.key.R:
            slow_motion = False
            last_direction_key = None
            simulation.frame_dt = simulation.config.frame_dt
            renderer.paused = True
            on_reset()
        elif symbol == pyglet.window.key.P:
            renderer.draw_grid = not renderer.draw_grid
        else:
            return None
        return pyglet.event.EVENT_HANDLED

    def on_key_release(symbol, _modifiers):
        pressed_keys.discard(symbol)

    def on_deactivate():
        pressed_keys.clear()

    renderer.register_key_press_callback(on_key_press)
    renderer.window.push_handlers(on_key_release=on_key_release, on_deactivate=on_deactivate)
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
    if args.scene == "tank-box":
        from Lab_Framework.pbf.scenes import tank_box_config
        return replace(tank_box_config(), **overrides)
    return PBFConfig(**overrides) if overrides else None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--scene", choices=("static", "tank-box", "stairs-rigids"), default="static",
                        help="Original stairs, coupled tank/box, or stairs with four free rigid bodies.")
    parser.add_argument("--headless", action="store_true", help="Advance without an OpenGL window.")
    parser.add_argument("--stage-path", default=None, help="Optional USD output path for headless execution.")
    parser.add_argument("--cuda-graph", action="store_true", help="Capture and replay complete simulation frames.")
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

    with wp.ScopedDevice(args.device):
        simulation = create_pbf_simulation(
            verbose=args.verbose, config=args.config, scene=args.scene,
        )
        if args.cuda_graph:
            simulation.capture_graph()
        if args.headless:
            run_headless(simulation, args)
            return
        scene_points, scene_indices = load_static_scene(simulation.BoundarySys)
        # Focus on the fluid near the floor; distance follows the container footprint.
        camera_azimuth_deg = 35.0
        camera_elevation_deg = 25.0
        camera_distance_scale = 1.75
        camera_target = wp.vec3(0.5 * simulation.width, 0.25 * simulation.width, 0.25 * simulation.length)
        if simulation.scene == "tank-box":
            # View through the open top; the opaque walls hide the fluid at
            # the stairs scene's low viewing angle. Camera controls are shared.
            camera_elevation_deg = 65.0
            camera_distance_scale = 2.3
            camera_target = wp.vec3(.5 * simulation.width, .45, .5 * simulation.length)
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
        )
        billboard_radius = simulation.particle_radius
        frame = 0
        reset_requested = False

        def request_reset():
            nonlocal reset_requested
            reset_requested = True

        try:
            renderer.render_mesh(
                name="terrain",
                points=scene_points,
                indices=scene_indices,
                colors=(0.55, 0.55, 0.55),
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                scale=(1.0, 1.0, 1.0),
                smooth_shading=False,
            )
            register_rigid_meshes(renderer, simulation)
            register_keyboard_controls(
                renderer, simulation, on_reset=request_reset,
            )
            while renderer.is_running() and (args.num_frames == 0 or frame < args.num_frames):
                if reset_requested:
                    reset_simulation(simulation)
                    reset_requested = False
                    renderer.paused = True
                # Render first so the initialized PBF particles are visible.
                renderer.begin_frame(simulation.sim_time)
                renderer.body_transforms = simulation.state.body_q
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
                    if args.cuda_graph:
                        if simulation.graph_needs_recapture():
                            simulation.capture_graph()
                        simulation.launch_graph()
                    else:
                        simulation.step()
                frame += 1
        finally:
            renderer.close()


def run_headless(simulation, args):
    """Optionally export USD without coupling rendering into the solver."""
    renderer = None
    if args.stage_path is not None:
        import warp.render
        renderer = wp.render.UsdRenderer(args.stage_path)
        rigid_meshes = load_rigid_scene(simulation)
        if rigid_meshes:
            points, indices = load_static_scene(simulation.BoundarySys)
            renderer.render_mesh("terrain", points, indices, colors=(.55, .55, .55))
    for _ in range(args.num_frames):
        if renderer is not None:
            renderer.begin_frame(simulation.sim_time)
            renderer.render_points(name="points", points=simulation.pos.numpy(),
                                   radius=simulation.particle_radius, colors=(.8, .3, .2))
            if rigid_meshes:
                body_q = simulation.state.body_q.numpy()
                for name, body, local, (points, indices) in rigid_meshes:
                    transform = wp.transform_multiply(wp.transform(*body_q[body]), wp.transform(*local))
                    renderer.render_mesh(name, points, indices, colors=(.85, .45, .15),
                        pos=tuple(wp.transform_get_translation(transform)),
                        rot=tuple(wp.transform_get_rotation(transform)), smooth_shading=False)
            renderer.end_frame()
        if args.cuda_graph:
            simulation.launch_graph()
        else:
            simulation.step()
    if renderer is not None:
        renderer.save()


if __name__ == "__main__":
    main()
