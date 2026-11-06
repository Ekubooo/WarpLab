"""OpenGL billboard display, mesh instances and native window input."""

import time
import numpy as np
import warp as wp
from demos.fluid.common.billboard_renderer import (
    BillboardRenderer, configure_nvidia_prime_render_offload,
)


def load_static_scene(boundary_system):
    """Load the mesh paired with the system's first static boundary asset."""
    instance = next((item for item in boundary_system.instances if not item.is_dynamic), None)
    if instance is None:
        raise ValueError("Static rendering requires a static boundary instance")
    path = boundary_system.assets[instance.asset_id].mesh_path
    import trimesh
    mesh = trimesh.load(path, force="mesh", process=False)
    return mesh.vertices, mesh.faces


def register_rigid_meshes(renderer, simulation):
    """Upload each asset once, with independent body instances driven on GPU."""
    from PBF.scene import load_render_meshes
    boundaries = simulation.BoundarySys
    meshes = load_render_meshes(boundaries)
    data = boundaries.device_data
    body_ids, local = data.body_ids.numpy(), data.local_transforms.numpy()
    bodies = [renderer.register_body(f"body_{i}") for i in range(simulation.model.body_count)]
    shapes = {}
    color = (.85, .45, .15)
    for index, instance in enumerate(boundaries.instances):
        if not instance.is_dynamic:
            continue
        body = int(body_ids[instance.rigid_id])
        pose = local[instance.rigid_id]
        name = f"rigid_{index}"
        if instance.asset_id not in shapes:
            shapes[instance.asset_id] = renderer.render_mesh(name, *meshes[index],
                colors=color, pos=tuple(pose[:3]), rot=tuple(pose[3:]),
                parent_body=f"body_{body}", smooth_shading=False)
        else:
            renderer.add_shape_instance(name, shapes[instance.asset_id], bodies[body],
                tuple(pose[:3]), tuple(pose[3:]), color1=color, color2=color)


class PBFRenderer(BillboardRenderer):
    """Let the PBF main loop handle pause and deferred reset requests."""

    def __init__(self, *args, particle_positions: wp.array | None = None,
                 speed_color_mid=2.0, speed_color_max=None, **kwargs):
        self._closed = False
        self.speed_color_mid = speed_color_mid
        self.speed_color_max = speed_color_max
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

    def initialize(self, simulation):
        points, indices = load_static_scene(simulation.BoundarySys)
        self.render_mesh(name="terrain", points=points, indices=indices,
            colors=(0.55, 0.55, 0.55), pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0), scale=(1.0, 1.0, 1.0), smooth_shading=False)
        if simulation.is_coupled:
            register_rigid_meshes(self, simulation)
        if self.speed_color_max is None:
            self.speed_color_max = simulation.config.max_speed
        self._keyboard_callbacks = {}
        self._pressed_keys = set()
        self.register_key_press_callback(self._dispatch_key)
        self.window.push_handlers(on_key_release=self._release_key,
                                  on_deactivate=self._release_keys)

    def render(self, simulation):
        self.begin_frame(simulation.sim_time)
        self.body_transforms = simulation.state.body_q
        self.render_billboards(name="points", points=simulation.pos, velocities=simulation.v,
            radius=simulation.particle_radius, speed_color_mid=self.speed_color_mid,
            speed_color_max=self.speed_color_max)
        self.end_frame()

    def register_keyboard_callback(self, key, callback, *, repeat=True):
        from pyglet.window import key as keys
        self._keyboard_callbacks[getattr(keys, key.upper())] = callback, repeat

    def _dispatch_key(self, symbol, modifiers):
        import pyglet
        if symbol not in self._keyboard_callbacks:
            return None
        callback, repeat = self._keyboard_callbacks[symbol]
        if repeat or symbol not in self._pressed_keys:
            self._pressed_keys.add(symbol)
            callback()
        return pyglet.event.EVENT_HANDLED

    def _release_key(self, symbol, modifiers):
        self._pressed_keys.discard(symbol)

    def _release_keys(self):
        self._pressed_keys.clear()

    def close(self):
        if not self._closed:
            self._closed = True
            super().close()

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
    speed_color_mid=2.0, speed_color_max=None,
):
    """Keep the shared frontend's view settings with PBF-specific pause handling."""
    configure_nvidia_prime_render_offload()
    return PBFRenderer(
        title=title, scaling=scaling, fps=120, up_axis="Y",
        screen_width=1280, screen_height=720, near_plane=0.1, far_plane=100.0,
        camera_pos=camera_pos, camera_front=camera_front, camera_up=(0.0, 1.0, 0.0),
        background_color=(0.0, 0.0, 0.0), vsync=True, device=device,
        particle_positions=particle_positions,
        speed_color_mid=speed_color_mid, speed_color_max=speed_color_max,
    )


