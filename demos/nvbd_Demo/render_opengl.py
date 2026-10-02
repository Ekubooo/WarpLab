"""Minimal interactive OpenGL viewer for the copied NanoVDB example."""

import argparse
import time
import weakref

import warp as wp
import warp.render

if __package__:
    from .example_nvdb import Example
else:
    from example_nvdb import Example


@wp.kernel
def sample_sdf(volume: wp.uint64, lower: wp.vec3, field: wp.array3d[float]):
    i, j, k = wp.tid()
    uvw = lower + wp.vec3(float(i), float(j), float(k))
    field[i, j, k] = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)


@wp.kernel
def mesh_to_world(volume: wp.uint64, lower: wp.vec3, vertices: wp.array[wp.vec3]):
    i = wp.tid()
    vertices[i] = wp.volume_index_to_world(volume, lower + vertices[i])


def collision_mesh(volume):
    """Extract the static zero surface once, without needing a USD reader."""
    voxels = volume.get_voxels().numpy()
    lower = voxels.min(axis=0) - 1
    upper = voxels.max(axis=0) + 1
    shape = tuple(int(n) for n in upper - lower + 1)
    origin = wp.vec3(*(float(n) for n in lower))
    field = wp.empty(shape, dtype=float, device=volume.device)
    wp.launch(sample_sdf, dim=shape, inputs=[volume.id, origin, field], device=volume.device)
    surface = wp.MarchingCubes(*shape)
    surface.surface(field, 0.0)
    wp.launch(mesh_to_world, dim=len(surface.verts), inputs=[volume.id, origin, surface.verts], device=volume.device)
    return surface.verts.numpy(), surface.indices.numpy()


def create_renderer(simulation, headless=False):
    points, indices = collision_mesh(simulation.volume)
    renderer = wp.render.OpenGLRenderer(
        title="Warp NanoVDB",
        screen_width=1280,
        screen_height=720,
        camera_pos=(45.0, 35.0, 65.0),
        camera_front=(-0.55, -0.25, -0.80),
        near_plane=0.1,
        far_plane=200.0,
        up_axis="Y",
        draw_sky=False,
        vsync=True,
        headless=headless,
        device=simulation.positions.device,
    )
    renderer.default_num_segments = 8
    renderer.render_ground(size=100.0)
    renderer.render_mesh("rocks", points, indices, colors=(0.35, 0.55, 0.9))
    return renderer


class Playback:
    """Pace fixed simulation steps independently of rendering and camera input."""

    def __init__(self, renderer):
        from pyglet.window import key

        self.renderer = weakref.proxy(renderer)
        self.paused = True
        self.slow = False
        self.accumulator = 0.0
        self.last_time = time.perf_counter()
        renderer.register_key_press_callback(self.on_key_press)
        # S controls playback; the down arrow still moves the camera backward.
        renderer.register_input_processor(lambda keys: keys.on_key_release(key.S, 0))
        self.update_caption()

    def update_caption(self):
        state = "Paused" if self.paused else "Playing"
        speed = "0.1x" if self.slow else "1x"
        self.renderer.window.set_caption(f"Warp NanoVDB | {state} | {speed} | Space: pause | S: speed")

    def on_key_press(self, symbol, modifiers):
        from pyglet import event
        from pyglet.window import key

        if symbol == key.SPACE:
            self.paused = not self.paused
        elif symbol == key.S:
            self.slow = not self.slow
        else:
            return
        self.last_time = time.perf_counter()
        self.accumulator = 0.0
        self.update_caption()
        return event.EVENT_HANDLED

    def advance(self, simulation):
        now = time.perf_counter()
        elapsed = now - self.last_time
        self.last_time = now
        if self.paused:
            return
        # Avoid a burst of catch-up steps after a stalled window or debugger.
        self.accumulator += min(elapsed, 0.25) * (0.1 if self.slow else 1.0)
        frame_dt = simulation.sim_dt * simulation.sim_substeps
        while self.accumulator >= frame_dt:
            simulation.step()
            self.accumulator -= frame_dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=0, help="Zero runs until the window is closed.")
    parser.add_argument("--headless", action="store_true", help="Render without showing a window; requires --num-frames.")
    args = parser.parse_args()
    if args.num_frames < 0 or (args.headless and args.num_frames == 0):
        parser.error("--num-frames must be nonnegative and positive for --headless")

    with wp.ScopedDevice(args.device):
        if not wp.get_device().is_cuda:
            parser.error("This viewer requires a CUDA device.")
        simulation = Example(stage_path=None)
        renderer = create_renderer(simulation, headless=args.headless)
        playback = Playback(renderer)
        frame = 0
        try:
            while renderer.is_running() and (args.num_frames == 0 or frame < args.num_frames):
                frame_start = time.perf_counter()
                playback.advance(simulation)
                renderer.begin_frame(simulation.sim_time)
                renderer.time = simulation.sim_time
                renderer.render_points("particles", simulation.positions, simulation.sim_margin, colors=(0.8, 0.3, 0.2))
                renderer.end_frame()
                if not renderer.is_running():
                    break
                frame += 1
                time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - frame_start)))
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
