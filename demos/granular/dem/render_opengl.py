"""Executable real-time OpenGL frontend for Warp's official DEM example."""

import argparse
import os

import numpy as np

import warp as wp
import warp.render
from warp._src.render.render_opengl import ShapeInstancer

from simulation import create_dem_simulation


PARTICLE_COLOR = (0.8, 0.3, 0.2)


def configure_nvidia_prime_render_offload():
    """Prefer NVIDIA's GLX provider before Pyglet creates a GL context."""
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


@wp.kernel
def update_cube_transforms(
    positions: wp.array(dtype=wp.vec3),
    half_extent: float,
    transforms: wp.array(dtype=wp.mat44),
):
    """Build world-axis-aligned cube transforms directly in the OpenGL VBO."""
    tid = wp.tid()
    p = positions[tid]

    # Warp matrices are laid out transposed when consumed as OpenGL instance
    # attributes. These rows become the basis and translation columns in GLSL.
    transforms[tid] = wp.mat44(
        half_extent,
        0.0,
        0.0,
        0.0,
        0.0,
        half_extent,
        0.0,
        0.0,
        0.0,
        0.0,
        half_extent,
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


class DemRenderer(wp.render.OpenGLRenderer):
    """OpenGL renderer for large DEM particle sets using upright cubes."""

    def clear(self):
        # CUDA must release registrations before ShapeInstancer deletes the GL buffers.
        for instancer in self._shape_instancers.values():
            instancer._instance_transform_cuda_buffer = None
        self._shape_instancers.clear()
        super().clear()

    def render_particles(self, name, points, radius, color=PARTICLE_COLOR):
        """Render fixed-size particles as upright cubes without a per-frame CPU copy."""
        if not isinstance(points, wp.array):
            raise TypeError("DEM particle positions must be a Warp array")
        if points.dtype != wp.vec3:
            raise TypeError("DEM particle positions must have dtype wp.vec3")
        if points.device != self._device:
            raise ValueError(
                f"DEM particles and renderer must use the same device ({points.device} != {self._device})"
            )
        if radius <= 0.0:
            raise ValueError("DEM particle radius must be greater than zero")
        if len(points) == 0:
            return

        if name not in self._shape_instancers:
            vertices, indices = self._create_box_mesh((1.0, 1.0, 1.0))

            instancer = ShapeInstancer(self._shape_shader, self._device)
            instancer.register_shape(vertices, indices, color1=color, color2=color)
            instancer.allocate_instances(
                np.zeros((len(points), 3), dtype=np.float32),
                colors1=color,
                colors2=color,
            )
            self._shape_instancers[name] = instancer
        else:
            instancer = self._shape_instancers[name]
            if len(points) != instancer.num_instances:
                raise ValueError("DEM particle count cannot change after OpenGL resources are allocated")

        with instancer:
            wp.launch(
                kernel=update_cube_transforms,
                dim=len(points),
                inputs=[points, radius],
                outputs=[instancer.vbo_transforms],
                device=self._device,
                record_tape=False,
            )


def create_dem_renderer(device=None):
    """Create the interactive DEM viewer on the selected Warp device."""
    configure_nvidia_prime_render_offload()

    return DemRenderer(
        title="Warp DEM",
        scaling=1.0,
        fps=60,
        up_axis="Y",
        screen_width=1280,
        screen_height=720,
        near_plane=0.1,
        far_plane=100.0,
        camera_pos=(11.0, 9.0, 24.0),
        camera_front=(-0.34, -0.15, -0.93),
        camera_up=(0.0, 1.0, 0.0),
        background_color=(0.02, 0.025, 0.035),
        draw_grid=True,
        draw_sky=False,
        draw_axis=True,
        vsync=True,
        enable_backface_culling=True,
        device=device,
    )


def nonnegative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
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
    return parser.parse_args()


def main():
    args = parse_args()

    with wp.ScopedDevice(args.device):
        simulation = create_dem_simulation()
        renderer = create_dem_renderer(device=wp.get_device())
        frame = 0

        try:
            while renderer.is_running() and (args.num_frames == 0 or frame < args.num_frames):
                renderer.begin_frame(simulation.sim_time)
                renderer.render_particles(
                    name="particles",
                    points=simulation.x,
                    radius=simulation.point_radius,
                )
                renderer.end_frame()

                simulation.step()
                frame += 1
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
