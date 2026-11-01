"""Shared OpenGL billboard renderer for Warp particle simulations.

The renderer keeps particle positions and velocities on the selected Warp
device.  A small Warp kernel writes camera-facing instance transforms and
speed-based colors directly into the CUDA/OpenGL shared buffers each frame.
"""

import os

import numpy as np

import warp as wp
import warp.render
from warp._src.render.render_opengl import ShapeInstancer, arr_pointer


LOW_SPEED_COLOR = (29.0 / 255.0, 119.0 / 255.0, 231.0 / 255.0)
MID_SPEED_COLOR = (160.0 / 255.0, 221.0 / 255.0, 1.0)
HIGH_SPEED_COLOR = (1.0, 1.0, 1.0)


def configure_nvidia_prime_render_offload():
    """Prefer the discrete NVIDIA GLX provider on PRIME systems."""
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


BILLBOARD_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;

layout (location = 3) in vec4 aInstanceTransform0;
layout (location = 4) in vec4 aInstanceTransform1;
layout (location = 5) in vec4 aInstanceTransform2;
layout (location = 6) in vec4 aInstanceTransform3;
layout (location = 7) in vec3 aObjectColor;

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

out vec3 ObjectColor;

void main()
{
    // Expand the square in view space, so camera rotation also works when
    // paused or redrawing without another particle-buffer update.
    vec4 center = view * model * aInstanceTransform3;
    vec2 size = vec2(
        length((model * aInstanceTransform0).xyz),
        length((model * aInstanceTransform1).xyz)
    );
    gl_Position = projection * (center + vec4(aPos.xy * size, 0.0, 0.0));
    ObjectColor = aObjectColor;
}
"""


BILLBOARD_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

in vec3 ObjectColor;

void main()
{
    FragColor = vec4(ObjectColor, 1.0);
}
"""


@wp.kernel
def update_billboard_transforms(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radius: float,
    speed_color_mid: float,
    speed_color_max: float,
    low_color: wp.vec3,
    mid_color: wp.vec3,
    high_color: wp.vec3,
    transforms: wp.array(dtype=wp.mat44),
    colors1: wp.array(dtype=wp.vec3),
    colors2: wp.array(dtype=wp.vec3),
):
    """Write particle centers, radii and speed-based instance colors."""
    tid = wp.tid()
    p = positions[tid]

    speed = wp.length(velocities[tid])
    normalized_speed = wp.clamp(speed / speed_color_max, 0.0, 1.0)
    normalized_mid = speed_color_mid / speed_color_max
    color = low_color
    if normalized_speed <= normalized_mid:
        color_fraction = normalized_speed / normalized_mid
        color = low_color * (1.0 - color_fraction) + mid_color * color_fraction
    else:
        color_fraction = (normalized_speed - normalized_mid) / (1.0 - normalized_mid)
        color = mid_color * (1.0 - color_fraction) + high_color * color_fraction
    colors1[tid] = color
    colors2[tid] = color

    # Warp matrices are laid out transposed when consumed as OpenGL instance
    # attributes. Store scale and translation; the shader supplies orientation.
    transforms[tid] = wp.mat44(
        radius,
        0.0,
        0.0,
        0.0,
        0.0,
        radius,
        0.0,
        0.0,
        0.0,
        0.0,
        radius,
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


class BillboardRenderer(wp.render.OpenGLRenderer):
    """Warp OpenGL renderer with instanced camera-facing square particles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from pyglet.graphics.shader import Shader, ShaderProgram

        self._billboard_shader = ShaderProgram(
            Shader(BILLBOARD_VERTEX_SHADER, "vertex"),
            Shader(BILLBOARD_FRAGMENT_SHADER, "fragment"),
        )
        gl = self.gl
        self._loc_billboard_model = gl.glGetUniformLocation(self._billboard_shader.id, b"model")
        self._loc_billboard_view = gl.glGetUniformLocation(self._billboard_shader.id, b"view")
        self._loc_billboard_projection = gl.glGetUniformLocation(self._billboard_shader.id, b"projection")
        self._billboard_color_resources = {}

    def clear(self):
        # Unregister CUDA/GL color resources before OpenGLRenderer deletes the
        # underlying color buffers.
        self._billboard_color_resources.clear()
        super().clear()

    def _upload_billboard_camera(self):
        """Use the same camera state as Warp's mesh shader for this draw."""
        gl = self.gl
        gl.glUseProgram(self._billboard_shader.id)
        gl.glUniformMatrix4fv(self._loc_billboard_model, 1, gl.GL_FALSE, arr_pointer(self._model_matrix))
        gl.glUniformMatrix4fv(self._loc_billboard_view, 1, gl.GL_FALSE, arr_pointer(self._view_matrix))
        gl.glUniformMatrix4fv(
            self._loc_billboard_projection,
            1,
            gl.GL_FALSE,
            arr_pointer(self._projection_matrix),
        )

    def _process_inputs(self):
        # Warp calls this once inside _draw, before uploading its mesh camera.
        super()._process_inputs()
        self._upload_billboard_camera()

    def _draw(self):
        self._switch_context()
        if self.headless:
            # Warp skips _process_inputs for hidden/offscreen renders.
            self._upload_billboard_camera()
        super()._draw()

    def _allocate_billboard_instances(self, instancer, points, color):
        """Initialize instance storage; subclasses may allocate it without host positions."""
        instancer.allocate_instances(points.numpy(), colors1=color, colors2=color)

    def _register_billboard_color_resources(self, name, instancer):
        self._billboard_color_resources[name] = (
            wp.RegisteredGLBuffer(
                int(instancer.instance_color1_buffer.value),
                self._device,
                flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            ),
            wp.RegisteredGLBuffer(
                int(instancer.instance_color2_buffer.value),
                self._device,
                flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            ),
        )

    def render_billboards(
        self,
        name,
        points,
        velocities,
        radius,
        speed_color_mid,
        speed_color_max,
        low_color=LOW_SPEED_COLOR,
        mid_color=MID_SPEED_COLOR,
        high_color=HIGH_SPEED_COLOR,
    ):
        """Render camera-facing squares colored by particle speed."""
        if len(points) == 0:
            return

        if len(velocities) != len(points):
            raise ValueError("Billboard positions and velocities must have the same length")
        if speed_color_mid <= 0.0:
            raise ValueError("speed_color_mid must be greater than zero")
        if speed_color_max <= speed_color_mid:
            raise ValueError("speed_color_max must be greater than speed_color_mid")
        if radius <= 0.0:
            raise ValueError("Billboard radius must be greater than zero")

        if not isinstance(points, wp.array):
            points = wp.array(points, dtype=wp.vec3, device=self._device)
        elif points.device != self._device:
            points = points.to(self._device)
        if not isinstance(velocities, wp.array):
            velocities = wp.array(velocities, dtype=wp.vec3, device=self._device)
        elif velocities.device != self._device:
            velocities = velocities.to(self._device)

        if name not in self._shape_instancers:
            # Unit square expanded in the current camera plane by the shader.
            vertices = np.array(
                [
                    (-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                    (1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
                    (1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                    (-1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
                ],
                dtype=np.float32,
            )
            indices = np.array((0, 1, 2, 0, 2, 3), dtype=np.uint32)

            instancer = ShapeInstancer(self._billboard_shader, self._device)
            instancer.register_shape(vertices, indices, color1=low_color, color2=low_color)
            self._allocate_billboard_instances(instancer, points, low_color)
            self._shape_instancers[name] = instancer
            self._register_billboard_color_resources(name, instancer)
        else:
            instancer = self._shape_instancers[name]
            if len(points) != instancer.num_instances:
                # Registered resources must be released before glBufferData()
                # potentially replaces their underlying color storage.
                self._billboard_color_resources.pop(name, None)
                self._allocate_billboard_instances(instancer, points, low_color)
                self._register_billboard_color_resources(name, instancer)

        with instancer:
            color1_resource, color2_resource = self._billboard_color_resources[name]
            colors1 = color1_resource.map(dtype=wp.vec3, shape=(len(points),))
            colors2 = None
            try:
                colors2 = color2_resource.map(dtype=wp.vec3, shape=(len(points),))
                wp.launch(
                    kernel=update_billboard_transforms,
                    dim=len(points),
                    inputs=[
                        points,
                        velocities,
                        radius,
                        speed_color_mid,
                        speed_color_max,
                        wp.vec3(*low_color),
                        wp.vec3(*mid_color),
                        wp.vec3(*high_color),
                    ],
                    outputs=[instancer.vbo_transforms, colors1, colors2],
                    device=self._device,
                    record_tape=False,
                )
            finally:
                if colors2 is not None:
                    color2_resource.unmap()
                color1_resource.unmap()


def create_billboard_renderer(
    device=None,
    title="Warp Fluid",
    *,
    scaling=0.05,
    camera_pos=(2.0, 3.0, 10.0),
    camera_front=(-0.1, -0.1, -1.0),
):
    """Create a renderer with optional scene scale and initial camera overrides."""
    # OpenGLRenderer imports Pyglet lazily, so setting PRIME here is early
    # enough to create its GL context on the same NVIDIA GPU as CUDA.
    configure_nvidia_prime_render_offload()

    return BillboardRenderer(
        title=title,
        scaling=scaling,
        fps=120,
        up_axis="Y",
        screen_width=1280,
        screen_height=720,
        near_plane=0.1,
        far_plane=100.0,
        camera_pos=camera_pos,
        camera_front=camera_front,
        camera_up=(0.0, 1.0, 0.0),
        background_color=(0.0, 0.0, 0.0),
        vsync=False,
        device=device,
    )
