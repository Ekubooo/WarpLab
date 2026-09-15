"""Billboard fluid, lit rigid meshes and a directional shadow map."""

import argparse
import ctypes
from pathlib import Path
import sys
import time

import numpy as np
import warp as wp

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demos.fluid.common.billboard_renderer import (
    BillboardRenderer,
    configure_nvidia_prime_render_offload,
)
from demos.fluid.MMPBF.render_opengl import (
    register_keyboard_controls,
    reverse_gravity,
    rotate_gravity,
)
from demos.fluid.PBF2WayCoupling.simulation import (
    add_simulation_arguments,
    config_from_args,
    create_pbf2way_simulation,
    positive_float,
    run_headless,
)

MESH_VERTEX = """#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec4 pose0;
layout(location=3) in vec4 pose1;
layout(location=4) in vec4 pose2;
layout(location=5) in vec4 pose3;
uniform mat4 view, projection, light_space;
out vec3 world_normal;
out vec4 light_position;
void main() {
    mat4 pose = mat4(pose0, pose1, pose2, pose3);
    vec4 world = pose * vec4(position, 1.0);
    world_normal = mat3(pose) * normal;
    light_position = light_space * world;
    gl_Position = projection * view * world;
}
"""
MESH_FRAGMENT = """#version 330 core
in vec3 world_normal;
in vec4 light_position;
uniform vec3 color;
uniform sampler2D shadow_map;
uniform bool shadows;
out vec4 frag_color;
void main() {
    vec3 n = normalize(world_normal);
    vec3 light_dir = normalize(vec3(5.0, 8.0, 6.0));
    float diffuse = max(dot(n, light_dir), 0.0);
    vec3 uv = light_position.xyz / light_position.w * 0.5 + 0.5;
    float shadow = 0.0;
    if(shadows && uv.z > 0.0 && uv.z < 1.0 && uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0) {
        vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0));
        float bias = max(0.0008 * (1.0 - diffuse), 0.00015);
        for(int x=-1; x<=1; x++) for(int y=-1; y<=1; y++)
            shadow += uv.z - bias > texture(shadow_map, uv.xy+vec2(x,y)*texel).r ? 1.0 : 0.0;
        shadow /= 9.0;
    }
    float lighting = 0.32 + 0.68 * diffuse * (1.0 - 0.75 * shadow);
    frag_color = vec4(color * lighting, 1.0);
}
"""
DEPTH_VERTEX = """#version 330 core
layout(location=0) in vec3 position;
layout(location=2) in vec4 pose0;
layout(location=3) in vec4 pose1;
layout(location=4) in vec4 pose2;
layout(location=5) in vec4 pose3;
uniform mat4 light_space;
void main() {
    mat4 pose = mat4(pose0, pose1, pose2, pose3);
    gl_Position = light_space * pose * vec4(position, 1.0);
}
"""
DEPTH_FRAGMENT = """#version 330 core
void main() {}
"""


@wp.kernel
def write_rigid_matrices(
    positions: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quat),
    body_ids: wp.array(dtype=int),
    columns: wp.array(dtype=wp.vec4),
):
    """Write column-major model matrices directly into a registered GL buffer."""
    i = wp.tid()
    body = body_ids[i]
    position = wp.vec3(0.0)
    rotation = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if body >= 0:
        position = positions[body]
        rotation = wp.quat_to_matrix(rotations[body])
    columns[4 * i] = wp.vec4(rotation[0, 0], rotation[1, 0], rotation[2, 0], 0.0)
    columns[4 * i + 1] = wp.vec4(rotation[0, 1], rotation[1, 1], rotation[2, 1], 0.0)
    columns[4 * i + 2] = wp.vec4(rotation[0, 2], rotation[1, 2], rotation[2, 2], 0.0)
    columns[4 * i + 3] = wp.vec4(position[0], position[1], position[2], 1.0)


def advance_and_render(simulation, renderer):
    """Exactly one complete simulation step per unpaused rendered frame."""
    if not renderer.paused:
        simulation.step()
    simulation.render()


def pace_frame(started, frame_dt, playback_speed):
    """Limit playback frequency without accumulating work or skipping states."""
    remaining = frame_dt / playback_speed - (time.perf_counter() - started)
    if remaining > 0:
        time.sleep(remaining)


def light_matrix(center):
    eye = np.asarray(center) + np.array([5.0, 8.0, 6.0])
    front = np.asarray(center) - eye
    front /= np.linalg.norm(front)
    right = np.cross(front, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, front)
    view = np.eye(4)
    view[:3, :3] = np.array([right, up, -front])
    view[:3, 3] = -view[:3, :3] @ eye
    projection = np.diag([1 / 6, 1 / 6, -2 / 30, 1])
    projection[2, 3] = -1
    return (projection @ view).astype(np.float32)


class CouplingRenderer(BillboardRenderer):
    """Own CUDA/GL mesh buffers and render exactly once per frontend iteration."""

    def __init__(self, device=None, hidden=False):
        if device is not None and not wp.get_device(device).is_cuda:
            raise ValueError(
                "The billboard frontend requires CUDA/OpenGL; use --headless for CPU simulation"
            )
        configure_nvidia_prime_render_offload()
        super().__init__(
            title="Warp PBF · Two-way coupling",
            scaling=1.0,
            fps=120,
            screen_width=1280,
            screen_height=800,
            near_plane=0.05,
            far_plane=50,
            camera_pos=(3.1, 2.8, 6.2),
            camera_front=(-0.40, -0.19, -0.90),
            background_color=(0.025, 0.04, 0.07),
            draw_grid=False,
            draw_sky=False,
            draw_axis=False,
            vsync=False,
            device=device,
            headless=hidden,
        )
        from pyglet.graphics.shader import Shader, ShaderProgram

        self.mesh_program = ShaderProgram(
            Shader(MESH_VERTEX, "vertex"), Shader(MESH_FRAGMENT, "fragment")
        )
        self.depth_program = ShaderProgram(
            Shader(DEPTH_VERTEX, "vertex"), Shader(DEPTH_FRAGMENT, "fragment")
        )
        self.mesh_buffers = []
        self.pose_resource = None
        self.pose_buffer = None
        self.colors = []
        self.light_space = light_matrix([0.0, 2.0, 0.0])
        self.shadows = True
        self.shadow_size = 2048
        self._create_shadow_map()
        self.render_3d_callbacks.append(self._draw_rigids)
        # Warp's default Tab shortcut skips rendering; this frontend always draws.
        import pyglet

        self.register_key_press_callback(
            lambda symbol, modifiers: (
                pyglet.event.EVENT_HANDLED if symbol == pyglet.window.key.TAB else None
            )
        )

    def end_frame(self):
        """Return after one draw, including while paused; the frontend owns pacing."""
        # Warp's base end_frame contains a nested pause loop. Keep its preparation
        # but leave pause handling to advance_and_render so reset remains responsive.
        self._last_end_frame_time = time.time()
        if self._add_shape_instances:
            self.allocate_shape_instances()
        if self._update_shape_instances:
            self.update_shape_instances()
        self.update()

    def _create_shadow_map(self):
        gl = self.gl
        self.shadow_texture, self.shadow_fbo = gl.GLuint(), gl.GLuint()
        gl.glGenTextures(1, self.shadow_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.shadow_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_DEPTH_COMPONENT24,
            self.shadow_size,
            self.shadow_size,
            0,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            None,
        )
        for param in (gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, param, gl.GL_NEAREST)
        for param in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, param, gl.GL_CLAMP_TO_EDGE)
        gl.glGenFramebuffers(1, self.shadow_fbo)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.shadow_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.shadow_texture, 0
        )
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Shadow framebuffer is incomplete")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _allocate_billboard_instances(self, instancer, points, color):
        # No initial transforms need copying: the billboard kernel fills every matrix.
        gl = self.gl
        instancer.num_instances = len(points)
        instancer._instance_transform_cuda_buffer = None
        if instancer.instance_transform_gl_buffer is None:
            instancer.instance_transform_gl_buffer = gl.GLuint()
            gl.glGenBuffers(1, instancer.instance_transform_gl_buffer)
        gl.glBindVertexArray(instancer.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, instancer.instance_transform_gl_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, len(points) * 64, None, gl.GL_DYNAMIC_DRAW)
        instancer._instance_transform_cuda_buffer = wp.RegisteredGLBuffer(
            int(instancer.instance_transform_gl_buffer.value),
            self._device,
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            fallback_to_copy=False,
        )
        instancer.update_colors(color, color)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, instancer.instance_transform_gl_buffer)
        for column in range(4):
            gl.glVertexAttribPointer(
                3 + column, 4, gl.GL_FLOAT, gl.GL_FALSE, 64, ctypes.c_void_p(column * 16)
            )
            gl.glEnableVertexAttribArray(3 + column)
            gl.glVertexAttribDivisor(3 + column, 1)
        gl.glBindVertexArray(0)

    def _register_billboard_color_resources(self, name, instancer):
        if instancer._instance_transform_cuda_buffer.resource is None:
            raise RuntimeError("PBF rendering requires CUDA/OpenGL interop without CPU fallback")
        self._billboard_color_resources[name] = tuple(
            wp.RegisteredGLBuffer(
                int(buffer.value),
                self._device,
                flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                fallback_to_copy=False,
            )
            for buffer in (instancer.instance_color1_buffer, instancer.instance_color2_buffer)
        )

    def _mesh(self, vertices, faces):
        gl = self.gl
        triangles = np.asarray(vertices)[faces]
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        data = (
            np.concatenate((triangles, np.repeat(normal[:, None, :], 3, axis=1)), axis=2)
            .reshape(-1, 6)
            .astype(np.float32)
        )
        vao, vbo = gl.GLuint(), gl.GLuint()
        gl.glGenVertexArrays(1, vao)
        gl.glBindVertexArray(vao)
        gl.glGenBuffers(1, vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data.ctypes.data, gl.GL_STATIC_DRAW)
        for index in (0, 1):
            gl.glEnableVertexAttribArray(index)
            gl.glVertexAttribPointer(
                index, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(index * 12)
            )
        gl.glBindVertexArray(0)
        return vao, vbo, len(data)

    def update_rigid_scene(self, simulation):
        if not self.mesh_buffers:
            self.visible_bodies = [
                i for i, body in enumerate(simulation.body_models) if not body.wall
            ]
            for i in self.visible_bodies:
                body = simulation.body_models[i]
                self.mesh_buffers.append(self._mesh(body.vertices, body.faces))
                self.colors.append(body.color)
            container_min, container_max = simulation.container_min, simulation.container_max
            # A visible receiving floor; walls stay in the physics model.
            floor = [
                [container_min[0], container_min[1] - 0.003, container_min[2]],
                [container_max[0], container_min[1] - 0.003, container_min[2]],
                [container_max[0], container_min[1] - 0.003, container_max[2]],
                [container_min[0], container_min[1] - 0.003, container_max[2]],
            ]
            self.mesh_buffers.append(self._mesh(floor, np.array([[0, 2, 1], [0, 3, 2]])))
            self.colors.append((0.25, 0.30, 0.36))
            # Each mesh draws one instance using its own matrix in this shared VBO.
            gl = self.gl
            self.pose_buffer = gl.GLuint()
            gl.glGenBuffers(1, self.pose_buffer)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pose_buffer)
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER, 64 * len(self.mesh_buffers), None, gl.GL_DYNAMIC_DRAW
            )
            for index, (vao, _vbo, _count) in enumerate(self.mesh_buffers):
                gl.glBindVertexArray(vao)
                for column in range(4):
                    location = 2 + column
                    gl.glEnableVertexAttribArray(location)
                    gl.glVertexAttribPointer(
                        location,
                        4,
                        gl.GL_FLOAT,
                        gl.GL_FALSE,
                        64,
                        ctypes.c_void_p(index * 64 + column * 16),
                    )
                    gl.glVertexAttribDivisor(location, 1)
            gl.glBindVertexArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            self.pose_resource = wp.RegisteredGLBuffer(
                int(self.pose_buffer.value),
                simulation.device,
                flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                fallback_to_copy=False,
            )
            self.pose_body_ids = wp.array(
                self.visible_bodies + [-1], dtype=int, device=simulation.device
            )
        columns = self.pose_resource.map(dtype=wp.vec4, shape=(4 * len(self.mesh_buffers),))
        try:
            wp.launch(
                write_rigid_matrices,
                len(self.mesh_buffers),
                [simulation.rigid.position, simulation.rigid.rotation, self.pose_body_ids, columns],
                device=simulation.device,
            )
        finally:
            self.pose_resource.unmap()

    def _matrix(self, program, name, matrix, column_major=False):
        gl = self.gl
        data = np.ascontiguousarray(matrix if column_major else matrix.T, dtype=np.float32)
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program.id, name.encode()),
            1,
            gl.GL_FALSE,
            data.ctypes.data_as(ctypes.POINTER(gl.GLfloat)),
        )

    def _draw(self):
        if self.mesh_buffers:
            gl = self.gl
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.shadow_fbo)
            gl.glViewport(0, 0, self.shadow_size, self.shadow_size)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glUseProgram(self.depth_program.id)
            self._matrix(self.depth_program, "light_space", self.light_space)
            for vao, _vbo, count in self.mesh_buffers[:-1]:
                gl.glBindVertexArray(vao)
                gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, count, 1)
            gl.glBindVertexArray(0)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glViewport(0, 0, self.screen_width, self.screen_height)
        super()._draw()

    def _draw_rigids(self):
        gl, program = self.gl, self.mesh_program
        gl.glUseProgram(program.id)
        gl.glDisable(gl.GL_CULL_FACE)
        self._matrix(program, "view", self._view_matrix, column_major=True)
        self._matrix(program, "projection", self._projection_matrix, column_major=True)
        self._matrix(program, "light_space", self.light_space)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.shadow_texture)
        gl.glUniform1i(gl.glGetUniformLocation(program.id, b"shadow_map"), 0)
        gl.glUniform1i(gl.glGetUniformLocation(program.id, b"shadows"), int(self.shadows))
        for (vao, _vbo, count), color in zip(self.mesh_buffers, self.colors):
            gl.glUniform3f(gl.glGetUniformLocation(program.id, b"color"), *color)
            gl.glBindVertexArray(vao)
            gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, count, 1)
        gl.glBindVertexArray(0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def screenshot(self, path):
        """Read our framebuffer with Pyglet, without optional image packages."""
        import pyglet

        gl = self.gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)
        pixels = (gl.GLubyte * (self.screen_width * self.screen_height * 3))()
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glReadPixels(
            0, 0, self.screen_width, self.screen_height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixels
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as stream:
            pyglet.image.ImageData(
                self.screen_width, self.screen_height, "RGB", bytes(pixels)
            ).save(str(path), file=stream)

    def close(self):
        if getattr(self, "_coupling_closed", False):
            return
        self._coupling_closed = True
        if hasattr(self, "mesh_buffers"):
            gl = self.gl
            self._switch_context()
            for vao, vbo, _ in self.mesh_buffers:
                gl.glDeleteVertexArrays(1, vao)
                gl.glDeleteBuffers(1, vbo)
            self.mesh_buffers.clear()
            # Release CUDA registration before deleting its OpenGL storage.
            self.pose_resource = None
            if self.pose_buffer is not None:
                gl.glDeleteBuffers(1, self.pose_buffer)
                self.pose_buffer = None
            gl.glDeleteTextures(1, self.shadow_texture)
            gl.glDeleteFramebuffers(1, self.shadow_fbo)
            self.mesh_program.delete()
            self.depth_program.delete()
        super().close()


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def nonnegative_float(value):
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_simulation_arguments(parser)
    parser.add_argument("--num-frames", type=nonnegative_int, default=0)
    parser.add_argument("--playback-speed", type=positive_float, default=0.5)
    parser.add_argument(
        "--headless", action="store_true", help="Run physics without constructing OpenGL"
    )
    parser.add_argument(
        "--hidden", action="store_true", help="Hidden GL window for screenshot verification"
    )
    parser.add_argument("--seconds", type=positive_float, default=10)
    parser.add_argument(
        "--warmup",
        type=nonnegative_float,
        default=0,
        help="Simulate this many seconds before the first frame",
    )
    parser.add_argument("--screenshot", help="Save the final frame as PNG")
    args = parser.parse_args()
    if args.hidden and not args.num_frames:
        args.num_frames = 1
    config = config_from_args(args)
    if args.headless:
        run_headless(config, args.device, args.seconds, verbose=args.verbose)
        return
    simulation = create_pbf2way_simulation(config=config, device=args.device, verbose=args.verbose)
    while simulation.sim_time < args.warmup:
        simulation.step()
    renderer = CouplingRenderer(device=simulation.device, hidden=args.hidden)
    simulation.renderer = renderer
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
    frame = 0
    try:
        while renderer.is_running() and (not args.num_frames or frame < args.num_frames):
            if reset_requested:
                simulation = create_pbf2way_simulation(
                    config=config, device=args.device, verbose=args.verbose
                )
                simulation.renderer = renderer
                reset_requested = False
                renderer.paused = True
            started = time.perf_counter()
            advance_and_render(simulation, renderer)
            frame += 1
            pace_frame(started, simulation.frame_dt, args.playback_speed)
        if args.screenshot:
            renderer.screenshot(args.screenshot)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
