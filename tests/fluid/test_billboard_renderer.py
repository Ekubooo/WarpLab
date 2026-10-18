"""CUDA/OpenGL regressions for camera motion and repeated billboard draws."""

import ctypes
import unittest

import numpy as np
import warp as wp

from demos.fluid.common.billboard_renderer import BillboardRenderer
from demos.fluid.PBF.render_opengl import PBFRenderer


class BillboardCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA/OpenGL interop is required")
        cls.device = wp.get_device("cuda:0")

    def make_renderer(self, renderer_type):
        renderer = renderer_type(
            title="Billboard camera regression", device=self.device, headless=True,
            scaling=1.0, up_axis="Y", camera_pos=(0, 0, 4), camera_front=(0, 0, -1),
            screen_width=320, screen_height=240, near_plane=0.1, far_plane=20,
            draw_grid=False, draw_sky=False, draw_axis=False, vsync=False,
            background_color=(0, 0, 0),
        )
        self.addCleanup(self.close_renderer, renderer)
        positions = wp.array([[0, 0, 0]], dtype=wp.vec3, device=self.device)
        velocities = wp.zeros(1, dtype=wp.vec3, device=self.device)
        renderer.begin_frame(0)
        renderer.render_billboards("points", positions, velocities, 0.4, 2.0, 7.5)
        renderer.end_frame()
        return renderer, positions, velocities

    @staticmethod
    def close_renderer(renderer):
        # Destroy instancers with their GL context still current.
        renderer.headless = True
        renderer._switch_context()
        renderer._shape_instancers.clear()
        renderer.close()

    @staticmethod
    def uniform(renderer, program, location):
        values = (ctypes.c_float * 16)()
        renderer.gl.glGetUniformfv(program.id, location, values)
        return np.ctypeslib.as_array(values).copy()

    def assert_camera_matches(self, renderer):
        billboard = self.uniform(renderer, renderer._billboard_shader, renderer._loc_billboard_view)
        mesh = self.uniform(renderer, renderer._shape_shader, renderer._loc_shape_view)
        np.testing.assert_allclose(billboard, mesh, atol=1e-7)
        np.testing.assert_allclose(billboard, renderer._view_matrix, atol=1e-7)
        projection = self.uniform(renderer, renderer._billboard_shader, renderer._loc_billboard_projection)
        mesh_projection = self.uniform(renderer, renderer._shape_shader, renderer._loc_shape_projection)
        np.testing.assert_allclose(projection, mesh_projection, atol=1e-7)

    def pixels(self, renderer):
        pixels = wp.empty((240, 320, 3), dtype=wp.uint8, device=self.device)
        self.assertTrue(renderer.get_pixels(pixels, split_up_tiles=False, use_uint8=True))
        return pixels.numpy()

    def test_key_motion_updates_once_and_aligns_both_shaders(self):
        from pyglet.window import key

        for renderer_type in (BillboardRenderer, PBFRenderer):
            with self.subTest(renderer=renderer_type.__name__):
                renderer, _, _ = self.make_renderer(renderer_type)
                # Exercise the actual key-hold draw path while keeping the window hidden.
                renderer._key_handler = key.KeyStateHandler()
                renderer.headless = False
                renderer._frame_speed = 5.0
                renderer._key_handler.on_key_press(key.D, 0)
                start = np.array(renderer.camera_pos)
                for draw in range(1, 4):
                    renderer._draw()
                    expected = start + [draw * renderer._camera_speed * 5.0, 0, 0]
                    np.testing.assert_allclose(renderer.camera_pos, expected, atol=1e-6)
                    self.assert_camera_matches(renderer)
                renderer._key_handler.on_key_release(key.D, 0)
                renderer._draw()
                np.testing.assert_allclose(renderer.camera_pos, expected, atol=1e-6)
                self.assert_camera_matches(renderer)

    def test_camera_rotation_after_submission_matches_fresh_submission(self):
        for renderer_type in (BillboardRenderer, PBFRenderer):
            with self.subTest(renderer=renderer_type.__name__):
                renderer, positions, velocities = self.make_renderer(renderer_type)
                renderer.paused = True
                renderer.camera_pos = (2, 1, 4)
                renderer.camera_front = (-2, -1, -4)
                renderer.camera_fov = 60
                renderer.update_projection_matrix()
                # A paused redraw must use the latest camera without resubmitting particles.
                renderer._draw()
                self.assert_camera_matches(renderer)
                redrawn = self.pixels(renderer)
                self.assertGreater(np.count_nonzero(redrawn), 100)
                rows, columns = np.where(np.any(redrawn != 0, axis=2))
                # A camera-facing square stays rectangular in screen space.
                self.assertEqual(len(rows), (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1))
                renderer.render_billboards("points", positions, velocities, 0.4, 2.0, 7.5)
                renderer._draw()
                np.testing.assert_array_equal(self.pixels(renderer), redrawn)
                self.assertEqual(renderer.gl.glGetError(), renderer.gl.GL_NO_ERROR)


if __name__ == "__main__":
    unittest.main()
