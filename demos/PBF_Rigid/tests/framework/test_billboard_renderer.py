"""CUDA/OpenGL regressions for camera motion and repeated billboard draws."""

import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import warp as wp

from demos.PBF_Rigid.billboard_renderer import BillboardRenderer
from demos.PBF_Rigid.render_opengl import PBFRenderer, register_keyboard_controls


class BillboardCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA/OpenGL interop is required")
        cls.device = wp.get_device("cuda:0")

    def make_renderer(self, renderer_type, **kwargs):
        renderer = renderer_type(
            title="Billboard camera regression", device=self.device, headless=True,
            scaling=1.0, up_axis="Y", camera_pos=(0, 0, 4), camera_front=(0, 0, -1),
            screen_width=320, screen_height=240, near_plane=0.1, far_plane=20,
            draw_grid=False, draw_sky=False, draw_axis=False, vsync=False,
            background_color=(0, 0, 0),
            **kwargs,
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

    def test_pbf_particle_count_is_optional_and_snapshots_length_once(self):
        class PositionsMetadata:
            def __init__(self, count):
                self.count = count
                self.reads = 0

            def __len__(self):
                self.reads += 1
                return self.count

        renderer, _, _ = self.make_renderer(PBFRenderer)
        self.assertIsNone(renderer._particle_count_label)
        self.assertNotIn(renderer._draw_particle_count, renderer.render_2d_callbacks)

        for count in (0, 125000):
            with self.subTest(count=count):
                metadata = PositionsMetadata(count)
                reference = weakref.ref(metadata)
                renderer, positions, velocities = self.make_renderer(
                    PBFRenderer, particle_positions=metadata,
                )
                expected = f"Particles: {count:,}"
                self.assertEqual(renderer._particle_count_label.text, expected)
                metadata.count = count + 1
                renderer.paused = True
                renderer._draw()
                renderer.render_billboards("points", positions, velocities, 0.4, 2.0, 7.5)
                renderer._draw()
                self.assertEqual(renderer._particle_count_label.text, expected)
                self.assertEqual(metadata.reads, 1)
                del metadata
                self.assertIsNone(reference())

    def test_pbf_particle_count_overlay_tracks_info_layout_and_visibility(self):
        from pyglet.window import key

        positions = wp.zeros(27, dtype=wp.vec3, device=self.device)
        renderer, _, _ = self.make_renderer(PBFRenderer, particle_positions=positions)
        label = renderer._particle_count_label
        normal_y = label.y
        renderer.paused = True
        renderer._draw()
        self.assertLess(label.y, normal_y)
        for height in (360, 480):
            renderer.window.set_size(480, height)
            renderer._window_resize_callback(480, height)
            renderer._draw()
            self.assertEqual(label.x, renderer._info_label.x)
            self.assertEqual(label.y, renderer._info_label.y - renderer._info_label.content_height - 4)
            self.assertGreater(label.y - label.content_height, 0)
            self.assertEqual(label.text, "Particles: 27")
            self.assertEqual(renderer.gl.glGetError(), renderer.gl.GL_NO_ERROR)

        with patch.object(label, 'draw', wraps=label.draw) as draw_label:
            renderer._key_press_callback(key.I, 0)
            renderer._draw()
            self.assertFalse(renderer.show_info)
            draw_label.assert_not_called()
            renderer._key_press_callback(key.I, 0)
            renderer._draw()
            self.assertTrue(renderer.show_info)
            draw_label.assert_called_once_with()
        self.assertTrue(renderer.gl.glIsEnabled(renderer.gl.GL_DEPTH_TEST))

    def test_static_particle_count_does_not_query_depth_or_update_positions(self):
        positions = wp.zeros(27, dtype=wp.vec3, device=self.device)
        renderer, _, _ = self.make_renderer(PBFRenderer, particle_positions=positions)
        label = renderer._particle_count_label
        with patch.object(label, '_update_translation', wraps=label._update_translation) as translate, \
                patch.object(renderer.gl, 'glIsEnabled', side_effect=AssertionError('Per-frame GL query')):
            for _ in range(3):
                renderer._draw_particle_count()
            translate.assert_not_called()
            renderer.paused = True
            renderer._draw()
            translate.assert_called_once_with()
        with patch.object(label, 'draw', side_effect=RuntimeError('Text draw failed')):
            with self.assertRaisesRegex(RuntimeError, 'Text draw failed'):
                renderer._draw_particle_count()
        self.assertTrue(renderer.gl.glIsEnabled(renderer.gl.GL_DEPTH_TEST))

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

    def test_pbf_s_and_down_move_back_x_preserves_other_camera_controls(self):
        from pyglet.window import key

        renderer, _, _ = self.make_renderer(PBFRenderer)
        renderer._key_handler = key.KeyStateHandler()
        renderer.headless = False
        renderer._frame_speed = 5.0
        start = np.array(renderer.camera_pos)
        front = np.array(renderer.camera_front)
        distance = renderer._camera_speed * renderer._frame_speed

        renderer._key_handler.on_key_press(key.S, 0)
        renderer._draw()
        np.testing.assert_allclose(renderer.camera_pos, start - front * distance, atol=1e-6)
        self.assertTrue(renderer._key_handler[key.S])
        renderer._key_handler.on_key_release(key.S, 0)

        renderer._key_handler.on_key_press(key.DOWN, 0)
        renderer._draw()
        np.testing.assert_allclose(renderer.camera_pos, start - front * distance * 2, atol=1e-6)
        renderer._key_handler.on_key_release(key.DOWN, 0)

        sim = SimpleNamespace(
            config=SimpleNamespace(frame_dt=0.02, gravity=(0.0, -9.81, 0.0)),
            substep=4, frame_dt=0.02, dt=0.005,
        )
        register_keyboard_controls(renderer, sim, on_reset=lambda: None)
        wireframe = renderer.render_wireframe
        for _ in range(3):
            renderer._key_press_callback(key.X, 0)
        self.assertEqual(sim.frame_dt, 0.01)
        self.assertEqual(renderer.render_wireframe, wireframe)
        renderer._key_handler.on_key_press(key.X, 0)
        renderer._draw()
        np.testing.assert_allclose(renderer.camera_pos, start - front * distance * 2, atol=1e-6)

        renderer._key_handler.on_key_press(key.W, 0)
        renderer._draw()
        np.testing.assert_allclose(renderer.camera_pos, start - front * distance, atol=1e-6)
        self.assert_camera_matches(renderer)
        renderer._key_handler.on_key_release(key.W, 0)

        renderer._key_handler.on_key_press(key.D, 0)
        renderer._draw()
        np.testing.assert_allclose(renderer.camera_pos, start - front * distance + [distance, 0, 0], atol=1e-6)
        self.assertTrue(renderer._key_handler[key.X])
        self.assert_camera_matches(renderer)


if __name__ == "__main__":
    unittest.main()
