"""PBF controls, reset invariants and the signed density constraint."""
import contextlib
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from demos.fluid.PBF import PBF as solver
from demos.fluid.PBF import render_opengl as frontend
from demos.fluid.PBF.pbf_helper import PBFConfig
from demos.fluid.PBF.simulation import reset_simulation, reverse_gravity
from tests.fluid.pbf.solver_fixture import bind_solver, self_neighbor


CONFIG = PBFConfig(
    particle_radius=0.05, container_size=(1.0, 1.0, 1.0),
    block_start=(0.1, 0.1, 0.1), block_end=(0.5, 0.5, 0.5),
)


class KeyboardControlsTest(unittest.TestCase):
    def setUp(self):
        import pyglet

        self.key = pyglet.window.key
        self.handled = pyglet.event.EVENT_HANDLED
        self.config = replace(CONFIG, frame_dt=0.02, substeps=4)
        self.sim = SimpleNamespace(
            config=self.config, substep=self.config.substeps,
            frame_dt=self.config.frame_dt, dt=self.config.frame_dt / self.config.substeps,
            gravity=wp.vec3(*self.config.gravity), sim_time=2.0,
        )
        self.renderer = SimpleNamespace(
            paused=False, draw_grid=True, window=Mock(), register_key_press_callback=Mock(),
        )
        self.reset = Mock()
        self.press = frontend.register_keyboard_controls(self.renderer, self.sim, on_reset=self.reset)
        handlers = self.renderer.window.push_handlers.call_args.kwargs
        self.release = handlers['on_key_release']
        self.deactivate = handlers['on_deactivate']

    def tap(self, symbol):
        self.assertEqual(self.press(symbol, 0), self.handled)
        self.assertIsNone(self.release(symbol, 0))

    def test_direction_selection_repeat_switch_and_reversal(self):
        strength = float(np.linalg.norm(self.config.gravity))
        left, right, down, up = (
            wp.vec3(-strength, 0, 0), wp.vec3(strength, 0, 0),
            wp.vec3(0, -strength, 0), wp.vec3(0, strength, 0),
        )
        for symbol, expected in (
            (self.key.Q, left), (self.key.Q, down), (self.key.Q, left),
            (self.key.E, right), (self.key.Q, left), (self.key.E, right),
            (self.key.E, down), (self.key.G, up), (self.key.G, down),
            (self.key.Q, left), (self.key.G, right), (self.key.Q, left),
            (self.key.E, right), (self.key.G, left), (self.key.E, right),
        ):
            with self.subTest(key=symbol, expected=tuple(expected)):
                self.tap(symbol)
                self.assertEqual(tuple(self.sim.gravity), tuple(expected))
                self.assertEqual(self.sim.frame_dt, self.config.frame_dt)
                self.assertFalse(self.renderer.paused)
        self.assertEqual(self.sim.sim_time, 2.0)
        self.reset.assert_not_called()

    def test_direction_repeat_guards_focus_loss_and_slow_motion(self):
        self.renderer.paused = True
        for symbol in (self.key.Q, self.key.E):
            self.press(symbol, 0)
            horizontal = tuple(self.sim.gravity)
            for _ in range(3):
                self.assertEqual(self.press(symbol, 0), self.handled)
                self.assertEqual(tuple(self.sim.gravity), horizontal)
            self.tap(self.key.X)
            self.assertEqual(tuple(self.sim.gravity), horizontal)
            self.assertIsNone(self.deactivate())
            self.tap(symbol)
            self.assertEqual(tuple(self.sim.gravity), tuple(wp.vec3(*self.config.gravity)))
        self.assertEqual(self.sim.frame_dt, self.config.frame_dt)
        self.assertEqual(self.sim.sim_time, 2.0)
        self.assertTrue(self.renderer.paused)

    def test_direction_uses_configured_magnitude_and_reset_clears_selection(self):
        self.sim.config = replace(self.config, gravity=(3.0, -4.0, 0.0))
        self.press = frontend.register_keyboard_controls(self.renderer, self.sim, on_reset=self.reset)
        self.release = self.renderer.window.push_handlers.call_args.kwargs['on_key_release']
        for symbol, expected in ((self.key.Q, (-5.0, 0.0, 0.0)),
                                 (self.key.Q, (0.0, -5.0, 0.0)),
                                 (self.key.E, (5.0, 0.0, 0.0))):
            self.tap(symbol)
            self.assertEqual(tuple(self.sim.gravity), expected)
        self.tap(self.key.R)
        self.reset.assert_called_once_with()
        self.sim.gravity = wp.vec3(*self.sim.config.gravity)  # Deferred reset in main().
        self.tap(self.key.E)
        self.assertEqual(tuple(self.sim.gravity), (5.0, 0.0, 0.0))
        self.assertEqual(self.sim.config.gravity, (3.0, -4.0, 0.0))
        self.assertTrue(self.renderer.paused)

    def test_half_speed_toggle_and_gravity_ignore_key_repeat(self):
        original = asdict(self.config)
        gravity = tuple(self.sim.gravity)
        for _ in range(3):
            self.assertEqual(self.press(self.key.X, 0), self.handled)
        self.assertEqual(self.sim.frame_dt, self.config.frame_dt * 0.5)
        self.assertEqual(self.sim.dt, self.config.frame_dt * 0.5 / self.sim.substep)
        for _ in range(3):
            self.press(self.key.G, 0)
        self.assertEqual(tuple(self.sim.gravity), tuple(-value for value in gravity))
        self.assertEqual(self.sim.frame_dt, self.config.frame_dt * 0.5)
        self.assertIsNone(self.release(self.key.G, 0))
        self.press(self.key.G, 0)
        self.assertEqual(tuple(self.sim.gravity), gravity)
        for scale in (1.0, 0.5, 1.0, 0.5, 1.0):
            self.assertIsNone(self.release(self.key.X, 0))
            self.press(self.key.X, 0)
            self.assertEqual(self.sim.frame_dt, self.config.frame_dt * scale)
            self.assertEqual(self.sim.dt, self.sim.frame_dt / self.sim.substep)
        self.assertEqual(asdict(self.config), original)
        self.assertEqual(self.sim.sim_time, 2.0)
        self.reset.assert_not_called()

    def test_paused_toggle_and_deferred_reset_restore_normal_speed(self):
        self.renderer.paused = True
        self.press(self.key.X, 0)
        self.assertTrue(self.renderer.paused)
        self.assertEqual(self.sim.sim_time, 2.0)
        self.reset.side_effect = lambda: self.assertEqual(
            (self.renderer.paused, self.sim.frame_dt, self.sim.dt),
            (True, self.config.frame_dt, self.config.frame_dt / self.sim.substep),
        )
        self.press(self.key.R, 0)
        self.reset.assert_called_once_with()
        self.assertEqual(self.sim.sim_time, 2.0)  # Actual reset remains deferred to main().
        self.release(self.key.X, 0)
        self.press(self.key.X, 0)
        self.assertEqual(self.sim.frame_dt, self.config.frame_dt * 0.5)
        self.assertTrue(self.renderer.paused)

    def test_focus_loss_clears_repeat_guards_and_other_keys_pass_through(self):
        gravity = tuple(self.sim.gravity)
        self.press(self.key.X, 0)
        self.press(self.key.G, 0)
        self.assertIsNone(self.deactivate())
        self.press(self.key.X, 0)
        self.press(self.key.G, 0)
        self.assertEqual(self.sim.frame_dt, self.config.frame_dt)
        self.assertEqual(tuple(self.sim.gravity), gravity)
        self.press(self.key.P, 0)
        self.assertFalse(self.renderer.draw_grid)
        for symbol in (self.key.SPACE, self.key.DOWN, self.key.ESCAPE, self.key.S):
            self.assertIsNone(self.press(symbol, 0))


class ControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def make_simulation(self, config=CONFIG):
        if self.device == "cpu":
            self.skipTest("The solver's NanoVDB boundary requires CUDA")
        with wp.ScopedDevice(self.device):
            # Keep controls tests independent of the scene while providing a valid SDF id.
            empty_sdf = wp.Volume.allocate([0, 0, 0], [0, 0, 0], 1.0, bg_value=10.0,
                                           translation=(100.0, 100.0, 100.0), device=self.device)
            with patch.object(solver.boundary_model, "load_static_sdf", return_value=empty_sdf):
                return solver.PBF(stage_path=None, config=config)

    def step(self, sim, count=1):
        bind_solver(sim.config, sim.grid, sim.static_sdf, sim.volume_map)
        with wp.ScopedDevice(self.device), contextlib.redirect_stdout(io.StringIO()):
            for _ in range(count):
                sim.step()

    def test_cli_and_boolean_validation(self):
        for argv, expected in (([], True), (["--clamp-negative-pressure"], True),
                               (["--no-clamp-negative-pressure"], False)):
            args = frontend.parse_args(argv)
            self.assertEqual((args.config or PBFConfig()).clamp_negative_pressure, expected)
        args = frontend.parse_args(["--particle-radius", "0.01", "--no-clamp-negative-pressure"])
        self.assertEqual(args.config.particle_radius, 0.01)
        self.assertFalse(args.config.clamp_negative_pressure)
        with self.assertRaisesRegex(ValueError, "must be a bool"):
            PBFConfig(clamp_negative_pressure=1)

    def test_gravity_reversal_and_reset_trajectory(self):
        sim = self.make_simulation()
        initial = sim.pos.numpy()
        gravity = tuple(sim.gravity)
        buffer_names = ("pos", "v", "pre_Pos", "delta_Pos", "lambda_Opt",
                        "delta_Vel", "curl", "volume_Rho", "volume_Grad", "temp_pos", "temp_pPos")
        allocation_pool = {getattr(sim, name).ptr for name in buffer_names}
        grid, sdf = sim.grid, sim.static_sdf
        sim.renderer = renderer = object()
        reverse_gravity(sim)
        self.assertEqual(tuple(sim.gravity), tuple(-value for value in gravity))
        np.testing.assert_array_equal(sim.pos.numpy(), initial)
        np.testing.assert_array_equal(sim.v.numpy(), 0)
        reverse_gravity(sim)
        self.assertEqual(tuple(sim.gravity), gravity)
        self.step(sim, 5)
        expected_pos, expected_vel = sim.pos.numpy(), sim.v.numpy()
        reverse_gravity(sim)
        np.testing.assert_array_equal(sim.pos.numpy(), expected_pos)
        np.testing.assert_array_equal(sim.v.numpy(), expected_vel)
        # Steps may swap position buffers; reset must preserve the current bindings.
        pointers = tuple(getattr(sim, name).ptr for name in buffer_names)
        with wp.ScopedDevice("cpu"), \
                patch.object(solver.boundary_model, "load_static_sdf",
                             side_effect=AssertionError("Reset reloaded SDF")):
            reset_simulation(sim)
            reset_simulation(sim)
            self.assertEqual(wp.get_device(), wp.get_device("cpu"))
        self.assertIs(sim.config, CONFIG)
        self.assertEqual(sim.sim_time, 0)
        self.assertEqual(tuple(sim.gravity), gravity)
        self.assertEqual(tuple(getattr(sim, name).ptr for name in buffer_names), pointers)
        self.assertEqual({getattr(sim, name).ptr for name in buffer_names}, allocation_pool)
        self.assertIs(sim.grid, grid)
        self.assertIs(sim.static_sdf, sdf)
        self.assertIs(sim.renderer, renderer)
        np.testing.assert_array_equal(sim.pos.numpy(), initial)
        for name in ("v", "pre_Pos", "delta_Pos", "lambda_Opt", "delta_Vel", "curl", "volume_Rho", "volume_Grad"):
            np.testing.assert_array_equal(getattr(sim, name).numpy(), 0)
        np.testing.assert_array_equal(sim.n_Cache.NeighborCount.numpy(), 0)
        self.step(sim, 5)
        np.testing.assert_array_equal(sim.pos.numpy(), expected_pos)
        np.testing.assert_array_equal(sim.v.numpy(), expected_vel)
        fresh = self.make_simulation()
        self.step(fresh, 5)
        np.testing.assert_array_equal(fresh.pos.numpy(), expected_pos)
        np.testing.assert_array_equal(fresh.v.numpy(), expected_vel)

    def test_reversed_gravity_affects_prediction(self):
        sim = self.make_simulation()
        before = sim.pos.numpy()
        reverse_gravity(sim)
        self.step(sim)
        self.assertGreater(float((sim.pos.numpy() - before)[:, 1].mean()), 0)

    def test_keyboard_horizontal_gravity_affects_prediction(self):
        import pyglet

        for symbol, direction in ((pyglet.window.key.Q, -1), (pyglet.window.key.E, 1)):
            with self.subTest(key=symbol):
                sim = self.make_simulation()
                before = sim.pos.numpy()
                renderer = SimpleNamespace(window=Mock(), register_key_press_callback=Mock())
                press = frontend.register_keyboard_controls(renderer, sim, on_reset=Mock())
                press(symbol, 0)
                self.step(sim)
                displacement = (sim.pos.numpy() - before).mean(axis=0)
                self.assertGreater(float(displacement[0]) * direction, 0)
                self.assertAlmostEqual(float(displacement[1]), 0, places=6)
                self.assertAlmostEqual(sim.sim_time, CONFIG.frame_dt)

    def test_reset_preserves_imported_startup_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "startup.json"
            path.write_text(json.dumps(asdict(CONFIG)), encoding="utf-8")
            config = frontend.parse_args(["--config", str(path)]).config
        sim = self.make_simulation(config)
        initial = sim.pos.numpy()
        self.step(sim, 3)
        expected_pos, expected_vel = sim.pos.numpy(), sim.v.numpy()
        reverse_gravity(sim)
        reset_simulation(sim)
        self.assertIs(sim.config, config)
        self.assertEqual(sim.config, CONFIG)
        self.assertEqual(sim.sim_time, 0)
        self.assertEqual(tuple(sim.gravity), tuple(wp.vec3(*config.gravity)))
        np.testing.assert_array_equal(sim.pos.numpy(), initial)
        self.step(sim, 3)
        np.testing.assert_array_equal(sim.pos.numpy(), expected_pos)
        np.testing.assert_array_equal(sim.v.numpy(), expected_vel)

    def test_clamp_preserves_zero_and_signed_constraint(self):
        sim = self.make_simulation()
        with wp.ScopedDevice(self.device):
            pos = wp.array([[0.4, 0.4, 0.4]], dtype=wp.vec3)
            grid = wp.HashGrid(8, 8, 8)
            grid.build(pos, sim.support_radius)
            result = wp.zeros(1, dtype=float)
            volume_rho = wp.zeros(1, dtype=float)
            volume_grad = wp.zeros(1, dtype=wp.vec3)
            neighbors = self_neighbor(self.device)
            values = []
            for clamp in (1, 0):
                bind_solver(replace(CONFIG, clamp_negative_pressure=bool(clamp)),
                            grid, sim.static_sdf, sim.volume_map)
                wp.launch(solver.volume_query, 1,
                          inputs=[pos], outputs=[volume_rho, volume_grad])
                wp.launch(solver.calc_lambda, 1,
                          inputs=[pos, neighbors, volume_rho, volume_grad], outputs=[result])
                values.append(float(result.numpy()[0]))
        self.assertEqual(values[0], 0)
        self.assertGreater(values[1], 0)
        volume = 0.8 * (2 * CONFIG.particle_radius)**3
        self_density = volume * 315 / (64 * np.pi * sim.support_radius**3)
        epsilon = CONFIG.lambda_regularization / sim.support_radius**2
        self.assertAlmostEqual(values[1], (1 - self_density) / epsilon, places=7)
        explicit = self.make_simulation(replace(CONFIG, clamp_negative_pressure=True))
        self.step(explicit, 3)
        default = self.make_simulation()
        self.step(default, 3)
        np.testing.assert_array_equal(default.pos.numpy(), explicit.pos.numpy())
        np.testing.assert_array_equal(default.v.numpy(), explicit.v.numpy())

    def test_main_loop_reset_while_running_and_paused(self):
        import pyglet
        key = pyglet.window.key
        sim = self.make_simulation()
        bind_solver(sim.config, sim.grid, sim.static_sdf, sim.volume_map)
        initial = sim.pos.numpy()
        events = (None, key.G, key.X, key.R, key.R, key.X, key.SPACE,
                  key.X, key.SPACE, None, key.P)
        states = []

        class Renderer:
            paused = False
            draw_grid = True
            frames = 0
            meshes = 0
            closed = False
            window = Mock()

            def render_mesh(self, **kwargs):
                self.meshes += 1

            def register_key_press_callback(self, callback):
                self.callback = callback

            def is_running(self):
                return self.frames < len(events)

            def begin_frame(self, time):
                states.append((time, self.paused, sim.pos.numpy(), tuple(sim.gravity), sim.frame_dt))

            def render_billboards(self, **kwargs):
                pass

            def end_frame(self):
                event = events[self.frames]
                if event is not None:
                    handled = self.callback(event, 0)
                    if handled != pyglet.event.EVENT_HANDLED and event == key.SPACE:
                        self.paused = not self.paused
                    self.window.push_handlers.call_args.kwargs['on_key_release'](event, 0)
                self.frames += 1

            def close(self):
                self.closed = True

        renderer = Renderer()
        args = frontend.parse_args([])
        with wp.ScopedDevice(self.device), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(frontend, 'parse_args', return_value=args), \
                patch.object(frontend, 'load_static_scene', return_value=(np.zeros((3, 3)), [[0, 1, 2]])), \
                patch.object(frontend, 'create_pbf_simulation', return_value=sim), \
                patch.object(frontend, 'create_pbf_renderer', return_value=renderer), \
                patch.object(wp, 'launch', wraps=wp.launch) as launching:
            frontend.main()
        for frame in (4, 5, 6):
            self.assertEqual(states[frame][0], 0)
            self.assertTrue(states[frame][1])
            np.testing.assert_array_equal(states[frame][2], initial)
            self.assertEqual(states[frame][3], tuple(wp.vec3(*CONFIG.gravity)))
        self.assertAlmostEqual(states[3][0], CONFIG.frame_dt * 2.5)
        self.assertEqual(states[4][4], CONFIG.frame_dt)
        self.assertEqual(states[6][4], CONFIG.frame_dt * 0.5)
        self.assertEqual(states[7][0], CONFIG.frame_dt * 0.5)
        self.assertEqual(states[8][0], CONFIG.frame_dt * 1.5)
        self.assertEqual(states[9][0], states[8][0])
        self.assertEqual(states[10][0], states[8][0])
        expected_dt = [
            CONFIG.frame_dt * scale / CONFIG.substeps
            for scale in (1.0, 1.0, 0.5, 0.5, 1.0)
            for _ in range(CONFIG.substeps)
        ]
        for kernel, argument in ((solver.apply_predict, 1), (solver.update_position, 0),
                                 (solver.calc_visvor, 0)):
            actual_dt = [
                call.kwargs['inputs'][argument] for call in launching.call_args_list
                if call.kwargs.get('kernel') is kernel
            ]
            self.assertEqual(actual_dt, expected_dt)
        self.assertFalse(renderer.draw_grid)
        self.assertEqual(renderer.meshes, 1)
        self.assertTrue(renderer.closed)


if __name__ == '__main__':
    unittest.main()
