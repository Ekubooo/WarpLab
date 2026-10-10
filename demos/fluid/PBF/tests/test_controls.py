"""PBF controls, reset invariants and the signed density constraint."""
import contextlib
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from .. import PBF as solver
from .. import render_opengl as frontend
from ..pbf_helper import PBFConfig


CONFIG = PBFConfig(
    particle_radius=0.05, container_size=(1.0, 1.0, 1.0),
    block_start=(0.1, 0.1, 0.1), block_end=(0.5, 0.5, 0.5),
)


class ControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def make_simulation(self, config=CONFIG):
        with wp.ScopedDevice(self.device):
            return solver.Example(stage_path=None, config=config)

    def step(self, sim, count=1):
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
        buffers = (sim.pos, sim.v, sim.pre_Pos, sim.pre_New, sim.delta_Pos,
                   sim.lambda_Opt, sim.delta_Vel, sim.curl)
        pointers = tuple(buffer.ptr for buffer in buffers)
        sim.reverse_gravity()
        self.assertEqual(tuple(sim.gravity), tuple(-value for value in gravity))
        np.testing.assert_array_equal(sim.pos.numpy(), initial)
        np.testing.assert_array_equal(sim.v.numpy(), 0)
        sim.reverse_gravity()
        self.assertEqual(tuple(sim.gravity), gravity)
        self.step(sim, 5)
        expected_pos, expected_vel = sim.pos.numpy(), sim.v.numpy()
        sim.reverse_gravity()
        with wp.ScopedDevice(self.device):
            sim.reset()
        self.assertIs(sim.config, CONFIG)
        self.assertEqual(sim.sim_time, 0)
        self.assertEqual(tuple(sim.gravity), gravity)
        self.assertEqual(tuple(buffer.ptr for buffer in buffers), pointers)
        np.testing.assert_array_equal(sim.pos.numpy(), initial)
        for buffer in buffers[1:]:
            np.testing.assert_array_equal(buffer.numpy(), 0)
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
        sim.reverse_gravity()
        self.step(sim)
        self.assertGreater(float((sim.pos.numpy() - before)[:, 1].mean()), 0)

    def test_reset_preserves_imported_startup_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "startup.json"
            path.write_text(json.dumps(asdict(CONFIG)), encoding="utf-8")
            config = frontend.parse_args(["--config", str(path)]).config
        sim = self.make_simulation(config)
        initial = sim.pos.numpy()
        self.step(sim, 3)
        expected_pos, expected_vel = sim.pos.numpy(), sim.v.numpy()
        sim.reverse_gravity()
        with wp.ScopedDevice(self.device):
            sim.reset()
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
            values = []
            for clamp in (1, 0):
                wp.launch(solver.calc_lambda, 1,
                          inputs=[grid.id, sim.support_radius, pos, pos, clamp], outputs=[result])
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
        initial = sim.pos.numpy()
        events = (None, key.G, key.R, key.R, key.SPACE, key.SPACE, None, key.P)
        states = []

        class Renderer:
            paused = False
            draw_grid = True
            frames = 0

            def register_key_press_callback(self, callback):
                self.callback = callback

            def is_running(self):
                return self.frames < len(events)

            def begin_frame(self, time):
                states.append((time, self.paused, sim.pos.numpy(), tuple(sim.gravity)))

            def render_billboards(self, **kwargs):
                pass

            def end_frame(self):
                event = events[self.frames]
                if event is not None:
                    handled = self.callback(event, 0)
                    if handled != pyglet.event.EVENT_HANDLED and event == key.SPACE:
                        self.paused = not self.paused
                self.frames += 1

            def close(self):
                pass

        renderer = Renderer()
        args = frontend.parse_args([])
        with wp.ScopedDevice(self.device), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(frontend, 'parse_args', return_value=args), \
                patch.object(frontend, 'create_pbf_simulation', return_value=sim), \
                patch.object(frontend, 'create_pbf_renderer', return_value=renderer):
            frontend.main()
        for frame in (3, 4):
            self.assertEqual(states[frame][0], 0)
            self.assertTrue(states[frame][1])
            np.testing.assert_array_equal(states[frame][2], initial)
            self.assertEqual(states[frame][3], tuple(wp.vec3(*CONFIG.gravity)))
        self.assertEqual(states[5][0], CONFIG.frame_dt)
        self.assertEqual(states[6][0], states[5][0])
        self.assertEqual(states[7][0], states[5][0])
        self.assertFalse(renderer.draw_grid)


if __name__ == '__main__':
    unittest.main()
