"""Manual full-step graph capture and ordinary-step equivalence."""
import contextlib
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from demos.fluid.PBF import PBF as solver
from demos.fluid.PBF.pbf_helper import PBFConfig
from demos.fluid.PBF.simulation import reset_simulation, reverse_gravity
from tests.fluid.pbf.solver_fixture import bind_solver


CONFIG = PBFConfig(
    particle_radius=0.05, container_size=(1.0, 1.0, 1.0),
    block_start=(0.1, 0.1, 0.1), block_end=(0.5, 0.5, 0.5),
)
BUFFERS = ("pos", "pre_Pos", "temp_pos", "temp_pPos")


class CUDAGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("PBF graph capture requires CUDA")
        cls.device = wp.get_device("cuda:0")

    def make_simulation(self, config=CONFIG):
        with wp.ScopedDevice(self.device), contextlib.redirect_stdout(io.StringIO()):
            sdf = wp.Volume.allocate([0, 0, 0], [0, 0, 0], 1.0, bg_value=10.0,
                                     translation=(100.0, 100.0, 100.0), device=self.device)
            with patch.object(solver.boundary_model, "load_boundary_volumes", return_value=(sdf, sdf)):
                sim = solver.PBF(stage_path=None, config=config)
            bind_solver(config, sim.grid, sim.static_sdf, sim.volume_map)
            # Exercise nonzero pressure, curl and viscosity rather than uniform free fall.
            positions = sim.pos.numpy()
            offsets = (positions - positions.mean(axis=0)) * 0.75
            sim.pos.assign(positions.mean(axis=0) + offsets)
            sim.v.assign(np.column_stack((0.2 * offsets[:, 2],
                                          0.1 * offsets[:, 0], -0.15 * offsets[:, 1])))
        return sim

    def pointers(self, sim):
        return tuple(getattr(sim, name).ptr for name in BUFFERS)

    def capture(self, sim):
        with contextlib.redirect_stdout(io.StringIO()):
            sim.capture_graph()

    def test_capture_preserves_state_and_restores_on_failure(self):
        sim = self.make_simulation()
        self.assertFalse(hasattr(sim, "step_graphs"))
        with wp.ScopedDevice(self.device), contextlib.redirect_stdout(io.StringIO()):
            sim.step()
        names = BUFFERS + ("v", "lambda_Opt", "delta_Pos", "volume_Rho",
                           "volume_Grad", "curl", "delta_Vel")
        expected = {name: getattr(sim, name).numpy() for name in names}
        counts = sim.n_Cache.NeighborCount.numpy()
        neighbors = sim.n_Cache.CacheBuffer.numpy()
        pointers, sim_time = self.pointers(sim), sim.sim_time

        with wp.ScopedDevice("cpu"):
            self.capture(sim)
            self.assertEqual(wp.get_device(), wp.get_device("cpu"))
        self.assertEqual(len(sim.step_graphs), 2)
        self.assertEqual(self.pointers(sim), pointers)
        self.assertEqual(sim.sim_time, sim_time)
        for name in names:
            np.testing.assert_array_equal(getattr(sim, name).numpy(), expected[name])
        np.testing.assert_array_equal(sim.n_Cache.NeighborCount.numpy(), counts)
        np.testing.assert_array_equal(sim.n_Cache.CacheBuffer.numpy(), neighbors)

        # A partial recapture must preserve good graphs and restore Python state.
        graphs = sim.step_graphs
        original_step = sim.step
        calls = 0
        def fail_second_capture():
            nonlocal calls
            calls += 1
            original_step()
            if calls == 2:
                raise RuntimeError("injected capture failure")
        with patch.object(sim, "step", side_effect=fail_second_capture):
            with self.assertRaisesRegex(RuntimeError, "injected capture failure"):
                self.capture(sim)
        self.assertIs(sim.step_graphs, graphs)
        self.assertEqual(self.pointers(sim), pointers)
        self.assertEqual(sim.sim_time, sim_time)
        for name in names:
            np.testing.assert_array_equal(getattr(sim, name).numpy(), expected[name])

    def test_replay_matches_step_for_odd_and_even_substeps_and_reset(self):
        for substeps in (3, 2):
            with self.subTest(substeps=substeps):
                sim = self.make_simulation(replace(CONFIG, substeps=substeps))
                start_pos, start_vel = sim.pos.numpy(), sim.v.numpy()
                expected = []
                with wp.ScopedDevice(self.device), contextlib.redirect_stdout(io.StringIO()):
                    for _ in range(5):
                        sim.step()
                        expected.append((sim.pos.numpy(), sim.v.numpy(), sim.sim_time))
                reset_simulation(sim)
                sim.pos.assign(start_pos)
                sim.v.assign(start_vel)
                self.capture(sim)
                graphs = sim.step_graphs
                for replay in range(2):
                    if replay:
                        reset_simulation(sim)
                        sim.pos.assign(start_pos)
                        sim.v.assign(start_vel)
                    initial = self.pointers(sim)
                    swapped = (initial[2], initial[3], initial[0], initial[1])
                    with patch.object(sim, "step", side_effect=AssertionError("step called during replay")):
                        for frame, (pos, vel, sim_time) in enumerate(expected, 1):
                            with wp.ScopedDevice("cpu"):
                                sim.launch_graph()
                                self.assertEqual(wp.get_device(), wp.get_device("cpu"))
                            np.testing.assert_allclose(sim.pos.numpy(), pos, rtol=2e-6, atol=1e-7)
                            np.testing.assert_allclose(sim.v.numpy(), vel, rtol=2e-6, atol=1e-7)
                            self.assertEqual(sim.sim_time, sim_time)
                            want = swapped if substeps % 2 and frame % 2 else initial
                            self.assertEqual(self.pointers(sim), want)
                    self.assertIs(sim.step_graphs, graphs)

    def test_manual_recapture_after_parameter_changes(self):
        sim = self.make_simulation()
        self.capture(sim)
        old_graphs = sim.step_graphs
        reverse_gravity(sim)
        sim.substep = 2
        sim.iterations = 2
        sim.frame_dt = CONFIG.frame_dt * 0.5
        sim.dt = sim.frame_dt / sim.substep
        start_pos, start_vel = sim.pos.numpy(), sim.v.numpy()
        start_time = sim.sim_time
        self.capture(sim)
        self.assertIsNot(sim.step_graphs, old_graphs)
        np.testing.assert_array_equal(sim.pos.numpy(), start_pos)
        self.assertEqual(sim.sim_time, start_time)
        with wp.ScopedDevice(self.device), contextlib.redirect_stdout(io.StringIO()):
            sim.step()
        expected_pos, expected_vel = sim.pos.numpy(), sim.v.numpy()
        sim.pos.assign(start_pos)
        sim.v.assign(start_vel)
        sim.sim_time = start_time
        sim.launch_graph()
        np.testing.assert_allclose(sim.pos.numpy(), expected_pos, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(sim.v.numpy(), expected_vel, rtol=2e-6, atol=1e-7)
        self.assertEqual(sim.sim_time, start_time + sim.frame_dt)

    def test_launch_requires_capture(self):
        sim = self.make_simulation()
        with self.assertRaisesRegex(RuntimeError, "capture_graph"):
            sim.launch_graph()


if __name__ == "__main__":
    unittest.main()
