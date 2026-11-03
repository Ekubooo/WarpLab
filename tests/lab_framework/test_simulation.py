"""Newton scene setup, frame parity, reset and standalone entry contracts."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from Lab_Framework.pbf.config import PBFConfig
from Lab_Framework.pbf import config_io, render_opengl
from Lab_Framework.pbf.simulation import Simulation
from Lab_Solver.pbf import boundary_model as boundary


class EntryTests(unittest.TestCase):
    def test_verbose_controls_frame_timer_without_synchronization(self):
        for verbose in (False, True):
            simulation = Simulation.__new__(Simulation)
            simulation.model = SimpleNamespace(device="cpu")
            simulation.verbose = verbose
            simulation.sim_time = 0.
            simulation.frame_dt = .01
            with patch.object(wp, "ScopedDevice", return_value=contextlib.nullcontext()), \
                    patch.object(wp, "ScopedTimer", return_value=contextlib.nullcontext()) as timer, \
                    patch.object(simulation, "_advance_frame") as advance:
                simulation.step()
            timer.assert_called_once_with("PBF frame", active=verbose, synchronize=False)
            advance.assert_called_once_with()
            self.assertEqual(simulation.sim_time, .01)

    def test_help_and_export_do_not_initialize_cuda(self):
        code = """
import contextlib, io, sys, tempfile
from pathlib import Path
from unittest.mock import patch
import warp as wp
from Lab_Framework.pbf import render_opengl as entry, config_io
with patch.object(wp, 'init', side_effect=AssertionError('CUDA initialized')), \
     patch.object(entry, 'create_pbf_simulation', side_effect=AssertionError('scene loaded')):
    with contextlib.redirect_stdout(io.StringIO()):
        try: entry.parse_args(['--help'])
        except SystemExit as error: assert error.code == 0
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(config_io, 'CONFIG_DIR', Path(directory)), \
         patch.object(sys, 'argv', ['pbf', '--export-config', 'test.json']), \
         contextlib.redirect_stdout(io.StringIO()):
        entry.main()
        assert (Path(directory) / 'test.json').is_file()
assert not any(name == 'demos' or name.startswith('demos.') for name in sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_original_json_fields_are_preserved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory, patch.object(config_io, "CONFIG_DIR", Path(directory)):
            config = replace(PBFConfig(), initial_velocity=(.1, .2, .3), viscosity=.2)
            self.assertEqual(config_io.load_config(config_io.save_config(config, "preset.json")), config)
        self.assertTrue(render_opengl.parse_args(["--headless", "--num-frames", "2"]).headless)

    def test_stairs_rigids_keeps_original_default_config_and_overrides(self):
        args = render_opengl.parse_args(["--scene", "stairs-rigids"])
        self.assertEqual(args.config or PBFConfig(), PBFConfig())
        args = render_opengl.parse_args(["--scene", "stairs-rigids", "--substeps", "2"])
        self.assertEqual(args.config, replace(PBFConfig(), substeps=2))


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA NanoVDB is required")
        cls.device = wp.get_device("cuda:0")
        with wp.ScopedDevice(cls.device):
            tiles = wp.array([[0, 0, 0]], dtype=wp.vec3i)
            cls.sdf = wp.Volume.allocate_by_tiles(tiles, .1, bg_value=10.)
            cls.volume_map = wp.Volume.allocate_by_tiles(tiles, .1, bg_value=0.)

    def make_simulation(self, substeps=3):
        asset = boundary.BoundaryAsset(Path("unused.nvdb"), Path("unused.obj"), self.sdf, self.volume_map)
        boundaries = boundary.BoundarySystem((asset,), (boundary.BoundaryInstance(0),), self.device)
        config = replace(PBFConfig(), particle_radius=.025, block_start=(.1, .1, .1),
                         block_end=(.3, .3, .3), substeps=substeps, initial_velocity=(.1, 0., 0.))
        with wp.ScopedDevice(self.device), contextlib.redirect_stdout(io.StringIO()):
            return Simulation(config=config, boundaries=boundaries)

    def test_lattice_mass_and_runtime_state_binding(self):
        sim = self.make_simulation()
        self.assertEqual(sim.n, 27)
        expected = (np.asarray(sim.config.block_start, dtype=np.float32).astype(np.float64)
                    + float(np.float32(.05)) * (np.indices((3, 3, 3)).reshape(3, -1).T + 1)).astype(np.float32)
        np.testing.assert_array_equal(sim.pos.numpy(), expected)
        np.testing.assert_allclose(sim.model.particle_mass.numpy(), .8 * .05**3 * 1000.)
        before = sim.pos
        sim.step()
        self.assertIs(sim.pos, sim.state.particle_q)
        self.assertIs(sim.pos, sim.solver.pos)
        self.assertIsNot(before, sim.pos)
        self.assertAlmostEqual(sim.sim_time, sim.frame_dt)

    def test_graph_replay_matches_eager_for_even_and_odd_substeps(self):
        for substeps in (2, 3):
            with self.subTest(substeps=substeps), contextlib.redirect_stdout(io.StringIO()):
                sim = self.make_simulation(substeps)
                expected = []
                for _ in range(4):
                    sim.step()
                    expected.append((sim.pos.numpy(), sim.v.numpy(), sim.sim_time))
                sim.reset()
                sim.capture_graph()
                for positions, velocities, time in expected:
                    sim.launch_graph()
                    np.testing.assert_array_equal(sim.pos.numpy(), positions)
                    np.testing.assert_array_equal(sim.v.numpy(), velocities)
                    self.assertEqual(sim.sim_time, time)

    def test_reset_and_graph_invalidation_preserve_buffers(self):
        sim = self.make_simulation()
        initial = sim.pos.numpy()
        pointers = {array.ptr for array in sim.solver.get_buffer_bindings()}
        sim.capture_graph()
        sim.gravity = (0., 9.81, 0.)
        with self.assertRaisesRegex(RuntimeError, "parameters changed"):
            sim.launch_graph()
        sim.step()
        sim.reset()
        np.testing.assert_array_equal(sim.pos.numpy(), initial)
        self.assertEqual(sim.sim_time, 0.)
        self.assertEqual(tuple(sim.gravity), tuple(wp.vec3(*sim.config.gravity)))
        self.assertEqual({array.ptr for array in sim.solver.get_buffer_bindings()}, pointers)
        sim.capture_graph()
        sim.frame_dt *= .5
        with self.assertRaisesRegex(RuntimeError, "Time step changed"):
            sim.launch_graph()

    def test_headless_entry_advances_the_current_state(self):
        sim = self.make_simulation()
        args = render_opengl.parse_args(["--headless", "--num-frames", "2"])
        with patch.object(render_opengl, "parse_args", return_value=args), \
                patch.object(render_opengl, "create_pbf_simulation", return_value=sim), \
                patch.object(render_opengl, "create_pbf_renderer", side_effect=AssertionError("window opened")):
            render_opengl.main()
        self.assertEqual(sim.sim_time, 2 * sim.frame_dt)


if __name__ == "__main__":
    unittest.main()
