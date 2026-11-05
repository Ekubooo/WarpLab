"""Full-scale static parity, State ownership and frame lifecycle."""

import ast
import json
from pathlib import Path
import unittest

import newton
import numpy as np
import warp as wp

from PBF import PBF as kernels, pbf_helper, boundary_model
from .reference import independent_pair, verify_pair, WORK_ARRAYS, ROOT


@wp.kernel(enable_backward=False)
def sample_sdf(grid: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=float)):
    i = wp.tid()
    values[i] = wp.volume_sample_f(grid, wp.volume_world_to_index(grid, points[i]), wp.Volume.LINEAR)


class StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pair = independent_pair()
        cls.reference, cls.sim = cls.pair.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.pair.__exit__(None, None, None)

    def setUp(self):
        self.sim.substep = self.sim.config.substeps
        self.sim.reset()

    def test_30_substeps_bitwise_parity(self):
        result = verify_pair(self.reference, self.sim)
        report = ROOT / "outputs/pbf-newton/parity.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2) + "\n")

    def test_original_initial_overlap(self):
        sim = self.sim
        distances = wp.empty(sim.n, dtype=float)
        wp.launch(sample_sdf, sim.n, inputs=[sim.solver.static_sdf.id, sim.pos, distances])
        self.assertEqual(int(np.count_nonzero(distances.numpy() < 0.)), 1411)
        self.assertEqual(sim.sim_time, 0.)
        self.assertEqual(sim.model.body_count, 0)
        self.assertEqual(sim.BoundarySys.rigid_count, 0)

    def test_graph_parities_capture_and_controls(self):
        sim = self.sim
        solver = sim.solver
        pointers = [(s.particle_q.ptr, s.particle_qd.ptr) for s in sim.states]
        for substeps in (1, 2, 3):
            sim.substep = substeps
            sim.reset()
            for gravity, slow in (((0., -9.81, 0.), False), ((9.81, 0., 0.), True)):
                sim.gravity = wp.vec3(*gravity)
                sim.frame_dt = sim.config.frame_dt * (.5 if slow else 1.)
                self.assertTrue(sim.graph_needs_recapture())
                for start_index in (0, 1):
                    for reverse_predict in (False, True):
                        sim._state_index = start_index
                        if reverse_predict:
                            solver.pre_Pos, solver.temp_pPos = solver.temp_pPos, solver.pre_Pos
                        index, prediction, time = sim._state_index, (solver.pre_Pos, solver.temp_pPos), sim.sim_time
                        arrays = [a for state in sim.states for a in (state.particle_q, state.particle_qd)]
                        arrays += [getattr(solver, n) for n in WORK_ARRAYS] + [solver.temp_pPos, solver.n_Cache.NeighborCount]
                        snapshots = [(a, wp.clone(a)) for a in arrays]
                        sim.capture_graph()
                        self.assertEqual((sim._state_index, sim.sim_time), (index, time))
                        for a, saved in snapshots:
                            np.testing.assert_array_equal(a.numpy(), saved.numpy())
                        sim.step()
                        expected = [a.numpy() for a in (sim.pos, sim.v)]
                        work = [getattr(solver, n).numpy() for n in WORK_ARRAYS]
                        counts = solver.n_Cache.NeighborCount.numpy()
                        for a, saved in snapshots:
                            wp.copy(a, saved)
                        sim._state_index, sim.sim_time = index, time
                        solver.pre_Pos, solver.temp_pPos = prediction
                        sim.launch_graph()
                        for a, value in zip((sim.pos, sim.v), expected):
                            np.testing.assert_array_equal(a.numpy(), value)
                        for n, value in zip(WORK_ARRAYS, work):
                            np.testing.assert_array_equal(getattr(solver, n).numpy(), value, err_msg=n)
                        np.testing.assert_array_equal(solver.n_Cache.NeighborCount.numpy(), counts)
                        self.assertEqual(sim.sim_time, time + sim.frame_dt)
        self.assertEqual([(s.particle_q.ptr, s.particle_qd.ptr) for s in sim.states], pointers)

    def test_reset_restores_both_states_and_scratch(self):
        sim = self.sim
        sim.launch_graph()
        sim.gravity = wp.vec3(9.81, 0., 0.)
        sim.frame_dt *= .5
        sim.reset()
        for state in sim.states:
            np.testing.assert_array_equal(state.particle_q.numpy(), sim.model.particle_q.numpy())
            np.testing.assert_array_equal(state.particle_qd.numpy(), sim.model.particle_qd.numpy())
        for name in WORK_ARRAYS + ("temp_pPos",):
            np.testing.assert_array_equal(getattr(sim.solver, name).numpy(), 0.)
        self.assertEqual(sim.sim_time, 0.)
        self.assertEqual(sim.frame_dt, sim.config.frame_dt)
        np.testing.assert_allclose(sim.model.gravity.numpy()[-1], sim.config.gravity, atol=1e-6)

    def test_solver_ownership_and_minimal_contract(self):
        sim = self.sim
        state_out = sim.state_out
        state_out.body_q = wp.array([wp.transform_identity()], dtype=wp.transform)
        state_out.joint_q = wp.array([7.], dtype=float)
        try:
            sim._substep(sim.dt)
            np.testing.assert_array_equal(state_out.body_q.numpy(), [[0., 0., 0., 0., 0., 0., 1.]])
            np.testing.assert_array_equal(state_out.joint_q.numpy(), [7.])
        finally:
            state_out.body_q = state_out.joint_q = None
        self.assertIsInstance(sim.solver, newton.solvers.SolverBase)
        self.assertFalse(hasattr(sim.solver, "pos"))
        self.assertFalse(hasattr(sim.solver, "v"))
        self.assertFalse(hasattr(sim.solver, "temp_pos"))
        with self.assertRaises(ValueError):
            sim.solver.step(sim.state, sim.state, None, None, sim.dt)
        with self.assertRaises(NotImplementedError):
            sim.solver.step(sim.state, sim.state_out, None, None, sim.dt, observables=object())
        with self.assertRaises(NotImplementedError):
            sim.solver.reset(sim.state, flags=newton.StateFlags.PARTICLE_Q)

    def test_forward_only_and_step_has_no_copies(self):
        for module in (kernels, pbf_helper, boundary_model):
            self.assertFalse(wp.get_module_options(module=module)['enable_backward'])
        tree = ast.parse(Path(kernels.__file__).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
        calls = [n.func.attr for n in ast.walk(step) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn("copy", calls)
        self.assertNotIn("numpy", calls)


if __name__ == "__main__":
    unittest.main()
