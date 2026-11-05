"""Newton state ownership, original launch sequence and instance isolation."""

import contextlib
from dataclasses import replace
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
import newton

from demos.PBF_Rigid.solver import SolverPBF
from .fixture import make_boundaries, make_solver


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA required")

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.boundaries = make_boundaries(directory.name)

    def test_two_states_and_exact_original_launch_order(self):
        solver, state = make_solver(self.boundaries)
        self.assertIsInstance(solver, newton.solvers.SolverBase)
        velocity = state.particle_qd
        output = solver.model.state()
        output_positions, output_velocities = output.particle_q, output.particle_qd
        input_positions, input_velocities = state.particle_q.numpy(), state.particle_qd.numpy()
        before = solver.get_buffer_bindings()
        launched = []
        original_launch = wp.launch
        def record(kernel, *args, **kwargs):
            if kernel in vars(solver.kernels).values():
                launched.append(next(name for name, value in vars(solver.kernels).items() if value is kernel))
            return original_launch(kernel, *args, **kwargs)
        with patch.object(wp, "launch", side_effect=record), contextlib.redirect_stdout(io.StringIO()):
            solver.step(state, output, None, None, 1./450.)
        expected = ["apply_predict", "reorder", "cache_neighbors"]
        expected += ["volume_query", "calc_lambda", "calc_deltaPos", "update_prePos"] * 3
        expected += ["update_position", "calc_curl", "calc_visvor", "update_velocity"]
        self.assertEqual(launched, expected)
        self.assertIs(output.particle_q, output_positions)
        self.assertIs(output.particle_qd, output_velocities)
        np.testing.assert_array_equal(state.particle_q.numpy(), input_positions)
        np.testing.assert_array_equal(state.particle_qd.numpy(), input_velocities)
        self.assertIs(state.particle_qd, velocity)
        self.assertEqual({a.ptr for a in before}, {a.ptr for a in solver.get_buffer_bindings()})
        self.assertTrue(np.isfinite(output.particle_q.numpy()).all())

    def test_explicitly_reject_unsupported_step_modes(self):
        solver, state = make_solver(self.boundaries)
        output = solver.model.state()
        with self.assertRaises(ValueError):
            solver.step(state, state, None, None, .01)
        with self.assertRaises(NotImplementedError):
            solver.step(state, output, None, object(), .01)
        for dt in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                solver.step(state, output, None, None, dt)

    def test_reset_reuses_buffers_and_respects_state_flags_and_mask(self):
        solver, state = make_solver(self.boundaries)
        initial = state.particle_q.numpy()
        buffers = {a.ptr for a in solver.get_buffer_bindings()}
        output = solver.model.state()
        with contextlib.redirect_stdout(io.StringIO()):
            solver.step(state, output, None, None, .002)
        state = output
        changed = state.particle_q.numpy()
        mask = wp.zeros(solver.model.world_count + 1, dtype=wp.bool, device=solver.device)
        solver.reset(state, world_mask=mask)
        np.testing.assert_array_equal(state.particle_q.numpy(), changed)
        solver.reset(state, flags=newton.StateFlags.PARTICLE_QD)
        np.testing.assert_array_equal(state.particle_q.numpy(), changed)
        np.testing.assert_array_equal(state.particle_qd.numpy(), np.zeros_like(initial))
        solver.reset(state)
        np.testing.assert_array_equal(state.particle_q.numpy(), initial)
        self.assertEqual({a.ptr for a in solver.get_buffer_bindings()}, buffers)

    def test_other_instance_does_not_replace_compiled_constants(self):
        first, state = make_solver(self.boundaries)
        config = replace(SolverPBF.Config(), viscosity=.1, artificial_pressure_strength=.002)
        second, other = make_solver(self.boundaries, radius=.01, config=config)
        output, other_output = first.model.state(), second.model.state()
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                first.step(state, output, None, None, .002)
                state, output = output, state
        expected = state.particle_q.numpy()
        first.reset(state)
        with contextlib.redirect_stdout(io.StringIO()):
            first.step(state, output, None, None, .002)
            state, output = output, state
            second.step(other, other_output, None, None, .001)
            first.step(state, output, None, None, .002)
            state = output
        np.testing.assert_array_equal(state.particle_q.numpy(), expected)

    def test_model_change_updates_gravity_or_requires_recreation(self):
        solver, state = make_solver(self.boundaries)
        solver.model.set_gravity((1., 0., 0.))
        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        self.assertEqual(tuple(solver.gravity), (1., 0., 0.))
        solver.model.particle_radius.fill_(.02)
        with self.assertRaisesRegex(ValueError, "recreate"):
            solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

    def test_invalid_particle_properties_are_rejected_before_specialization(self):
        solver, _ = make_solver(self.boundaries)
        for name in ("particle_mass", "particle_radius"):
            array = getattr(solver.model, name)
            original = array.numpy()
            for value in (0., -1., float("nan"), float(original[0]) * 2):
                changed = original.copy()
                changed[0] = value
                array.assign(changed)
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    SolverPBF(solver.model, boundaries=self.boundaries)
            array.assign(original)

    def test_automatic_differentiation_is_rejected_explicitly(self):
        solver, _ = make_solver(self.boundaries)
        unbound = SolverPBF(solver.model, boundaries=self.boundaries)
        with self.assertRaisesRegex(ValueError, "differentiation"):
            unbound.step(solver.model.state(requires_grad=True), solver.model.state(), None, None, .002)


if __name__ == "__main__":
    unittest.main()
