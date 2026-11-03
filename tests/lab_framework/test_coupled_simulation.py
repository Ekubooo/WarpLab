"""Shared-State weak coupling, graph replay and complete scene lifecycle."""

import contextlib
from dataclasses import replace
import io
import tempfile
import unittest

import numpy as np
import newton
import warp as wp
from scipy.spatial import cKDTree

from Lab_Framework.pbf.scenes import BOX_HALF_EXTENTS, tank_box_config, load_tank_box_boundaries
from Lab_Framework.pbf.simulation import Simulation
from Lab_Solver.pbf.boundary_model import BoundarySystem


class CoupledSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA NanoVDB and MuJoCo Warp are required")
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.config = replace(tank_box_config(), particle_radius=.025)
        with wp.ScopedDevice("cuda:0"), contextlib.redirect_stdout(io.StringIO()):
            cls.template = load_tank_box_boundaries(cls.config, "cuda:0",
                cache_dir=cls.directory.name, voxel_size=.025)

    def make_simulation(self, *, substeps=3, feedback=True):
        with wp.ScopedDevice("cuda:0"), contextlib.redirect_stdout(io.StringIO()):
            boundaries = BoundarySystem(self.template.assets, self.template.instances, "cuda:0")
            return Simulation(scene="tank-box", config=replace(self.config, substeps=substeps),
                              boundaries=boundaries, feedback=feedback)

    @staticmethod
    def snapshot(sim):
        return {name: getattr(sim.state, name).numpy().copy()
                for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")}

    def test_disjoint_solver_writes_preserve_input_and_particle_bindings(self):
        sim = self.make_simulation()
        before = self.snapshot(sim)
        pointers = [(state.particle_q.ptr, state.particle_qd.ptr) for state in sim.states]
        rigid_before = {name: getattr(sim.state_out, name).numpy().copy()
                        for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
        sim.fluid_solver.step(sim.state, sim.state_out, None, None, sim.dt)
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(sim.state, name).numpy(), values)
        for name, values in rigid_before.items():
            np.testing.assert_array_equal(getattr(sim.state_out, name).numpy(), values)
        particles = (sim.state_out.particle_q.numpy().copy(), sim.state_out.particle_qd.numpy().copy())
        sim.solver.rigid.step(sim.state, sim.state_out, sim.control, None, sim.dt)
        np.testing.assert_array_equal(sim.state_out.particle_q.numpy(), particles[0])
        np.testing.assert_array_equal(sim.state_out.particle_qd.numpy(), particles[1])
        self.assertEqual([(state.particle_q.ptr, state.particle_qd.ptr) for state in sim.states], pointers)

    def test_graph_capture_and_replay_match_eager_even_and_odd(self):
        for substeps in (2, 3):
            with self.subTest(substeps=substeps):
                sim = self.make_simulation(substeps=substeps)
                sim.capture_graph()
                for _ in range(40):
                    sim.launch_graph()
                start = self.snapshot(sim)
                initial_time = sim.sim_time
                rigid_count = sim.solver.rigid._step
                private = {name: getattr(sim.solver.rigid.mjw_data, name).numpy().copy()
                           for name in ("qpos", "qvel", "qacc_warmstart", "ctrl", "xfrc_applied")}
                sim.capture_graph()
                self.assertEqual(sim.sim_time, initial_time)
                self.assertEqual(sim.solver.rigid._step, rigid_count)
                for name, values in start.items():
                    np.testing.assert_array_equal(getattr(sim.state, name).numpy(), values)
                for name, values in private.items():
                    np.testing.assert_array_equal(getattr(sim.solver.rigid.mjw_data, name).numpy(), values)

                # Compare whole frames from identical warm-start inputs.
                # Long independent trajectories can diverge when atomic roundoff
                # crosses neighbor/collision thresholds; that is not a graph bug.
                for _ in range(4):
                    state_index = sim._state_index
                    scratch = sim.fluid_solver.get_buffer_bindings()
                    time = sim.sim_time
                    counter = sim.solver.rigid._step
                    arrays = [value for state in sim.states for value in vars(state).values()
                              if isinstance(value, wp.array)]
                    arrays += [value for value in vars(sim.solver.rigid.mjw_data).values()
                               if isinstance(value, wp.array)]
                    saved = [(value, wp.clone(value)) for value in arrays]
                    sim.step()
                    expected = self.snapshot(sim)
                    for target, source in saved:
                        wp.copy(target, source)
                    sim._state_index = state_index
                    sim.fluid_solver.set_buffer_bindings(scratch)
                    sim.sim_time = time
                    sim.solver.rigid._step = counter
                    sim.launch_graph()
                    # Small atomic differences can move a particle across a
                    # hash-cell edge and permute slots. IDs are not persistent.
                    positions = sim.pos.numpy()
                    distance, match = cKDTree(expected["particle_q"]).query(positions)
                    self.assertEqual(len(np.unique(match)), len(match))
                    self.assertLess(float(distance.max()), 2e-5)
                    np.testing.assert_allclose(sim.v.numpy(), expected["particle_qd"][match],
                                               atol=2e-4, rtol=2e-4)
                    for name, values in expected.items():
                        if name.startswith("particle_"):
                            continue
                        np.testing.assert_allclose(getattr(sim.state, name).numpy(), values,
                                                   atol=2e-4 if name.endswith("qd") else 2e-5,
                                                   rtol=2e-4, err_msg=name)

    def test_reset_restores_both_states_and_boundary_pose(self):
        sim = self.make_simulation()
        initial = self.snapshot(sim)
        pointers = [(state.particle_q.ptr, state.particle_qd.ptr, state.body_q.ptr) for state in sim.states]
        sim.capture_graph()
        for _ in range(45):
            sim.launch_graph()
        self.assertFalse(np.array_equal(sim.state.body_q.numpy(), initial["body_q"]))
        sim.gravity = (1., 2., 3.)
        sim.frame_dt *= .5
        sim.reset()
        self.assertEqual(sim.sim_time, 0.)
        self.assertEqual(sim.frame_dt, sim.config.frame_dt)
        self.assertTrue(sim.graph_needs_recapture())
        np.testing.assert_allclose(tuple(sim.gravity), sim.config.gravity)
        for state in sim.states:
            for name, values in initial.items():
                np.testing.assert_array_equal(getattr(state, name).numpy(), values)
            np.testing.assert_array_equal(state.body_f.numpy(), 0.)
            np.testing.assert_array_equal(state.particle_f.numpy(), 0.)
        np.testing.assert_array_equal(sim.solver.boundary_forces.numpy(), 0.)
        rigid_instance = sim.BoundarySys.device_data.rigid_instance_ids.numpy()[0]
        np.testing.assert_allclose(sim.BoundarySys.device_data.transforms.numpy()[rigid_instance],
                                   initial["body_q"][0], atol=1e-7)
        self.assertEqual([(state.particle_q.ptr, state.particle_qd.ptr, state.body_q.ptr)
                          for state in sim.states], pointers)

    def test_gravity_and_slow_motion_invalidate_graph_for_both_solvers(self):
        sim = self.make_simulation()
        sim.capture_graph()
        self.assertFalse(sim.graph_needs_recapture())
        sim.gravity = (0., 9.81, 0.)
        self.assertTrue(sim.graph_needs_recapture())
        with self.assertRaisesRegex(RuntimeError, "parameters changed"):
            sim.launch_graph()
        np.testing.assert_allclose(tuple(sim.fluid_solver.gravity), (0., 9.81, 0.))
        np.testing.assert_allclose(sim.solver.rigid.mjw_model.opt.gravity.numpy(), [[0., 9.81, 0.]], atol=1e-6)
        sim.capture_graph()
        sim.launch_graph()
        self.assertGreater(sim.state.body_qd.numpy()[0, 1], 0.)
        sim.frame_dt *= .5
        self.assertTrue(sim.graph_needs_recapture())
        with self.assertRaisesRegex(RuntimeError, "Time step changed"):
            sim.launch_graph()
        sim.capture_graph()
        before = sim.sim_time
        sim.launch_graph()
        self.assertAlmostEqual(sim.sim_time - before, sim.frame_dt)

    def test_direct_reset_body_force_flag_respects_world_mask(self):
        sim = self.make_simulation()
        sim.step()
        initial = self.snapshot(sim)
        force = np.ones((sim.model.body_count, 6), dtype=np.float32)
        sim.state.body_f.assign(force)
        mask = wp.zeros(sim.model.world_count + 1, dtype=wp.bool, device=sim.model.device)
        sim.solver.reset(sim.state, world_mask=mask, flags=newton.StateFlags.BODY_F)
        np.testing.assert_array_equal(sim.state.body_f.numpy(), force)
        mask.fill_(True)
        sim.solver.reset(sim.state, world_mask=mask, flags=newton.StateFlags.BODY_F)
        np.testing.assert_array_equal(sim.state.body_f.numpy(), 0.)
        for name, values in initial.items():
            np.testing.assert_allclose(getattr(sim.state, name).numpy(), values, atol=1e-7, rtol=1e-6)

    def test_mujoco_alone_free_fall_and_tank_floor_contact(self):
        sim = self.make_simulation()
        solver = sim.solver.rigid
        initial_y = float(sim.state.body_q.numpy()[0, 1])
        particles = [state.particle_q.numpy().copy() for state in sim.states]

        def rigid_step():
            sim.state.clear_forces()
            solver.step(sim.state, sim.state_out, sim.control, None, sim.dt)
            sim._state_index = 1 - sim._state_index

        # No fluid advancement or pressure feedback: this tests the backend's
        # mass, gravity and collision setup independently of the fluid solver.
        rigid_step()
        np.testing.assert_allclose(sim.state.body_qd.numpy()[0, 1],
                                   sim.config.gravity[1] * sim.dt, atol=3e-6)
        self.assertLess(float(sim.state.body_q.numpy()[0, 1]), initial_y)
        counter = solver._step
        with wp.ScopedDevice(sim.model.device), wp.ScopedCapture() as capture:
            rigid_step()
            rigid_step()
        solver._step = counter
        for _ in range(round(2. / (2. * sim.dt))):
            wp.capture_launch(capture.graph)
            solver._step += 2
        pose, velocity = sim.state.body_q.numpy(), sim.state.body_qd.numpy()
        self.assertTrue(np.isfinite(pose).all() and np.isfinite(velocity).all())
        self.assertAlmostEqual(float(pose[0, 1]), BOX_HALF_EXTENTS[1], delta=.01)
        self.assertLess(float(np.linalg.norm(velocity[0, :3])), .05)
        for state, initial in zip(sim.states, particles):
            np.testing.assert_array_equal(state.particle_q.numpy(), initial)

    def test_pressure_feedback_keeps_box_afloat_against_one_way_baseline(self):
        final_heights = {}
        for feedback in (False, True):
            with self.subTest(feedback=feedback):
                sim = self.make_simulation(feedback=feedback)
                sim.capture_graph()
                peak_force = 0.
                for frame in range(round(2. / sim.frame_dt)):
                    sim.launch_graph()
                    if frame % 15 == 0:
                        for values in (sim.pos.numpy(), sim.v.numpy(),
                                       sim.state.body_q.numpy(), sim.state.body_qd.numpy()):
                            self.assertTrue(np.isfinite(values).all())
                        peak_force = max(peak_force,
                                         float(sim.solver.boundary_forces.numpy()[0, 1]))
                self.assertTrue(np.isfinite(sim.pos.numpy()).all())
                self.assertTrue(np.isfinite(sim.v.numpy()).all())
                final_heights[feedback] = float(sim.state.body_q.numpy()[0, 1])
                self.assertGreater(peak_force, 1.)
        self.assertAlmostEqual(final_heights[False], BOX_HALF_EXTENTS[1], delta=.01)
        # Broad physical bounds tolerate chaotic free-surface detail, while
        # distinguishing buoyant support from simply colliding with the floor.
        self.assertGreater(final_heights[True], .30)
        self.assertLess(final_heights[True], .70)
        self.assertGreater(final_heights[True] - final_heights[False], .15)


if __name__ == "__main__":
    unittest.main()
