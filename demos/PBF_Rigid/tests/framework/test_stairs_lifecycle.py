"""Full-scale coupled lifecycle, enabled explicitly for the integration run."""

import os
import unittest

import newton
import numpy as np
from scipy.spatial import cKDTree
import warp as wp

from demos.PBF_Rigid.simulation import Simulation


@unittest.skipUnless(os.environ.get("PBF_FULL_SCENE_TESTS") == "1", "enable PBF_FULL_SCENE_TESTS=1")
class StairsLifecycleTests(unittest.TestCase):
    @staticmethod
    def snapshot(sim):
        return {name: getattr(sim.state, name).numpy().copy()
                for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")}

    def test_graph_contact_frame_reset_and_controls(self):
        with wp.ScopedDevice("cuda:0"):
            sim = Simulation(scene="stairs-rigids")
            initial = self.snapshot(sim)
            pointers = [(s.particle_q.ptr, s.particle_qd.ptr, s.body_q.ptr) for s in sim.states]
            sim.capture_graph()
            # Use an actual contact state, not only the initial airborne frame.
            for frame in range(300):
                sim.launch_graph()
                if frame >= 60 and sim.solver.rigid.mjw_data.nacon.numpy()[0] > 0:
                    break
            self.assertGreater(sim.solver.rigid.mjw_data.nacon.numpy()[0], 0)
            for substeps in (2, 3):
                sim.substep = substeps
                before = self.snapshot(sim)
                time, counter = sim.sim_time, sim.solver.rigid._step
                sim.capture_graph()
                self.assertEqual((sim.sim_time, sim.solver.rigid._step), (time, counter))
                for name, values in before.items():
                    np.testing.assert_array_equal(getattr(sim.state, name).numpy(), values)
                for _ in range(2):
                    index, scratch = sim._state_index, sim.fluid_solver.get_buffer_bindings()
                    time, counter = sim.sim_time, sim.solver.rigid._step
                    arrays = [a for s in sim.states for a in vars(s).values() if isinstance(a, wp.array)]
                    arrays += [a for a in vars(sim.solver.rigid.mjw_data).values() if isinstance(a, wp.array)]
                    saved = [(a, wp.clone(a)) for a in arrays]
                    sim.step()
                    expected = self.snapshot(sim)
                    for target, source in saved:
                        wp.copy(target, source)
                    sim._state_index, sim.sim_time = index, time
                    sim.solver.rigid._step = counter
                    sim.fluid_solver.set_buffer_bindings(scratch)
                    sim.launch_graph()
                    distance, match = cKDTree(expected["particle_q"]).query(sim.pos.numpy())
                    self.assertEqual(len(np.unique(match)), sim.n)
                    self.assertLess(float(distance.max()), 2e-5)
                    np.testing.assert_allclose(sim.v.numpy(), expected["particle_qd"][match], atol=2e-4, rtol=2e-4)
                    for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                        np.testing.assert_allclose(getattr(sim.state, name).numpy(), expected[name],
                                                   atol=2e-4 if name.endswith("qd") else 2e-5, rtol=2e-4)
            sim.substep = sim.config.substeps
            sim.gravity = (9.81, 0., 0.)
            self.assertTrue(sim.graph_needs_recapture())
            np.testing.assert_allclose(sim.solver.rigid.mjw_model.opt.gravity.numpy(), [[9.81, 0., 0.]], atol=1e-6)
            sim.frame_dt *= .5
            sim.capture_graph()
            before = sim.sim_time
            sim.launch_graph()
            self.assertAlmostEqual(sim.sim_time - before, sim.frame_dt)
            sim.reset()
            self.assertEqual(sim.sim_time, 0.)
            self.assertEqual(sim.frame_dt, sim.config.frame_dt)
            self.assertTrue(sim.graph_needs_recapture())
            for state in sim.states:
                for name, values in initial.items():
                    np.testing.assert_array_equal(getattr(state, name).numpy(), values)
                np.testing.assert_array_equal(state.body_f.numpy(), 0.)
            np.testing.assert_array_equal(sim.solver.boundary_forces.numpy(), 0.)
            data = sim.BoundarySys.device_data
            np.testing.assert_allclose(data.transforms.numpy()[data.rigid_instance_ids.numpy()],
                                       sim.state.body_q.numpy()[data.body_ids.numpy()], atol=1e-7)
            self.assertEqual([(s.particle_q.ptr, s.particle_qd.ptr, s.body_q.ptr) for s in sim.states], pointers)
            # A force-only reset must preserve the four independent poses.
            sim.state.body_f.fill_(wp.spatial_vector(1.))
            sim.solver.reset(sim.state, flags=newton.StateFlags.BODY_F)
            np.testing.assert_array_equal(sim.state.body_f.numpy(), 0.)
            np.testing.assert_array_equal(sim.state.body_q.numpy(), initial["body_q"])


if __name__ == "__main__":
    unittest.main()
