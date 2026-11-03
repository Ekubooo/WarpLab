"""Controlled contacts and force ownership using the real stairs/rigid assets.

The contact tests advance MuJoCo alone, keeping one diagnostic fluid particle
unchanged. Pressure exchange is checked separately, away from every contact.
"""

import contextlib
import io
import math
import unittest

import newton
from newton.solvers import SolverMuJoCo
import numpy as np
import warp as wp

from Lab_Framework.pbf.config import PBFConfig
from Lab_Framework.pbf.stairs_scene import (
    prepare_stairs_assets, add_stairs_rigids, load_stairs_boundaries,
)
from Lab_Solver.pbf import SolverPBF


class StairsCouplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA NanoVDB and MuJoCo Warp are required")
        cls.device = wp.get_device("cuda:0")
        cls.config = PBFConfig()
        cls.dt = cls.config.frame_dt / cls.config.substeps
        with wp.ScopedDevice(cls.device), contextlib.redirect_stdout(io.StringIO()):
            cls.assets = prepare_stairs_assets(cls.config, cls.device)
            builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
            builder.gravity = (0., 0., 0.)
            SolverMuJoCo.register_custom_attributes(builder)
            cls.bodies = add_stairs_rigids(builder, cls.assets)
            builder.add_particles(pos=[(1., 2., .8)], vel=[(0., 0., 0.)],
                                  mass=[.0125], radius=[.0125])
            cls.model = builder.finalize(device=cls.device)
            cls.states = (cls.model.state(), cls.model.state())
            cls.control = cls.model.control()
            cls.rigid = SolverMuJoCo(cls.model, use_mujoco_cpu=False,
                use_mujoco_contacts=True, update_data_interval=1, enable_sleeping=False,
                nconmax=512, njmax=2048)
            cls.boundaries = load_stairs_boundaries(cls.assets, cls.config, cls.device)
            cls.boundaries.bind_bodies(cls.model, cls.bodies)
            cls.fluid = SolverPBF(cls.model, boundaries=cls.boundaries,
                                 grid_bounds=((0., 0., 0.), cls.config.container_size))
        cls.com = cls.model.body_com.numpy()
        cls.shape_body = cls.model.shape_body.numpy()
        cls.geom_shape = cls.rigid.mjc_geom_to_newton_shape.numpy()[0]
        children = cls.model.joint_child.numpy()
        cls.body_joint = {int(child): joint for joint, child in enumerate(children)}
        cls.q_start = cls.model.joint_q_start.numpy()
        cls.qd_start = cls.model.joint_qd_start.numpy()

    @property
    def state(self):
        return self.states[self.index]

    def place(self, centers, velocities=None, gravity=(0., 0., 0.)):
        """Place selected COMs; park the other bodies clear of all surfaces."""
        self.index = 0
        self.model.set_gravity(gravity)
        self.rigid.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        q = self.model.joint_q.numpy().copy()
        qd = np.zeros_like(self.model.joint_qd.numpy())
        velocities = {} if velocities is None else velocities
        for local, body in enumerate(self.bodies):
            center = centers.get(local, (.5 + .65 * local, 3.2, .35))
            joint = self.body_joint[body]
            start, dof = self.q_start[joint], self.qd_start[joint]
            q[start:start + 3] = np.asarray(center) - self.com[body]
            q[start + 3:start + 7] = (0., 0., 0., 1.)
            qd[dof:dof + 3] = velocities.get(local, (0., 0., 0.))
        for state in self.states:
            self.rigid.reset(state)
            state.clear_forces()
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        self.boundaries.sync_body_poses(self.state)

    def advance(self):
        self.state.clear_forces()
        self.rigid.step(self.state, self.states[1 - self.index], self.control, None, self.dt)
        self.index = 1 - self.index

    def contacts(self):
        """Return actual contacting Newton shapes, not broadphase candidates."""
        data = self.rigid.mjw_data
        count = int(data.nacon.numpy()[0])
        self.assertLess(count, data.naconmax, "MuJoCo contact capacity exhausted")
        self.assertLess(int(data.nefc.numpy().max()), data.njmax,
                        "MuJoCo constraint capacity exhausted")
        if not count:
            return []
        geoms = data.contact.geom.numpy()[:count]
        distances = data.contact.dist.numpy()[:count]
        self.assertGreater(float(distances.min()), -.02, "Excessive controlled-contact penetration")
        pairs = []
        for geoms, distance in zip(geoms, distances):
            if distance <= .001:
                self.assertTrue(np.all(geoms >= 0))
                shapes = self.geom_shape[geoms]
                self.assertTrue(np.all(shapes >= 0))
                pairs.append(tuple(int(shape) for shape in shapes))
        return pairs

    def run_motion(self, seconds, observe=None):
        particles = [state.particle_q.numpy().copy() for state in self.states]
        self.advance()  # Warm compilation before recording two state parities.
        current = self.contacts()
        pairs = set(current)
        if observe is not None:
            observe(current, self.dt)
        index, counter = self.index, self.rigid._step
        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            self.advance()
            self.advance()
        self.index, self.rigid._step = index, counter
        for step in range(math.ceil((seconds / self.dt - 1.) / 2.)):
            wp.capture_launch(capture.graph)
            self.rigid._step += 2
            current = self.contacts()
            pairs.update(current)
            if observe is not None:
                observe(current, (3 + 2 * step) * self.dt)
        for state, values in zip(self.states, particles):
            np.testing.assert_array_equal(state.particle_q.numpy(), values)
        self.assertTrue(np.isfinite(self.state.body_q.numpy()).all())
        self.assertTrue(np.isfinite(self.state.body_qd.numpy()).all())
        return pairs

    def test_box_sphere_and_spot_transfer_momentum_to_another_body(self):
        for moving, target in ((0, 1), (2, 0), (3, 0)):
            with self.subTest(moving=moving, target=target):
                self.place({moving: (1.15, 2., .8), target: (1.65, 2., .8)},
                           {moving: (.6, 0., 0.)})
                pairs = self.run_motion(.8)
                expected = {self.bodies[moving], self.bodies[target]}
                self.assertTrue(any(set(self.shape_body[list(pair)]) == expected for pair in pairs),
                                "The intended dynamic bodies never contacted")
                velocity = self.state.body_qd.numpy()
                self.assertGreater(float(velocity[self.bodies[target], 0]), .01)
                self.assertLess(float(velocity[self.bodies[moving], 0]), .59)
                self.assertGreater(float(self.state.body_q.numpy()[self.bodies[target], 0]), 1.655)

    def test_stairs_and_wall_contacts_stop_free_motion(self):
        cases = (
            ("stairs", (2.1, .8, .525), (0., 0., 0.), (0., -9.81, 0.), .7),
            ("wall", (2.55, 2., .8), (1., 0., 0.), (0., 0., 0.), .7),
        )
        for name, center, velocity, gravity, seconds in cases:
            with self.subTest(surface=name):
                self.place({0: center}, {0: velocity}, gravity)
                pairs = self.run_motion(seconds)
                contacts = [self.model.shape_label[shape] for pair in pairs
                            if self.bodies[0] in self.shape_body[list(pair)]
                            for shape in pair if self.shape_body[shape] == -1]
                self.assertTrue(any(label.startswith("stairs_solid_") for label in contacts))
                q, qd = self.state.body_q.numpy()[self.bodies[0]], self.state.body_qd.numpy()[self.bodies[0]]
                if name == "stairs":
                    self.assertAlmostEqual(float(q[1]), .55, delta=.02)
                    self.assertLess(abs(float(qd[1])), .1)
                else:
                    self.assertLessEqual(float(q[0]), 2.885)
                    self.assertGreater(float(q[0]), 2.80)
                    self.assertLess(abs(float(qd[0])), .15)

    def test_platform_bunny_convex_parts_produce_contacts(self):
        self.place({0: (1.55, 1.6, .8)}, gravity=(0., -9.81, 0.))
        contact_response = []

        def observe(pairs, time):
            for pair in pairs:
                labels = [self.model.shape_label[shape] for shape in pair]
                if (self.bodies[0] in self.shape_body[list(pair)]
                        and any(label.startswith("static_bunny_") for label in labels)):
                    velocity = float(self.state.body_qd.numpy()[self.bodies[0], 1])
                    contact_response.append(velocity + 9.81 * time)
                    break

        pairs = self.run_motion(.7, observe)
        touched = [self.model.shape_label[shape] for pair in pairs
                   if self.bodies[0] in self.shape_body[list(pair)]
                   for shape in pair if self.shape_body[shape] == -1]
        self.assertTrue(any(label.startswith("static_bunny_") for label in touched),
                        "Falling box missed the platform model")
        # The curved surface can tip the box off the platform. Check the
        # departure from free fall while touching the bunny, not its eventual
        # height after it has slid away onto another stair.
        self.assertGreater(max(contact_response), .5)

    def test_shared_box_asset_has_independent_pose_and_force_slots(self):
        instances = self.boundaries.instances
        self.assertEqual(instances[1].asset_id, instances[2].asset_id)
        self.assertNotEqual(instances[1].rigid_id, instances[2].rigid_id)
        for selected in (0, 1):
            with self.subTest(pressured_body=selected):
                center = np.array([1.1, 2., .8])
                self.place({selected: center})
                data = self.boundaries.device_data
                transforms = data.transforms.numpy()
                np.testing.assert_allclose(transforms[1 + selected, :3], center, atol=1e-7)
                self.assertGreater(np.linalg.norm(transforms[1, :3] - transforms[2, :3]), .5)
                # One pressure sample above one box; every other instance and
                # the static landscape are outside its support radius.
                point = center + [.05, .17, 0.]
                self.fluid.pre_Pos.assign([point])
                self.fluid.lambda_Opt.fill_(-.002)
                self.fluid.n_Cache.NeighborCount.zero_()
                self.fluid.boundary_forces.zero_()
                wp.launch(self.fluid.kernels.volume_query, 1,
                    inputs=[self.fluid.pre_Pos, self.fluid.boundary_data],
                    outputs=[self.fluid.volume_Rho, self.fluid.volume_Grad], device=self.device)
                wp.launch(self.fluid.kernels.calc_deltaPos, 1,
                    inputs=[self.fluid.pre_Pos, self.fluid.lambda_Opt, self.fluid.n_Cache,
                            self.fluid.volume_Grad, self.dt, self.fluid.boundary_data],
                    outputs=[self.fluid.delta_Pos], device=self.device)
                reaction = self.fluid.boundary_forces.numpy()
                body = self.bodies[selected]
                self.assertLess(float(reaction[body, 1]), 0.)
                np.testing.assert_array_equal(np.delete(reaction, body, axis=0), 0.)
                np.testing.assert_allclose(reaction[body, :3] * self.dt,
                    -self.fluid.fluid_mass * self.fluid.delta_Pos.numpy()[0] / self.dt,
                    atol=2e-7, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
