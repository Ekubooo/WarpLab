"""Pressure exchange, transformed boundary geometry and particle-state ownership."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import newton
import numpy as np
import warp as wp

from demos.PBF_Rigid.solver import SolverPBF
from demos.PBF_Rigid.solver import boundary_model as boundary
from .fixture import make_boundaries


class CouplingForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA volume boundaries required")

    def make_solver(self, *, transformed=False, mujoco=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        plane = make_boundaries(directory.name)
        tiles = plane.assets[0].sdf.get_tiles()
        empty_sdf = wp.Volume.allocate_by_tiles(tiles, .05, bg_value=1.,
                                                translation=(-.2,) * 3, device=plane.device)
        empty_map = wp.Volume.allocate_by_tiles(tiles, .05, bg_value=0.,
                                                translation=(-.2,) * 3, device=plane.device)
        static = boundary.BoundaryAsset(Path("empty.nvdb"), Path("empty.obj"), empty_sdf, empty_map)
        boundaries = boundary.BoundarySystem((static, plane.assets[0]),
            (boundary.BoundaryInstance(0), boundary.BoundaryInstance(1, 0)), plane.device)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        if mujoco:
            from newton.solvers import SolverMuJoCo
            SolverMuJoCo.register_custom_attributes(builder)
        builder.gravity = (0., 0., 0.)
        inertia = wp.mat33(1., 0., 0., 0., 1., 0., 0., 0., 1.)
        builder.add_body(mass=1., inertia=inertia, label="unmapped")
        transform = wp.transform_identity()
        local = wp.transform_identity()
        if transformed:
            transform = wp.transform(wp.vec3(.2, .3, .1), wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), .7))
            local = wp.transform(wp.vec3(.03, .04, .02), wp.quat_from_axis_angle(wp.vec3(1., 0., 0.), .2))
        body = builder.add_body(xform=transform, com=wp.vec3(.02, -.01, .03),
                                mass=1., inertia=inertia, label="boundary")
        point = wp.transform_point(transform * local, wp.vec3(.1, .02, .04))
        builder.add_particles(pos=[tuple(point)], vel=[(0., 0., 0.)], mass=[.0125], radius=[.0125])
        model = builder.finalize(device=plane.device)
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        boundaries.bind_bodies(model, [body], [tuple(local)])
        boundaries.sync_body_poses(state_in)
        solver = SolverPBF(model, boundaries=boundaries, grid_bounds=((-1.,) * 3, (1.,) * 3))
        solver.pre_Pos.assign([tuple(point)])
        solver.lambda_Opt.fill_(-.002)
        solver.n_Cache.NeighborCount.zero_()
        wp.launch(solver.kernels.volume_query, 1, inputs=[solver.pre_Pos, solver.boundary_data],
                  outputs=[solver.volume_Rho, solver.volume_Grad], device=model.device)
        return solver, state_in, state_out, transform, local

    def pressure(self, solver, dt):
        wp.launch(solver.kernels.calc_deltaPos, solver.n,
            inputs=[solver.pre_Pos, solver.lambda_Opt, solver.n_Cache, solver.volume_Grad,
                    dt, solver.boundary_data], outputs=[solver.delta_Pos], device=solver.device)

    def test_pressure_impulse_balance_torque_and_body_mapping(self):
        for transformed in (False, True):
            with self.subTest(transformed=transformed), contextlib.redirect_stdout(io.StringIO()):
                solver, state, _, transform, local = self.make_solver(transformed=transformed)
                dt = .002
                self.pressure(solver, dt)
                force = solver.boundary_forces.numpy()
                correction = solver.delta_Pos.numpy()[0]
                np.testing.assert_allclose(force[1, :3] * dt,
                    -solver.fluid_mass * correction / dt, rtol=2e-6, atol=2e-7)
                np.testing.assert_array_equal(force[0], np.zeros(6))
                normal = np.asarray(wp.transform_vector(transform * local, wp.vec3(0., 1., 0.)))
                point = solver.pre_Pos.numpy()[0]
                boundary_point = point - max(.02 + .125 * .05, .5 * .05) * normal
                com = np.asarray(wp.transform_point(transform, wp.vec3(.02, -.01, .03)))
                np.testing.assert_allclose(force[1, 3:], np.cross(boundary_point - com, force[1, :3]),
                                            rtol=2e-5, atol=2e-6)
                self.assertLess(np.dot(force[1, :3], normal), 0.)
                np.testing.assert_allclose(solver.boundaries.device_data.world_com.numpy()[0], com, atol=1e-7)

    def test_all_iterations_accumulate_and_use_substep_squared(self):
        solver, *_ = self.make_solver()
        self.pressure(solver, .002)
        first = solver.boundary_forces.numpy()
        self.pressure(solver, .002)
        np.testing.assert_allclose(solver.boundary_forces.numpy(), 2 * first, rtol=2e-6)
        solver.boundary_forces.zero_()
        self.pressure(solver, .004)
        np.testing.assert_allclose(solver.boundary_forces.numpy(), .25 * first, rtol=2e-6)

    def test_step_preserves_input_and_rigid_output_and_clears_previous_force(self):
        solver, source, target, *_ = self.make_solver()
        snapshots = {name: getattr(source, name).numpy().copy()
                     for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")}
        target.body_qd.fill_(wp.spatial_vector(3.))
        target_rigid = target.body_qd.numpy().copy()
        pointers = target.particle_q.ptr, target.particle_qd.ptr
        solver.boundary_forces.fill_(wp.spatial_vector(100.))
        solver.step(source, target, None, None, .002)
        first_force = solver.boundary_forces.numpy().copy()
        solver.boundary_forces.fill_(wp.spatial_vector(200.))
        solver.step(source, target, None, None, .002)
        np.testing.assert_allclose(solver.boundary_forces.numpy(), first_force, rtol=2e-6, atol=1e-7)
        for name, expected in snapshots.items():
            np.testing.assert_array_equal(getattr(source, name).numpy(), expected, err_msg=name)
        np.testing.assert_array_equal(target.body_qd.numpy(), target_rigid)
        self.assertEqual((target.particle_q.ptr, target.particle_qd.ptr), pointers)

    def test_projection_fallback_does_not_add_rigid_force(self):
        solver, *_ = self.make_solver()
        solver.pre_Pos.assign([[.1, -.01, .04]])
        solver.delta_Pos.zero_()
        wp.launch(solver.kernels.update_prePos, 1,
            inputs=[solver.delta_Pos, solver.boundary_data], outputs=[solver.pre_Pos], device=solver.device)
        self.assertAlmostEqual(solver.pre_Pos.numpy()[0, 1], solver.particle_radius, places=6)
        np.testing.assert_array_equal(solver.boundary_forces.numpy(), np.zeros((2, 6)))

    def test_off_center_pressure_drives_mujoco_rotation_and_preserves_external_force(self):
        from newton.solvers import SolverMuJoCo
        from demos.PBF_Rigid.solver.solver_coupled import add_boundary_forces

        solver, source, target, *_ = self.make_solver(mujoco=True)
        # Isolate the pressure impulse: no projection, velocity clipping,
        # gravity, rigid contacts or non-pressure particle kernels are run.
        dt = .002
        self.pressure(solver, dt)
        reaction = solver.boundary_forces.numpy().copy()
        self.assertGreater(np.linalg.norm(reaction[1, 3:]), 0.)
        external = np.array([[.01, -.02, .03, 0., 0., 0.],
                             [.1, .2, .3, 0., 0., 0.]], dtype=np.float32)
        source.body_f.assign(external)
        wp.launch(add_boundary_forces, solver.model.body_count,
                  inputs=[solver.boundary_forces], outputs=[source.body_f], device=solver.device)
        np.testing.assert_allclose(source.body_f.numpy(), external + reaction, rtol=1e-6, atol=1e-7)

        rigid = SolverMuJoCo(solver.model, use_mujoco_cpu=False,
            use_mujoco_contacts=True, disable_contacts=True, enable_sleeping=False,
            update_data_interval=1, integrator="euler")
        rigid.step(source, target, solver.model.control(), None, dt)
        velocity = target.body_qd.numpy()[1]
        # The fixture has identity orientation and isotropic inertia, but a
        # nonzero local COM: the application-point torque must survive the
        # Newton -> MuJoCo frame/reference-point conversion.
        inertia = solver.model.body_inertia.numpy()[1]
        expected_omega = np.linalg.solve(inertia, reaction[1, 3:]) * dt
        np.testing.assert_allclose(velocity[3:], expected_omega, rtol=2e-4, atol=2e-6)
        self.assertGreater(np.dot(velocity[3:], reaction[1, 3:]), 0.)
        expected_linear = (external[1, :3] + reaction[1, :3]) * dt / solver.model.body_mass.numpy()[1]
        np.testing.assert_allclose(velocity[:3], expected_linear, rtol=2e-4, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
