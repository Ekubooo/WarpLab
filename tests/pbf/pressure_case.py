"""Isolated, single-instance checks of dynamic volume-map pressure exchange.

Run in its own process: PBF deliberately specializes module constants once.
The analytic plane avoids mesh baking, projection, limiting and contacts when
checking the pressure impulse itself.
"""

import json
from pathlib import Path
from unittest.mock import patch

import newton
import numpy as np
import warp as wp

from PBF import PBF as kernels, boundary_model as boundary
from PBF.pbf_helper import PBFConfig


@wp.kernel(enable_backward=False)
def fill_plane(sdf: wp.uint64, volume_map: wp.uint64, tiles: wp.array(dtype=wp.vec3i)):
    i = wp.tid()
    local = i % 512
    cell = tiles[i // 512] + wp.vec3i(local // 64, (local // 8) % 8, local % 8)
    point = wp.volume_index_to_world(sdf, wp.vec3(float(cell[0]), float(cell[1]), float(cell[2])))
    wp.volume_store_f(sdf, cell[0], cell[1], cell[2], point[1])
    wp.volume_store_f(volume_map, cell[0], cell[1], cell[2], .00002)


def main():
    from newton.solvers import SolverMuJoCo
    from PBF.coupling import add_boundary_forces

    wp.init()
    with wp.ScopedDevice("cuda:0"):
        config = PBFConfig(gravity=(0., 0., 0.))
        mass = .8 * (2 * config.particle_radius)**3 * config.rest_density
        tiles = wp.array([[x, y, z] for x in (0, 8) for y in (0, 8) for z in (0, 8)], dtype=wp.vec3i)
        sdf = wp.Volume.allocate_by_tiles(tiles, .05, bg_value=1., translation=(-.2,) * 3)
        volume = wp.Volume.allocate_by_tiles(tiles, .05, bg_value=0., translation=(-.2,) * 3)
        wp.launch(fill_plane, len(tiles) * 512, inputs=[sdf.id, volume.id, tiles])
        empty_sdf = wp.Volume.allocate_by_tiles(tiles, .05, bg_value=1., translation=(-.2,) * 3)
        empty_volume = wp.Volume.allocate_by_tiles(tiles, .05, bg_value=0., translation=(-.2,) * 3)
        assets = (
            boundary.BoundaryAsset(Path("empty.nvdb"), Path("empty.obj"), empty_sdf, empty_volume),
            boundary.BoundaryAsset(Path("plane.nvdb"), Path("plane.obj"), sdf, volume),
        )
        # This analytic fixture represents a finite patch of a plane, not an OBJ.
        mesh = (np.array([[-.2, -.2, -.2], [.6, .6, .6]]), np.empty((0, 3), dtype=int))
        with patch.object(boundary.BoundarySystem, "mesh_data", return_value=mesh):
            boundaries = boundary.BoundarySystem(assets, (
                boundary.BoundaryInstance(0), boundary.BoundaryInstance(1, 0),
                boundary.BoundaryInstance(1, 1)), "cuda:0")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.gravity = (0., 0., 0.)
        inertia = wp.mat33(1., 0., 0., 0., 1., 0., 0., 0., 1.)
        builder.add_body(mass=1., inertia=inertia, label="unmapped")
        builder.add_body(com=wp.vec3(.02, -.01, .03), mass=1., inertia=inertia, label="pressed")
        builder.add_body(xform=wp.transform((5., 0., 0.), wp.quat_identity()),
                         mass=1., inertia=inertia, label="same_asset_far_away")
        builder.add_particles(pos=[(.1, .02, .04)], vel=[(0., 0., 0.)],
                              mass=[mass], radius=[config.particle_radius])
        model = builder.finalize(device="cuda:0")
        source, target = model.state(), model.state()
        boundaries.bind_bodies(model, [2, 1])
        solver = kernels.PBF(model, config=config, boundaries=boundaries)

        def place(transform, local):
            poses = model.body_q.numpy().copy()
            poses[1] = tuple(transform)
            source.body_q.assign(poses)
            boundaries.device_data.local_transforms.assign([tuple(wp.transform_identity()), tuple(local)])
            boundaries.sync_body_poses(source)
            point = wp.transform_point(transform * local, wp.vec3(.1, .02, .04))
            solver.pre_Pos.assign([tuple(point)])
            solver.lambda_Opt.fill_(-.002)
            solver.n_Cache.NeighborCount.zero_()
            solver.boundary_forces.zero_()
            wp.launch(kernels.volume_query, 1, inputs=[solver.pre_Pos, solver.boundary_data],
                      outputs=[solver.volume_Rho, solver.volume_Grad])
            return np.asarray(point)

        def pressure(dt):
            previous = solver.boundary_forces.numpy().copy()
            wp.launch(kernels.calc_deltaPos, 1,
                      inputs=[solver.pre_Pos, solver.lambda_Opt, solver.n_Cache,
                              solver.volume_Grad], outputs=[solver.delta_Pos])
            np.testing.assert_array_equal(solver.boundary_forces.numpy(), previous)
            wp.launch(kernels.calc_boundaryForce, 1,
                      inputs=[solver.pre_Pos, solver.lambda_Opt, dt, solver.boundary_data])

        dt = .002
        transforms = (
            (wp.transform_identity(), wp.transform_identity()),
            (wp.transform((.2, .3, .1), wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), .7)),
             wp.transform((.03, .04, .02), wp.quat_from_axis_angle(wp.vec3(1., 0., 0.), .2))),
        )
        for transform, local in transforms:
            point = place(transform, local)
            pressure(dt)
            forces = solver.boundary_forces.numpy()
            correction = solver.delta_Pos.numpy()[0]
            np.testing.assert_allclose(forces[1, :3] * dt, -mass * correction / dt, rtol=3e-6, atol=3e-7)
            np.testing.assert_array_equal(forces[[0, 2]], np.zeros((2, 6)))
            normal = np.asarray(wp.transform_vector(transform * local, wp.vec3(0., 1., 0.)))
            boundary_point = point - max(.02 + .125 * .05, .5 * .05) * normal
            com = np.asarray(wp.transform_point(transform, wp.vec3(.02, -.01, .03)))
            np.testing.assert_allclose(forces[1, 3:], np.cross(boundary_point - com, forces[1, :3]),
                                       rtol=3e-5, atol=3e-6)
            np.testing.assert_allclose(boundaries.device_data.world_com.numpy()[1], com, atol=1e-7)
            assert np.dot(forces[1, :3], normal) < 0.
            assert np.linalg.norm(forces[1, 3:]) > 0.
            pressure(dt)
            np.testing.assert_allclose(solver.boundary_forces.numpy(), 2 * forces, rtol=3e-6)
            solver.boundary_forces.zero_()
            pressure(2 * dt)
            np.testing.assert_allclose(solver.boundary_forces.numpy(), .25 * forces, rtol=3e-6)

        place(*transforms[0])
        solver.pre_Pos.assign([(.1, -.01, .04)])
        solver.delta_Pos.zero_()
        wp.launch(kernels.update_prePos, 1, inputs=[solver.delta_Pos, solver.boundary_data], outputs=[solver.pre_Pos])
        np.testing.assert_allclose(solver.pre_Pos.numpy()[0, 1], config.particle_radius, atol=1e-7)
        np.testing.assert_array_equal(solver.boundary_forces.numpy(), np.zeros((3, 6)))

        # PBF owns only particle outputs. The input and rigid sentinel survive.
        snapshots = {name: getattr(source, name).numpy().copy()
                     for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")}
        target.body_qd.fill_(wp.spatial_vector(3.))
        sentinel = target.body_qd.numpy().copy()
        pointers = target.particle_q.ptr, target.particle_qd.ptr
        solver.step(source, target, None, None, dt)
        for name, expected in snapshots.items():
            np.testing.assert_array_equal(getattr(source, name).numpy(), expected)
        np.testing.assert_array_equal(target.body_qd.numpy(), sentinel)
        assert pointers == (target.particle_q.ptr, target.particle_qd.ptr)

        # Pure pressure -> rigid response, with no contacts or gravity.
        place(*transforms[0])
        pressure(dt)
        reaction = solver.boundary_forces.numpy().copy()
        external = np.array([[.01, -.02, .03, 0., 0., 0.],
                             [.1, .2, .3, 0., 0., 0.], [0.] * 6], dtype=np.float32)
        source.body_f.assign(external)
        wp.launch(add_boundary_forces, model.body_count,
                  inputs=[solver.boundary_forces], outputs=[source.body_f])
        np.testing.assert_allclose(source.body_f.numpy(), external + reaction, rtol=1e-6, atol=1e-7)
        particle_outputs = target.particle_q.numpy().copy(), target.particle_qd.numpy().copy()
        for use_cpu in (False, True):
            rigid = SolverMuJoCo(model, use_mujoco_cpu=use_cpu, use_mujoco_contacts=True,
                                 disable_contacts=True, enable_sleeping=False,
                                 update_data_interval=1, integrator="euler")
            rigid.step(source, target, model.control(), None, dt)
            velocity = target.body_qd.numpy()[1]
            expected_omega = np.linalg.solve(model.body_inertia.numpy()[1], reaction[1, 3:]) * dt
            np.testing.assert_allclose(velocity[3:], expected_omega, rtol=2e-4, atol=2e-6)
            expected_linear = (external[1, :3] + reaction[1, :3]) * dt / model.body_mass.numpy()[1]
            np.testing.assert_allclose(velocity[:3], expected_linear, rtol=2e-4, atol=2e-6)
            np.testing.assert_array_equal(target.particle_q.numpy(), particle_outputs[0])
            np.testing.assert_array_equal(target.particle_qd.numpy(), particle_outputs[1])

        report = dict(pressure_impulse_balance=True, translated_rotated_asset=True,
                      nonzero_asset_offset=True, eccentric_com_torque=True, body_mapping=True,
                      shared_asset_independent=True, iteration_accumulation=True, dt_squared=True,
                      projection_without_reaction=True, field_ownership=True,
                      external_force_preserved=True, mujoco_pressure_rotation=True, cpu_pressure_rotation=True)
        output = Path(__file__).resolve().parents[2] / "outputs/pbf-newton/pressure.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
