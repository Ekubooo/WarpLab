"""Independent formula and rigid-body checks; CPU by default, CUDA parity if available."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from . import PBF2WayCoupling as solver
from . import coupling_initialization as init
from . import coupling_functions as fn
from .simulation import diagnostics


@wp.kernel
def evaluate_kernels(
    x: wp.array(dtype=wp.vec3),
    h: float,
    values: wp.array(dtype=float),
    gradients: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    values[i] = fn.poly6(wp.length(x[i]), h)
    gradients[i] = fn.spiky_gradient(x[i], h)


@wp.kernel
def query_distances(mesh: wp.uint64, x: wp.array(dtype=wp.vec3), distances: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_normal(mesh, x[i], 10.0)
    p = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    distances[i] = q.sign * wp.length(p - x[i])


def w(r, h):
    return 315 / (64 * np.pi * h**9) * max(h * h - r * r, 0) ** 3


def grad(x, h):
    r = np.linalg.norm(x)
    return -45 / (np.pi * h**6) * (h - r) ** 2 * x / r if 1e-9 < r < h else np.zeros(3)


class CouplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.scene = Path(cls.temp.name) / "small.json"
        assets = init.ROOT / "assets"
        cls.scene.write_text(
            json.dumps(
                {
                    "RigidBodies": [
                        {
                            "geometryFile": str(assets / "UnitBox.obj"),
                            "scale": [0.8, 1, 0.8],
                            "translation": [0, 0.5, 0],
                            "isWall": True,
                        },
                        {
                            "geometryFile": str(assets / "sphere.obj"),
                            "scale": [0.12] * 3,
                            "translation": [0.12, 0.35, 0],
                            "density": 500,
                            "isDynamic": True,
                        },
                    ],
                    "FluidBlocks": [{"start": [-0.3, 0, -0.3], "end": [0.3, 0.6, 0.3]}],
                }
            )
        )
        cls.config = solver.PBF2WayCouplingConfig(scene=str(cls.scene), particle_radius=0.05)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make(self, **overrides):
        return solver.Example(replace(self.config, **overrides), device="cpu")

    @staticmethod
    def initial_order(simulation, array):
        values = array.numpy()
        particle_ids = simulation.particle_ids.numpy()
        restored = np.empty_like(values)
        restored[particle_ids] = values
        return restored

    def search(self, s):
        s.fluid_grid.build(s.positions, s.support_radius)
        s.boundary_grid.build(s.boundary.position, s.support_radius)
        wp.launch(
            solver.cache_neighbors,
            s.num_particles,
            [
                s.fluid_grid.id,
                s.boundary_grid.id,
                s.positions,
                s.old_to_sorted,
                s.boundary,
                s.support_radius,
                s.neighbors,
            ],
            device=s.device,
        )

    def solve_contacts(self, s):
        wp.launch(solver.prepare_rigid_contacts, len(s.body_models), [s.rigid], device=s.device)
        wp.launch(
            solver.prepare_contacts, s.config.max_contacts, [s.rigid, s.contacts], device=s.device
        )
        wp.launch(solver.solve_contacts, 1, [s.rigid, s.contacts, 5], device=s.device)

    def test_fixed_step_counts_and_no_readback(self):
        s = self.make()
        with patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected GPU readback")):
            with patch.object(wp, "launch", wraps=wp.launch) as launches:
                s.step()
            self.assertEqual(
                sum(call.args[0] is solver.predict for call in launches.call_args_list), 3
            )
            self.assertEqual(
                sum(call.args[0] is solver.reorder_fluid for call in launches.call_args_list), 3
            )
            self.assertEqual(
                sum(call.args[0] is solver.pressure_correction for call in launches.call_args_list),
                9,
            )
            self.assertEqual(
                sum(call.args[0] is solver.solve_contacts for call in launches.call_args_list), 3
            )
            for _ in range(89):
                s.step()
        self.assertEqual(s.sim_time, 1.0)
        self.assertEqual(s.total_steps, 90)
        self.assertEqual(s.total_substeps, 270)
        self.assertEqual(s.current_dt, 1 / 90)
        self.assertEqual(s.substep_dt, 1 / 270)

    def test_hash_order_reorders_persistent_state_and_inverse_map(self):
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                s = solver.Example(self.config, device=device)
                count = s.num_particles
                positions = s.positions.numpy().copy()
                velocities = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
                old_positions = -positions
                s.velocities.assign(velocities)
                s.old_positions.assign(old_positions)
                s.fluid_grid.build(s.positions, s.support_radius)
                s._reorder_fluid()

                particle_ids = s.particle_ids.numpy()
                np.testing.assert_array_equal(np.sort(particle_ids), np.arange(count))
                np.testing.assert_array_equal(s.positions.numpy(), positions[particle_ids])
                np.testing.assert_array_equal(s.velocities.numpy(), velocities[particle_ids])
                np.testing.assert_array_equal(s.old_positions.numpy(), old_positions[particle_ids])
                inverse = s.old_to_sorted.numpy()
                np.testing.assert_array_equal(particle_ids[inverse], np.arange(count))

    def test_hash_grid_dimensions_cover_container_query_span(self):
        s = self.make()
        query_min = np.floor((s.container_min - s.support_radius) / s.support_radius)
        query_max = np.floor((s.container_max + s.support_radius) / s.support_radius)
        span = (query_max - query_min + 1).astype(int)
        self.assertEqual(solver.HASH_GRID_DIMS, (128, 160, 128))
        self.assertTrue(
            np.all(span < np.asarray(solver.HASH_GRID_DIMS)),
            (span, solver.HASH_GRID_DIMS),
        )

    def test_hash_grid_rejects_periodic_aliasing_span(self):
        with patch.object(solver, "HASH_GRID_DIMS", (1, 160, 128)):
            with self.assertRaisesRegex(ValueError, "must exceed the container query span"):
                self.make()

    def test_reordered_neighbor_cache_matches_geometry(self):
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                s = solver.Example(self.config, device=device)
                s.fluid_grid.build(s.positions, s.support_radius)
                s.boundary_grid.build(s.boundary.position, s.support_radius)
                s._reorder_fluid()
                wp.launch(
                    solver.cache_neighbors,
                    s.num_particles,
                    [
                        s.fluid_grid.id,
                        s.boundary_grid.id,
                        s.positions,
                        s.old_to_sorted,
                        s.boundary,
                        s.support_radius,
                        s.neighbors,
                    ],
                    device=s.device,
                )
                positions = s.positions.numpy()
                boundary = s.boundary.position.numpy()
                particle_ids = s.particle_ids.numpy()
                fluid_neighbors = s.neighbors.fluid.numpy()
                fluid_counts = s.neighbors.fluid_count.numpy()
                boundary_neighbors = s.neighbors.boundary.numpy()
                boundary_counts = s.neighbors.boundary_count.numpy()
                for i, position in enumerate(positions):
                    expected = set(
                        particle_ids[
                            np.flatnonzero(
                                (np.linalg.norm(positions - position, axis=1) < s.support_radius)
                                & (np.arange(s.num_particles) != i)
                            )
                        ]
                    )
                    actual = set(particle_ids[fluid_neighbors[i, : fluid_counts[i]]])
                    self.assertEqual(actual, expected)
                    expected_boundary = set(
                        np.flatnonzero(
                            np.linalg.norm(boundary - position, axis=1) < s.support_radius
                        )
                    )
                    actual_boundary = set(boundary_neighbors[i, : boundary_counts[i]])
                    self.assertEqual(actual_boundary, expected_boundary)

    def test_configurable_fixed_pressure_iterations(self):
        s = self.make(pressure_iterations=4)
        for iterations in (4, 2):
            s.config = replace(s.config, pressure_iterations=iterations)
            with patch.object(wp, "launch", wraps=wp.launch) as launches:
                s.step()
            self.assertEqual(s.iterations, iterations)
            self.assertEqual(
                sum(call.args[0] is solver.pressure_correction for call in launches.call_args_list),
                s.config.substeps * iterations,
            )

    def test_latched_nonfinite_fault(self):
        s = self.make()
        velocity = s.rigid.velocity.numpy()
        velocity[1, 0] = np.nan
        s.rigid.velocity.assign(velocity)
        # Actual device audit must discover the invalid velocity before mesh queries.
        with patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected readback")):
            s.step()
        self.assertEqual(s.invalid_state.numpy()[0], 1)
        positions = s.positions.numpy().copy()
        particle_ids = s.particle_ids.numpy().copy()
        rotations = s.rigid.rotation.numpy().copy()
        with patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected readback")):
            s.step()
        with self.assertRaises(FloatingPointError):
            diagnostics(s)
        np.testing.assert_array_equal(s.positions.numpy(), positions)
        np.testing.assert_array_equal(s.particle_ids.numpy(), particle_ids)
        np.testing.assert_array_equal(s.rigid.rotation.numpy(), rotations)
        self.assertEqual(s.rigid.fault.numpy()[1], 1)

    def test_obj_mass_inertia_and_sampling(self):
        v, f = init.load_obj(init.ROOT / "assets/UnitBox.obj")
        v = v * [2, 3, 4] + [5, -2, 7]
        mass, center, inertia = init.mass_properties(v, f, 10)
        self.assertAlmostEqual(mass, 240, places=9)
        np.testing.assert_allclose(center, [5, -2, 7], atol=1e-12)
        np.testing.assert_allclose(inertia, np.diag([500, 400, 260]), atol=1e-9)
        v, f = init.load_obj(init.ROOT / "assets/sphere.obj")
        mass, center, inertia = init.mass_properties(v * 0.2, f, 1000)
        self.assertAlmostEqual(mass / (1000 * 4 * np.pi * 0.2**3 / 3), 1, delta=0.03)
        np.testing.assert_allclose(np.diag(inertia), np.full(3, 0.4 * mass * 0.2**2), rtol=0.03)
        samples = init.sample_surface(v, f, 0.2)
        np.testing.assert_array_equal(samples, init.sample_surface(v, f, 0.2))
        self.assertEqual(len(np.unique(np.round(samples, 5), axis=0)), len(samples))

    def test_kernel_origin_support_edge_and_outside(self):
        s = self.make()
        x = np.array(
            [[0, 0, 0], [0.03, 0.02, 0.01], [0.199999, 0, 0], [0.2, 0, 0], [0.3, 0, 0]],
            dtype=np.float32,
        )
        values = wp.zeros(len(x), dtype=float, device="cpu")
        gradients = wp.zeros(len(x), dtype=wp.vec3, device="cpu")
        wp.launch(
            evaluate_kernels,
            len(x),
            [wp.array(x, dtype=wp.vec3, device="cpu"), 0.2, values, gradients],
            device="cpu",
        )
        np.testing.assert_allclose(
            values.numpy(),
            [w(np.linalg.norm(a.astype(float)), 0.2) for a in x],
            rtol=1e-5,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            gradients.numpy(), [grad(a.astype(float), 0.2) for a in x], rtol=1e-5, atol=1e-7
        )

    def test_torus_hole_is_free_space(self):
        self.make()  # Initialize Warp and its local cache.
        v, f = init.load_obj(init.ROOT / "assets/torus.obj")
        mesh = wp.Mesh(
            wp.array(v * 0.2, dtype=wp.vec3, device="cpu"),
            wp.array(f.flatten(), dtype=int, device="cpu"),
        )
        # The official torus lies in XZ, with major radius 1 and tube radius .5.
        points = wp.array([[0, 0, 0], [0.2, 0, 0], [0.4, 0, 0]], dtype=wp.vec3, device="cpu")
        distances = wp.zeros(3, dtype=float, device="cpu")
        wp.launch(query_distances, 3, [mesh.id, points, distances], device="cpu")
        self.assertGreater(distances.numpy()[0], 0.09)
        self.assertLess(distances.numpy()[1], -0.09)
        self.assertGreater(distances.numpy()[2], 0.09)

    def test_standard_viscosity_and_reaction_against_numpy(self):
        s = self.make(boundary_viscosity=0.04)
        rng = np.random.default_rng(7)
        vel = rng.normal(0, 0.2, (s.num_particles, 3)).astype(np.float32)
        s.velocities.assign(vel)
        self.search(s)
        wp.launch(
            solver.density_lambda,
            s.num_particles,
            [
                s.positions,
                s.boundary,
                s.neighbors,
                s.fluid_volume,
                s.support_radius,
                s.densities,
                s.lambdas,
            ],
            device="cpu",
        )
        wp.launch(
            solver.viscosity,
            s.num_particles,
            [
                s.positions,
                s.velocities,
                s.boundary,
                s.rigid,
                s.neighbors,
                s.densities,
                s.fluid_volume,
                s.fluid_mass,
                s.support_radius,
                0.01,
                0.04,
                1,
                s.acceleration,
            ],
            device="cpu",
        )
        x = s.positions.numpy().astype(float)
        bx = s.boundary.position.numpy().astype(float)
        bv = s.boundary.volume.numpy()
        density = s.densities.numpy()
        ids = s.boundary.body.numpy()
        acc = np.zeros_like(x)
        force = np.zeros((2, 3))
        h = s.support_radius
        for i, xi in enumerate(x):
            for j in np.flatnonzero(np.linalg.norm(x - xi, axis=1) < h):
                if i != j:
                    delta = xi - x[j]
                    acc[i] += (
                        10
                        * 0.01
                        * s.fluid_volume
                        / density[j]
                        * np.dot(vel[i] - vel[j], delta)
                        / (delta @ delta + 0.01 * h * h)
                        * grad(delta, h)
                    )
            for j in np.flatnonzero(np.linalg.norm(bx - xi, axis=1) < h):
                delta = xi - bx[j]
                a = (
                    10
                    * 0.04
                    * bv[j]
                    / density[i]
                    * np.dot(vel[i], delta)
                    / (delta @ delta + 0.01 * h * h)
                    * grad(delta, h)
                )
                acc[i] += a
                if ids[j] == 1:
                    force[1] -= s.fluid_mass * a
        np.testing.assert_allclose(s.acceleration.numpy(), acc, rtol=1e-5, atol=3e-6)
        np.testing.assert_allclose(s.rigid.force.numpy(), force, rtol=2e-5, atol=3e-6)

    def test_density_lambda_correction_and_force_against_brute_force(self):
        s = self.make()
        # Compress the cloud to produce positive density constraints.
        points = s.positions.numpy()
        points[:, 0] *= 0.6
        s.positions.assign(points)
        self.search(s)
        self.assertFalse(s.neighbors.overflow.numpy().any())
        wp.launch(
            solver.density_lambda,
            s.num_particles,
            [
                s.positions,
                s.boundary,
                s.neighbors,
                s.fluid_volume,
                s.support_radius,
                s.densities,
                s.lambdas,
            ],
            device=s.device,
        )
        x, bx, bv = (
            points.astype(float),
            s.boundary.position.numpy().astype(float),
            s.boundary.volume.numpy().astype(float),
        )
        h, volume, mass = s.support_radius, s.fluid_volume, s.fluid_mass
        densities, lambdas = [], []
        for i, xi in enumerate(x):
            rho = volume * w(0, h)
            gi, denominator = np.zeros(3), 0.0
            for j in np.flatnonzero(np.linalg.norm(x - xi, axis=1) < h):
                if i == j:
                    continue
                rho += volume * w(np.linalg.norm(xi - x[j]), h)
                gj = -volume * grad(xi - x[j], h)
                gi -= gj
                denominator += gj @ gj
            for j in np.flatnonzero(np.linalg.norm(bx - xi, axis=1) < h):
                rho += bv[j] * w(np.linalg.norm(xi - bx[j]), h)
                gi += bv[j] * grad(xi - bx[j], h)
            densities.append(rho)
            lambdas.append(-max(rho - 1, 0) / (denominator + gi @ gi + 1e-6))
        np.testing.assert_allclose(s.densities.numpy(), densities, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(s.lambdas.numpy(), lambdas, rtol=3e-5, atol=1e-8)
        self.assertGreater(max(densities), 1.05)
        correction = np.zeros_like(x)
        forces, torques = np.zeros((2, 3)), np.zeros((2, 3))
        bodies = s.boundary.body.numpy()
        centers = s.rigid.position.numpy()
        for i, xi in enumerate(x):
            for j in np.flatnonzero(np.linalg.norm(x - xi, axis=1) < h):
                if i != j:
                    correction[i] += (lambdas[i] + lambdas[j]) * volume * grad(xi - x[j], h)
            for j in np.flatnonzero(np.linalg.norm(bx - xi, axis=1) < h):
                dx = lambdas[i] * bv[j] * grad(xi - bx[j], h)
                correction[i] += dx
                b = bodies[j]
                if b == 1:
                    force = -mass * dx / s.substep_dt**2
                    forces[b] += force
                    torques[b] += np.cross(bx[j] - centers[b], force)
        wp.launch(
            solver.pressure_correction,
            s.num_particles,
            [
                s.positions,
                s.boundary,
                s.rigid,
                s.neighbors,
                volume,
                mass,
                h,
                s.substep_dt,
                1,
                s.lambdas,
                s.corrections,
            ],
            device=s.device,
        )
        np.testing.assert_allclose(s.corrections.numpy(), correction, rtol=2e-4, atol=1e-7)
        np.testing.assert_allclose(s.rigid.force.numpy(), forces, rtol=1e-4, atol=0.1)
        np.testing.assert_allclose(s.rigid.torque.numpy(), torques, rtol=1e-4, atol=0.02)

    def test_boundary_volume_groups_and_rigid_transformation(self):
        s = self.make()
        bx, group = s.boundary.position.numpy(), s.boundary.body.numpy()
        volume = s.boundary.volume.numpy()
        for i in [0, 100, len(bx) - 1]:
            expected = 1 / sum(
                w(float(np.linalg.norm(bx[i] - bx[j])), s.support_radius)
                for j in np.flatnonzero(group == group[i])
            )
            self.assertAlmostEqual(volume[i] / expected, 1, delta=2e-5)
        q = init.rotation_quaternion([0, 1, 0], 0.7)
        s.rigid.rotation.assign(np.array([[0, 0, 0, 1], q], dtype=np.float32))
        s.rigid.velocity.assign(np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32))
        s.rigid.omega.assign(np.array([[0, 0, 0], [0, 2, 0]], dtype=np.float32))
        wp.launch(solver.update_boundary, len(bx), [s.rigid, s.boundary], device=s.device)
        r = s.boundary.local.numpy()[group == 1] @ init.rotation_matrix(q).T
        np.testing.assert_allclose(
            s.boundary.position.numpy()[group == 1], r + s.rigid.position.numpy()[1], atol=1e-7
        )
        np.testing.assert_allclose(
            s.boundary.velocity.numpy()[group == 1], np.cross([0, 2, 0], r) + [1, 2, 3], atol=5e-7
        )
        np.testing.assert_array_equal(s.boundary.volume.numpy(), volume)

    def test_pair_impulse_balance_and_offcenter_torque(self):
        s = self.make()
        # Artificial single boundary sample isolates the sign, dt^2 and lever arm.
        b = int(np.flatnonzero(s.boundary.body.numpy() == 1)[0])
        bx = s.boundary.position.numpy()[b]
        x = s.positions.numpy()
        x[0] = bx + [0.04, 0.03, 0.02]
        s.positions.assign(x)
        ids = s.neighbors.boundary.numpy()
        ids[0, 0] = b
        s.neighbors.boundary.assign(ids)
        count = np.zeros(s.num_particles, dtype=np.int32)
        count[0] = 1
        s.neighbors.boundary_count.assign(count)
        lambdas = np.zeros(s.num_particles, dtype=np.float32)
        lambdas[0] = -0.002
        s.lambdas.assign(lambdas)
        wp.launch(
            solver.pressure_correction,
            1,
            [
                s.positions,
                s.boundary,
                s.rigid,
                s.neighbors,
                s.fluid_volume,
                s.fluid_mass,
                s.support_radius,
                0.002,
                1,
                s.lambdas,
                s.corrections,
            ],
            device=s.device,
        )
        fluid_impulse = s.fluid_mass * s.corrections.numpy()[0] / 0.002
        rigid_impulse = s.rigid.force.numpy()[1] * 0.002
        np.testing.assert_allclose(fluid_impulse + rigid_impulse, np.zeros(3), atol=2e-5)
        self.assertGreater(np.linalg.norm(s.rigid.torque.numpy()[1]), 0.01)
        expected_torque = np.cross(bx - s.rigid.position.numpy()[1], s.rigid.force.numpy()[1])
        np.testing.assert_allclose(s.rigid.torque.numpy()[1], expected_torque, atol=1e-3)

    def test_static_body_and_quaternion_after_integration(self):
        s = self.make()
        initial = s.rigid.position.numpy().copy()
        force = np.array([[100, 100, 100], [1, 0, 0]], dtype=np.float32)
        s.rigid.force.assign(force)
        s.rigid.torque.assign(np.array([[100, 100, 100], [0, 0.01, 0]], dtype=np.float32))
        wp.launch(solver.integrate_rigid, 2, [s.rigid, wp.vec3(0, 0, 0), 0.001], device=s.device)
        np.testing.assert_array_equal(s.rigid.position.numpy()[0], initial[0])
        self.assertGreater(s.rigid.velocity.numpy()[1, 0], 0)
        self.assertGreater(s.rigid.omega.numpy()[1, 1], 0)
        np.testing.assert_allclose(np.linalg.norm(s.rigid.rotation.numpy(), axis=1), 1, atol=1e-7)

    def test_neighbor_overflow_is_reported(self):
        s = self.make(max_fluid_neighbors=1, max_boundary_neighbors=1)
        s.step()
        with self.assertRaisesRegex(RuntimeError, "Neighbor capacity"):
            diagnostics(s)
        positions = s.positions.numpy().copy()
        velocities = s.velocities.numpy().copy()
        s.step()
        np.testing.assert_array_equal(s.positions.numpy(), positions)
        np.testing.assert_array_equal(s.velocities.numpy(), velocities)
        self.assertEqual(s.rigid.fault.numpy()[1], 1)

    def test_contact_floor_restitution_and_friction(self):
        s = self.make()
        s.contacts.count.assign(np.array([1], dtype=np.int32))
        for name, value in (("a", 1), ("b", 0), ("gap", -0.001), ("target", 0.6)):
            data = getattr(s.contacts, name).numpy()
            data[0] = value
            getattr(s.contacts, name).assign(data)
        p = s.contacts.point.numpy()
        p[0] = s.rigid.position.numpy()[1] - [0, 0.12, 0]
        s.contacts.point.assign(p)
        normal = s.contacts.normal.numpy()
        normal[0] = [0, 1, 0]
        s.contacts.normal.assign(normal)
        s.rigid.friction.assign(np.array([0.5, 0.5], dtype=np.float32))
        s.rigid.velocity.assign(np.array([[0, 0, 0], [1, -1, 0]], dtype=np.float32))
        self.solve_contacts(s)
        self.assertAlmostEqual(s.rigid.velocity.numpy()[1, 1], 0.6, places=5)
        self.assertLess(s.rigid.velocity.numpy()[1, 0], 1)
        np.testing.assert_array_equal(s.rigid.velocity.numpy()[0], [0, 0, 0])
        self.assertLessEqual(
            np.linalg.norm(s.contacts.tangent_impulse.numpy()[0]),
            0.5 * s.contacts.normal_impulse.numpy()[0] + 1e-6,
        )

    def test_precomputed_fused_contacts_match_original_sweeps(self):
        from .test_contact_reference import reference_contact_sweep

        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                s = solver.Example(self.config, device=device)
                count = 6
                s.contacts.count.assign(np.array([count], dtype=np.int32))
                center = s.rigid.position.numpy()[1]
                normals = np.array(
                    [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [-1, 0, 0], [0, 0, -1]],
                    dtype=np.float32,
                )
                offsets = np.array(
                    [
                        [0.03, -0.12, 0],
                        [-0.1, 0.02, 0],
                        [0.02, 0, -0.11],
                        [-0.02, -0.12, 0.03],
                        [0.1, 0, 0.02],
                        [0, 0.02, 0.1],
                    ],
                    dtype=np.float32,
                )
                for name, values in (
                    ("a", np.ones(count)),
                    ("b", np.zeros(count)),
                    ("point", center + offsets),
                    ("normal", normals),
                    ("target", np.linspace(0, 0.2, count)),
                ):
                    data = getattr(s.contacts, name).numpy()
                    data[:count] = values
                    getattr(s.contacts, name).assign(data)
                s.rigid.velocity.assign(np.array([[0, 0, 0], [0.8, -1, 0.3]], dtype=np.float32))
                s.rigid.omega.assign(np.array([[0, 0, 0], [0.2, 0.3, -0.4]], dtype=np.float32))
                fields = (
                    s.rigid.velocity,
                    s.rigid.omega,
                    s.contacts.normal_impulse,
                    s.contacts.tangent_impulse,
                )
                before = [a.numpy().copy() for a in fields]
                for _ in range(5):
                    wp.launch(reference_contact_sweep, 1, [s.rigid, s.contacts], device=device)
                expected = [a.numpy().copy() for a in fields]
                for a, data in zip(fields, before):
                    a.assign(data)
                self.solve_contacts(s)
                for a, data in zip(fields, expected):
                    np.testing.assert_allclose(a.numpy(), data, rtol=2e-5, atol=2e-6)

    def test_mesh_contacts_and_overflow(self):
        s = self.make(max_contacts=1)
        p = s.rigid.position.numpy()
        p[1, 1] = 0.1  # Sphere radius .12: intersects the floor.
        s.rigid.position.assign(p)
        wp.launch(
            solver.update_boundary, len(s.boundary.local), [s.rigid, s.boundary], device="cpu"
        )
        wp.launch(
            solver.detect_contacts,
            (len(s.boundary.local), 2),
            [s.rigid, s.boundary, s.contacts, 0.06, 0.002],
            device="cpu",
        )
        self.assertEqual(s.contacts.overflow.numpy()[0], 1)
        self.assertGreater(s.contacts.count.numpy()[0], 1)
        wp.launch(
            solver.check_faults, 1, [s.neighbors, s.contacts, s.invalid_state], device=s.device
        )
        with self.assertRaisesRegex(RuntimeError, "Contact capacity"):
            diagnostics(s)

    def test_mutual_mesh_contact_transfers_momentum(self):
        data = json.loads(self.scene.read_text())
        second = dict(data["RigidBodies"][1])
        data["RigidBodies"][1]["translation"] = [-0.11, 0.4, 0]
        second["translation"] = [0.11, 0.4, 0]
        data["RigidBodies"].append(second)
        path = Path(self.temp.name) / "pair.json"
        path.write_text(json.dumps(data))
        s = self.make(scene=str(path))
        s.rigid.velocity.assign(np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.float32))
        wp.launch(
            solver.update_boundary, len(s.boundary.local), [s.rigid, s.boundary], device="cpu"
        )
        wp.launch(
            solver.detect_contacts,
            (len(s.boundary.local), 3),
            [s.rigid, s.boundary, s.contacts, 0.06, 0.002],
            device="cpu",
        )
        self.assertGreater(s.contacts.count.numpy()[0], 0)
        count = int(s.contacts.count.numpy()[0])
        self.assertTrue(np.any(s.contacts.gap.numpy()[:count] < 0))
        self.solve_contacts(s)
        v = s.rigid.velocity.numpy()
        masses = np.array([b.mass for b in s.body_models])
        np.testing.assert_allclose((masses[:, None] * v).sum(axis=0), [0, 0, 0], atol=2e-5)
        self.assertLess(v[1, 0], 0)
        self.assertGreater(v[2, 0], 0)

    def test_cpu_cuda_short_trajectory(self):
        cpu = self.make()
        if not wp.is_cuda_available():
            self.skipTest("CUDA not available")
        gpu = solver.Example(self.config, device="cuda:0")
        for _ in range(20):
            cpu.step()
            gpu.step()
        np.testing.assert_allclose(
            self.initial_order(cpu, cpu.positions),
            self.initial_order(gpu, gpu.positions),
            rtol=2e-4,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            cpu.rigid.position.numpy(), gpu.rigid.position.numpy(), atol=5e-5
        )
        self.assertEqual(cpu.iterations, gpu.iterations)
        self.assertEqual(cpu.iterations, 3)
        self.assertTrue(np.isfinite(diagnostics(cpu)["density_error_percent"]))

    def test_invalid_configuration(self):
        for overrides in (
            {"particle_radius": 0},
            {"gravity": (0, float("nan"), 0)},
            {"viscosity": -1},
            {"frame_dt": 0},
            {"substeps": 0},
            {"substeps": 4},
            {"frame_dt": 1 / 30},
            {"pressure_iterations": 0},
            {"pressure_iterations": -1},
            {"pressure_iterations": 1.5},
            {"contact_iterations": 4},
            {"max_contacts": 0},
        ):
            with self.assertRaises(ValueError):
                replace(self.config, **overrides)


if __name__ == "__main__":
    unittest.main()
