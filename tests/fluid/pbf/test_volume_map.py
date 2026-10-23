"""Gauss volume integration and NanoVDB coordinate/background regression tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from demos.fluid.PBF import boundary_model as boundary
from demos.fluid.PBF import pbf_helper as fn
from tests.fluid.pbf.solver_fixture import bind_solver, self_neighbor


@wp.kernel
def _probe_occupancy(distances: wp.array(dtype=float), radius: float, values: wp.array(dtype=float)):
    i = wp.tid()
    values[i] = boundary._volume_occupancy(distances[i], radius)


@wp.kernel
def _fill_plane(
    volume: wp.uint64, tiles: wp.array(dtype=wp.vec3i),
    normal: wp.vec3, origin: wp.vec3, distance_offset: float,
):
    tid = wp.tid()
    local = tid % 512
    cell = tiles[tid // 512] + wp.vec3i(local // 64, (local // 8) % 8, local % 8)
    x = wp.volume_index_to_world(volume, wp.vec3(float(cell[0]), float(cell[1]), float(cell[2])))
    wp.volume_store_f(volume, cell[0], cell[1], cell[2], wp.dot(x - origin, normal) + distance_offset)


@wp.kernel
def _lookup(volume: wp.uint64, cells: wp.array(dtype=wp.vec3i), values: wp.array(dtype=float)):
    i = wp.tid()
    cell = cells[i]
    values[i] = wp.volume_lookup_f(volume, cell[0], cell[1], cell[2])


@wp.kernel
def _probe_volume_boundary(
    sdf: wp.uint64, positions: wp.array(dtype=wp.vec3), radius: float,
    valid: wp.array(dtype=int), distances: wp.array(dtype=float), directions: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    ok, distance, direction = fn.query_volume_boundary(sdf, positions[i], radius)
    valid[i] = int(ok)
    distances[i] = distance
    directions[i] = direction


@wp.kernel
def _probe_collision(
    sdf: wp.uint64, positions: wp.array(dtype=wp.vec3), radius: float,
    corrected: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    corrected[i] = fn.apply_SDF(sdf, positions[i], radius)


@wp.kernel
def _probe_position_kernels(
    positions: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=wp.vec2), gradients: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    origin = wp.vec3(0.0)
    position = positions[i]
    values[i] = wp.vec2(fn.Poly6(origin, position), fn.Poly6(position, origin))
    gradients[i] = fn.GradPow3(origin, position)


def _occupancy_reference(distance, radius):
    q = np.maximum(np.asarray(distance, dtype=np.float64) / radius, 0)
    return np.where(q <= 0.5, 1 - 6 * q**2 + 6 * q**3,
                    np.where(q < 1, 2 * (1 - q)**3, 0))


class VolumeQuadratureTests(unittest.TestCase):
    def test_default_ball_points_symmetry_and_cubic_scaling(self):
        radius = 0.05
        offsets, weights = boundary._build_volume_quadrature(radius, 16)
        self.assertEqual(offsets.shape, (1008, 3))
        self.assertEqual(weights.shape, (1008,))
        self.assertEqual(offsets.dtype, np.float32)
        self.assertEqual(weights.dtype, np.float32)
        self.assertTrue(np.all(np.linalg.norm(offsets, axis=1) <= radius))
        self.assertTrue(np.all(weights > 0))
        np.testing.assert_allclose(offsets, -offsets[::-1], atol=1e-9)
        np.testing.assert_array_equal(weights, weights[::-1])
        larger_offsets, larger_weights = boundary._build_volume_quadrature(2 * radius, 16)
        np.testing.assert_allclose(larger_offsets, 2 * offsets, rtol=1e-6)
        np.testing.assert_allclose(larger_weights, 8 * weights, rtol=1e-6)
        # Cube quadrature with a hard ball cutoff is not exact sphere cubature.
        self.assertAlmostEqual(weights.sum(dtype=np.float64) / radius**3, 4.29907477768956, places=6)
        self.assertAlmostEqual(weights.sum(dtype=np.float64) / (4 * np.pi * radius**3 / 3),
                               1.0263285023864768, places=6)

    def test_invalid_radius_and_point_count(self):
        for radius in (0, -0.1, float("nan"), float("inf")):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, "support_radius"):
                boundary._build_volume_quadrature(radius, 16)
        for count in (0, -1, 2.5, True, "16"):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "quadrature_points"):
                boundary._build_volume_quadrature(0.05, count)
        offsets, weights = boundary._build_volume_quadrature(0.05, np.int64(1))
        np.testing.assert_array_equal(offsets, np.zeros((1, 3)))
        np.testing.assert_allclose(weights, [8 * 0.05**3])
        with self.assertRaisesRegex(TypeError, "wp.Volume"):
            boundary.precompute_volumeMap(None, 0.05)

    def test_invalid_quadrature_result_is_rejected_before_return(self):
        for weight in (0.0, float("nan"), float("inf")):
            with self.subTest(weight=weight), \
                    patch.object(np.polynomial.legendre, "leggauss",
                                 return_value=(np.array([0.0]), np.array([weight]))), \
                    self.assertRaisesRegex(ValueError, "finite float32 weights"):
                boundary._build_volume_quadrature(0.05, 1)

    def test_occupancy_values_and_piecewise_joins_on_cpu(self):
        wp.init()
        radius = 0.2
        fractions = np.array([-1, 0, 1e-5, 0.25, 0.5 - 1e-5, 0.5,
                              0.5 + 1e-5, 0.75, 1 - 1e-5, 1, 2], dtype=np.float32)
        distances = wp.array(radius * fractions, dtype=float, device="cpu")
        values = wp.empty(len(fractions), dtype=float, device="cpu")
        wp.launch(_probe_occupancy, len(fractions), inputs=[distances, radius, values], device="cpu")
        actual = values.numpy()
        np.testing.assert_allclose(actual, _occupancy_reference(radius * fractions, radius), atol=2e-7)
        np.testing.assert_allclose(actual[[0, 1, 5, 7, 9, 10]], [1, 1, 0.25, 0.03125, 0, 0])
        self.assertLess(abs(actual[4] - actual[6]), 4e-5)


class VolumeMapGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is required for NanoVDB volume maps")
        cls.device = wp.get_device("cuda:0")

    def allocate(self, tiles, *, background, translation=(0, 0, 0), transform=None):
        return wp.Volume.allocate_by_tiles(
            wp.array(tiles, dtype=wp.vec3i, device=self.device),
            voxel_size=0.1 if transform is None else None, translation=translation, transform=transform,
            bg_value=background, device=self.device,
        )

    def fill(self, volume, normal=(0, 0, 0), origin=(0, 0, 0), offset=0):
        tiles = wp.empty(volume.get_tile_count(), dtype=wp.vec3i, device=self.device)
        volume.get_tiles(out=tiles)
        wp.launch(_fill_plane, 512 * len(tiles),
                  inputs=[volume.id, tiles, wp.vec3(*normal), wp.vec3(*origin), float(offset)],
                  device=self.device)

    def lookup(self, volume, cells):
        cells = wp.array(cells, dtype=wp.vec3i, device=self.device)
        values = wp.empty(len(cells), dtype=float, device=self.device)
        wp.launch(_lookup, len(cells), inputs=[volume.id, cells, values], device=self.device)
        return values.numpy()

    def probe_boundary(self, sdf, positions, radius):
        positions = wp.array(positions, dtype=wp.vec3, device=self.device)
        valid = wp.empty(len(positions), dtype=int, device=self.device)
        distances = wp.empty(len(positions), dtype=float, device=self.device)
        directions = wp.empty(len(positions), dtype=wp.vec3, device=self.device)
        wp.launch(_probe_volume_boundary, len(positions),
                  inputs=[sdf.id, positions, radius, valid, distances, directions], device=self.device)
        return valid.numpy(), distances.numpy(), directions.numpy()

    def test_runtime_rejects_degenerate_and_nonfinite_samples(self):
        sdf = self.allocate([[0, 0, 0]], background=1.0)
        for slope, offset in ((0.0, 0.02), (1e-7, 0.02), (float("nan"), 0.02),
                              (0.0, float("inf")), (0.0, -0.01), (0.0, 0.0), (0.0, 0.05)):
            with self.subTest(slope=slope, offset=offset):
                self.fill(sdf, normal=(0, slope, 0), offset=offset)
                valid, distances, directions = self.probe_boundary(sdf, [[0.15, 0.2, 0.25]], 0.05)
                np.testing.assert_array_equal(valid, [0])
                np.testing.assert_array_equal(distances, [0])
                np.testing.assert_array_equal(directions, [[0, 0, 0]])
        self.fill(sdf, normal=(0, 1e-5, 0), offset=0.02)
        valid, _, directions = self.probe_boundary(sdf, [[0.15, 0.2, 0.25]], 0.05)
        np.testing.assert_array_equal(valid, [1])
        np.testing.assert_allclose(directions, [[0, -1, 0]], atol=1e-6)

    def test_pressure_distance_protection_uses_true_distance_for_support(self):
        sdf = self.allocate([[0, 0, 0]], background=1.0)
        self.fill(sdf, normal=(0, 1, 0), origin=(0, 0.2, 0))
        distances = np.array([-0.01, 0, 0.001, 0.018, 0.019, 0.025, 0.049, 0.051])
        valid, protected, directions = self.probe_boundary(
            sdf, [[0.15, 0.2 + d, 0.15] for d in distances], 0.05,
        )
        inside = (distances > 0) & (distances < 0.05)
        expected = np.where(inside, np.maximum(distances + 0.00625, 0.025), 0)
        np.testing.assert_array_equal(valid, inside.astype(int))
        np.testing.assert_allclose(protected, expected, atol=2e-7)
        np.testing.assert_allclose(directions[inside], np.tile([0, -1, 0], (inside.sum(), 1)), atol=1e-6)
        np.testing.assert_array_equal(directions[~inside], 0)

    def test_collision_preserves_positions_without_valid_normals(self):
        sdf = self.allocate([[0, 0, 0]], background=1.0)
        positions = wp.array([[0.15, 0.02, 0.15], [0.13, 0.023, 0.17]],
                             dtype=wp.vec3, device=self.device)
        corrected = wp.empty(2, dtype=wp.vec3, device=self.device)
        for slope, offset in ((0.0, 0.01), (0.0, -0.01), (1e-7, 0.01),
                              (float("nan"), 0.01), (0.0, float("inf"))):
            with self.subTest(slope=slope, offset=offset):
                self.fill(sdf, normal=(0, slope, 0), offset=offset)
                wp.launch(_probe_collision, 2, inputs=[sdf.id, positions, 0.0125, corrected],
                          device=self.device)
                np.testing.assert_array_equal(corrected.numpy(), positions.numpy())

    def test_collision_projects_penetrated_surface_and_translated_plane(self):
        sdf = self.allocate([[0, 0, 0]], background=1.0)
        self.fill(sdf, normal=(0, 1, 0), origin=(0, 0.2, 0))
        positions = wp.array([[0.15, y, 0.15] for y in (0.18, 0.2, 0.205, 0.25)],
                             dtype=wp.vec3, device=self.device)
        corrected = wp.empty(4, dtype=wp.vec3, device=self.device)
        wp.launch(_probe_collision, 4, inputs=[sdf.id, positions, 0.0125, corrected], device=self.device)
        expected = positions.numpy().copy()
        expected[:3, 1] = 0.2125
        np.testing.assert_allclose(corrected.numpy(), expected, atol=2e-7)

        origin = np.array([1.3, -0.7, 0.4])
        normal = np.array([1, 2, -1]) / np.sqrt(6)
        sdf = self.allocate([[0, 0, 0]], background=1.0, translation=origin)
        self.fill(sdf, normal=normal, origin=origin)
        relative = np.array([0.15, 0.15, 0.15])
        position = origin + relative
        positions = wp.array([position], dtype=wp.vec3, device=self.device)
        corrected = wp.empty(1, dtype=wp.vec3, device=self.device)
        wp.launch(_probe_collision, 1, inputs=[sdf.id, positions, 0.15, corrected], device=self.device)
        expected = position + (0.15 - relative @ normal) * normal
        np.testing.assert_allclose(corrected.numpy(), [expected], atol=2e-7)

    def test_runtime_world_normal_with_translation_and_uniform_voxel_scaling(self):
        origin = np.array([1.3, -0.7, 0.4])
        normal = np.array([1, 2, -1]) / np.sqrt(6)
        sdf = self.allocate([[0, 0, 0]], background=1.0, translation=origin)
        self.fill(sdf, normal=normal, origin=origin)
        relative = np.array([0.15, 0.15, 0.15])
        valid, distances, directions = self.probe_boundary(sdf, [origin + relative], 0.2)
        np.testing.assert_array_equal(valid, [1])
        expected_distance = max(relative @ normal + 0.025, 0.1)
        np.testing.assert_allclose(distances, [expected_distance], atol=2e-7)
        np.testing.assert_allclose(directions, [-normal], atol=2e-6)

        # A tiny index-space gradient is valid when its world-space norm is one.
        tiny = wp.Volume.allocate_by_tiles(
            wp.array([[0, 0, 0]], dtype=wp.vec3i, device=self.device),
            voxel_size=1e-8, bg_value=1.0, device=self.device,
        )
        self.fill(tiny, normal=(0, 1, 0))
        valid, distances, directions = self.probe_boundary(tiny, [[2e-8, 2e-8, 2e-8]], 1e-7)
        np.testing.assert_array_equal(valid, [1])
        np.testing.assert_allclose(distances, [5e-8], rtol=1e-6, atol=0)
        np.testing.assert_allclose(directions, [[0, -1, 0]], atol=1e-6)

    def test_position_kernels_preserve_self_density_gradient_direction_and_support(self):
        config = fn.PBFConfig()
        radius = 4 * config.particle_radius
        positions = np.array([[0, 0, 0], [0.0125, 0, 0], [-0.0125, 0, 0],
                              [0, 0.025, 0], [0.02, 0.01, -0.01],
                              [radius, 0, 0], [1.1 * radius, 0, 0]], dtype=np.float32)
        grid = wp.HashGrid(8, 8, 8, device=self.device)
        sdf = self.allocate([[0, 0, 0]], background=1.0)
        bind_solver(config, grid, sdf, sdf)
        wp.get_module(__name__).unload()
        values = wp.empty(len(positions), dtype=wp.vec2, device=self.device)
        gradients = wp.empty(len(positions), dtype=wp.vec3, device=self.device)
        wp.launch(_probe_position_kernels, len(positions),
                  inputs=[wp.array(positions, dtype=wp.vec3, device=self.device)],
                  outputs=[values, gradients], device=self.device)

        offsets = positions.astype(float)
        squared = np.sum(offsets**2, axis=1)
        distances = np.sqrt(squared)
        expected_values = 315 / (64 * np.pi * radius**9) * np.maximum(radius**2 - squared, 0)**3
        direction = np.divide(offsets, distances[:, None], out=np.zeros_like(offsets),
                              where=distances[:, None] > 0)
        expected_gradients = direction * (-45 / (np.pi * radius**6) * np.maximum(radius - distances, 0)**2)[:, None]
        np.testing.assert_allclose(values.numpy(), np.column_stack([expected_values, expected_values]), rtol=3e-6, atol=1e-7)
        np.testing.assert_allclose(gradients.numpy(), expected_gradients, rtol=3e-6, atol=1e-7)
        self.assertGreater(values.numpy()[0, 0], 0)
        np.testing.assert_array_equal(gradients.numpy()[0], 0)
        np.testing.assert_array_equal(values.numpy()[-2:], 0)
        np.testing.assert_array_equal(gradients.numpy()[-2:], 0)

    def test_runtime_pressure_plane_and_degenerate_gradient(self):
        from demos.fluid.PBF import PBF as solver

        sim = SimpleNamespace(config=solver.fn.PBFConfig())
        solver.fn.init_parameters(sim)
        radius = sim.smoothing_length
        sdf = self.allocate([[0, 0, 0]], background=1.0)
        volume_map = self.allocate([[0, 0, 0]], background=0.0)
        volume = 2e-4
        self.fill(volume_map, offset=volume)
        positions = wp.array([[0.15, 0.0125, 0.15]], dtype=wp.vec3, device=self.device)
        grid = wp.HashGrid(16, 16, 16, device=self.device)
        grid.build(positions, radius)
        neighbors = self_neighbor(self.device)
        bind_solver(sim.config, grid, sdf, volume_map)
        lambdas = wp.empty(1, dtype=float, device=self.device)
        corrections = wp.empty(1, dtype=wp.vec3, device=self.device)
        volume_rho = wp.full(1, 99.0, dtype=float, device=self.device)
        volume_grad = wp.full(1, wp.vec3(99.0), dtype=wp.vec3, device=self.device)

        def solve():
            wp.launch(solver.volume_query, 1,
                      inputs=[positions],
                      outputs=[volume_rho, volume_grad], device=self.device)
            wp.launch(solver.calc_lambda, 1,
                      inputs=[positions, neighbors, volume_rho, volume_grad],
                      outputs=[lambdas], device=self.device)
            wp.launch(solver.calc_deltaPos, 1,
                      inputs=[positions, lambdas, neighbors, volume_grad],
                      outputs=[corrections], device=self.device)
            return lambdas.numpy()[0], corrections.numpy()[0]

        self.fill(sdf, normal=(0, 1, 0))
        actual_lambda, actual_delta = solve()
        distance = max(0.0125 + 0.5 * sim.config.particle_radius, 2 * sim.config.particle_radius)
        poly6 = lambda d: 315 / (64 * np.pi * radius**9) * (radius**2 - d**2)**3
        gradient = volume * 45 / (np.pi * radius**6) * (radius - distance)**2
        mass = 0.8 * (2 * sim.config.particle_radius)**3 * sim.config.rest_density
        constraint = mass / sim.config.rest_density * poly6(0) + volume * poly6(distance) - 1
        epsilon = sim.config.lambda_regularization / radius**2
        expected_lambda = -max(constraint, 0) / (gradient**2 + epsilon)
        self.assertLess(expected_lambda, 0)
        np.testing.assert_allclose(actual_lambda, expected_lambda, rtol=2e-5)
        np.testing.assert_allclose(actual_delta, [0, -expected_lambda * gradient, 0], rtol=2e-5, atol=1e-8)
        np.testing.assert_allclose(volume_grad.numpy(), [[0, gradient, 0]], rtol=2e-5)
        np.testing.assert_allclose(volume_rho.numpy(), [volume * sim.config.rest_density * poly6(distance)], rtol=2e-5)

        # Delta consumes the saved result even if the volume changes after lambda.
        self.fill(sdf, offset=0.01)
        lambdas.fill_(-0.1)
        wp.launch(solver.calc_deltaPos, 1,
                  inputs=[positions, lambdas, neighbors, volume_grad],
                  outputs=[corrections], device=self.device)
        np.testing.assert_allclose(corrections.numpy(), [[0, 0.1 * gradient, 0]], rtol=2e-5, atol=1e-8)

        # The next volume query must overwrite a previously valid cache with zero.
        actual_lambda, actual_delta = solve()
        self.assertEqual(actual_lambda, 0)
        np.testing.assert_array_equal(actual_delta, [0, 0, 0])
        np.testing.assert_array_equal(volume_grad.numpy(), [[0, 0, 0]])
        np.testing.assert_array_equal(volume_rho.numpy(), [0])
        # An externally nonzero lambda must still produce zero boundary motion.
        lambdas.fill_(-0.1)
        wp.launch(solver.calc_deltaPos, 1,
                  inputs=[positions, lambdas, neighbors, volume_grad],
                  outputs=[corrections], device=self.device)
        np.testing.assert_array_equal(corrections.numpy(), [[0, 0, 0]])

    def test_constant_fields_all_voxels_and_zero_output_background(self):
        radius = 0.15
        for distance, occupied in ((-radius, True), (2 * radius, False)):
            with self.subTest(distance=distance):
                sdf = self.allocate([[-8, 0, -8], [8, 0, 0]], background=float(distance))
                self.fill(sdf, offset=distance)
                before = sdf.array().numpy().copy()
                result = boundary.precompute_volumeMap(sdf, radius)
                cells = sdf.get_voxels().numpy()
                expected = (boundary._build_volume_quadrature(radius, 16)[1].sum(dtype=np.float64)
                            if occupied else 0)
                np.testing.assert_allclose(self.lookup(result, cells), expected, rtol=2e-5, atol=1e-9)
                np.testing.assert_array_equal(sdf.array().numpy(), before)
                np.testing.assert_array_equal(result.get_tiles().numpy(), sdf.get_tiles().numpy())
                self.assertEqual(result.dtype, wp.float32)
                self.assertEqual(result.device, sdf.device)
                self.assertNotEqual(result.id, sdf.id)
                np.testing.assert_array_equal(self.lookup(result, [[200, 200, 200]]), [0])

    def test_plane_world_transform_and_negative_indices(self):
        radius = 0.16
        origin = (1.3, -0.7, 0.4)
        transform = wp.mat33(0, -0.1, 0, 0.125, 0, 0, 0, 0, 0.075)
        normal = np.array([1, 2, -1], dtype=np.float64) / np.sqrt(6)
        tiles = [(i, j, k) for i in (-8, 0) for j in (-8, 0) for k in (-8, 0)]
        sdf = self.allocate(tiles, background=2.0, translation=origin, transform=transform)
        self.fill(sdf, normal=normal, origin=origin)
        before = sdf.array().numpy().copy()
        result = boundary.precompute_volumeMap(sdf, radius)
        cells = np.array([[-2, -1, 0], [0, 0, 0], [1, -2, 1], [2, 1, -1]])
        matrix = np.asarray(transform, dtype=np.float64).reshape(3, 3)
        positions_relative_to_plane = cells @ matrix.T
        offsets, weights = boundary._build_volume_quadrature(radius, 16)
        distances = (positions_relative_to_plane[:, None, :] + offsets[None, :, :]) @ normal
        expected = _occupancy_reference(distances, radius) @ weights.astype(np.float64)
        np.testing.assert_allclose(self.lookup(result, cells), expected, rtol=1e-4, atol=radius**3 * 1e-6)
        np.testing.assert_array_equal(sdf.array().numpy(), before)
        np.testing.assert_array_equal(result.get_tiles().numpy(), sdf.get_tiles().numpy())
        original_info, output_info = sdf.get_grid_info(), result.get_grid_info()
        np.testing.assert_array_equal(output_info.translation, original_info.translation)
        np.testing.assert_array_equal(output_info.transform_matrix, original_info.transform_matrix)
        np.testing.assert_array_equal(result.get_voxel_size(), sdf.get_voxel_size())

    def test_input_background_is_used_outside_allocated_tiles(self):
        radius = 0.2
        sdf = self.allocate([[0, 0, 0]], background=2.0)
        self.fill(sdf, offset=-radius)
        result = boundary.precompute_volumeMap(sdf, radius)
        offsets, weights = boundary._build_volume_quadrature(radius, 16)
        # At index (0,0,0), negative coordinates interpolate with background.
        index_points = offsets.astype(np.float64) / 0.1
        fractions_inside = np.prod(np.clip(index_points + 1, 0, 1), axis=1)
        distances = fractions_inside * (-radius) + (1 - fractions_inside) * 2.0
        expected = np.dot(weights, _occupancy_reference(distances, radius))
        actual = self.lookup(result, [[0, 0, 0]])[0]
        self.assertAlmostEqual(actual, expected, delta=radius**3 * 1e-5)
        self.assertLess(actual, weights.sum() / 2)

    def test_reject_cpu_vector_and_index_inputs(self):
        sdf = self.allocate([[0, 0, 0]], background=1.0)
        cpu = wp.Volume(sdf.array().to("cpu"))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            boundary.precompute_volumeMap(cpu, 0.1)
        for background in (wp.vec3(1), None):
            invalid = self.allocate([[0, 0, 0]], background=background)
            with self.subTest(background=background), self.assertRaisesRegex(ValueError, "float32 scalar"):
                boundary.precompute_volumeMap(invalid, 0.1)


if __name__ == "__main__":
    unittest.main()
