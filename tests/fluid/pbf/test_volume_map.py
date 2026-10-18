"""Gauss volume integration and NanoVDB coordinate/background regression tests."""

import unittest

import numpy as np
import warp as wp

from demos.fluid.PBF import boundary_model as boundary


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
