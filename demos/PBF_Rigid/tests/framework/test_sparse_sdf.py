"""Sparse positive-band SDF regression against an analytic hollow box."""

import contextlib
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import trimesh
import warp as wp

from demos.PBF_Rigid.sim_ready import config, sdf
from demos.PBF_Rigid.sim_ready.__main__ import parse_args
from demos.PBF_Rigid.solver import boundary_model


@wp.kernel
def _lookup(grid: wp.uint64, cells: wp.array(dtype=wp.vec3i), values: wp.array(dtype=float)):
    i = wp.tid()
    cell = cells[i]
    values[i] = wp.volume_lookup_f(grid, cell[0], cell[1], cell[2])


@wp.func
def _box_distance_wp(point: wp.vec3, half: float):
    q = wp.abs(point) - wp.vec3(half)
    return wp.length(wp.max(q, wp.vec3(0.0))) + wp.min(wp.max(wp.max(q[0], q[1]), q[2]), 0.0)


@wp.kernel
def _fill_reference(grid: wp.uint64, cells: wp.array(dtype=wp.vec3i)):
    i = wp.tid()
    cell = cells[i]
    point = wp.volume_index_to_world(grid, wp.vec3(float(cell[0]), float(cell[1]), float(cell[2])))
    distance = wp.max(_box_distance_wp(point, 1.0), -_box_distance_wp(point, 0.7))
    wp.volume_store_f(grid, cell[0], cell[1], cell[2], distance)


@wp.kernel
def _sample(grid: wp.uint64, positions: wp.array(dtype=wp.vec3), values: wp.array(dtype=float),
            gradients: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    index = wp.volume_world_to_index(grid, positions[i])
    gradient = wp.vec3(0.0)
    values[i] = wp.volume_sample_grad_f(grid, index, wp.Volume.LINEAR, gradient)
    gradients[i] = wp.volume_world_to_index_dir(grid, gradient)


def _box_distance(points, half):
    q = np.abs(points) - half
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


class SparseSDFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA required")
        outer = trimesh.creation.box(extents=(2, 2, 2))
        inner = trimesh.creation.box(extents=(1.4, 1.4, 1.4))
        inner.invert()
        hollow = trimesh.util.concatenate((outer, inner))
        cls.mesh = SimpleNamespace(vertices=np.asarray(hollow.vertices, dtype=np.float32),
                                   faces=np.asarray(hollow.faces, dtype=np.int32),
                                   bounds=np.asarray(hollow.bounds, dtype=np.float64))
        cls.options = config.BakeOptions(voxel_size=0.04, max_support_radius=0.05)
        cls.background = 2 * cls.options.max_support_radius + 2 * math.sqrt(3) * cls.options.voxel_size
        cls.volume = sdf.bake(cls.mesh, options=cls.options)
        cls.padding = cls.options.padding_voxels
        upper = np.ceil((cls.mesh.bounds[1] - cls.mesh.bounds[0]) / cls.options.voxel_size).astype(int)
        upper += 2 * cls.padding
        cls.dimensions = (upper // 8 + 1) * 8

    def lookup(self, grid, cells):
        cells = wp.array(cells, dtype=wp.vec3i, device="cuda:0")
        values = wp.empty(len(cells), dtype=float, device="cuda:0")
        wp.launch(_lookup, len(cells), inputs=[grid.id, cells, values], device="cuda:0")
        return values.numpy()

    def sample(self, grid, positions):
        positions = wp.array(positions, dtype=wp.vec3, device="cuda:0")
        values = wp.empty(len(positions), dtype=float, device="cuda:0")
        gradients = wp.empty(len(positions), dtype=wp.vec3, device="cuda:0")
        wp.launch(_sample, len(positions), inputs=[grid.id, positions, values, gradients], device="cuda:0")
        return values.numpy(), gradients.numpy()

    def reference_volume(self, extra_tiles=0):
        halo = extra_tiles * 8
        dense = wp.Volume.allocate([-halo] * 3, (self.dimensions + halo - 1).tolist(),
                                   self.options.voxel_size, bg_value=10.0,
                                   translation=self.volume.get_grid_info().translation, device="cuda:0")
        cells = wp.empty(dense.get_voxel_count(), dtype=wp.vec3i, device="cuda:0")
        dense.get_voxels(out=cells)
        wp.launch(_fill_reference, len(cells), inputs=[dense.id, cells], device="cuda:0")
        return dense

    def boundary_positions(self):
        h = self.options.max_support_radius
        distances = np.array([0.1, 0.5, 0.98, 1.02, 1.5, 1.98, 2.1]) * h
        positions = []
        for axis in range(3):
            for sign in (-1, 1):
                for d in distances:
                    point = [.12, .18, .16]
                    point[axis] = sign * (.7 - d)
                    positions.append(point)
        # Cross a tangential tile seam while remaining inside the valid band.
        seam = float(self.volume.get_grid_info().translation[1]) + 4 * 8 * self.options.voxel_size
        positions += [[.65, seam + e, .18] for e in (-1e-4, 0, 1e-4)]
        return positions

    def test_all_solid_distances_and_positive_band_match_analytic_field(self):
        # Compare the entire candidate lattice, including omitted tiles. This
        # catches tile-center classification accidentally dropping thin solids.
        cells = np.stack(np.meshgrid(*(np.arange(n, dtype=np.int32) for n in self.dimensions),
                                    indexing="ij"), axis=-1).reshape(-1, 3)
        origin = np.asarray(self.volume.get_grid_info().translation, dtype=np.float64)
        positions = origin + cells.astype(np.float64) * self.options.voxel_size
        reference = np.maximum(_box_distance(positions, 1), -_box_distance(positions, 0.7))
        expected = np.minimum(reference, self.background)
        actual = self.lookup(self.volume, cells)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
        self.assertTrue(np.any(reference < -0.1))
        self.assertTrue(np.any(reference > self.background))

    def test_sparse_tiles_background_and_grid_alignment(self):
        dense_tile_count = int(np.prod(self.dimensions // 8))
        self.assertLess(self.volume.get_tile_count(), dense_tile_count)
        # The lattice remains an integer-voxel extension of the source bounds.
        info = self.volume.get_grid_info()
        expected_origin = self.mesh.bounds[0] - self.padding * self.options.voxel_size
        np.testing.assert_allclose(info.translation, expected_origin, atol=1e-7)
        np.testing.assert_allclose(self.volume.get_voxel_size(), [self.options.voxel_size] * 3)
        actual = self.lookup(self.volume, [[-100, -100, -100], [1000, 1000, 1000]])
        np.testing.assert_allclose(actual, self.background)

    def test_positive_band_gradients_and_normals_match_dense_reference(self):
        dense = self.reference_volume()
        positions = self.boundary_positions()
        actual_values, actual_gradients = self.sample(self.volume, positions)
        expected_values, expected_gradients = self.sample(dense, positions)
        np.testing.assert_allclose(actual_values, expected_values, atol=2e-6)
        np.testing.assert_allclose(actual_gradients, expected_gradients, atol=1e-4)
        actual_normals = actual_gradients / np.linalg.norm(actual_gradients, axis=1, keepdims=True)
        expected_normals = expected_gradients / np.linalg.norm(expected_gradients, axis=1, keepdims=True)
        np.testing.assert_allclose(actual_normals, expected_normals, atol=1e-4)

    def test_default_quadrature_volume_map_matches_dense_reference(self):
        # Extend the reference independently of the baker: no shared domain
        # clipping may hide an integration or interpolation error.
        dense = self.reference_volume(extra_tiles=1)
        h = self.options.max_support_radius
        sparse_map = boundary_model.precompute_volumeMap(self.volume, h)
        dense_map = boundary_model.precompute_volumeMap(dense, h)
        # The contract covers the cavity, including unallocated far-field nodes.
        cells = dense.get_voxels().numpy()
        origin = np.asarray(dense.get_grid_info().translation)
        positions = origin + cells * self.options.voxel_size
        cells = cells[np.all(np.abs(positions) <= .7 + 1e-6, axis=1)]
        np.testing.assert_allclose(self.lookup(sparse_map, cells), self.lookup(dense_map, cells),
                                   rtol=5e-4, atol=h**3 * 1e-5)
        actual_values, _ = self.sample(sparse_map, self.boundary_positions())
        expected_values, _ = self.sample(dense_map, self.boundary_positions())
        np.testing.assert_allclose(actual_values, expected_values, rtol=5e-4, atol=h**3 * 1e-5)

    def test_support_radius_keeps_original_domain_and_cavity_map_correct(self):
        # Here the fixed .12 m padding is less than h=.2 m, so evaluating
        # against a same-sized reference would miss exterior clipping bugs.
        h = .2
        volume = sdf.bake(self.mesh, options=config.BakeOptions(
            voxel_size=self.options.voxel_size, max_support_radius=h))
        np.testing.assert_allclose(volume.get_grid_info().translation,
                                   self.mesh.bounds[0] - 3 * self.options.voxel_size, atol=1e-7)
        for grid in (self.volume, volume):
            tiles = grid.get_tiles().numpy()
            np.testing.assert_array_equal(tiles.min(axis=0), [0, 0, 0])
            np.testing.assert_array_equal(tiles.max(axis=0) + 8, self.dimensions)
        dense = self.reference_volume(extra_tiles=1)
        sparse_map = boundary_model.precompute_volumeMap(volume, h)
        dense_map = boundary_model.precompute_volumeMap(dense, h)
        cells = dense.get_voxels().numpy()
        origin = np.asarray(dense.get_grid_info().translation)
        positions = origin + cells * self.options.voxel_size
        cavity = np.all(np.abs(positions) <= .7 + 1e-6, axis=1)
        np.testing.assert_allclose(self.lookup(sparse_map, cells[cavity]),
                                   self.lookup(dense_map, cells[cavity]), rtol=5e-4, atol=h**3 * 1e-5)
        distances = np.array([.1, .98, 1.02, 1.98, 2.1]) * h
        probes = [[sign * (.7 - d), .12, .18] for sign in (-1, 1) for d in distances]
        seam = float(volume.get_grid_info().translation[1]) + 4 * 8 * self.options.voxel_size
        probes += [[.65, seam + e, .18] for e in (-1e-4, 0, 1e-4)]
        actual_values, actual_gradients = self.sample(volume, probes)
        expected_values, expected_gradients = self.sample(dense, probes)
        np.testing.assert_allclose(actual_values, expected_values, atol=2e-6)
        np.testing.assert_allclose(actual_gradients, expected_gradients, atol=1e-4)
        actual_map, _ = self.sample(sparse_map, probes)
        expected_map, _ = self.sample(dense_map, probes)
        np.testing.assert_allclose(actual_map, expected_map, rtol=5e-4, atol=h**3 * 1e-5)

    def test_saved_metadata_roundtrip_and_gpu_input_unchanged(self):
        before = self.volume.array().numpy().copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hollow.nvdb"
            sdf.save_volume(self.volume, path)
            with path.open("rb") as stream:
                loaded = wp.Volume.load_from_nvdb(stream, device="cuda:0")
            self.assertEqual(loaded.get_grid_info().name,
                             "PBF_SDF|v=1|hmax=" + self.options.max_support_radius.hex())
            self.assertEqual(loaded.get_grid_info().grid_count, 1)
            self.assertIsNone(loaded.load_next_grid())
            self.assertEqual(loaded.get_tile_count(), self.volume.get_tile_count())
            cells = self.volume.get_voxels().numpy()[::127]
            np.testing.assert_array_equal(self.lookup(loaded, cells), self.lookup(self.volume, cells))
            dense = self.reference_volume()
            dense_path = Path(directory) / "dense.nvdb"
            sdf.save_volume(dense, dense_path)
            self.assertLess(path.stat().st_size, dense_path.stat().st_size)
            self.assertLess(self.volume.get_grid_info().size_in_bytes, dense.get_grid_info().size_in_bytes)
        np.testing.assert_array_equal(self.volume.array().numpy(), before)


class SparseSDFOptionsTests(unittest.TestCase):
    def test_maximum_support_radius_cli(self):
        self.assertEqual(parse_args([]).sdf_max_support_radius, 0.05)
        self.assertEqual(parse_args(["--sdf-max-support-radius", "0.075"]).sdf_max_support_radius, 0.075)
        for value in ("0", "-1", "nan", "inf"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--sdf-max-support-radius", value])


if __name__ == "__main__":
    unittest.main()
