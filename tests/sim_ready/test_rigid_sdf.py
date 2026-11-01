"""Exterior rigid SDF and compatibility with PBF's unchanged volume-map cache."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
import warp as wp

from demos.sim_Ready import config, geometry, nvdb_io, rigid_pipeline, sdf, validation
from demos.fluid.PBF import boundary_model


CENTER = wp.constant(wp.vec3(.17, -.11, .23))
HALF = wp.constant(wp.vec3(.12, .08, .16))


@wp.func
def _analytic_box(position: wp.vec3):
    q = wp.abs(position - CENTER) - HALF
    return wp.length(wp.max(q, wp.vec3(0.))) + wp.min(wp.max(wp.max(q[0], q[1]), q[2]), 0.)


@wp.kernel
def _fill_reference(grid: wp.uint64, cells: wp.array(dtype=wp.vec3i)):
    i = wp.tid()
    cell = cells[i]
    position = wp.volume_index_to_world(grid, wp.vec3(float(cell[0]), float(cell[1]), float(cell[2])))
    wp.volume_store_f(grid, cell[0], cell[1], cell[2], _analytic_box(position))


@wp.kernel
def _sample(grid: wp.uint64, points: wp.array(dtype=wp.vec3),
            values: wp.array(dtype=float), normals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    gradient = wp.vec3(0.)
    index = wp.volume_world_to_index(grid, points[i])
    values[i] = wp.volume_sample_grad_f(grid, index, wp.Volume.LINEAR, gradient)
    normals[i] = wp.volume_world_to_index_dir(grid, gradient)


class RigidSDFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA required for NanoVDB baking")
        cls.center = np.array((.17, -.11, .23))
        cls.half = np.array((.12, .08, .16))
        cls.mesh = trimesh.creation.box(extents=2 * cls.half)
        cls.mesh.apply_translation(cls.center)
        cls.checked = validation.validate_solid(cls.mesh)
        cls.options = config.BakeOptions(voxel_size=.04, max_support_radius=.08)
        cls.volume = sdf.bake_rigid(cls.checked, options=cls.options)

    def sample(self, grid, positions):
        points = wp.array(positions, dtype=wp.vec3, device="cuda:0")
        values = wp.empty(len(points), dtype=float, device="cuda:0")
        normals = wp.empty(len(points), dtype=wp.vec3, device="cuda:0")
        wp.launch(_sample, len(points), inputs=[grid.id, points], outputs=[values, normals])
        return values.numpy(), normals.numpy()

    def exterior_probes(self, h):
        positions, distances, normals = [], [], []
        for axis in range(3):
            for sign in (-1, 1):
                for distance in (.2 * h, .75 * h, .98 * h, 1.5 * h, 1.98 * h):
                    point = self.center.copy()
                    point[axis] += sign * (self.half[axis] + distance)
                    normal = np.zeros(3)
                    normal[axis] = sign
                    positions.append(point)
                    distances.append(distance)
                    normals.append(normal)
        return positions, distances, normals

    def reference(self, volume):
        # Enlarge the reference by a full tile on every side, avoiding a
        # comparison that would hide clipping at the rigid baker's domain edge.
        upper = volume.get_tiles().numpy().max(axis=0) + 8
        dense = wp.Volume.allocate([-8] * 3, (upper + 8 - 1).tolist(),
            self.options.voxel_size, bg_value=10.,
            translation=volume.get_grid_info().translation, device="cuda:0")
        cells = wp.empty(dense.get_voxel_count(), dtype=wp.vec3i, device="cuda:0")
        dense.get_voxels(out=cells)
        wp.launch(_fill_reference, len(cells), inputs=[dense.id, cells])
        return dense

    def test_exterior_distance_normals_background_and_domain(self):
        positions, distances, normals = self.exterior_probes(self.options.max_support_radius)
        values, gradients = self.sample(self.volume, positions)
        np.testing.assert_allclose(values, distances, atol=2e-6)
        unit = gradients / np.linalg.norm(gradients, axis=1, keepdims=True)
        np.testing.assert_allclose(unit, normals, atol=1e-4)
        margin = 2 * self.options.max_support_radius + 2 * np.sqrt(3) * self.options.voxel_size
        origin = np.asarray(self.volume.get_grid_info().translation)
        far = origin + (self.volume.get_tiles().numpy().max(0) + 7) * self.options.voxel_size
        self.assertTrue(np.all(self.checked.bounds[0] - origin >= margin - 1e-6))
        self.assertTrue(np.all(far - self.checked.bounds[1] >= margin - 1e-6))
        values, gradients = self.sample(self.volume, [self.center, self.center + 10])
        self.assertLess(values[0], 0.)
        self.assertAlmostEqual(values[1], margin, places=6)
        np.testing.assert_array_equal(gradients[1], [0, 0, 0])
        # The existing container baker must continue to use its three-voxel halo.
        container = sdf.bake(self.checked, options=self.options)
        np.testing.assert_allclose(container.get_grid_info().translation,
            self.checked.bounds[0] - 3 * self.options.voxel_size, atol=1e-7)

    def test_exterior_volume_maps_match_larger_dense_reference(self):
        for h in (.04, .12):
            with self.subTest(support_radius=h):
                volume = sdf.bake_rigid(self.checked, options=replace(self.options, max_support_radius=h))
                dense = self.reference(volume)
                actual_map = boundary_model.precompute_volumeMap(volume, h)
                expected_map = boundary_model.precompute_volumeMap(dense, h)
                positions, _, _ = self.exterior_probes(h)
                # Add corners/edges and both sides of a tangential tile seam.
                positions += [self.center + self.half + d for d in (.01, .025)]
                seam = float(volume.get_grid_info().translation[1]) + 8 * self.options.voxel_size
                positions += [[self.center[0] + self.half[0] + .3 * h, seam + e, self.center[2]]
                              for e in (-1e-4, 0, 1e-4)]
                values, gradients = self.sample(volume, positions)
                expected, expected_gradients = self.sample(dense, positions)
                np.testing.assert_allclose(values, expected, atol=2e-6)
                np.testing.assert_allclose(gradients, expected_gradients, atol=1e-4)
                actual, _ = self.sample(actual_map, positions)
                reference, _ = self.sample(expected_map, positions)
                np.testing.assert_allclose(actual, reference, rtol=5e-4, atol=h**3 * 1e-5)

    def test_compound_union_does_not_leave_false_internal_surfaces(self):
        second = self.mesh.copy()
        second.apply_translation((self.half[0], 0, 0))
        parts = validation.validate_rigid_parts(trimesh.util.concatenate((self.mesh, second)))
        combined = validation.validate_solid(geometry.union(parts))
        volume = sdf.bake_rigid(combined, options=self.options)
        # This point was on the first box's +X face, now inside the union.
        internal_face = self.center + [self.half[0], 0, 0]
        outside = self.center + [2 * self.half[0] + .04, 0, 0]
        values, _ = self.sample(volume, [internal_face, outside])
        self.assertAlmostEqual(values[0], -self.half[1], places=6)
        self.assertAlmostEqual(values[1], .04, places=6)

    def test_pipeline_roundtrip_then_pbf_cache_writeback_and_rebake(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source.obj"
            self.mesh.export(source, digits=17)
            original = source.read_bytes()
            output, consumer = root / "output/rigid", root / "consumer/Rigid"
            with patch.object(config, "RIGID_OUTPUT", output), \
                    patch.object(config, "RIGID_OUTPUT_DIRECTORIES", (output, consumer)), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = rigid_pipeline.build_rigid_asset(source, name="rigid_box", bake_options=self.options)
                self.assertEqual(nvdb_io.grid_count(result.sdf), 1)
                self.assertEqual(nvdb_io.grid_count(consumer / result.sdf.name), 1)
                with result.sdf.open("rb") as stream:
                    loaded = wp.Volume.load_from_nvdb(stream, device="cuda:0")
                self.assertIsNone(loaded.load_next_grid())
                self.assertEqual(loaded.get_grid_info().name,
                    "PBF_SDF|v=1|hmax=" + self.options.max_support_radius.hex())
                positions, _, _ = self.exterior_probes(self.options.max_support_radius)
                np.testing.assert_allclose(self.sample(loaded, positions)[0],
                                           self.sample(self.volume, positions)[0], atol=2e-6)
                _, volume_map = boundary_model.load_boundary_volumes(
                    result.sdf, self.options.max_support_radius, device="cuda:0")
                self.assertEqual(nvdb_io.grid_count(result.sdf), 2)
                self.assertEqual(nvdb_io.grid_count(consumer / result.sdf.name), 1)
                cached_bytes = result.sdf.read_bytes()
                with patch.object(boundary_model, "precompute_volumeMap", side_effect=AssertionError("cache miss")):
                    _, cached_map = boundary_model.load_boundary_volumes(
                        result.sdf, self.options.max_support_radius, device="cuda:0")
                self.assertEqual(result.sdf.read_bytes(), cached_bytes)
                np.testing.assert_array_equal(self.sample(cached_map, positions)[0],
                                              self.sample(volume_map, positions)[0])
                rigid_pipeline.build_rigid_asset(source, name="rigid_box", bake_options=self.options)
                self.assertEqual(nvdb_io.grid_count(result.sdf), 1)
                self.assertEqual(nvdb_io.grid_count(consumer / result.sdf.name), 1)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
