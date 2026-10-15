"""GPU distance, cavity-sign, and existence-only cache regression tests."""

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import trimesh
import warp as wp

from demos.fluid.boundary import sdf_baker as baker


@wp.kernel
def _sample(volume: wp.uint64, points: wp.array(dtype=wp.vec3), output: wp.array(dtype=float)):
    tid = wp.tid()
    index = wp.volume_world_to_index(volume, points[tid])
    output[tid] = wp.volume_sample_f(volume, index, wp.Volume.LINEAR)


def _box_distance(points, half):
    q = np.abs(points) - half
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


class SDFBakerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is required by NanoVDB allocation")
        cls.device = wp.get_device("cuda:0")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.source = self.directory / "container.obj"
        self.source.write_text("source placeholder, validation mocked", encoding="utf-8")
        outer = trimesh.creation.box(extents=(2, 2, 2))
        inner = trimesh.creation.box(extents=(1.4, 1.4, 1.4))
        inner.invert()
        mesh = trimesh.util.concatenate([outer, inner])
        self.validated = types.SimpleNamespace(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32), bounds=mesh.bounds,
        )
        module = types.ModuleType("demos.fluid.boundary.mesh_validation")
        module.load_validate_container_obj = mock.Mock(return_value=self.validated)
        self.validator = module.load_validate_container_obj
        self.modules_patch = mock.patch.dict(sys.modules, {module.__name__: module})
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.options = baker.SDFOptions(voxel_size=0.05)

    def load(self, options=None, source=None):
        return baker.load_or_bake_sdf(
            source or self.source, cache_dir=self.directory,
            options=options or self.options, device=self.device,
        )

    def sample(self, volume, points):
        points = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=self.device)
        output = wp.zeros(len(points), dtype=float, device=self.device)
        wp.launch(_sample, len(points), inputs=[volume.id, points, output], device=self.device)
        return output.numpy()

    def test_bake_roundtrip_cavity_and_all_allocated_voxels(self):
        result = self.load()
        self.assertFalse(result.cache_hit)
        self.validator.assert_called_once_with(self.source.resolve())
        self.assertEqual(result.cache_path, self.source.with_suffix(".nvdb"))
        self.assertEqual(result.volume.device, self.device)
        points = np.array([[0, 0, 0], [0.8, 0, 0], [1.1, 0, 0],
                           [0, 0.8, 0], [0, 0, -0.9], [0.3, 0.2, 0.1]])
        expected = np.maximum(_box_distance(points, 1), -_box_distance(points, 0.7))
        distances = self.sample(result.volume, points)
        np.testing.assert_allclose(distances, expected, atol=2e-6)
        self.assertGreater(distances[0], 0)  # Enclosed fluid cavity is not solid.
        self.assertLess(distances[1], 0)
        self.assertGreater(distances[2], 0)

        # The final tile corner lies beyond requested bounds: it too was baked.
        tiles = result.volume.get_tiles().numpy()
        indices = np.vstack([tiles.min(axis=0), tiles.max(axis=0) + 7])
        origin = self.validated.bounds[0] - 3 * self.options.voxel_size
        corners = origin + indices * self.options.voxel_size
        expected_corners = _box_distance(corners, 1)
        np.testing.assert_allclose(self.sample(result.volume, corners), expected_corners, atol=2e-6)

        loaded = self.load()
        self.assertTrue(loaded.cache_hit)
        self.validator.assert_called_once()
        np.testing.assert_array_equal(self.sample(loaded.volume, points), distances)
        self.assertEqual(sorted(path.suffix for path in self.directory.iterdir()), [".nvdb", ".obj"])

    def test_existing_cache_ignores_source_and_changed_options(self):
        self.load()
        self.source.unlink()
        self.validator.reset_mock()
        loaded = self.load(options=baker.SDFOptions(voxel_size=0.1))
        self.assertTrue(loaded.cache_hit)
        self.validator.assert_not_called()
        np.testing.assert_allclose(loaded.volume.get_voxel_size(), [0.05] * 3)

    def test_corrupt_cache_requires_explicit_rebuild(self):
        self.source.with_suffix(".nvdb").write_bytes(b"broken")
        with self.assertRaisesRegex(RuntimeError, "force rebuild"):
            self.load()
        self.validator.assert_not_called()
        rebuilt = self.load(baker.SDFOptions(voxel_size=0.05, force_rebuild=True))
        self.assertFalse(rebuilt.cache_hit)
        self.assertGreater(rebuilt.cache_path.stat().st_size, 100)

    def test_force_rebuild_revalidates_and_changes_resolution(self):
        self.load()
        self.validator.reset_mock()
        rebuilt = self.load(baker.SDFOptions(voxel_size=0.1, force_rebuild=True))
        self.validator.assert_called_once()
        self.assertFalse(rebuilt.cache_hit)
        np.testing.assert_allclose(rebuilt.volume.get_voxel_size(), [0.1] * 3)

    def test_failed_query_does_not_save(self):
        def fail_queries(*args, **kwargs):
            kwargs["inputs"][-1].assign(np.array([1], dtype=np.int32))
        with mock.patch.object(wp, "launch", side_effect=fail_queries):
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                self.load()
        self.assertFalse(self.source.with_suffix(".nvdb").exists())

    def test_failed_save_keeps_previous_cache_and_removes_temporary(self):
        target = self.source.with_suffix(".nvdb")
        target.write_bytes(b"previous")
        def failed_save(path, **kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("write interrupted")
        volume = mock.Mock()
        volume.save_to_nvdb.side_effect = failed_save
        with self.assertRaisesRegex(OSError, "interrupted"):
            baker._save_atomic(volume, target)
        self.assertEqual(target.read_bytes(), b"previous")
        self.assertFalse(list(self.directory.glob("*.tmp")))

    def test_cpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            baker.load_or_bake_sdf(self.source, cache_dir=self.directory, device="cpu")

    def test_invalid_options_do_not_validate_or_save(self):
        for h in (0, -1, float("nan"), float("inf")):
            with self.subTest(voxel_size=h):
                with self.assertRaisesRegex(ValueError, "voxel_size"):
                    self.load(baker.SDFOptions(voxel_size=h))
        self.validator.assert_not_called()
        self.assertFalse(self.source.with_suffix(".nvdb").exists())


if __name__ == "__main__":
    unittest.main()
