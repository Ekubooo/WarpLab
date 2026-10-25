"""Single-segment NanoVDB packing and parameter-aware boundary cache tests."""

import contextlib
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from demos.fluid.PBF import boundary_model as boundary
from demos.sim_Ready import nvdb_io


@wp.kernel
def _fill_constant(volume: wp.uint64, tiles: wp.array(dtype=wp.vec3i), value: float):
    tid = wp.tid()
    local = tid % 512
    cell = tiles[tid // 512] + wp.vec3i(local // 64, (local // 8) % 8, local % 8)
    wp.volume_store_f(volume, cell[0], cell[1], cell[2], value)


@wp.kernel
def _lookup(volume: wp.uint64, cells: wp.array(dtype=wp.vec3i), values: wp.array(dtype=float)):
    i = wp.tid()
    cell = cells[i]
    values[i] = wp.volume_lookup_f(volume, cell[0], cell[1], cell[2])


class BoundaryCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is required for boundary grids")
        cls.device = wp.get_device("cuda:0")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "boundary.nvdb"

    def volume(self, value, background=1.0):
        volume = wp.Volume.allocate_by_tiles(
            wp.array([[-8, 0, 0], [0, 0, 0]], dtype=wp.vec3i, device=self.device),
            voxel_size=0.1, translation=(1.0, 2.0, 3.0), bg_value=background, device=self.device,
        )
        tiles = wp.empty(volume.get_tile_count(), dtype=wp.vec3i, device=self.device)
        volume.get_tiles(out=tiles)
        wp.launch(_fill_constant, volume.get_tile_count() * 512,
                  inputs=[volume.id, tiles, value], device=self.device)
        wp.synchronize_device(self.device)
        return volume

    def values(self, volume):
        cells = wp.array([[-8, 0, 0], [0, 0, 0], [100, 100, 100]],
                         dtype=wp.vec3i, device=self.device)
        result = wp.empty(3, dtype=float, device=self.device)
        wp.launch(_lookup, 3, inputs=[volume.id, cells, result], device=self.device)
        return result.numpy()

    def read_pair(self):
        with self.path.open("rb") as stream:
            first = wp.Volume.load_from_nvdb(stream, device=self.device)
        return first, first.load_next_grid()

    def load(self, radius=0.05, points=16):
        with contextlib.redirect_stdout(io.StringIO()):
            return boundary.load_boundary_volumes(
                self.path, radius, quadrature_points=points, device=self.device,
            )

    def test_pair_roundtrip_and_aliased_source_replacement(self):
        sdf, vmap = self.volume(-0.125), self.volume(0.002, background=0.0)
        original = [volume.array().numpy().copy() for volume in (sdf, vmap)]
        nvdb_io.save_volumes((sdf, vmap), self.path, grid_names=("SDF", "old-map"))
        self.assertEqual(nvdb_io.grid_count(self.path), 2)
        a, b = self.read_pair()
        self.assertIsNone(b.load_next_grid())
        for expected, actual in ((sdf, a), (vmap, b)):
            np.testing.assert_array_equal(self.values(actual), self.values(expected))
            np.testing.assert_array_equal(actual.get_tiles().numpy(), expected.get_tiles().numpy())
            np.testing.assert_array_equal(actual.get_grid_info().transform_matrix,
                                          expected.get_grid_info().transform_matrix)
            np.testing.assert_array_equal(actual.get_grid_info().translation,
                                          expected.get_grid_info().translation)
        self.assertGreater(a.array().shape[0], a.get_grid_info().size_in_bytes)
        fresh = self.volume(0.003, background=0.0)
        nvdb_io.save_volumes((a, fresh), self.path, grid_names=("SDF", "new-map"))
        a2, b2 = self.read_pair()
        self.assertEqual(nvdb_io.grid_count(self.path), 2)
        self.assertIsNone(b2.load_next_grid())
        np.testing.assert_array_equal(self.values(a2), self.values(sdf))
        np.testing.assert_array_equal(self.values(b2), self.values(fresh))
        for volume, before in zip((sdf, vmap), original):
            np.testing.assert_array_equal(volume.array().numpy(), before)

    def test_cpu_source_is_not_changed_and_bad_names_are_rejected(self):
        nvdb_io.save_volumes([self.volume(0.025)], self.path, grid_names=["original"])
        with self.path.open("rb") as stream:
            cpu = wp.Volume.load_from_nvdb(stream, device="cpu")
        before = cpu.array().numpy().copy()
        nvdb_io.save_volumes([cpu], self.path, grid_names=["renamed"])
        np.testing.assert_array_equal(cpu.array().numpy(), before)
        raw = self.path.read_bytes()
        for names in (["a" * 256], ["null\0name"], []):
            with self.subTest(names=names), self.assertRaises(ValueError):
                nvdb_io.save_volumes([cpu], self.path, grid_names=names)
            self.assertEqual(self.path.read_bytes(), raw)

    def test_first_build_then_hit_and_each_parameter_invalidates(self):
        nvdb_io.save_volumes([self.volume(0.025)], self.path)
        compute = boundary.precompute_volumeMap
        with patch.object(boundary, "precompute_volumeMap", wraps=compute) as mocked:
            first, vmap = self.load()
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(nvdb_io.grid_count(self.path), 2)
            self.assertEqual(vmap.get_grid_info().name, boundary._volume_map_name(0.05, 16))
            old = self.path.read_bytes()
            second, cached = self.load()
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(self.path.read_bytes(), old)
            np.testing.assert_array_equal(self.values(first), self.values(second))
            np.testing.assert_array_equal(self.values(vmap), self.values(cached))
            self.load(radius=0.04)
            self.assertEqual(mocked.call_count, 2)
            self.load(radius=0.04, points=8)
            self.assertEqual(mocked.call_count, 3)
            with patch.object(boundary, "VOLUME_MAP_ALGORITHM_VERSION", 2):
                self.load(radius=0.04, points=8)
            self.assertEqual(mocked.call_count, 4)

    def test_sparse_sdf_metadata_survives_cache_and_rejects_large_h(self):
        name = f"PBF_SDF|v=1|hmax={float(0.05).hex()}"
        nvdb_io.save_volumes([self.volume(0.025)], self.path, grid_names=[name])
        sdf, _ = self.load()
        self.assertEqual(sdf.get_grid_info().name, name)
        before = self.path.read_bytes()
        with patch.object(boundary, "precompute_volumeMap", side_effect=AssertionError("computed")):
            with self.assertRaisesRegex(RuntimeError, "larger max support radius"):
                self.load(radius=0.051)
        self.assertEqual(self.path.read_bytes(), before)

    def test_sparse_bake_volume_map_matches_before_and_after_write(self):
        import trimesh
        from demos.sim_Ready import sdf as baker
        from demos.sim_Ready.config import BakeOptions
        from types import SimpleNamespace
        mesh = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
        validated = SimpleNamespace(vertices=np.asarray(mesh.vertices, dtype=np.float32),
                                    faces=np.asarray(mesh.faces, dtype=np.int32), bounds=mesh.bounds)
        sparse = baker.bake(validated, options=BakeOptions(voxel_size=0.05, max_support_radius=0.05),
                            device=self.device)
        expected = boundary.precompute_volumeMap(sparse, 0.05)
        baker.save_volume(sparse, self.path)
        loaded, actual = self.load()
        np.testing.assert_array_equal(loaded.get_tiles().numpy(), sparse.get_tiles().numpy())
        np.testing.assert_array_equal(actual.get_tiles().numpy(), sparse.get_tiles().numpy())
        cells = wp.empty(sparse.get_voxel_count(), dtype=wp.vec3i, device=self.device)
        sparse.get_voxels(out=cells)
        a = wp.empty(cells.shape[0], dtype=float, device=self.device)
        b = wp.empty_like(a)
        wp.launch(_lookup, cells.shape[0], inputs=[expected.id, cells, a], device=self.device)
        wp.launch(_lookup, cells.shape[0], inputs=[actual.id, cells, b], device=self.device)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_failed_save_keeps_original_and_removes_temporary(self):
        volume = self.volume(0.025)
        nvdb_io.save_volumes([volume], self.path)
        before = self.path.read_bytes()
        for stage in ("save", "replace", "verify"):
            if stage == "save":
                target, attribute = wp.Volume, "save_to_nvdb"
            elif stage == "replace":
                target, attribute = nvdb_io.os, "replace"
            else:
                target, attribute = wp.Volume, "load_from_nvdb"
            with self.subTest(stage=stage), patch.object(target, attribute, side_effect=OSError(stage)):
                with self.assertRaises(OSError):
                    nvdb_io.save_volumes([volume], self.path)
            self.assertEqual(self.path.read_bytes(), before)
            self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_corruption_and_extra_segments_are_not_overwritten(self):
        nvdb_io.save_volumes([self.volume(0.025)], self.path)
        valid = self.path.read_bytes()
        for data in (b"broken", valid[:-1], valid + valid):
            self.path.write_bytes(data)
            with self.subTest(size=len(data)), self.assertRaisesRegex(RuntimeError, "Cannot load boundary"):
                self.load()
            self.assertEqual(self.path.read_bytes(), data)

    def test_file_and_grid_header_counts_must_agree(self):
        nvdb_io.save_volumes([self.volume(0.025)], self.path)
        data = bytearray(self.path.read_bytes())
        name_size = struct.unpack_from("<I", data, 152)[0]
        struct.pack_into("<I", data, 192 + name_size + 28, 2)
        self.path.write_bytes(data)
        with self.assertRaisesRegex(RuntimeError, "headers disagree"):
            self.load()
        self.assertEqual(self.path.read_bytes(), data)

    def test_warp_zip_file_header_is_recognized(self):
        volume = self.volume(0.025)
        volume.save_to_nvdb(str(self.path), codec="zip")
        self.assertEqual(nvdb_io.grid_count(self.path), 1)
        sdf, vmap = self.load(points=8)
        np.testing.assert_array_equal(self.values(sdf), self.values(volume))
        self.assertEqual(vmap.get_grid_info().name, boundary._volume_map_name(0.05, 8))

    def test_paired_zip_cache_is_loaded_without_precomputation(self):
        name = boundary._volume_map_name(0.05, 16)
        nvdb_io.save_volumes((self.volume(0.025), self.volume(0.002, background=0.0)),
                             self.path, grid_names=("SDF", name))
        pair, _ = self.read_pair()
        pair.save_to_nvdb(str(self.path), codec="zip")
        self.assertEqual(nvdb_io.grid_count(self.path), 2)
        with patch.object(boundary, "precompute_volumeMap", side_effect=AssertionError("computed")):
            _, cached = self.load()
        self.assertEqual(cached.get_grid_info().name, name)

    def test_layout_mismatch_and_reversed_grids_are_rejected(self):
        name = boundary._volume_map_name(0.05, 16)
        sdf, vmap = self.volume(0.025), self.volume(0.002, background=0.0)
        unrelated = wp.Volume.allocate_by_tiles(
            wp.array([[0, 0, 0]], dtype=wp.vec3i, device=self.device),
            voxel_size=0.2, bg_value=0.0, device=self.device,
        )
        for pair, names, message in (
            ((sdf, unrelated), ("SDF", name), "coordinate transform"),
            ((vmap, sdf), (name, "SDF"), "first NanoVDB grid"),
        ):
            nvdb_io.save_volumes(pair, self.path, grid_names=names)
            before = self.path.read_bytes()
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.load()
            self.assertEqual(self.path.read_bytes(), before)

    def test_rebuild_write_failure_propagates_without_using_memory_result(self):
        nvdb_io.save_volumes([self.volume(0.025)], self.path)
        before = self.path.read_bytes()
        with patch.object(nvdb_io, "save_volumes", side_effect=OSError("read-only asset")):
            with self.assertRaisesRegex(RuntimeError, "read-only asset"):
                self.load(points=8)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
