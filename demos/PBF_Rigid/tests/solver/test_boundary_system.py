"""Shared assets, count discovery and GPU instance transforms without dynamics."""

import contextlib
from dataclasses import replace
import gc
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import warp as wp

from demos.PBF_Rigid.solver import boundary_model as boundary
from demos.sim_Ready import nvdb_io


@wp.kernel
def _fill_plane(sdf: wp.uint64, volume_map: wp.uint64, tiles: wp.array(dtype=wp.vec3i)):
    i = wp.tid()
    cell = tiles[i // 512] + wp.vec3i((i % 512) // 64, (i % 64) // 8, i % 8)
    local = wp.volume_index_to_world(sdf, wp.vec3(float(cell[0]), float(cell[1]), float(cell[2])))
    wp.volume_store_f(sdf, cell[0], cell[1], cell[2], local[1])
    wp.volume_store_f(volume_map, cell[0], cell[1], cell[2], .003)


@wp.kernel
def _query_instances(data: boundary.BoundaryDeviceData, ids: wp.array(dtype=int),
                     points: wp.array(dtype=wp.vec3), distances: wp.array(dtype=float),
                     gradients: wp.array(dtype=wp.vec3), volumes: wp.array(dtype=float)):
    i = wp.tid()
    distance, gradient = boundary.sample_instance_sdf(data, ids[i], points[i])
    distances[i] = distance
    gradients[i] = gradient
    volumes[i] = boundary.sample_instance_volume(data, ids[i], points[i])


class BoundarySystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA required for boundary Volume resources")
        cls.device = wp.get_device("cuda:0")
        cls.options = boundary.BoundaryLoadOptions(.05, device=cls.device)
        tiles = wp.array([[0, 0, 0], [8, 0, 0]], dtype=wp.vec3i, device=cls.device)
        cls.sdf = wp.Volume.allocate_by_tiles(tiles, .05, bg_value=1., translation=(-.2,) * 3,
                                              device=cls.device)
        cls.volume_map = wp.Volume.allocate_by_tiles(tiles, .05, bg_value=0., translation=(-.2,) * 3,
                                                     device=cls.device)
        wp.launch(_fill_plane, len(tiles) * 512, inputs=[cls.sdf.id, cls.volume_map.id, tiles], device=cls.device)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.rigid = self.root / "Rigid"

    def asset(self, name, *, static=False, single=False):
        directory = self.root if static else self.rigid
        directory.mkdir(exist_ok=True)
        path = directory / (name + ".nvdb")
        if not static:
            # A path-only asset loader must not try parsing this OBJ.
            path.with_suffix(".obj").write_text("mesh loading belongs to a later stage", encoding="utf-8")
        if single:
            nvdb_io.save_volumes([self.sdf], path, grid_names=["SDF"])
        else:
            nvdb_io.save_volumes((self.sdf, self.volume_map), path,
                                grid_names=("SDF", boundary._volume_map_name(.05, 16)))
        return path

    def load(self, scene=None, options=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return boundary.load_boundaries(scene or boundary.BoundaryScene(self.root, ), options or self.options)

    def test_missing_and_empty_rigid_directory_allow_empty_system(self):
        for exists in (False, True):
            if exists:
                self.rigid.mkdir()
            with self.subTest(directory_exists=exists), patch.object(boundary, "load_boundary_volumes") as loader:
                system = self.load()
                loader.assert_not_called()
            self.assertEqual(system.assets, ())
            self.assertEqual(system.instances, ())
            self.assertEqual((system.static_count, system.rigid_count), (0, 0))
            self.assertEqual(system.device_data.transforms.shape, (0,))
            system.set_rigid_poses(np.empty((0, 7)))
            system.reset()

    def test_default_count_is_one_and_scanning_is_nonrecursive(self):
        path = self.asset("box")
        (self.rigid / "nested").mkdir()
        (self.rigid / "nested/ignored.obj").write_text("ignored", encoding="utf-8")
        (self.rigid / "orphan.nvdb").write_bytes(b"not an OBJ asset")
        system = self.load()
        self.assertEqual(system.rigid_count, 1)
        self.assertEqual(len(system.assets), 1)
        self.assertEqual(system.assets[0].sdf_path, path)
        self.assertEqual(system.assets[0].mesh_path, path.with_suffix(".obj"))
        self.assertTrue(system.instances[0].is_dynamic)

    def test_counts_order_sharing_and_static_nvdb_without_obj(self):
        self.asset("room", static=True)
        self.asset("zebra")
        self.asset("alpha")
        (self.rigid / "disabled.obj").write_text("no NVDB needed when disabled", encoding="utf-8")
        scene = boundary.BoundaryScene(self.root, static_assets=("room.nvdb",), rigid_counts={"zebra": 3, "disabled": 0})
        with patch.object(boundary, "load_boundary_volumes", wraps=boundary.load_boundary_volumes) as loader:
            system = self.load(scene)
        self.assertEqual(loader.call_count, 3)
        self.assertEqual([a.sdf_path.stem for a in system.assets], ["room", "alpha", "zebra"])
        self.assertEqual((system.static_count, system.rigid_count), (1, 4))
        np.testing.assert_array_equal(system.device_data.asset_ids.numpy(), [0, 1, 2, 2, 2])
        np.testing.assert_array_equal(system.device_data.rigid_ids.numpy(), [-1, 0, 1, 2, 3])
        np.testing.assert_array_equal(system.device_data.rigid_instance_ids.numpy(), [1, 2, 3, 4])
        self.assertFalse(system.assets[0].mesh_path.exists())
        self.assertFalse(system.instances[0].is_dynamic)

    def test_large_count_reuses_one_asset(self):
        self.asset("box")
        with patch.object(boundary, "load_boundary_volumes", wraps=boundary.load_boundary_volumes) as loader:
            system = self.load(boundary.BoundaryScene(self.root, rigid_counts={"box": 10000}))
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(system.rigid_count, 10000)
        self.assertEqual(len(system.assets), 1)
        self.assertEqual(system.device_data.transforms.shape, (10000,))

    def test_duplicate_static_references_share_one_asset(self):
        self.asset("room", static=True)
        with patch.object(boundary, "load_boundary_volumes", wraps=boundary.load_boundary_volumes) as loader:
            system = self.load(boundary.BoundaryScene(self.root, static_assets=("room.nvdb", "room.nvdb")))
        self.assertEqual(loader.call_count, 1)
        self.assertEqual((len(system.assets), system.static_count, system.rigid_count), (1, 2, 0))

    def test_zero_count_skips_missing_nvdb_and_volume_allocation(self):
        self.rigid.mkdir()
        (self.rigid / "disabled.obj").write_text("unused", encoding="utf-8")
        with patch.object(boundary, "load_boundary_volumes", side_effect=AssertionError("loaded disabled asset")):
            system = self.load(boundary.BoundaryScene(self.root, default_rigid_count=0))
        self.assertEqual(system.assets, ())
        self.assertEqual(system.rigid_count, 0)

    def test_discovery_errors_precede_device_initialization_and_cache_writes(self):
        self.asset("box")
        for scene, message in (
            (boundary.BoundaryScene(self.root, default_rigid_count=-1), "nonnegative"),
            (boundary.BoundaryScene(self.root, default_rigid_count=True), "nonnegative"),
            (boundary.BoundaryScene(self.root, rigid_counts={"box": 1.5}), "nonnegative"),
            (boundary.BoundaryScene(self.root, rigid_counts={"box": False}), "nonnegative"),
            (boundary.BoundaryScene(self.root, rigid_counts={"typo": 0}), "Unknown rigid"),
            (boundary.BoundaryScene(self.root, rigid_counts={"box": 2**31}), "int32"),
            (boundary.BoundaryScene(self.root, static_assets="room.nvdb"), "sequence"),
        ):
            with self.subTest(message=message), \
                    patch.object(wp, "get_device", side_effect=AssertionError("device touched")), \
                    self.assertRaisesRegex(ValueError, message):
                self.load(scene)
        (self.rigid / "box.nvdb").unlink()
        with patch.object(wp, "get_device", side_effect=AssertionError("device touched")), \
                self.assertRaisesRegex(FileNotFoundError, "paired NVDB"):
            self.load()

    def test_invalid_load_options_are_rejected_even_for_empty_scene(self):
        for options in (replace(self.options, support_radius=0), replace(self.options, support_radius=float("nan")),
                        replace(self.options, quadrature_points=0), replace(self.options, quadrature_points=True),
                        replace(self.options, device="cpu")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.load(options=options)

    def test_cache_is_built_once_per_asset_and_written_to_that_file(self):
        path = self.asset("box", single=True)
        scene = boundary.BoundaryScene(self.root, rigid_counts={"box": 3})
        options = replace(self.options, quadrature_points=4)
        with patch.object(boundary, "precompute_volumeMap", wraps=boundary.precompute_volumeMap) as compute:
            system = self.load(scene, options)
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(nvdb_io.grid_count(path), 2)
            cached = path.read_bytes()
            self.load(scene, options)
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(path.read_bytes(), cached)
            self.load(scene, replace(options, support_radius=.04))
            self.assertEqual(compute.call_count, 2)
            self.load(scene, replace(options, support_radius=.04, quadrature_points=8))
            self.assertEqual(compute.call_count, 3)
        self.assertEqual(system.assets[0].sdf_path, path)
        self.assertEqual({p.name for p in self.rigid.iterdir()}, {"box.obj", "box.nvdb"})

    def test_cpu_pose_update_normalizes_copy_and_keeps_static_pose(self):
        self.asset("room", static=True)
        self.asset("box")
        system = self.load(boundary.BoundaryScene(self.root, static_assets=("room.nvdb",), rigid_counts={"box": 2}))
        poses = np.array([[1, 2, 3, 0, 0, 1, 1], [-1, 0, 2, 0, 0, 0, 1]], dtype=np.float64)
        original = poses.copy()
        pointer = system.device_data.transforms.ptr
        system.set_rigid_poses(poses)
        result = system.device_data.transforms.numpy()
        np.testing.assert_array_equal(poses, original)
        np.testing.assert_allclose(result[0], [0, 0, 0, 0, 0, 0, 1])
        expected = poses.copy()
        expected[:, 3:] /= np.linalg.norm(expected[:, 3:], axis=1, keepdims=True)
        np.testing.assert_allclose(result[1:], expected, atol=1e-7)
        self.assertEqual(system.device_data.transforms.ptr, pointer)

    def test_gpu_pose_update_has_no_readback_and_reset_reuses_resources(self):
        self.asset("box")
        system = self.load()
        poses = wp.array([[1, 2, 3, 0, 0, 0, 1]], dtype=wp.transform, device=self.device)
        system.set_rigid_poses(poses)  # Warm up before prohibiting readback.
        system.reset()
        pointers = (system.device_data.transforms.ptr, system.device_data.sdf_ids.ptr)
        resource_ids = (system.assets[0].sdf.id, system.assets[0].volume_map.id)
        with patch.object(wp.array, "numpy", side_effect=AssertionError("GPU readback")):
            system.set_rigid_poses(poses)
            wp.synchronize_device(self.device)
        poses.fill_(wp.transform_identity())
        np.testing.assert_array_equal(system.device_data.transforms.numpy(), [[1, 2, 3, 0, 0, 0, 1]])
        with patch.object(boundary, "load_boundary_volumes", side_effect=AssertionError("reset reloaded")):
            system.reset()
        np.testing.assert_array_equal(system.device_data.transforms.numpy(), [[0, 0, 0, 0, 0, 0, 1]])
        self.assertEqual(pointers, (system.device_data.transforms.ptr, system.device_data.sdf_ids.ptr))
        self.assertEqual(resource_ids, (system.assets[0].sdf.id, system.assets[0].volume_map.id))

    def test_bad_pose_shapes_values_and_quaternions_are_rejected(self):
        self.asset("box")
        system = self.load()
        for poses in ([], np.zeros((2, 7)), [[0, 0, 0, 0, 0, 0, 0]],
                      [[float("nan"), 0, 0, 0, 0, 0, 1]]):
            with self.subTest(poses=poses), self.assertRaises(ValueError):
                system.set_rigid_poses(poses)
        with self.assertRaises(ValueError):
            system.set_rigid_poses(wp.zeros(1, dtype=wp.vec3, device=self.device))

    def test_translated_rotated_instance_queries_preserve_distance_and_volume(self):
        self.asset("room", static=True)
        self.asset("box")
        system = self.load(boundary.BoundaryScene(self.root, static_assets=("room.nvdb",)))
        system.set_rigid_poses([[1, 2, 3, 0, 0, np.sqrt(.5), np.sqrt(.5)]])
        ids = wp.array([0, 1], dtype=int, device=self.device)
        points = wp.array([[.05, .05, 0], [.95, 2.05, 3]], dtype=wp.vec3, device=self.device)
        distance = wp.empty(2, dtype=float, device=self.device)
        gradient = wp.empty(2, dtype=wp.vec3, device=self.device)
        volume = wp.empty(2, dtype=float, device=self.device)
        wp.launch(_query_instances, 2, inputs=[system.device_data, ids, points],
                  outputs=[distance, gradient, volume], device=self.device)
        np.testing.assert_allclose(distance.numpy(), [.05, .05], atol=1e-6)
        np.testing.assert_allclose(gradient.numpy(), [[0, 1, 0], [-1, 0, 0]], atol=2e-6)
        np.testing.assert_allclose(volume.numpy(), [.003, .003], atol=1e-8)

    def test_system_owns_volume_lifetimes_until_released(self):
        self.asset("box")
        system = self.load(boundary.BoundaryScene(self.root, rigid_counts={"box": 3}))
        refs = [weakref.ref(system.assets[0].sdf), weakref.ref(system.assets[0].volume_map)]
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in refs))
        del system
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

