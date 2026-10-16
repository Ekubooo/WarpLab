"""Geometry contracts, CLI boundaries, failure-safe publication and CUDA baking."""

import contextlib
from dataclasses import replace
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
import warp as wp

from demos.sim_Ready import config, geometry, pipeline, sdf, validation
from demos.sim_Ready.__main__ import parse_args


OPTIONS = config.GeometryOptions()
SOURCE = config.ASSETS / "bunny_flat_bottom.obj"
REPOSITORY = config.ROOT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = geometry.prepare_scene(SOURCE)
        cls.collision, cls.bounds = geometry.container(cls.scene, OPTIONS)

    def test_terrain_source_is_an_exact_copy(self):
        original = REPOSITORY / "demos/fluid/boundary/terrain_generator.py"
        self.assertEqual(digest(original), digest(config.ROOT / "terrain_generator.py"))

    def test_default_model_and_container_match_current_assets(self):
        expected = trimesh.load(REPOSITORY / "demos/fluid/boundary/Assets/output/stairs_scene_bool_simplified.obj",
                                force="mesh", process=False)
        visible = geometry.landscape(self.scene, OPTIONS)
        np.testing.assert_allclose(visible.bounds, expected.bounds, atol=2e-6)
        self.assertAlmostEqual(visible.volume, expected.volume, delta=2e-5)
        from scipy.spatial import cKDTree
        # Face/vertex numbering and triangulation may change; geometric samples must agree.
        from trimesh.proximity import closest_point_naive
        for a, b in ((visible, expected), (expected, visible)):
            distances = cKDTree(b.vertices).query(a.vertices)[0]
            points = a.vertices[distances > OPTIONS.simplify_tolerance]
            for chunk in np.array_split(points, max(1, len(points) // 30 + 1)):
                if len(chunk):
                    self.assertLessEqual(closest_point_naive(b, chunk)[1].max(), 0.0011)
        checked = validation.validate_container(self.collision, self.bounds)
        np.testing.assert_allclose(checked.cavity_bounds, [[0.075, 0, 0.075], [3.025, 4.5, 1.525]], atol=1e-6)
        np.testing.assert_allclose(checked.bounds, [[-0.175, -0.2, -0.175], [3.275, 4.75, 1.775]], atol=1e-6)

    def test_arbitrary_model_scaling_support_and_z_up(self):
        with tempfile.TemporaryDirectory() as directory:
            model = trimesh.creation.box((10, 2, 4))
            model.apply_translation([6, 7, -3])
            y_path, z_path = Path(directory) / "y.obj", Path(directory) / "z.obj"
            model.export(y_path)
            model.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            model.export(z_path)
            for yaw in (0, 37, 90):
                options = replace(OPTIONS, model_yaw=yaw)
                y = geometry.prepare_scene(y_path, options=options)
                z = geometry.prepare_scene(z_path, options=options, obj_up_axis="Z")
                self.assertAlmostEqual(y.model.extents[[0, 2]].max(), 0.665, places=6)
                np.testing.assert_allclose(geometry.support_center(y.model), [1.55, 0.497, 0.8], atol=1e-7)
                np.testing.assert_allclose(y.model.bounds, z.model.bounds, atol=1e-7)

    def test_nonflat_support_uses_section_centroid(self):
        model = trimesh.convex.convex_hull([[0, 0, 0], [1, 2, 0], [3, 2, 0], [1, 2, -2]])
        np.testing.assert_allclose(geometry.support_center(model, 0.5), [5 / 12, 0, -1 / 6], atol=1e-10)

    def test_wall_rules_across_sizes_and_overlaps(self):
        for size in ((3.1, 1.6), (2.4, 2.0)):
            for delta in (0.025, 0.075, 0.2):
                options = replace(OPTIONS, size=size, overlap=delta)
                scene = geometry.Scene(geometry.terrain_mesh(options), self.scene.model, 1)
                # Use terrain alone: a model is irrelevant to wall/base topology.
                parts, bounds = geometry.container_parts(scene.terrain, options)
                length, width = bounds[1, [0, 2]] - bounds[0, [0, 2]]
                self.assertAlmostEqual(parts[0].extents[0], length)
                self.assertAlmostEqual(parts[2].extents[2], min(size[1] + delta, width), places=6)
                for long in parts[:2]:
                    for short in parts[2:4]:
                        overlap = np.minimum(long.bounds[1], short.bounds[1]) - np.maximum(long.bounds[0], short.bounds[0])
                        self.assertTrue(np.all(overlap > 0))
                solid = geometry.simplify(geometry.union([scene.terrain, *parts]), 0.001)
                checked = validation.validate_container(solid, bounds)
                self.assertLess(checked.cavity_bounds[1, 1], checked.bounds[1, 1])

    def test_open_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "open.obj"
            mesh = trimesh.creation.box()
            mesh.update_faces(np.arange(len(mesh.faces) - 1))
            mesh.export(path)
            with self.assertRaisesRegex(ValueError, "closed"):
                geometry.prepare_scene(path)

    def test_validation_rejects_bad_topology_and_float32_collapse(self):
        for mutate in (lambda m: m.update_faces(np.arange(len(m.faces) - 1)),
                       lambda m: setattr(m, "faces", np.vstack((m.faces, m.faces[0]))),
                       lambda m: setattr(m, "faces", m.faces[:, ::-1])):
            mesh = self.collision.copy()
            mutate(mesh)
            with self.assertRaises(ValueError):
                validation.validate_container(mesh, self.bounds)
        # Both source shells are valid but the thin outer/inner separation collapses on upload.
        outer = trimesh.creation.box((1e-6, 1, 1))
        outer.apply_translation([100, 0, 0])
        inner = trimesh.creation.box((0.5e-6, 0.5, 0.5))
        inner.apply_translation([100, 0, 0])
        inner.invert()
        mesh = trimesh.util.concatenate([outer, inner])
        with self.assertRaisesRegex(ValueError, "float32"):
            validation.validate_container(mesh, outer.bounds)


class InterfaceTests(unittest.TestCase):
    def test_names_and_obj_resolution(self):
        for name in ("", "..", "x.obj", "../a", "x/y", "x\\y", "C:x", " bad", "CON", "a?"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pipeline.validate_name(name)
        self.assertEqual(pipeline.validate_name("stairs_bunny"), "stairs_bunny")
        self.assertEqual(pipeline.resolve_obj("bunny.obj"), config.ASSETS / "bunny.obj")
        self.assertEqual(pipeline.resolve_obj("./local.obj"), (Path.cwd() / "local.obj").resolve())
        self.assertEqual(pipeline.resolve_obj(SOURCE), SOURCE)
        self.assertEqual(parse_args([]).name, "stairs_scene")

    def test_help_and_imports_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(REPOSITORY))
            commands = (["-m", "demos.sim_Ready", "--help"], [str(config.ROOT / "__main__.py"), "--help"])
            for command in commands:
                result = subprocess.run([sys.executable, *command], cwd=directory, env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--obj", result.stdout)
                self.assertNotIn("Warp", result.stdout)
            script = """
import sys
from demos.sim_Ready.__main__ import parse_args
parse_args([])
assert 'warp' not in sys.modules
from demos.sim_Ready import geometry, validation, sdf, pipeline
assert not any(n.startswith(('demos.fluid.boundary', 'demos.fluid.PBF')) for n in sys.modules)
"""
            result = subprocess.run([sys.executable, "-c", script], cwd=directory, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = geometry.prepare_scene(SOURCE)
        cls.visible = geometry.landscape(cls.scene, OPTIONS)
        cls.collision, cls.bounds = geometry.container(cls.scene, OPTIONS)
        cls.checked = validation.validate_container(cls.collision, cls.bounds)

    def build(self, root, **kwargs):
        with patch.object(config, "OUTPUT", root / "output"), \
                patch.object(config, "OUTPUT_DIRECTORIES", (root / "output", root / "consumer")), \
                patch.object(geometry, "prepare_scene", return_value=self.scene), \
                patch.object(geometry, "landscape", return_value=self.visible), \
                patch.object(geometry, "container", return_value=(self.collision, self.bounds)), \
                contextlib.redirect_stdout(io.StringIO()):
            return pipeline.build_assets(SOURCE, name="test_scene", **kwargs)

    def test_success_always_rebakes_and_publishes_matching_pair(self):
        class Volume:
            def save_to_nvdb(self, path, *, codec):
                Path(path).write_bytes(b"new grid")
        with tempfile.TemporaryDirectory() as directory, patch.object(sdf, "bake", return_value=Volume()) as bake:
            root = Path(directory)
            for _ in range(2):
                self.build(root)
            self.assertEqual(bake.call_count, 2)
            for target in (root / "output", root / "consumer"):
                self.assertEqual({p.name for p in target.iterdir()}, {"test_scene.obj", "test_scene.nvdb"})
                self.assertEqual((target / "test_scene.nvdb").read_bytes(), b"new grid")
            self.assertEqual(digest(root / "output/test_scene.obj"), digest(root / "consumer/test_scene.obj"))

    def test_validation_and_bake_failures_leave_consumer_pair_untouched(self):
        for stage in ("validation", "bake"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for target in (root / "output", root / "consumer"):
                    target.mkdir()
                    (target / "test_scene.nvdb").write_bytes(b"old grid")
                (root / "consumer/test_scene.obj").write_bytes(b"old mesh")
                target = validation if stage == "validation" else sdf
                attribute = "validate_container" if stage == "validation" else "bake"
                with patch.object(target, attribute, side_effect=ValueError("injected failure")), \
                        self.assertRaisesRegex(RuntimeError, "failed"):
                    self.build(root)
                self.assertTrue((root / "output/test_scene.obj").is_file())
                self.assertEqual((root / "output/test_scene.nvdb").read_bytes(), b"old grid")
                self.assertEqual((root / "consumer/test_scene.obj").read_bytes(), b"old mesh")
                self.assertEqual((root / "consumer/test_scene.nvdb").read_bytes(), b"old grid")

    def test_atomic_volume_save_failure_preserves_old_file(self):
        class Broken:
            def save_to_nvdb(self, path, *, codec):
                Path(path).write_bytes(b"partial")
                raise OSError("injected disk error")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.nvdb"
            path.write_bytes(b"old")
            with self.assertRaises(OSError):
                sdf.save_volume(Broken(), path)
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_publication_rolls_back_on_second_directory_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = (root / "x.obj", root / "x.nvdb")
            destinations = (root / "a", root / "b")
            for source in sources:
                source.write_bytes(b"new")
            for target in destinations:
                target.mkdir()
                for source in sources:
                    (target / source.name).write_bytes(b"old")
            original = os.replace
            failed = False
            def replace_file(a, b):
                nonlocal failed
                if Path(b) == destinations[1] / "x.nvdb" and not failed:
                    failed = True
                    raise OSError("publication failure")
                return original(a, b)
            with patch.object(pipeline.os, "replace", side_effect=replace_file), self.assertRaises(OSError):
                pipeline.publish_pair(sources, destinations)
            for target in destinations:
                self.assertEqual({p.name for p in target.iterdir()}, {"x.obj", "x.nvdb"})
                for path in target.iterdir():
                    self.assertEqual(path.read_bytes(), b"old")

    def test_input_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.obj"
            source.write_bytes(b"input")
            with patch.object(config, "OUTPUT", root), \
                    patch.object(config, "OUTPUT_DIRECTORIES", (root,)), \
                    self.assertRaisesRegex(ValueError, "overwrite"):
                pipeline.build_assets(source, name="model")
            self.assertEqual(source.read_bytes(), b"input")

    def test_failed_rollback_keeps_recovery_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = (root / "x.obj", root / "x.nvdb")
            destination = root / "consumer"
            destination.mkdir()
            for source in sources:
                source.write_bytes(b"new")
                (destination / source.name).write_bytes(b"old")
            real_replace = os.replace
            calls = 0
            def fail_after_first(a, b):
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise OSError("persistent disk failure")
                return real_replace(a, b)
            with patch.object(pipeline.os, "replace", side_effect=fail_after_first), \
                    self.assertRaisesRegex(RuntimeError, "recovery copies"):
                pipeline.publish_pair(sources, (destination,))
            backups = list(destination.glob("*.asset.tmp"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b"old")


@wp.kernel
def probe(volume: wp.uint64, positions: wp.array(dtype=wp.vec3), values: wp.array(dtype=float)):
    i = wp.tid()
    index = wp.volume_world_to_index(volume, positions[i])
    values[i] = wp.volume_sample_f(volume, index, wp.Volume.LINEAR)


class BakeTests(unittest.TestCase):
    def test_cuda_round_trip_and_signs(self):
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        scene = geometry.prepare_scene(SOURCE)
        collision, bounds = geometry.container(scene, OPTIONS)
        checked = validation.validate_container(collision, bounds)
        volume = sdf.bake(checked, options=config.BakeOptions(voxel_size=0.025))
        points = wp.array([[1.55, 3, 0.8], [0, 2, 0.8], [1.55, 4.6, 0.8],
                           [1.55, 0.25, 0.8], [1.55, -0.1, 0.8]], dtype=wp.vec3, device="cuda:0")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.nvdb"
            sdf.save_volume(volume, path)
            with path.open("rb") as stream:
                loaded = wp.Volume.load_from_nvdb(stream, device="cuda:0")
            self.assertEqual(loaded.dtype, wp.float32)
            self.assertFalse(loaded.is_index)
            for grid in (volume, loaded):
                values = wp.empty(len(points), dtype=float, device="cuda:0")
                wp.launch(probe, dim=len(points), inputs=[grid.id, points, values], device="cuda:0")
                distances = values.numpy()
                self.assertGreater(distances[0], 0)
                self.assertTrue(np.all(distances[1:] < 0), distances)
                self.assertTrue(np.isfinite(distances).all())

    def test_invalid_bake_options(self):
        for options in (config.BakeOptions(voxel_size=0), config.BakeOptions(voxel_size=float("nan")),
                        config.BakeOptions(padding_voxels=0), config.BakeOptions(padding_voxels=True)):
            with self.assertRaises(ValueError):
                sdf.check_options(options)
