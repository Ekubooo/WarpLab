"""Rigid local frames, compound solids, CLI boundaries and publication failures."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from demos.sim_Ready import config, geometry, pipeline, rigid_pipeline, sdf, validation


REPOSITORY = config.ROOT.parents[1]


class RigidGeometryTests(unittest.TestCase):
    def test_default_preserves_offset_origin_and_explicit_transforms(self):
        mesh = trimesh.creation.box(extents=(.4, .6, .8))
        mesh.apply_translation((1.7, -2.3, .5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offset.obj"
            mesh.export(path, digits=17)
            original = path.read_bytes()
            loaded = geometry.prepare_rigid(path)
            np.testing.assert_allclose(loaded.bounds, mesh.bounds, atol=1e-14)
            np.testing.assert_allclose(loaded.center_mass, mesh.center_mass, atol=1e-14)
            converted = geometry.prepare_rigid(path, scale=2, obj_up_axis="Z")
            points = np.asarray(mesh.vertices) * 2
            expected = np.column_stack((points[:, 0], points[:, 2], -points[:, 1]))
            np.testing.assert_allclose(converted.bounds, [expected.min(0), expected.max(0)], atol=1e-14)
            self.assertEqual(path.read_bytes(), original)
            for scale in (0, -1, float("inf"), float("nan")):
                with self.subTest(scale=scale), self.assertRaises(ValueError):
                    geometry.prepare_rigid(path, scale=scale)
            with self.assertRaises(ValueError):
                geometry.prepare_rigid(path, obj_up_axis="X")

    def test_compound_obj_union_overlap_face_contact_and_separation(self):
        for offset, volume, shells in ((.5, 1.5, 1), (1., 2., 1), (2., 2., 2)):
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as directory:
                first, second = trimesh.creation.box(), trimesh.creation.box()
                second.apply_translation((offset, 0, 0))
                path = Path(directory) / "compound.obj"
                trimesh.util.concatenate((first, second)).export(path)
                source = geometry.prepare_rigid(path)
                parts = validation.validate_rigid_parts(source)
                self.assertEqual(len(parts), 2)
                result = geometry.union(parts)
                self.assertAlmostEqual(result.volume, volume)
                self.assertEqual(len(result.split(only_watertight=False, repair=False)), shells)
                checked = validation.validate_solid(result)
                self.assertFalse(hasattr(checked, "cavity_bounds"))
                np.testing.assert_allclose(checked.bounds, [[-.5, -.5, -.5], [offset + .5, .5, .5]])

    def test_exact_vertex_seams_are_welded_without_repairing_faces(self):
        box = trimesh.creation.box()
        # Every triangle owns its vertices, as in an OBJ with normal/UV seams.
        soup = trimesh.Trimesh(box.triangles.reshape(-1, 3),
                              np.arange(len(box.faces) * 3).reshape(-1, 3), process=False)
        parts = validation.validate_rigid_parts(soup)
        self.assertEqual(len(parts), 1)
        self.assertAlmostEqual(parts[0].volume, 1.)

    def test_invalid_inputs_are_rejected_before_boolean_union(self):
        box = trimesh.creation.box()
        missing = box.copy()
        missing.update_faces(np.arange(len(box.faces) - 1))
        reversed_part = box.copy()
        reversed_part.invert()
        other = box.copy()
        other.apply_translation((3, 0, 0))
        mixed = trimesh.util.concatenate((other, reversed_part))
        duplicate = trimesh.Trimesh(box.vertices, np.vstack((box.faces, box.faces[0])), process=False)
        degenerate = trimesh.Trimesh(box.vertices, np.vstack((box.faces, [0, 0, 1])), process=False)
        for mesh in (missing, reversed_part, mixed, duplicate, degenerate):
            with self.subTest(faces=len(mesh.faces)), self.assertRaises(ValueError):
                validation.validate_rigid_parts(mesh)

    def test_point_contact_and_float32_collapse_are_rejected_after_union(self):
        first, second = trimesh.creation.box(), trimesh.creation.box()
        second.apply_translation((1, 1, 1))
        with self.assertRaisesRegex(ValueError, "vertex"):
            validation.validate_solid(geometry.union([first, second]))
        tiny = trimesh.creation.box(extents=(1e-6, 1, 1))
        tiny.apply_translation((100, 0, 0))
        with self.assertRaisesRegex(ValueError, "float32"):
            validation.validate_solid(tiny)

    def test_inward_cavity_is_not_silently_reinterpreted_as_solid(self):
        outer = trimesh.creation.box()
        inner = trimesh.creation.box(extents=(.5, .5, .5))
        inner.invert()
        with self.assertRaisesRegex(ValueError, "positive"):
            validation.validate_rigid_parts(trimesh.util.concatenate((outer, inner)))


class RigidInterfaceTests(unittest.TestCase):
    def test_cli_defaults_required_arguments_and_invalid_numbers(self):
        args = rigid_pipeline.parse_args(["--obj", "part.obj", "--name", "rigid_part"])
        self.assertEqual(args.scale, 1.)
        self.assertIsNone(args.obj_up_axis)
        self.assertEqual(args.sdf_voxel_size, config.BakeOptions.voxel_size)
        self.assertEqual(args.sdf_max_support_radius, config.BakeOptions.max_support_radius)
        for options in ([], ["--obj", "part.obj"],
                        ["--obj", "part.obj", "--name", "../bad"],
                        ["--obj", "part.obj", "--name", "x", "--scale", "nan"],
                        ["--obj", "part.obj", "--name", "x", "--sdf-voxel-size", "0"],
                        ["--obj", "part.obj", "--name", "x", "--sdf-max-support-radius", "inf"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                rigid_pipeline.parse_args(options)

    def test_module_and_script_help_do_not_load_warp_or_geometry(self):
        for command in (["-m", "demos.sim_Ready.rigid_pipeline", "--help"],
                        [str(config.ROOT / "rigid_pipeline.py"), "--help"]):
            result = subprocess.run([sys.executable, *command], cwd=REPOSITORY,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--scale", result.stdout)
            self.assertNotIn("Warp", result.stdout)
        script = """
import sys
from demos.sim_Ready.rigid_pipeline import parse_args
parse_args(['--obj', 'part.obj', '--name', 'rigid_part'])
assert 'warp' not in sys.modules
assert 'trimesh' not in sys.modules
assert not any(n.startswith('demos.fluid') for n in sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=REPOSITORY,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rigid_padding_is_conservative_and_does_not_change_container_options(self):
        for dx, h, requested in ((.0075, .05, 3), (.04, .2, 3), (.04, .05, 50)):
            options = config.BakeOptions(voxel_size=dx, max_support_radius=h, padding_voxels=requested)
            sentinel = object()
            with patch.object(sdf, "bake", return_value=sentinel) as bake:
                self.assertIs(sdf.bake_rigid(object(), options=options), sentinel)
            actual = bake.call_args.kwargs["options"]
            self.assertGreaterEqual(actual.padding_voxels, requested)
            self.assertGreaterEqual(actual.padding_voxels * dx, 2 * h + 2 * np.sqrt(3) * dx)
            self.assertEqual(options.padding_voxels, requested)
            self.assertEqual(replace(actual, padding_voxels=requested), options)


class RigidPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "input.obj"
        box = trimesh.creation.box()
        box.apply_translation((1, 2, 3))
        box.export(self.source)
        self.source_bytes = self.source.read_bytes()
        self.output, self.consumer = self.root / "output/rigid", self.root / "consumer/Rigid"
        for mocker in (patch.object(config, "RIGID_OUTPUT", self.output),
                       patch.object(config, "RIGID_OUTPUT_DIRECTORIES", (self.output, self.consumer))):
            mocker.start()
            self.addCleanup(mocker.stop)

    def build(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return rigid_pipeline.build_rigid_asset(self.source, name="rigid_test", **kwargs)

    @staticmethod
    def fake_volume():
        class Volume:
            def save_to_nvdb(self, path, *, codec):
                Path(path).write_bytes(b"new SDF")
        return Volume()

    def old_assets(self):
        for target in (self.output, self.consumer):
            target.mkdir(parents=True, exist_ok=True)
            (target / "rigid_test.obj").write_bytes(b"old mesh")
            (target / "rigid_test.nvdb").write_bytes(b"old SDF")

    def assert_consumer_unchanged(self):
        self.assertEqual((self.consumer / "rigid_test.obj").read_bytes(), b"old mesh")
        self.assertEqual((self.consumer / "rigid_test.nvdb").read_bytes(), b"old SDF")
        self.assertEqual({p.name for p in self.consumer.iterdir()}, {"rigid_test.obj", "rigid_test.nvdb"})

    def test_success_rebakes_and_publishes_pair_without_modifying_source(self):
        with patch.object(sdf, "bake_rigid", return_value=self.fake_volume()) as bake:
            for _ in range(2):
                result = self.build()
        self.assertEqual(bake.call_count, 2)
        self.assertEqual(result.mesh, self.output / "rigid_test.obj")
        self.assertEqual(result.sdf, self.output / "rigid_test.nvdb")
        self.assertEqual(result.destinations, (self.output.resolve(), self.consumer.resolve()))
        for target in result.destinations:
            np.testing.assert_allclose(trimesh.load(target / result.mesh.name, process=False).bounds,
                                       [[.5, 1.5, 2.5], [1.5, 2.5, 3.5]])
            self.assertEqual((target / result.sdf.name).read_bytes(), b"new SDF")
        self.assertEqual(self.source.read_bytes(), self.source_bytes)
        self.assertEqual(result.mesh.read_bytes(), (self.consumer / result.mesh.name).read_bytes())

    def test_validation_and_bake_failures_never_publish(self):
        self.old_assets()
        for target, function in ((validation, "validate_rigid_parts"), (validation, "validate_solid"),
                                 (sdf, "bake_rigid")):
            with self.subTest(function=function), patch.object(target, function, side_effect=ValueError("bad")), \
                    self.assertRaisesRegex(RuntimeError, "failed"):
                self.build()
            self.assert_consumer_unchanged()
            self.assertEqual((self.output / "rigid_test.nvdb").read_bytes(), b"old SDF")

    def test_publication_failure_restores_matching_consumer_pair(self):
        self.old_assets()
        original_replace = pipeline.os.replace
        failed = False

        def fail_one_copy(source, destination):
            nonlocal failed
            if Path(destination) == self.consumer / "rigid_test.nvdb" and not failed:
                failed = True
                raise OSError("injected publication failure")
            return original_replace(source, destination)

        with patch.object(sdf, "bake_rigid", return_value=self.fake_volume()), \
                patch.object(pipeline.os, "replace", side_effect=fail_one_copy), \
                self.assertRaisesRegex(RuntimeError, "rigid asset publication"):
            self.build()
        self.assert_consumer_unchanged()

    def test_input_overwrite_is_rejected_before_geometry_or_cuda(self):
        with patch.object(config, "RIGID_OUTPUT", self.root), \
                patch.object(config, "RIGID_OUTPUT_DIRECTORIES", (self.root,)), \
                patch.object(geometry, "prepare_rigid", side_effect=AssertionError("geometry touched")), \
                self.assertRaisesRegex(ValueError, "overwrite"):
            rigid_pipeline.build_rigid_asset(self.source, name="input")
        self.assertEqual(self.source.read_bytes(), self.source_bytes)


if __name__ == "__main__":
    unittest.main()
