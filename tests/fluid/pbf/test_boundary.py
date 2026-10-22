"""Code-selected static assets, resource lifetime and world-space projection."""

import contextlib
from dataclasses import replace
import gc
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
import warp as wp

from demos.fluid.PBF import boundary_model as boundary
from demos.fluid.PBF import config_io
from demos.fluid.PBF import PBF as solver
from demos.fluid.PBF import render_opengl as frontend
from demos.fluid.PBF.simulation import create_pbf_simulation, reset_simulation
from demos.fluid.PBF.pbf_helper import PBFConfig
from tests.fluid.pbf.solver_fixture import bind_solver


@wp.kernel
def probe_sdf(
    data: boundary.SDFDeviceData, points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float), gradients: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    distance, gradient = boundary.sample_boundary_sdf(data, points[i])
    distances[i] = distance
    gradients[i] = gradient


class BoundaryInterfaceTest(unittest.TestCase):
    def test_asset_validation_and_cli_removal_without_file_access(self):
        invalid = ("", ".", "..", "room.obj", "room.vdb", "room", "../room.nvdb",
                   "./room.nvdb", "sub/room.nvdb", "sub\\room.nvdb", "/room.nvdb",
                   "C:\\room.nvdb", "C:room.nvdb", "\\\\server\\room.nvdb")
        with patch.object(wp, "get_device", side_effect=AssertionError("GPU touched")):
            for entry in (solver, frontend):
                self.assertFalse(hasattr(entry.parse_args([]), "sdf_filename"))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    entry.parse_args(["--sdf-file", "another.nvdb"])
                self.assertEqual(caught.exception.code, 2)
            for name in invalid:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    boundary.load_static_sdf(name)

    def test_missing_selection_does_not_fall_back_to_old_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "Boundary"
            old.mkdir()
            (old / solver.SDF_FILENAME).write_bytes(b"old grid")
            container = root / "Container"
            container.mkdir()
            (container / solver.SDF_FILENAME).write_bytes(b"old grid")
            with patch.object(boundary, "ASSET_DIR", root), \
                    patch.object(wp, "get_device", side_effect=AssertionError("GPU touched")):
                with self.assertRaises(FileNotFoundError) as caught:
                    boundary.load_static_sdf(solver.SDF_FILENAME)
            self.assertIn(str(root / solver.SDF_FILENAME), str(caught.exception))

    def test_opengl_selects_matching_obj_and_reports_its_path(self):
        import trimesh
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(boundary, "ASSET_DIR", root), patch.object(solver, "SDF_FILENAME", "spot.nvdb"):
                self.assertEqual(boundary.resolve_scene_path("spot.nvdb"), root / "spot.obj")
                self.assertEqual(boundary.resolve_scene_path(solver.SDF_FILENAME), root / "spot.obj")
                with self.assertRaises(FileNotFoundError) as caught:
                    frontend.load_static_scene()
                self.assertIn(str(root / "spot.obj"), str(caught.exception))
                expected = trimesh.creation.box()
                expected.export(root / "spot.obj")
                points, faces = frontend.load_static_scene()
                np.testing.assert_allclose(points, expected.vertices)
                np.testing.assert_array_equal(faces, expected.faces)
                (root / "spot.obj").write_text("invalid mesh", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "spot.obj"):
                    frontend.load_static_scene()

    def test_minimal_signatures_and_removed_baking_options(self):
        self.assertEqual(list(inspect.signature(solver.PBF).parameters),
                         ["stage_path", "verbose", "config"])
        self.assertEqual(list(inspect.signature(create_pbf_simulation).parameters),
                         ["verbose", "config"])
        for entry in (solver, frontend):
            for option in (["--boundary-obj", "room.obj"], ["--sdf-voxel-size", "0.02"],
                           ["--sdf-force-rebuild"]):
                error = io.StringIO()
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                    entry.parse_args(option)
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("unrecognized arguments", error.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            preset = Path(directory) / "preset.json"
            preset.write_text(json.dumps({"substeps": 2}), encoding="utf-8")
            for entry in (solver, frontend):
                self.assertEqual(entry.parse_args(["--config", str(preset)]).config.substeps, 2)
                with patch.object(sys, "argv", [entry.__name__, "--export-config", "test.json"]), \
                        patch.object(config_io, "CONFIG_DIR", Path(directory)), \
                        patch.object(boundary, "load_static_sdf", side_effect=AssertionError("SDF touched")), \
                        patch.object(wp, "ScopedDevice", side_effect=AssertionError("GPU touched")), \
                        contextlib.redirect_stdout(io.StringIO()):
                    entry.main()
                self.assertEqual(config_io.load_config(Path(directory) / "test.json"), PBFConfig())

    def test_fixed_directory_and_loading_are_independent_of_cwd(self):
        expected = Path(boundary.__file__).resolve().parent / "Assets"
        self.assertEqual(boundary.ASSET_DIR, expected)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "static.nvdb"
            source.write_bytes(b"mock grid")
            elsewhere = root / "another-cwd"
            elsewhere.mkdir()
            selected = SimpleNamespace(is_cuda=True)
            volume = SimpleNamespace(dtype=wp.float32, is_index=False,
                                     get_voxel_size=lambda: (0.0075,) * 3)
            read_paths = []
            def read(stream, *, device):
                read_paths.append(Path(stream.name))
                self.assertIs(device, selected)
                return volume
            previous = Path.cwd()
            try:
                os.chdir(elsewhere)
                with patch.object(boundary, "ASSET_DIR", root), \
                        patch.object(solver, "SDF_FILENAME", source.name), \
                        patch.object(wp, "get_device", return_value=selected), \
                        patch.object(wp.Volume, "load_from_nvdb", side_effect=read), \
                        patch.object(wp, "synchronize_device"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertIs(boundary.load_static_sdf(solver.SDF_FILENAME, device="cuda:0"), volume)
            finally:
                os.chdir(previous)
            self.assertEqual(read_paths, [source])
            self.assertEqual({p.name for p in root.iterdir()}, {"static.nvdb", "another-cwd"})

    def test_missing_corrupt_nonfloat_and_cpu_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "static.nvdb"
            with patch.object(boundary, "ASSET_DIR", source.parent), \
                    patch.object(solver, "SDF_FILENAME", source.name), \
                    patch.object(wp, "get_device", side_effect=AssertionError("GPU touched")):
                with self.assertRaisesRegex(FileNotFoundError, "Static SDF not found"):
                    boundary.load_static_sdf(solver.SDF_FILENAME)
            source.write_bytes(b"broken")
            with patch.object(boundary, "ASSET_DIR", source.parent), \
                    patch.object(solver, "SDF_FILENAME", source.name), \
                    patch.object(wp, "get_device", return_value=SimpleNamespace(is_cuda=False)):
                with self.assertRaisesRegex(ValueError, "CUDA"):
                    boundary.load_static_sdf(solver.SDF_FILENAME, device="cpu")
            with patch.object(boundary, "ASSET_DIR", source.parent), \
                    patch.object(solver, "SDF_FILENAME", source.name), \
                    patch.object(wp, "get_device", return_value=SimpleNamespace(is_cuda=True)), \
                    patch.object(wp.Volume, "load_from_nvdb", side_effect=ValueError("bad grid")):
                with self.assertRaisesRegex(RuntimeError, "Cannot load static SDF.*bad grid"):
                    boundary.load_static_sdf(solver.SDF_FILENAME)
            with patch.object(boundary, "ASSET_DIR", source.parent), \
                    patch.object(solver, "SDF_FILENAME", source.name), \
                    patch.object(wp, "get_device", return_value=SimpleNamespace(is_cuda=True)), \
                    patch.object(wp.Volume, "load_from_nvdb", return_value=SimpleNamespace(dtype=wp.int32)):
                with self.assertRaisesRegex(RuntimeError, "float32 scalar"):
                    boundary.load_static_sdf(solver.SDF_FILENAME)
            self.assertEqual(source.read_bytes(), b"broken")

    def test_constructor_owns_volume_until_instance_is_released(self):
        class Resource:
            id = 1
        refs = []
        def load(filename=None, *, device):
            resource = Resource()
            refs.append(weakref.ref(resource))
            return resource
        device = object()
        with patch.object(solver.fn, "init_parameters",
                          side_effect=lambda sim: setattr(sim, "smoothing_length", 0.05)), \
                patch.object(solver.fn, "init_device_buffers"), \
                patch.object(wp, "get_device", return_value=device), \
                patch.object(boundary, "load_static_sdf", side_effect=load) as loading, \
                patch.object(boundary, "precompute_volumeMap", return_value=Resource()) as precomputing:
            sim = solver.PBF(stage_path=None)
        loading.assert_called_once_with(solver.SDF_FILENAME, device=device)
        precomputing.assert_called_once_with(sim.static_sdf, sim.smoothing_length)
        # Mock call history must not hold the SDF alive after the instance dies.
        precomputing.reset_mock()
        gc.collect()
        self.assertIs(refs[0](), sim.static_sdf)
        del sim
        gc.collect()
        self.assertIsNone(refs[0]())

    def test_both_main_functions_delegate_loading_to_constructor(self):
        for entry in (solver, frontend):
            args = entry.parse_args(["--num-frames", "1"])
            if entry is solver:
                constructor = "PBF"
                fake = SimpleNamespace(renderer=None, render=Mock(), step=Mock())
            else:
                constructor = "create_pbf_simulation"
                fake = SimpleNamespace(width=3.0, length=2.0, particle_radius=0.01)
            renderer = Mock()
            renderer.is_running.return_value = False
            with patch.object(entry, "parse_args", return_value=args), \
                    patch.object(entry.wp, "ScopedDevice", return_value=contextlib.nullcontext()), \
                    patch.object(boundary, "load_static_sdf",
                                 side_effect=AssertionError("Entry loaded SDF twice")) as loading, \
                    patch.object(entry, constructor, return_value=fake) as creating, \
                    contextlib.ExitStack() as stack:
                if entry is frontend:
                    stack.enter_context(patch.object(entry, "load_static_scene", return_value=([], [])))
                    stack.enter_context(patch.object(entry, "create_pbf_renderer", return_value=renderer))
                    stack.enter_context(patch.object(entry, "register_keyboard_controls"))
                entry.main()
            loading.assert_not_called()
            expected = dict(verbose=args.verbose, config=args.config)
            if entry is solver:
                expected["stage_path"] = args.stage_path
            creating.assert_called_once_with(**expected)


class BoundaryGPUIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is required for NanoVDB")
        cls.device = "cuda:0"

    def test_load_prebaked_volume_and_project_in_world_coordinates(self):
        import trimesh
        from demos.fluid.boundary import sdf_baker

        config = replace(PBFConfig(), particle_radius=0.05, container_size=(1.0, 1.0, 1.0),
                         block_start=(0.1, 0.1, 0.1), block_end=(0.5, 0.5, 0.5))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "room.obj"
            outer = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
            inner = trimesh.creation.box(extents=(1.6, 1.6, 1.6))
            inner.invert()
            trimesh.util.concatenate([outer, inner]).export(source)
            with wp.ScopedDevice(self.device), contextlib.redirect_stdout(io.StringIO()):
                baked = sdf_baker.load_or_bake_sdf(source, cache_dir=root,
                                                  options=sdf_baker.SDFOptions(voxel_size=0.05))
                source.unlink()
                path = baked.cache_path
                del baked
                with patch.object(boundary, "ASSET_DIR", root), \
                        patch.object(solver, "SDF_FILENAME", path.name), \
                        patch.object(sdf_baker, "load_or_bake_sdf", side_effect=AssertionError("Startup baked")):
                    sim = solver.PBF(stage_path=None, config=config)
                volume_ref = weakref.ref(sim.static_sdf)
                self.assertIsInstance(sim.static_sdf, wp.Volume)
                self.assertAlmostEqual(sim.static_sdf.get_voxel_size()[0], 0.05, places=6)
                with patch.object(boundary, "load_static_sdf", side_effect=AssertionError("Reset reloaded SDF")):
                    reset_simulation(sim)
                gc.collect()
                self.assertIs(volume_ref(), sim.static_sdf)
                volume = sim.static_sdf
                data = boundary.SDFDeviceData()
                data.volume_id = volume.id
                data.inverse_voxel_size = wp.vec3(*(1.0 / v for v in volume.get_voxel_size()))
                points = wp.array([[0.4, 0.1, 0.2], [0.9, 0.1, 0.2], [1.04, 0.1, 0.2]], dtype=wp.vec3)
                distances, gradients = wp.empty(3, dtype=float), wp.empty(3, dtype=wp.vec3)
                wp.launch(probe_sdf, dim=3, inputs=[data, points, distances, gradients])
                np.testing.assert_allclose(distances.numpy(), [0.4, -0.1, 0.04], atol=0.015)
                np.testing.assert_allclose(gradients.numpy()[0], [-1, 0, 0], atol=0.02)
                # A second explicitly selected file must not replace the first instance's SDF.
                other_source = root / "other.obj"
                other_inner = trimesh.creation.box(extents=(1.2, 1.2, 1.2))
                other_inner.invert()
                trimesh.util.concatenate([outer, other_inner]).export(other_source)
                other_baked = sdf_baker.load_or_bake_sdf(
                    other_source, cache_dir=root, options=sdf_baker.SDFOptions(voxel_size=0.05))
                other_source.unlink()
                del other_baked
                default_name = solver.SDF_FILENAME
                with patch.object(boundary, "ASSET_DIR", root), \
                        patch.object(solver, "SDF_FILENAME", "other.nvdb"), \
                        patch.object(sdf_baker, "load_or_bake_sdf", side_effect=AssertionError("Startup baked")):
                    other = create_pbf_simulation(config=config)
                self.assertEqual(solver.SDF_FILENAME, default_name)
                self.assertNotEqual(other.static_sdf.id, sim.static_sdf.id)
                other_data = boundary.SDFDeviceData()
                other_data.volume_id = other.static_sdf.id
                other_data.inverse_voxel_size = data.inverse_voxel_size
                wp.launch(probe_sdf, dim=3, inputs=[other_data, points, distances, gradients])
                np.testing.assert_allclose(distances.numpy(), [0.2, -0.1, 0.04], atol=0.015)
                wp.launch(probe_sdf, dim=3, inputs=[data, points, distances, gradients])
                np.testing.assert_allclose(distances.numpy(), [0.4, -0.1, 0.04], atol=0.015)
                other_volume = other.static_sdf
                with patch.object(boundary, "load_static_sdf", side_effect=AssertionError("Reset reloaded SDF")):
                    reset_simulation(other)
                self.assertIs(other.static_sdf, other_volume)
                # These world coordinates must project against the unshifted Volume.
                projected = wp.array([[0.79, 0.1, 0.2], [0.82, 0.1, 0.2], [-0.82, 0.1, 0.2]], dtype=wp.vec3)
                bind_solver(sim.config, sim.grid, volume, sim.volume_map)
                wp.launch(solver.update_prePos, dim=3,
                          inputs=[wp.zeros(3, dtype=wp.vec3)],
                          outputs=[projected])
                np.testing.assert_allclose(projected.numpy(),
                                           [[0.75, 0.1, 0.2], [0.75, 0.1, 0.2], [-0.75, 0.1, 0.2]], atol=2e-5)
                for _ in range(2):
                    bind_solver(sim.config, sim.grid, sim.static_sdf, sim.volume_map)
                    sim.step()
                    bind_solver(other.config, other.grid, other.static_sdf, other.volume_map)
                    other.step()
                self.assertTrue(np.isfinite(sim.pos.numpy()).all())
                self.assertTrue(np.isfinite(sim.v.numpy()).all())
                self.assertTrue(np.isfinite(other.pos.numpy()).all())


if __name__ == "__main__":
    unittest.main()
