"""Boundary CLI, resource ownership and GPU sampling integration."""

import contextlib
from dataclasses import replace
import gc
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from demos.fluid.PBF import boundary_model as boundary
from demos.fluid.PBF import config_io
from demos.fluid.PBF import PBF as solver
from demos.fluid.PBF import render_opengl as frontend
from demos.fluid.PBF.pbf_helper import PBFConfig


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
    def test_cli_defaults_and_config_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            preset = Path(directory) / "preset.json"
            preset.write_text(json.dumps({"substeps": 2}), encoding="utf-8")
            for entry in (solver, frontend):
                self.assertIsNone(entry.parse_args([]).boundary_obj)
                self.assertEqual(entry.parse_args([]).sdf_voxel_size, 0.0075)
                args = entry.parse_args([
                    "--config", str(preset), "--boundary-obj", "room.obj",
                    "--sdf-voxel-size", "0.02", "--sdf-force-rebuild",
                ])
                self.assertEqual(args.config.substeps, 2)
                self.assertEqual(boundary.boundary_arguments(args), dict(
                    boundary_obj="room.obj", sdf_voxel_size=0.02, sdf_force_rebuild=True,
                ))
                with patch.object(sys, "argv", [entry.__name__, "--boundary-obj", "missing.obj",
                                                "--export-config", "test.json"]), \
                        patch.object(config_io, "CONFIG_DIR", Path(directory)), \
                        patch.object(boundary, "load_static_boundary", side_effect=AssertionError("SDF touched")), \
                        patch.object(wp, "ScopedDevice", side_effect=AssertionError("GPU touched")), \
                        contextlib.redirect_stdout(io.StringIO()):
                    entry.main()
                self.assertEqual(config_io.load_config(Path(directory) / "test.json"), PBFConfig())

    def test_path_defaults_are_stable_across_working_directories(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(boundary.resolve_container_path("room.obj"), boundary.CONTAINER_DIR / "room.obj")
                self.assertEqual(boundary.resolve_container_path("models/room.obj"), (Path(directory) / "models/room.obj").resolve())
                absolute = Path(directory) / "absolute.obj"
                self.assertEqual(boundary.resolve_container_path(absolute), absolute.resolve())
            finally:
                os.chdir(previous)
        self.assertEqual(boundary.BOUNDARY_DIR, Path(boundary.__file__).resolve().parent / "Assets" / "Boundary")

    def test_both_main_functions_forward_boundary_options_without_rendering(self):
        for entry in (solver, frontend):
            args = entry.parse_args(["--boundary-obj", "room.obj", "--sdf-voxel-size", "0.03",
                                     "--sdf-force-rebuild", "--num-frames", "0"])
            if entry is solver:
                constructor = "Example"
                fake = SimpleNamespace(renderer=None)
            else:
                constructor = "create_pbf_simulation"
                fake = SimpleNamespace(width=3.0, length=2.0, particle_radius=0.01,
                                       reverse_gravity=Mock())
            renderer = Mock()
            renderer.is_running.return_value = False
            with patch.object(entry, "parse_args", return_value=args), \
                    patch.object(entry.wp, "ScopedDevice", return_value=contextlib.nullcontext()), \
                    patch.object(entry, constructor, return_value=fake) as create, \
                    contextlib.ExitStack() as stack:
                if entry is frontend:
                    stack.enter_context(patch.object(entry, "create_pbf_renderer", return_value=renderer))
                    stack.enter_context(patch.object(entry, "register_keyboard_controls"))
                entry.main()
            for key, value in boundary.boundary_arguments(args).items():
                self.assertEqual(create.call_args.kwargs[key], value)


class BoundaryGPUIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA is required for NanoVDB")
        cls.device = "cuda:0"

    def test_hollow_container_bake_load_and_sample_without_changing_solver(self):
        import trimesh

        config = replace(PBFConfig(), particle_radius=0.05, container_size=(1.0, 1.0, 1.0),
                         block_start=(0.1, 0.1, 0.1), block_end=(0.5, 0.5, 0.5))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "room.obj"
            outer = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
            inner = trimesh.creation.box(extents=(1.6, 1.6, 1.6))
            inner.invert()
            trimesh.util.concatenate([outer, inner]).export(source)
            with wp.ScopedDevice(self.device), patch.object(boundary, "BOUNDARY_DIR", root / "cache"), \
                    contextlib.redirect_stdout(io.StringIO()):
                sim = solver.Example(stage_path=None, config=config, boundary_obj=source, sdf_voxel_size=0.05)
                self.assertFalse(sim.static_boundary.cache_hit)
                self.assertTrue(sim.static_boundary.cache_path.is_file())
                # A cache hit must remain usable even if the source disappears.
                source.unlink()
                loaded = solver.Example(stage_path=None, config=config, boundary_obj=source)
                self.assertTrue(loaded.static_boundary.cache_hit)
                self.assertAlmostEqual(loaded.static_boundary.volume.get_voxel_size()[0], 0.05, places=6)
                gc.collect()
                points = wp.array([[0.4, 0.1, 0.2], [0.9, 0.1, 0.2], [1.04, 0.1, 0.2]], dtype=wp.vec3)
                distances, gradients = wp.empty(3, dtype=float), wp.empty(3, dtype=wp.vec3)
                wp.launch(probe_sdf, dim=3,
                          inputs=[loaded.static_boundary.device_data, points, distances, gradients])
                np.testing.assert_allclose(distances.numpy(), [0.4, -0.1, 0.04], atol=0.015)
                np.testing.assert_allclose(gradients.numpy()[0], [-1, 0, 0], atol=0.02)
                original = solver.Example(stage_path=None, config=config)
                self.assertIsNone(original.static_boundary)
                for _ in range(2):
                    sim.step()
                    original.step()
                np.testing.assert_array_equal(sim.pos.numpy(), original.pos.numpy())
                np.testing.assert_array_equal(sim.v.numpy(), original.v.numpy())
            with self.assertRaisesRegex(ValueError, "CUDA"):
                boundary.load_static_boundary(source, device="cpu")


if __name__ == "__main__":
    unittest.main()
