"""Independent baking CLI paths, isolation, and real cache/error behavior."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import trimesh
import warp as wp

from demos.fluid.PBF import bake_sdf as cli
from demos.fluid.PBF import boundary_model
from demos.fluid.boundary import sdf_baker


class BakeCLIInterfaceTests(unittest.TestCase):
    def test_required_argument_defaults_and_options(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli.parse_args([])
        self.assertEqual(caught.exception.code, 2)
        args = cli.parse_args(["--boundary-obj", "room.obj"])
        self.assertEqual((args.device, args.sdf_voxel_size, args.sdf_force_rebuild),
                         ("cuda:0", 0.0075, False))
        args = cli.parse_args(["--boundary-obj", "room.obj", "--device", "cuda:1",
                               "--sdf-voxel-size", "0.01", "--sdf-force-rebuild"])
        self.assertEqual((args.device, args.sdf_voxel_size, args.sdf_force_rebuild),
                         ("cuda:1", 0.01, True))

    def test_filename_directory_and_absolute_resolution(self):
        self.assertEqual(cli.CONTAINER_DIR, Path(cli.__file__).resolve().parent / "Assets/Container")
        self.assertEqual(cli.BOUNDARY_DIR, Path(cli.__file__).resolve().parent / "Assets/Boundary")
        self.assertEqual(cli.resolve_container_path("room.obj"), cli.CONTAINER_DIR / "room.obj")
        for relative in ("./room.obj", "subdir/room.obj", ".\\room.obj"):
            self.assertEqual(cli.resolve_container_path(relative), Path(relative).resolve())
        absolute = Path(tempfile.gettempdir()).resolve() / "room.obj"
        self.assertEqual(cli.resolve_container_path(str(absolute)), absolute)

    def test_module_and_direct_help_do_not_import_simulation_or_warp(self):
        root = Path(__file__).resolve().parents[3]
        for target, cwd in ((["-m", "demos.fluid.PBF.bake_sdf"], root),
                            ([str(Path(cli.__file__).resolve())], Path(tempfile.gettempdir()))):
            result = subprocess.run([sys.executable, *target, "--help"], cwd=cwd,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--boundary-obj", result.stdout)
            self.assertNotIn("Warp", result.stdout)
        # Fresh interpreter: even executing the CLI must not import PBF or OpenGL.
        script = (
            "import runpy, sys\n"
            "from unittest.mock import patch\n"
            "from types import SimpleNamespace as S\n"
            "from demos.fluid.boundary import sdf_baker as b\n"
            "r=S(cache_hit=True, cache_path='mock.nvdb', "
            "volume=S(get_voxel_size=lambda: (0.0075,)*3))\n"
            "sys.argv=['bake_sdf', '--boundary-obj', 'mock.obj']\n"
            "with patch.object(b, 'load_or_bake_sdf', return_value=r):\n"
            " runpy.run_module('demos.fluid.PBF.bake_sdf', run_name='__main__')\n"
            "assert 'demos.fluid.PBF.PBF' not in sys.modules\n"
            "assert 'demos.fluid.PBF.render_opengl' not in sys.modules\n"
            "assert 'demos.fluid.PBF.boundary_model' not in sys.modules\n"
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=root,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class BakeCLIGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("NanoVDB requires CUDA")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.inputs = self.root / "Container"
        self.outputs = self.root / "Boundary"
        self.inputs.mkdir()
        self.source = self.inputs / "room.obj"
        outer = trimesh.creation.box(extents=(1, 1, 1))
        inner = trimesh.creation.box(extents=(0.6, 0.6, 0.6))
        inner.invert()
        trimesh.util.concatenate([outer, inner]).export(self.source)
        for name, value in (("CONTAINER_DIR", self.inputs), ("BOUNDARY_DIR", self.outputs)):
            patcher = patch.object(cli, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.target = self.outputs / "room.nvdb"

    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = cli.main(["--boundary-obj", "room.obj", "--sdf-voxel-size", "0.05", *args])
        self.assertEqual(result, self.target)
        self.assertIn(str(self.target), output.getvalue())
        return output.getvalue()

    def test_missing_cache_bakes_hit_skips_missing_source_and_force_rebuilds(self):
        original_static = boundary_model.STATIC_SDF_PATH
        self.assertIn("baked and cached", self.run_cli())
        saved = self.target.read_bytes()
        self.source.unlink()
        with patch.object(sdf_baker, "_bake_validated_mesh", side_effect=AssertionError("rebaked")):
            self.assertIn("cache loaded", self.run_cli("--sdf-voxel-size", "0.1"))
        self.assertEqual(self.target.read_bytes(), saved)
        with self.assertRaisesRegex(SystemExit, "Cannot bake/load SDF"):
            self.run_cli("--sdf-force-rebuild")
        self.assertEqual(self.target.read_bytes(), saved)
        self.assertEqual(boundary_model.STATIC_SDF_PATH, original_static)

    def test_corrupt_cache_errors_and_explicit_rebuild_replaces(self):
        self.outputs.mkdir()
        self.target.write_bytes(b"broken cache")
        with self.assertRaisesRegex(SystemExit, "force rebuild"):
            self.run_cli()
        self.assertEqual(self.target.read_bytes(), b"broken cache")
        self.assertIn("baked and cached", self.run_cli("--sdf-force-rebuild"))
        self.assertGreater(self.target.stat().st_size, 100)

    def test_invalid_obj_cannot_replace_valid_cache_or_create_new_cache(self):
        self.run_cli()
        saved = self.target.read_bytes()
        trimesh.creation.box().export(self.source)  # No enclosed fluid cavity.
        with self.assertRaisesRegex(SystemExit, "Cannot bake/load SDF"):
            self.run_cli("--sdf-force-rebuild")
        self.assertEqual(self.target.read_bytes(), saved)
        self.target.unlink()
        with self.assertRaisesRegex(SystemExit, "Cannot bake/load SDF"):
            self.run_cli()
        self.assertFalse(self.target.exists())

    def test_cpu_rejected_without_output(self):
        with self.assertRaisesRegex(SystemExit, "CUDA"):
            self.run_cli("--device", "cpu")
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
