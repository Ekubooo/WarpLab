"""CLI entrypoints must work without creating a scene for help/config export."""

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


class EntrypointTests(unittest.TestCase):
    def test_module_and_direct_script_help(self):
        for command in (("-m", "PBF.PBF"), ("-m", "PBF.render_opengl"),
                        (str(ROOT / "PBF/PBF.py"),), (str(ROOT / "PBF/render_opengl.py"),)):
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, "-B", *command, "--help"],
                                        cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--config", result.stdout)
                self.assertIn("--scene", result.stdout)
                self.assertNotIn("Boundary cache", result.stdout)

    def test_config_export_and_import_do_not_initialize_cuda(self):
        code = '''
import contextlib, io, sys, tempfile
from pathlib import Path
from unittest.mock import patch
import warp as wp
from PBF import PBF as entry, config_io, render_opengl
with tempfile.TemporaryDirectory() as directory, \\
     patch.object(wp, 'init', side_effect=AssertionError('CUDA initialized')), \\
     patch.object(config_io, 'CONFIG_DIR', Path(directory)), \\
     patch.object(sys, 'argv', ['PBF', '--export-config', 'test.json']), \\
     contextlib.redirect_stdout(io.StringIO()):
    entry.main()
    assert config_io.load_config(Path(directory) / 'test.json') == entry.fn.PBFConfig()
assert not any(n.startswith(('mujoco', 'demos.PBF_Rigid')) for n in sys.modules)
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
