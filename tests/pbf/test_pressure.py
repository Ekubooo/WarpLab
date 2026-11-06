"""Run the single-configuration PBF pressure fixture in its own process."""

from pathlib import Path
import subprocess
import sys
import unittest


class PressureTests(unittest.TestCase):
    def test_pressure_coupling(self):
        result = subprocess.run([sys.executable, "-B", "-m", "tests.pbf.pressure_case"],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
