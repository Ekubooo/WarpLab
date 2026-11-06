"""Isolate the fixed-configuration coupled scene from static solver tests."""

from pathlib import Path
import subprocess
import sys
import unittest


class CoupledLifecycleTests(unittest.TestCase):
    def test_cpu_graph_contact_state_controls_and_reset(self):
        result = subprocess.run([sys.executable, "-B", "-m", "tests.pbf.coupled_lifecycle_case",
            "--cpu", "--output", "outputs/pbf-newton/cpu-backend/lifecycle-cpu.json"],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_graph_contact_state_controls_and_reset(self):
        result = subprocess.run([sys.executable, "-B", "-m", "tests.pbf.coupled_lifecycle_case"],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
