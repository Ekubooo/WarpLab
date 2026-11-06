"""Run controlled native rigid contacts in their own GPU process."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


class RigidCollisionTests(unittest.TestCase):
    def test_stairs_and_dynamic_contact_responses(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, "-B", "-m", "tests.pbf.rigid_collision_case"],
                                cwd=root, text=True, capture_output=True)
        (root / "outputs/pbf-newton/rigid-collisions.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        report = json.loads((root / "outputs/pbf-newton/rigid-collisions.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["cases"]), 6)


if __name__ == "__main__":
    unittest.main()
