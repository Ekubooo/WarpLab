"""Regression checks for random terrain geometry and CLI import entry points."""

from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import trimesh

from demos.fluid.boundary import creator1, scene_creator, terrain_generator


class BoundaryTests(unittest.TestCase):
    def test_random_cell_is_closed_at_positive_zero_and_negative_heights(self):
        for height in (-0.2, 0.0, 0.2):
            with self.subTest(height=height):
                vertices, indices = terrain_generator.create_mesh_terrain(
                    grid_size=(1, 1),
                    block_size=(1.0, 1.0),
                    terrain_types="random_grid",
                    terrain_params={
                        "random_grid": {
                            "grid_width": 1.0,
                            "grid_height_range": (height, height),
                        }
                    },
                )
                mesh = trimesh.Trimesh(vertices=vertices, faces=indices.reshape(-1, 3), process=False)
                mesh = scene_creator.prepare_solid(mesh)
                self.assertTrue(mesh.is_watertight)
                self.assertTrue(mesh.is_volume)
                np.testing.assert_allclose(mesh.bounds[:, 2], [-1.0, height], atol=1e-7)
                self.assertAlmostEqual(mesh.volume, 1.0 + height, places=6)

    def test_seeded_random_grid_cells_are_closed(self):
        vertices, indices = terrain_generator.create_mesh_terrain(
            grid_size=(1, 1), block_size=(2.0, 2.0), terrain_types="random_grid", seed=42,
        )
        for cell, faces in enumerate(indices.reshape(-1, 12, 3)):
            with self.subTest(cell=cell):
                vertex_ids, local_indices = np.unique(faces, return_inverse=True)
                mesh = trimesh.Trimesh(
                    vertices=vertices[vertex_ids], faces=local_indices.reshape(-1, 3), process=False,
                )
                self.assertTrue(scene_creator.prepare_solid(mesh).is_volume)

    def test_cli_supports_script_and_package_entry_points(self):
        repo_root = Path(__file__).resolve().parents[3]
        for module, option in ((scene_creator, "--obj"), (creator1, "--terrain")):
            for entry in ([module.__file__], ["-m", module.__name__]):
                with self.subTest(entry=entry):
                    result = subprocess.run(
                        [sys.executable, "-B", *entry, "--help"],
                        cwd=repo_root, capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
