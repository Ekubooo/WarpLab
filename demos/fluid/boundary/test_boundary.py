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

    @unittest.skipUnless(scene_creator.DEFAULT_OBJ_PATH.is_file(), "Default OBJ asset not installed")
    def test_default_model_fits_and_contacts_the_central_platform(self):
        terrain = scene_creator.generate_stairs()
        platform_z = terrain.bounds[1, 2]
        platform_vertices = terrain.vertices[np.isclose(terrain.vertices[:, 2], platform_z)]
        platform_min = platform_vertices[:, :2].min(axis=0)
        platform_max = platform_vertices[:, :2].max(axis=0)
        obj = scene_creator.load_and_place_obj(scene_creator.DEFAULT_OBJ_PATH, terrain)
        obj = scene_creator.prepare_solid(obj)
        self.assertTrue(obj.is_volume)
        np.testing.assert_allclose(obj.bounds[:, :2].mean(axis=0), (platform_min + platform_max) / 2)
        clearance = (platform_max - platform_min) * 0.05
        self.assertTrue(np.all(obj.bounds[0, :2] > platform_min + clearance))
        self.assertTrue(np.all(obj.bounds[1, :2] < platform_max - clearance))
        self.assertGreater(obj.extents[2], 0.5)
        self.assertLess(obj.extents[2], 1.0)
        self.assertLess(obj.bounds[0, 2], platform_z)
        self.assertGreater(obj.bounds[0, 2], platform_z - 0.02)
        # 裁平的封口完全位于平台内部，且所有底面三角形都在同一个平面。
        bottom_z = obj.bounds[0, 2]
        bottom_faces = np.all(np.isclose(obj.triangles[:, :, 2], bottom_z,
                                        atol=1e-10, rtol=0), axis=1)
        self.assertGreater(np.count_nonzero(bottom_faces), 0)
        np.testing.assert_allclose(obj.face_normals[bottom_faces],
                                   np.tile([0, 0, -1], (bottom_faces.sum(), 1)), atol=1e-10)
        solid_terrain = scene_creator.union_stair_boxes(terrain)
        intersection = trimesh.boolean.intersection([solid_terrain, obj], engine="manifold")
        self.assertTrue(intersection.is_volume)
        self.assertGreater(intersection.volume, 0)
        scene = scene_creator.boolean_union_meshes([solid_terrain, obj], "Default bunny placement")
        self.assertEqual(len(scene.split(only_watertight=False)), 1)
        self.assertTrue(scene.is_volume)
        # 不能只检查连通性：残留兔子底面仍可能接在外壳上。
        retained_caps = np.all(np.isclose(scene.triangles[:, :, 2], bottom_z,
                                         atol=1e-7, rtol=0), axis=1)
        self.assertFalse(retained_caps.any())
        # 平台内部也不能留下倾斜底面或空腔的侧面。
        centers = scene.triangles_center
        inside_xy = np.all((centers[:, :2] > platform_min + 1e-7)
                           & (centers[:, :2] < platform_max - 1e-7), axis=1)
        inside_z = ((centers[:, 2] > solid_terrain.bounds[0, 2] + 1e-7)
                    & (centers[:, 2] < platform_z - 1e-7))
        self.assertFalse(np.any(inside_xy & inside_z))
        self.assertAlmostEqual(scene.volume, solid_terrain.volume + obj.volume - intersection.volume,
                               places=6)

    def test_default_bunny_asset_has_a_planar_closed_bottom(self):
        self.assertTrue(scene_creator.DEFAULT_OBJ_PATH.is_file(), "Flattened default asset is required")
        mesh = scene_creator.prepare_solid(trimesh.load(scene_creator.DEFAULT_OBJ_PATH,
                                                       force="mesh", process=False))
        self.assertTrue(mesh.is_volume)
        self.assertEqual(len(mesh.split(only_watertight=False)), 1)
        self.assertAlmostEqual(mesh.bounds[0, 1], 0.04, places=12)
        bottom = np.all(np.isclose(mesh.triangles[:, :, 1], 0.04, atol=1e-12, rtol=0), axis=1)
        self.assertGreater(mesh.area_faces[bottom].sum(), 0.006)
        np.testing.assert_allclose(mesh.face_normals[bottom],
                                   np.tile([0, -1, 0], (bottom.sum(), 1)), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
