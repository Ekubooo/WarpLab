"""Regression checks for random terrain geometry and CLI import entry points."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

from demos.fluid.boundary import creator1, scene_creator, terrain_generator


class BoundaryTests(unittest.TestCase):
    @staticmethod
    def make_stairs(platform_width, block_size=(6.0, 4.0)):
        vertices, indices = terrain_generator.create_mesh_terrain(
            grid_size=(1, 1), block_size=block_size, terrain_types="pyramid_stairs",
            terrain_params={"pyramid_stairs": {
                "step_width": 0.2, "step_height": 0.1, "platform_width": platform_width,
            }},
        )
        return trimesh.Trimesh(vertices=vertices, faces=indices.reshape(-1, 3), process=False)

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
                        cwd=repo_root, capture_output=True, text=True, encoding="utf-8",
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(option, result.stdout)
                    if module is scene_creator:
                        self.assertIn("--scale-with-platform", result.stdout)
                        self.assertIn("--no-scale-with-platform", result.stdout)

    def test_model_tracks_platform_width_and_ignores_outer_terrain_size(self):
        source = trimesh.load(scene_creator.DEFAULT_OBJ_PATH, force="mesh", process=False)
        reference_extents = source.extents[[0, 2, 1]] * scene_creator.DEFAULT_OBJ_SCALE
        for width in (0.4, 1.0, 2.0):
            with self.subTest(width=width):
                terrain = self.make_stairs(width)
                original_vertices, original_faces = terrain.vertices.copy(), terrain.faces.copy()
                obj = scene_creator.load_and_place_obj(scene_creator.DEFAULT_OBJ_PATH, terrain)
                np.testing.assert_allclose(obj.extents, reference_extents * width, rtol=1e-6)
                self.assertAlmostEqual(obj.extents[0] / width, 0.95, places=6)
                self.assertAlmostEqual(terrain.bounds[1, 2] - obj.bounds[0, 2],
                                       scene_creator.DEFAULT_OBJ_SINK, places=10)
                np.testing.assert_array_equal(terrain.vertices, original_vertices)
                np.testing.assert_array_equal(terrain.faces, original_faces)
        small_ground = self.make_stairs(0.4)
        large_ground = self.make_stairs(0.4, block_size=(10.0, 8.0))
        small_obj = scene_creator.load_and_place_obj(scene_creator.DEFAULT_OBJ_PATH, small_ground)
        large_obj = scene_creator.load_and_place_obj(scene_creator.DEFAULT_OBJ_PATH, large_ground)
        # 生成器是 float32，较大坐标下平台边界的舍入误差约为亚微米。
        np.testing.assert_allclose(small_obj.extents, large_obj.extents, atol=1e-6, rtol=1e-6)

    def test_independent_obj_scale_and_disabled_platform_scaling(self):
        terrain = self.make_stairs(0.4)
        original_vertices, original_faces = terrain.vertices.copy(), terrain.faces.copy()
        source = trimesh.load(scene_creator.DEFAULT_OBJ_PATH, force="mesh", process=False)
        scale3 = scene_creator.load_and_place_obj(scene_creator.DEFAULT_OBJ_PATH, terrain, scale=3)
        scale6 = scene_creator.load_and_place_obj(scene_creator.DEFAULT_OBJ_PATH, terrain, scale=6)
        np.testing.assert_allclose(scale6.extents, scale3.extents * 2)
        fixed = scene_creator.load_and_place_obj(scene_creator.DEFAULT_OBJ_PATH, terrain,
                                                 scale=5, scale_with_platform=False)
        np.testing.assert_allclose(fixed.extents, source.extents[[0, 2, 1]] * 5)
        np.testing.assert_array_equal(terrain.vertices, original_vertices)
        np.testing.assert_array_equal(terrain.faces, original_faces)

    def test_custom_models_axes_yaw_and_explicit_position(self):
        terrain = self.make_stairs(0.4)
        original_vertices, original_faces = terrain.vertices.copy(), terrain.faces.copy()
        with tempfile.TemporaryDirectory() as directory:
            for axis, extents in (("Z", (2, 3, 4)), ("Y", (2, 4, 3))):
                with self.subTest(axis=axis):
                    path = Path(directory) / f"custom_{axis}.obj"
                    trimesh.creation.box(extents=extents).export(path)
                    obj = scene_creator.load_and_place_obj(path, terrain, scale=5, yaw_deg=90,
                                                          obj_up_axis=axis, position=(7, 8, 9), sink=0.1)
                    np.testing.assert_allclose(obj.extents, [6, 4, 8], rtol=1e-6)
                    np.testing.assert_allclose(np.r_[obj.bounds[:, :2].mean(axis=0), obj.bounds[0, 2]],
                                               [7, 8, 9], atol=1e-10)
        np.testing.assert_array_equal(terrain.vertices, original_vertices)
        np.testing.assert_array_equal(terrain.faces, original_faces)

    def test_rectangular_off_center_platform_uses_short_edge_and_actual_center(self):
        ground = trimesh.creation.box(extents=(10, 8, 0.1))
        platform = trimesh.creation.box(extents=(0.4, 0.8, 0.2))
        platform.apply_translation([1, 2, 0.15])
        terrain = trimesh.util.concatenate([ground, platform])
        obj = scene_creator.load_and_place_obj(scene_creator.DEFAULT_OBJ_PATH, terrain)
        source = trimesh.load(scene_creator.DEFAULT_OBJ_PATH, force="mesh", process=False)
        np.testing.assert_allclose(obj.extents, source.extents[[0, 2, 1]]
                                   * scene_creator.DEFAULT_OBJ_SCALE * 0.4)
        np.testing.assert_allclose(scene_creator.get_support_center(obj)[:2], [1, 2])
        self.assertAlmostEqual(obj.bounds[0, 2], 0.25 - scene_creator.DEFAULT_OBJ_SINK)

    def test_missing_platform_errors_only_when_scaling_is_enabled(self):
        for terrain in (trimesh.creation.icosphere(subdivisions=1),
                        trimesh.Trimesh(vertices=[[0, 0, 1], [1, 0, 1], [2, 0, 1]], faces=[[0, 1, 2]],
                                        process=False), trimesh.Trimesh()):
            with self.subTest(terrain=terrain):
                with self.assertRaisesRegex(ValueError, "有效水平平台"):
                    scene_creator.load_and_place_obj(scene_creator.DEFAULT_OBJ_PATH, terrain,
                                                     position=(1, 2, 3))
                fixed = scene_creator.load_and_place_obj(scene_creator.DEFAULT_OBJ_PATH, terrain,
                                                         position=(1, 2, 3), scale_with_platform=False)
                self.assertGreater(len(fixed.faces), 0)

    def test_cli_passes_platform_scaling_toggle_to_scene_creation(self):
        for options, enabled in (([], True), (["--scale-with-platform"], True),
                                 (["--no-scale-with-platform"], False)):
            with self.subTest(options=options):
                with mock.patch.object(sys, "argv", ["scene_creator", *options]), \
                        mock.patch.object(scene_creator, "create_scene", return_value=Path("scene.obj")) as create:
                    scene_creator.main()
                self.assertIs(create.call_args.kwargs["scale_with_platform"], enabled)

    def test_asymmetric_support_uses_area_centroid_after_rotation(self):
        # L 形底面由面积 2 和 1 的矩形组成，重心为 (-1/6, 5/6)，上部向右伸出。
        parts = []
        for size, center in (((2, 1, 1), (0, 0.5, 0.5)),
                             ((1, 1, 1), (-0.5, 1.5, 0.5)),
                             ((4, 2, 1), (2, 1, 1.5))):
            part = trimesh.creation.box(extents=size)
            part.apply_translation(center)
            parts.append(part)
        mesh = trimesh.boolean.union(parts, engine="manifold")
        expected = [-1 / 6, 5 / 6, 0]
        np.testing.assert_allclose(scene_creator.get_support_center(mesh), expected, atol=1e-10)
        # 改变局部三角形密度不应改变支撑面面积重心。
        vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces,
                                                  face_index=np.flatnonzero(mesh.triangles_center[:, 0] > 0))
        refined = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        np.testing.assert_allclose(scene_creator.get_support_center(refined), expected, atol=1e-10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overhanging.obj"
            mesh.export(path)
            placed = scene_creator.load_and_place_obj(path, self.make_stairs(0.4), scale=1,
                                                       obj_up_axis="Z", yaw_deg=90, position=(7, 8, 9),
                                                       scale_with_platform=False)
            rotated = mesh.copy()
            rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
            np.testing.assert_allclose(placed.vertices, rotated.vertices + [7 + 5 / 6, 8 + 1 / 6, 9],
                                       atol=1e-8)
            np.testing.assert_allclose(scene_creator.get_support_center(placed), [7, 8, 9], atol=1e-10)

    def test_nonplanar_bottom_uses_penetration_section_or_lowest_contact(self):
        points = np.array([[0, 0, 0], [1, 0, 2], [3, 0, 2], [1, 2, 2]], dtype=float)
        mesh = trimesh.convex.convex_hull(points)
        self.assertTrue(mesh.is_volume)
        # 高度 0.5 处为顶面三角形的 1/4 缩放，XY 面积重心可直接计算。
        np.testing.assert_allclose(scene_creator.get_support_center(mesh, contact_depth=0.5),
                                   [5 / 12, 1 / 6, 0], atol=1e-10)
        np.testing.assert_allclose(scene_creator.get_support_center(mesh), [0, 0, 0], atol=1e-10)
        mesh.apply_translation([3, -2, 5])
        np.testing.assert_allclose(scene_creator.get_support_center(mesh, contact_depth=0.5),
                                   [3 + 5 / 12, -2 + 1 / 6, 5], atol=1e-10)

    @unittest.skipUnless(scene_creator.DEFAULT_OBJ_PATH.is_file(), "Default OBJ asset not installed")
    def test_default_model_fits_and_contacts_the_central_platform(self):
        terrain = scene_creator.generate_stairs()
        original_vertices, original_faces = terrain.vertices.copy(), terrain.faces.copy()
        platform_z = terrain.bounds[1, 2]
        platform_vertices = terrain.vertices[np.isclose(terrain.vertices[:, 2], platform_z)]
        platform_min = platform_vertices[:, :2].min(axis=0)
        platform_max = platform_vertices[:, :2].max(axis=0)
        obj = scene_creator.load_and_place_obj(scene_creator.DEFAULT_OBJ_PATH, terrain)
        obj = scene_creator.prepare_solid(obj)
        self.assertTrue(obj.is_volume)
        np.testing.assert_allclose(scene_creator.get_support_center(obj)[:2], (platform_min + platform_max) / 2)
        self.assertGreater(np.linalg.norm(obj.bounds[:, :2].mean(axis=0)
                                         - (platform_min + platform_max) / 2), 0.03)
        clearance = (platform_max - platform_min) * 0.02
        self.assertTrue(np.all(obj.bounds[0, :2] > platform_min + clearance))
        self.assertTrue(np.all(obj.bounds[1, :2] < platform_max - clearance))
        source = trimesh.load(scene_creator.DEFAULT_OBJ_PATH, force="mesh", process=False)
        platform_ratio = np.min(platform_max - platform_min) / scene_creator.REFERENCE_PLATFORM_WIDTH
        self.assertAlmostEqual(obj.extents[0] / np.min(platform_max - platform_min), 0.95, places=6)
        np.testing.assert_allclose(obj.extents, source.extents[[0, 2, 1]]
                                   * scene_creator.DEFAULT_OBJ_SCALE * platform_ratio)
        self.assertLess(obj.bounds[0, 2], platform_z)
        self.assertGreater(obj.bounds[0, 2], platform_z - 0.02)
        # 裁平的封口完全位于平台内部，且所有底面三角形都在同一个平面。
        bottom_z = obj.bounds[0, 2]
        bottom_faces = np.all(np.isclose(obj.triangles[:, :, 2], bottom_z,
                                        atol=1e-10, rtol=0), axis=1)
        self.assertGreater(np.count_nonzero(bottom_faces), 0)
        np.testing.assert_allclose(np.average(obj.triangles_center[bottom_faces, :2], axis=0,
                                             weights=obj.area_faces[bottom_faces]),
                                   (platform_min + platform_max) / 2)
        np.testing.assert_allclose(obj.face_normals[bottom_faces],
                                   np.tile([0, 0, -1], (bottom_faces.sum(), 1)), atol=1e-10)
        solid_terrain = scene_creator.union_stair_boxes(terrain)
        self.assertEqual(len(solid_terrain.split(only_watertight=False)), 1)
        np.testing.assert_array_equal(terrain.vertices, original_vertices)
        np.testing.assert_array_equal(terrain.faces, original_faces)
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
