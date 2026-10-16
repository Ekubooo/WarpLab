"""双分支场景的几何、空腔和输出回归；使用 Conda warplab 运行。"""

from pathlib import Path
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from demos.fluid.boundary import scene_container_creator as creator
from demos.fluid.boundary import scene_creator as original


def make_floor(length=3.1, width=1.6, depth=0.2):
    mesh = trimesh.creation.box(extents=(length, depth, width))
    mesh.apply_translation((length / 2, -depth / 2, width / 2))
    return mesh


def sorted_triangles(mesh):
    """按几何比较，忽略顶点和三角形编号及面内循环顺序。"""
    triangles = []
    for triangle in mesh.triangles:
        triangle = triangle[np.lexsort(triangle.T[::-1])]
        triangles.append(triangle.reshape(-1))
    triangles = np.asarray(triangles)
    return triangles[np.lexsort(triangles.T[::-1])]


class ContainerTests(unittest.TestCase):
    def test_terrain_is_rigidly_converted_without_changing_generator(self):
        old = original.generate_stairs()
        terrain = creator.generate_stairs()
        width = old.bounds[1, 1]
        expected = np.column_stack((old.vertices[:, 0], old.vertices[:, 2], width - old.vertices[:, 1]))
        np.testing.assert_allclose(terrain.vertices, expected, atol=1e-12)
        np.testing.assert_array_equal(terrain.faces, old.faces)
        np.testing.assert_allclose(creator.get_platform_bounds(terrain).mean(axis=0), [1.55, 0.5, 0.8], atol=1e-7)

    @unittest.skipUnless(creator.DEFAULT_OBJ_PATH.is_file(), "Default OBJ asset not installed")
    def test_model_scale_support_and_yaw_match_previous_geometry(self):
        terrain = creator.generate_stairs()
        old_terrain = original.generate_stairs()
        width = old_terrain.bounds[1, 1]
        for yaw in (0, 37, 90):
            for linked in (True, False):
                old = original.load_and_place_obj(original.DEFAULT_OBJ_PATH, old_terrain,
                                                   yaw_deg=yaw, scale_with_platform=linked)
                new = creator.load_and_place_obj(creator.DEFAULT_OBJ_PATH, terrain,
                                                  yaw_deg=yaw, scale_with_platform=linked)
                expected = np.column_stack((old.vertices[:, 0], old.vertices[:, 2], width - old.vertices[:, 1]))
                np.testing.assert_allclose(new.vertices, expected, atol=1e-8)
                np.testing.assert_allclose(creator.get_support_center(new), [1.55, 0.497, 0.8], atol=1e-7)
        with tempfile.TemporaryDirectory() as directory:
            source = trimesh.load(creator.DEFAULT_OBJ_PATH, force="mesh", process=False)
            source.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            path = Path(directory) / "z-up.obj"
            source.export(path)
            z_input = creator.load_and_place_obj(path, terrain, obj_up_axis="Z", position=(7, 8, 9))
            y_input = creator.load_and_place_obj(creator.DEFAULT_OBJ_PATH, terrain, position=(7, 8, 9))
            np.testing.assert_allclose(z_input.vertices, y_input.vertices, atol=1e-7)
            np.testing.assert_allclose(creator.get_support_center(z_input), [7, 8, 9], atol=1e-8)

    def test_support_section_centroid_and_lowest_contact_in_y_up(self):
        # A nonplanar bottom exercises the signed (Z, X) section integration.
        mesh = trimesh.convex.convex_hull([[0, 0, 0], [1, 2, 0], [3, 2, 0], [1, 2, -2]])
        np.testing.assert_allclose(creator.get_support_center(mesh, contact_depth=0.5),
                                   [5 / 12, 0, -1 / 6], atol=1e-10)
        np.testing.assert_allclose(creator.get_support_center(mesh), [0, 0, 0], atol=1e-10)
        mesh.apply_translation([3, 5, -2])
        np.testing.assert_allclose(creator.get_support_center(mesh, contact_depth=0.5),
                                   [3 + 5 / 12, 5, -2 - 1 / 6], atol=1e-10)

    def assert_cuboid_exterior(self, mesh, bounds):
        """检查真实外表面，不能只用整个模型的 AABB 证明外观为长方体。"""
        exteriors = [surface for surface in mesh.split(only_watertight=False, repair=False)
                     if surface.volume > 0]
        self.assertEqual(len(exteriors), 1)
        exterior = exteriors[0]
        extents = bounds[1] - bounds[0]
        np.testing.assert_allclose(exterior.bounds, bounds, rtol=0, atol=1e-6)
        np.testing.assert_allclose(exterior.volume, np.prod(extents), rtol=0, atol=3e-6)
        expected_area = 2 * (extents[0] * extents[1] + extents[0] * extents[2] + extents[1] * extents[2])
        np.testing.assert_allclose(exterior.area, expected_area, rtol=0, atol=3e-6)
        on_faces = np.any(np.all(np.isclose(
            exterior.triangles[:, :, :, None], bounds.T[None, None, :, :], rtol=0, atol=1e-6),
            axis=1), axis=(1, 2))
        self.assertTrue(on_faces.all(), "外表面必须全部处于六个外平面上")

    def test_walls_overlap_floor_each_other_and_lid(self):
        floor = make_floor()
        parts = creator.build_container_parts(floor)
        long0, long1, short0, short1, lid = parts
        np.testing.assert_allclose(long0.extents, [3.45, 3.275, 0.25])
        np.testing.assert_allclose(short0.extents, [0.25, 3.275, 1.675])
        np.testing.assert_allclose(lid.bounds, [[-0.175, 3.0, -0.175], [3.275, 3.25, 1.775]])
        for wall in parts[:4]:
            overlap = trimesh.boolean.intersection([floor, wall], engine="manifold")
            self.assertGreater(overlap.volume, 0)
            overlap = trimesh.boolean.intersection([wall, lid], engine="manifold")
            self.assertGreater(overlap.volume, 0)
            self.assertAlmostEqual(wall.bounds[0, 1], floor.bounds[0, 1])
        for long_wall in (long0, long1):
            for short_wall in (short0, short1):
                self.assertGreater(trimesh.boolean.intersection(
                    [long_wall, short_wall], engine="manifold").volume, 0)
        np.testing.assert_allclose(short0.bounds[1, 0], 0.075)
        np.testing.assert_allclose(short1.bounds[0, 0], 3.1 - 0.075)
        np.testing.assert_allclose(long0.bounds[1, 2], 0.075)
        np.testing.assert_allclose(long1.bounds[0, 2], 1.6 - 0.075)

    def test_configuration_linkage_and_translated_terrain(self):
        floor = make_floor()
        shifted = floor.copy()
        shifted.apply_translation((7, 2, -3))
        for width in (0.1, 0.2):
            terrain = creator.generate_stairs(step_width=width)
            walls = creator.build_container_parts(terrain, wall_thickness=width)
            self.assertAlmostEqual(walls[0].extents[2], width)
            self.assertAlmostEqual(walls[2].extents[0], width)
            self.assertAlmostEqual(walls[2].bounds[1, 0] - terrain.bounds[0, 0], 0.075)
            # 台阶外圈的内边缘随同一个宽度变化。
            self.assertTrue(np.any(np.isclose(terrain.vertices[:, 0], width, atol=1e-7)))
        before = creator.build_container_parts(floor, overlap_depth=0.04)
        after = creator.build_container_parts(shifted, container_height=5.0, overlap_depth=0.04)
        for a, b in zip(before, after):
            np.testing.assert_allclose(b.vertices, a.vertices + (7, 2, -3))
        np.testing.assert_allclose(before[0].extents[[0, 2]], [3.52, 0.25])

    def test_outer_contour_is_cuboid_for_different_overlap_ratios_and_sizes(self):
        for length, width, w, ratio in ((3.1, 1.6, 0.15, 0.1), (3.1, 1.6, 0.15, 0.5),
                                       (3.1, 1.6, 0.15, 0.9), (1.6, 3.1, 0.2, 0.9),
                                       (4, 2, 0.1, 0.25)):
            with self.subTest(length=length, width=width, thickness=w, ratio=ratio):
                floor = make_floor(length, width)
                depth = w * ratio
                parts = creator.build_container_parts(floor, w, overlap_depth=depth)
                expected = np.array([[-(w - depth), -0.2, -(w - depth)],
                                     [length + w - depth, 3.0 + w, width + w - depth]])
                for part in parts:
                    self.assertTrue(np.all(part.bounds[0, [0, 2]] >= expected[0, [0, 2]] - 1e-10))
                    self.assertTrue(np.all(part.bounds[1, [0, 2]] <= expected[1, [0, 2]] + 1e-10))
                # 长墙端面、短墙外侧、顶盖边缘都对齐同一对 X 外平面。
                self.assertAlmostEqual(parts[0].bounds[0, 0], parts[2].bounds[0, 0])
                self.assertAlmostEqual(parts[1].bounds[1, 0], parts[3].bounds[1, 0])
                with contextlib.redirect_stdout(io.StringIO()):
                    container = creator.boolean_union_meshes([floor, *parts], "test")
                    simplified = creator.simplify_mesh_manifold(container)
                    for mesh in (container, simplified):
                        creator.validate_container_mesh(mesh, "test", outer_bounds=expected)
                        self.assert_cuboid_exterior(mesh, expected)

    def test_old_protruding_walls_fail_exterior_validation_even_with_correct_aabb(self):
        floor = make_floor()
        parts = creator.build_container_parts(floor)
        for index in (0, 1, 4):
            bounds = parts[index].bounds
            extents = parts[index].extents.copy()
            extents[0] = floor.extents[0] + 2 * creator.DEFAULT_STEP_WIDTH + creator.DEFAULT_OVERLAP_DEPTH
            parts[index] = trimesh.creation.box(extents=extents)
            parts[index].apply_translation(bounds.mean(axis=0))
        with contextlib.redirect_stdout(io.StringIO()):
            container = creator.boolean_union_meshes([floor, *parts], "old layout")
            with self.assertRaisesRegex(ValueError, "完整长方体"):
                creator.validate_container_mesh(container, "old layout", outer_bounds=container.bounds)

    def test_closed_cavity_volume_and_material_probes_before_and_after_simplify(self):
        floor = make_floor()
        with contextlib.redirect_stdout(io.StringIO()):
            container = creator.boolean_union_meshes([floor, *creator.build_container_parts(floor)], "test")
            simplified = creator.simplify_mesh_manifold(container)
        # 整体为外长方体减内空腔；凸出墙头即使封闭，也无法满足这个独立体积关系。
        expected_solid = 3.45 * 1.95 * 3.45 - 2.95 * 1.45 * 3.0
        probes = [((1.55, 0.75, 0.8), False),  # 空腔
                  ((1.55, -0.1, 0.8), True),  # 地板
                  ((0, -0.1, 0.8), True),     # 墙与底座接缝
                  ((0.02, 2.5, 0.8), True),   # 加高后的侧墙
                  ((0.02, 0.75, 0.02), True), # 墙角
                  ((1.55, 3.125, 0.8), True), # 顶盖
                  ((1.55, 3.3, 0.8), False)] # 容器外
        for mesh in (container, simplified):
            with contextlib.redirect_stdout(io.StringIO()):
                creator.validate_container_mesh(mesh, "test")
            # Trimesh 的 Boolean 桥接使用 float32，体积比较允许其坐标舍入误差。
            np.testing.assert_allclose(mesh.volume, expected_solid, rtol=0, atol=2e-6)
            surfaces = mesh.split(only_watertight=False, repair=False)
            self.assertEqual(len(surfaces), 2)
            self.assert_cuboid_exterior(mesh, np.array([[-0.175, -0.2, -0.175], [3.275, 3.25, 1.775]]))
            cavity = [surface for surface in surfaces if surface.volume < 0][0]
            np.testing.assert_allclose(-cavity.volume, 2.95 * 1.45 * 3.0, rtol=0, atol=2e-6)
            for center, solid in probes:
                probe = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
                probe.apply_translation(center)
                intersection = trimesh.boolean.intersection([mesh, probe], engine="manifold")
                volume = intersection.volume if len(intersection.faces) else 0.0
                self.assertAlmostEqual(volume, probe.volume if solid else 0, places=10)
        with self.assertRaisesRegex(ValueError, "空腔"):
            creator.validate_container_mesh(floor, "filled")

    def test_invalid_geometry_parameters(self):
        for options in ({"wall_thickness": 0}, {"wall_thickness": float("nan")},
                        {"overlap_depth": 0}, {"overlap_depth": 0.25},
                        {"overlap_depth": float("inf")}, {"container_height": 0},
                        {"container_height": float("nan")}, {"wall_thickness": 0.05}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                creator.build_container_parts(make_floor(), **options)

    def test_output_paths_cannot_overwrite_input_or_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, scene, container = (directory / name for name in ("input.obj", "scene.obj", "container.obj"))
            creator.validate_output_paths(source, scene, container)
            for outputs in ((source, container), (scene, scene), (scene.with_suffix(".nvdb"), container)):
                with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                    creator.validate_output_paths(source, *outputs)
            source.write_text("preserved source", encoding="utf-8")
            os.link(source, scene)
            with self.assertRaises(ValueError):
                creator.validate_output_paths(source, scene, container)
            self.assertEqual(source.read_text(encoding="utf-8"), "preserved source")

    @unittest.skipUnless(creator.DEFAULT_OBJ_PATH.is_file(), "Default OBJ asset not installed")
    def test_two_outputs_match_original_landscape_and_preserve_cavity_in_both_axes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            terrain = original.generate_stairs()
            solid_terrain = original.union_stair_boxes(terrain)
            obj = original.prepare_solid(original.load_and_place_obj(original.DEFAULT_OBJ_PATH, terrain))
            scene = original.boolean_union_meshes([solid_terrain, obj], "reference")
            expected = original.simplify_mesh_manifold(scene)
        actual_z = None
        for axis in ("Z", "Y"):
            with tempfile.TemporaryDirectory() as directory:
                scene_path, container_path = (Path(directory) / name for name in ("landscape.obj", "container.obj"))
                with contextlib.redirect_stdout(io.StringIO()):
                    outputs = creator.create_scene(output_path=scene_path,
                                                   container_output_path=container_path, up_axis=axis)
                self.assertEqual(outputs, (scene_path, container_path))
                self.assertEqual({path.name for path in Path(directory).iterdir()}, {"landscape.obj", "container.obj"})
                landscape = trimesh.load(scene_path, force="mesh", process=False)
                container = trimesh.load(container_path, force="mesh", process=False)
                if axis == "Y":
                    inverse = np.array([[1, 0, 0, 0], [0, 0, -1, terrain.bounds[1, 1]],
                                        [0, 1, 0, 0], [0, 0, 0, 1]])
                    landscape.apply_transform(inverse)
                    container.apply_transform(inverse)
                # Boolean triangulation can change after a rotation; compare surfaces.
                for a, b in ((landscape, expected), (expected, landscape)):
                    for start in range(0, len(a.vertices), 64):
                        _, distances, _ = trimesh.proximity.closest_point_naive(b, a.vertices[start:start + 64])
                        self.assertLess(distances.max(), 0.0011)
                self.assertAlmostEqual(landscape.volume, expected.volume, delta=2e-5)
                with contextlib.redirect_stdout(io.StringIO()):
                    creator.validate_container_mesh(container, "exported")
                bounds = np.array([
                    [*(solid_terrain.bounds[0, :2] - 0.175), solid_terrain.bounds[0, 2] - 0.1],
                    [*(solid_terrain.bounds[1, :2] + 0.175), 3.25],
                ])
                self.assert_cuboid_exterior(container, bounds)
                cavity = [part for part in container.split(only_watertight=False, repair=False)
                          if part.volume < 0][0]
                self.assertAlmostEqual(cavity.bounds[1, 2], 3.0, places=7)
                self.assertAlmostEqual(cavity.bounds[0, 2], 0.0, places=7)
                # 原坑底厚 0.10；墙厚从 0.15 增到 0.25，外底面同步外移 0.10。
                self.assertAlmostEqual(cavity.bounds[0, 2] - container.bounds[0, 2], 0.2, places=7)
                if actual_z is None:
                    actual_z = sorted_triangles(container)
                else:
                    # Both exports represent the same already-built geometry.
                    for a, b in ((sorted_triangles(container), actual_z), (actual_z, sorted_triangles(container))):
                        self.assertLess(cKDTree(b.reshape(-1, 3)).query(a.reshape(-1, 3))[0].max(), 1e-7)

    def test_invalid_requests_do_not_export_and_main_has_no_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            scene_path, container_path = (Path(directory) / name for name in ("scene.obj", "container.obj"))
            for options in ({"simplify_tolerance": 0}, {"container_height": 0.6}):
                with self.subTest(options=options), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    creator.create_scene(output_path=scene_path, container_output_path=container_path, **options)
                self.assertEqual(list(Path(directory).iterdir()), [])
        with mock.patch.object(creator.sys, "argv", ["script.py", "--simplify"]), self.assertRaises(SystemExit):
            creator.main()
        with mock.patch.object(creator.sys, "argv", ["script.py"]), \
                mock.patch.object(creator, "create_scene", return_value=(Path("scene.obj"), Path("container.obj"))) as create, \
                contextlib.redirect_stdout(io.StringIO()):
            creator.main()
        self.assertEqual(create.call_args.kwargs["simplify_tolerance"], original.DEFAULT_SIMPLIFY_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
