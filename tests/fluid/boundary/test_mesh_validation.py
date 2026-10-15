"""容器读取、精确拓扑与 GPU 转换回归；不将其视为自交认证。"""

from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import trimesh

from demos.fluid.boundary import mesh_validation
from demos.fluid.boundary.mesh_validation import load_validate_container_obj


def container():
    outer = trimesh.creation.box(extents=(4, 4, 4))
    inner = trimesh.creation.box(extents=(2, 2, 2))
    inner.invert()
    return trimesh.util.concatenate([outer, inner])


class MeshValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "container.obj"

    def validate(self, mesh):
        mesh.export(self.path)
        return load_validate_container_obj(self.path)

    def test_nested_shells_preserve_cavity_and_gpu_types(self):
        mesh = container()
        result = self.validate(mesh)
        self.assertEqual(result.vertices.dtype, np.float32)
        self.assertEqual(result.faces.dtype, np.int32)
        np.testing.assert_allclose(sorted(result.signed_volumes), [-8, 64])
        np.testing.assert_array_equal(result.bounds, [[-2, -2, -2], [2, 2, 2]])

    def test_exact_split_vertices_are_merged_without_repair(self):
        mesh = container()
        split = trimesh.Trimesh(vertices=mesh.triangles.reshape(-1, 3),
                                faces=np.arange(len(mesh.faces) * 3).reshape(-1, 3), process=False)
        result = self.validate(split)
        self.assertEqual(len(result.vertices), 16)

    def test_near_vertices_are_not_welded(self):
        mesh = container()
        vertices = mesh.triangles.reshape(-1, 3).copy()
        vertices[0, 0] += 1e-6
        split = trimesh.Trimesh(vertices=vertices,
                                faces=np.arange(len(mesh.faces) * 3).reshape(-1, 3), process=False)
        with self.assertRaisesRegex(ValueError, "不闭合|非流形"):
            self.validate(split)

    def test_missing_face_is_rejected(self):
        mesh = container()
        mesh.update_faces(np.arange(len(mesh.faces)) != 0)
        with self.assertRaisesRegex(ValueError, "不闭合"):
            self.validate(mesh)

    def test_reversed_face_is_rejected(self):
        mesh = container()
        mesh.faces[0] = mesh.faces[0, ::-1]
        with self.assertRaisesRegex(ValueError, "绕序"):
            self.validate(mesh)

    def test_duplicate_and_reversed_duplicate_are_rejected(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                mesh = container()
                face = mesh.faces[0, ::-1] if reverse else mesh.faces[0]
                mesh.faces = np.vstack([mesh.faces, face])
                with self.assertRaisesRegex(ValueError, "重复三角面"):
                    self.validate(mesh)

    def test_repeated_index_and_collinear_faces_are_rejected(self):
        mesh = container()
        mesh.faces[0, 1] = mesh.faces[0, 0]
        with self.assertRaisesRegex(ValueError, "退化面"):
            self.validate(mesh)
        mesh = container()
        a, b, c = mesh.faces[0]
        mesh.vertices[c] = (mesh.vertices[a] + mesh.vertices[b]) / 2
        with self.assertRaisesRegex(ValueError, "退化面"):
            self.validate(mesh)

    def test_nonfinite_and_invalid_indices_are_rejected(self):
        self.path.touch()
        for bad in ("nan", "index"):
            with self.subTest(bad=bad):
                mesh = container()
                if bad == "nan":
                    mesh.vertices[0, 0] = np.nan
                else:
                    mesh.faces[0, 0] = len(mesh.vertices) + 10
                with mock.patch("trimesh.load", return_value=mesh):
                    with self.assertRaisesRegex(ValueError, "非有限|索引越界"):
                        load_validate_container_obj(self.path)

    def test_empty_input_is_rejected(self):
        self.path.write_text("# empty OBJ\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_validate_container_obj(self.path)

    def test_solid_box_and_wrong_shell_orientation_are_rejected(self):
        solid = trimesh.creation.box(extents=(4, 4, 4))
        with self.assertRaisesRegex(ValueError, "空腔"):
            self.validate(solid)
        wrong = container()
        wrong.faces[12:] = wrong.faces[12:, ::-1]
        with self.assertRaisesRegex(ValueError, "空腔"):
            self.validate(wrong)
        wrong = container()
        wrong.invert()
        with self.assertRaisesRegex(ValueError, "正体积"):
            self.validate(wrong)

    def test_multiple_cavities_are_rejected(self):
        mesh = container()
        extra = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
        extra.invert()
        extra.apply_translation((1.5, 0, 0))
        with self.assertRaisesRegex(ValueError, "空腔"):
            self.validate(trimesh.util.concatenate([mesh, extra]))

    def test_pinched_vertex_is_rejected_even_with_two_faces_per_edge(self):
        first = trimesh.creation.box(extents=(2, 2, 2))
        second = first.copy()
        second.apply_translation((2, 2, 2))
        with self.assertRaisesRegex(ValueError, "非流形顶点"):
            self.validate(trimesh.util.concatenate([first, second]))

    def test_float32_collapse_is_rejected(self):
        mesh = container()
        mesh.apply_translation((1e8, 1e8, 1e8))
        with self.assertRaisesRegex(ValueError, "float32.*退化"):
            self.validate(mesh)

    def test_skinny_nondegenerate_triangles_only_warn(self):
        mesh = container()
        mesh.apply_scale((1, 1, 1e-5))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.validate(mesh)
        self.assertTrue(result.diagnostics)
        self.assertTrue(any("瘦长" in str(item.message) for item in caught))

    def test_current_container_passes(self):
        path = Path(mesh_validation.__file__).parent / "Assets/output/stairs_container_bool_simplified.obj"
        if not path.is_file():
            self.skipTest("当前生成容器未导出")
        with warnings.catch_warnings(record=True):
            result = load_validate_container_obj(path)
        self.assertGreater(len(result.faces), 100)
        self.assertEqual(len(result.signed_volumes), 2)
        self.assertTrue(result.diagnostics)


if __name__ == "__main__":
    unittest.main()
