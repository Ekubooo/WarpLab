"""Published geometry and native-shape mass checks, outside the runtime path."""

import hashlib
import json
import math
from pathlib import Path
import unittest

import newton
import numpy as np
import trimesh
import warp as wp

from PBF import scene


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class SceneAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = scene.load_rigid_assets()

    def test_published_meshes_and_collision(self):
        for name in ("stairs_scene", "Rigid/stairs_box", "Rigid/stairs_sphere"):
            stem = self.assets.root / name
            metadata = json.loads(stem.with_suffix(".json").read_text())
            self.assertEqual(digest(stem.with_suffix(".obj")), metadata["mesh_sha256"])
            mesh = trimesh.load(stem.with_suffix(".obj"), force="mesh", process=False)
            self.assertTrue(np.isfinite(mesh.vertices).all())
            if name.startswith("Rigid/"):
                np.testing.assert_allclose(mesh.bounds, [[-.15] * 3, [.15] * 3], atol=1e-7)
        metadata = json.loads((self.assets.root / "stairs_scene.json").read_text())
        self.assertEqual(digest(self.assets.root / "stairs_scene.collision.npz"), metadata["collision_sha256"])
        self.assertEqual(metadata["build"]["geometry"]["step_width"], .15)
        self.assertTrue(np.isfinite(self.assets.static_boxes).all())
        self.assertTrue(np.all(self.assets.static_boxes[:, 1] > 0))
        for vertices, faces in self.assets.bunny_hulls:
            self.assertTrue(np.isfinite(vertices).all())
            self.assertGreaterEqual(faces.min(), 0)
            self.assertLess(faces.max(), len(vertices))
            hull = trimesh.Trimesh(vertices, faces, process=False)
            self.assertTrue(hull.is_convex)
            self.assertTrue(hull.is_volume)
            self.assertGreater(hull.volume, 0.)

    def test_native_shape_mass_inertia_and_mapping(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        body_ids = scene.add_rigid_scene(builder, self.assets)
        model = builder.finalize(device="cpu")
        self.assertEqual(body_ids, [0, 1, 2])
        self.assertEqual(model.body_count, 3)
        expected_mass = np.array([500 * .3**3, 1200 * .3**3, 750 * 4 / 3 * math.pi * .15**3])
        expected_inertia = expected_mass * np.array([.3**2 / 6, .3**2 / 6, 2 * .15**2 / 5])
        np.testing.assert_allclose(model.body_mass.numpy(), expected_mass, rtol=1e-6)
        np.testing.assert_allclose(model.body_com.numpy(), 0., atol=1e-7)
        np.testing.assert_allclose(model.body_inertia.numpy(), expected_inertia[:, None, None] * np.eye(3), rtol=1e-6)
        np.testing.assert_allclose(model.body_q.numpy()[:, :3], [item.center for item in scene.RIGIDS], atol=1e-7)
        self.assertEqual(np.count_nonzero(model.shape_body.numpy() == -1), 24 + len(self.assets.bunny_hulls))
        for body in body_ids:
            self.assertEqual(np.count_nonzero(model.shape_body.numpy() == body), 1)

    def test_sdf_identity_independent_of_volume_map(self):
        # NVDB's second grid is mutable; normalize only the first grid's
        # checksum, flags/index/count before comparing its published identity.
        with wp.ScopedDevice("cuda:0"):
            for name in ("stairs_scene", "Rigid/stairs_box", "Rigid/stairs_sphere"):
                stem = self.assets.root / name
                metadata = json.loads(stem.with_suffix(".json").read_text())
                with stem.with_suffix(".nvdb").open("rb") as stream:
                    sdf = wp.Volume.load_from_nvdb(stream)
                size = sdf.get_grid_info().size_in_bytes
                raw = wp.array(ptr=sdf.array().ptr, dtype=wp.uint8, shape=size, device=sdf.device).numpy()
                checksum = hashlib.sha256()
                for chunk in (raw[:8], bytes(8), raw[16:20], bytes(12), raw[32:]):
                    checksum.update(chunk)
                self.assertEqual(checksum.hexdigest(), metadata["sdf_sha256"], name)


if __name__ == "__main__":
    unittest.main()
