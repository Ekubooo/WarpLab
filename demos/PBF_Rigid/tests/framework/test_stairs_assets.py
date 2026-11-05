"""Explicit multi-body assembly, original stair geometry and asset reuse."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import newton
import numpy as np
import trimesh
import warp as wp

from demos.PBF_Rigid.config import PBFConfig
from demos.PBF_Rigid import stairs_scene
from demos.PBF_Rigid.sim_ready import config, geometry, rigid_assets
from demos.PBF_Rigid.sim_ready.pbf_assets import build_scene_assets


class StairGeometryTests(unittest.TestCase):
    def test_aligned_boxes_preserve_original_landscape(self):
        scene = geometry.prepare_scene(config.ASSETS / "bunny_flat_bottom.obj")
        boxes = geometry.terrain_boxes(config.GeometryOptions())
        self.assertEqual(len(boxes), 18)
        rebuilt = geometry.union(boxes)
        np.testing.assert_array_equal(rebuilt.vertices, scene.terrain.vertices)
        np.testing.assert_array_equal(rebuilt.faces, scene.terrain.faces)
        visible = geometry.landscape(scene, config.GeometryOptions())
        stored = trimesh.load(stairs_scene.ROOT / "Assets" / "stairs_scene.obj", force="mesh", process=False)
        np.testing.assert_array_equal(np.round(visible.vertices, 8), stored.vertices)
        np.testing.assert_array_equal(visible.faces, stored.faces)

    def test_hull_cache_records_options_and_rejects_failed_decomposition(self):
        mesh = trimesh.creation.box(extents=(.3,) * 3)
        with tempfile.TemporaryDirectory() as directory:
            returned = [(mesh.vertices.copy(), mesh.faces.copy())]
            with patch("coacd.run_coacd", return_value=returned) as decompose:
                first = rigid_assets.prepare_hulls(mesh, "test_box", directory)
                self.assertEqual(decompose.call_args.kwargs, rigid_assets.COACD_OPTIONS)
            with patch("coacd.run_coacd", side_effect=AssertionError("cache miss")):
                second = rigid_assets.prepare_hulls(mesh, "test_box", directory)
            np.testing.assert_array_equal(first[0].vertices, second[0].vertices)
            changed = mesh.copy()
            changed.apply_scale(1.1)
            with patch("coacd.run_coacd", return_value=[]):
                with self.assertRaisesRegex(ValueError, "convex solids"):
                    rigid_assets.prepare_hulls(changed, "test_box", directory)
            self.assertEqual(len(list(Path(directory).rglob("hulls.npz"))), 1)


class StairsAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA volume assets are required")
        cls.config = PBFConfig()
        with wp.ScopedDevice("cuda:0"):
            build_scene_assets("stairs-rigids", cls.config)
            cls.assets = stairs_scene.load_stairs_assets(cls.config)

    def build(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        ids = stairs_scene.add_stairs_rigids(builder, self.assets)
        return builder, ids

    def test_native_stairs_and_walls_preserve_open_cavity(self):
        builder, ids = self.build()
        self.assertEqual(ids, [0, 1, 2, 3])
        self.assertEqual(len(self.assets.static_boxes), 24)
        for i, (center, half) in enumerate(self.assets.static_boxes):
            self.assertEqual(builder.shape_body[i], -1)
            np.testing.assert_allclose(builder.shape_transform[i][:3], center, atol=1e-7)
            np.testing.assert_allclose(builder.shape_scale[i], half, atol=1e-7)
        # This free point must not be swallowed by a convex hull of the tank.
        point = np.array([2.1, 2., .8])
        self.assertFalse(any(np.all(np.abs(point - center) < half)
                             for center, half in self.assets.static_boxes))

    def test_mass_com_and_full_inertia_counted_once(self):
        builder, ids = self.build()
        np.testing.assert_allclose(builder.body_mass[:2], [.3**3 * 500., .3**3 * 1200.], rtol=2e-6)
        self.assertAlmostEqual(builder.body_mass[2], 4 / 3 * np.pi * .15**3 * 750., places=4)
        mesh = trimesh.load(self.assets.boundaries["spot"].mesh_path, force="mesh", process=False)
        self.assertAlmostEqual(mesh.extents.max(), .35, places=8)
        self.assertAlmostEqual(builder.body_mass[3], mesh.volume * 500., places=5)
        np.testing.assert_allclose(builder.body_com[3], mesh.center_mass, atol=1e-7)
        inertia = np.asarray(builder.body_inertia[3]).reshape(3, 3)
        np.testing.assert_allclose(inertia, mesh.moment_inertia * 500., rtol=2e-6, atol=1e-8)
        self.assertTrue(np.all(np.linalg.eigvalsh(inertia) > 0.))
        self.assertGreater(abs(inertia[1, 2]), .001)
        for body, item in zip(ids, stairs_scene.RIGIDS):
            center = wp.transform_point(builder.body_q[body], wp.vec3(*builder.body_com[body]))
            np.testing.assert_allclose(center, item.center, atol=2e-7)

    def test_published_proxies_and_runtime_never_prepare_geometry(self):
        for hulls in (self.assets.bunny_hulls, self.assets.spot_hulls):
            self.assertGreater(len(hulls), 1)
            rigid_assets._validate_hulls([trimesh.Trimesh(h.vertices, h.faces, process=False) for h in hulls])
        with patch("coacd.run_coacd", side_effect=AssertionError("cached decomposition reran")), \
                patch("demos.PBF_Rigid.sim_ready.sdf.bake_rigid", side_effect=AssertionError("SDF rebaked")), \
                patch.object(geometry, "prepare_scene", side_effect=AssertionError("terrain regenerated")), \
                patch.object(geometry, "prepare_rigid", side_effect=AssertionError("source geometry loaded")), \
                patch.object(geometry, "union", side_effect=AssertionError("geometry union ran")), \
                patch.object(trimesh.Trimesh, "mass_properties",
                             new=property(lambda mesh: self.fail("runtime mass integration"))):
            cached = stairs_scene.load_stairs_assets(self.config)
            builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
            stairs_scene.add_stairs_rigids(builder, cached)
        self.assertEqual([v.sdf_path for v in cached.boundaries.values()],
                         [v.sdf_path for v in self.assets.boundaries.values()])

    def test_explicit_instances_share_box_volume_but_keep_independent_poses(self):
        with wp.ScopedDevice("cuda:0"):
            boundaries = stairs_scene.load_stairs_boundaries(self.assets, self.config, "cuda:0")
            self.assertEqual(len(boundaries.assets), 4)
            self.assertEqual([(i.asset_id, i.rigid_id) for i in boundaries.instances],
                             [(0, -1), (1, 0), (1, 1), (2, 2), (3, 3)])
            builder, ids = self.build()
            model = builder.finalize(device="cuda:0")
            state = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            boundaries.bind_bodies(model, ids)
            boundaries.sync_body_poses(state)
            transforms = boundaries.device_data.transforms.numpy()
            np.testing.assert_allclose(transforms[1:], state.body_q.numpy(), atol=1e-7)
            self.assertFalse(np.array_equal(transforms[1], transforms[2]))
            np.testing.assert_allclose(boundaries.device_data.world_com.numpy(),
                [i.center for i in stairs_scene.RIGIDS], atol=2e-7)


if __name__ == "__main__":
    unittest.main()
