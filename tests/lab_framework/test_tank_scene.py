"""Shared primitive geometry, physical mass and cached coupling assets."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import newton
import warp as wp

from Lab_Framework.pbf import config_io, scenes


class TankSceneTests(unittest.TestCase):
    def test_preset_roundtrip_and_initial_clearance(self):
        config = scenes.tank_box_config()
        preset = Path(scenes.__file__).parent / "config" / "tank_box.json"
        self.assertEqual(config_io.load_config(preset), config)
        self.assertGreater(scenes.BOX_POSITION[1] - scenes.BOX_HALF_EXTENTS[1], config.block_end[1])

    def test_collision_shapes_match_mesh_bounds_and_mass_once(self):
        config = scenes.tank_box_config()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        body_id = scenes.add_tank_box(builder, config)
        self.assertEqual(len(builder.body_mass), 1)
        self.assertEqual(len(builder.shape_body), 6)
        self.assertEqual(builder.shape_body, [-1] * 5 + [body_id])
        for i, (center, half) in enumerate(scenes.tank_walls(config)):
            np.testing.assert_allclose(builder.shape_transform[i][:3], center)
            np.testing.assert_allclose(builder.shape_scale[i], half)
            mesh = scenes._box_mesh(center, half)
            np.testing.assert_allclose(mesh.bounds, [np.array(center)-half, np.array(center)+half])
        mass = .5 * config.rest_density * np.prod(2. * np.asarray(scenes.BOX_HALF_EXTENTS))
        self.assertAlmostEqual(builder.body_mass[body_id], mass, places=5)
        np.testing.assert_allclose(builder.body_com[body_id], 0., atol=1e-7)
        inertia = mass * (.24**2 + .24**2) / 12.
        np.testing.assert_allclose(np.asarray(builder.body_inertia[body_id]).reshape(3, 3),
                                   np.eye(3)*inertia, rtol=2e-6)

    def test_wall_union_has_open_top_and_expected_volume(self):
        from Lab_Framework.sim_ready import geometry, validation
        config = scenes.tank_box_config()
        mesh = geometry.union([scenes._box_mesh(center, half)
                               for center, half in scenes.tank_walls(config)])
        checked = validation.validate_solid(mesh)
        np.testing.assert_allclose(checked.bounds, [[-.1, -.1, -.1], [1.1, 1., 1.1]], atol=1e-7)
        self.assertAlmostEqual(mesh.volume, 1.2*1.1*1.2 - 1., places=6)
        # No horizontal face seals the opening at y=1.
        top = np.isclose(mesh.triangles[:, :, 1], 1.).all(axis=1)
        centers = mesh.triangles_center[top]
        inside = (centers[:, 0] > 0.) & (centers[:, 0] < 1.) & (centers[:, 2] > 0.) & (centers[:, 2] < 1.)
        self.assertFalse(inside.any())

    def test_cache_identity_depends_on_geometry_and_support_not_density(self):
        config = scenes.tank_box_config()
        original = scenes.asset_cache_path(config)
        self.assertEqual(original, scenes.asset_cache_path(replace(config, rest_density=900.)))
        self.assertNotEqual(original, scenes.asset_cache_path(replace(config, particle_radius=.015)))
        self.assertNotEqual(original, scenes.asset_cache_path(replace(config, container_size=(1.2, 1., 1.))))
        self.assertNotEqual(original, scenes.asset_cache_path(config, voxel_size=.01))
        with self.assertRaisesRegex(ValueError, "fit"):
            scenes.tank_walls(replace(config, container_size=(.6, 1., 1.), block_end=(.5, .55, .95)))


class TankBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA NanoVDB is required")

    def test_bake_load_then_cache_reuse(self):
        from Lab_Framework.sim_ready import sdf
        config = scenes.tank_box_config()
        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cuda:0"):
            boundaries = scenes.load_tank_box_boundaries(config, "cuda:0", cache_dir=directory, voxel_size=.04)
            self.assertEqual((boundaries.static_count, boundaries.rigid_count), (1, 1))
            self.assertEqual([(i.asset_id, i.rigid_id) for i in boundaries.instances], [(0, -1), (1, 0)])
            for asset in boundaries.assets:
                self.assertTrue(asset.mesh_path.is_file())
                self.assertTrue(asset.sdf_path.is_file())
                self.assertGreater(asset.sdf.get_tile_count(), 0)
            with patch.object(sdf, "bake_rigid", side_effect=AssertionError("cached SDF was rebaked")):
                again = scenes.load_tank_box_boundaries(config, "cuda:0", cache_dir=directory, voxel_size=.04)
            self.assertEqual([a.sdf_path for a in boundaries.assets], [a.sdf_path for a in again.assets])


if __name__ == "__main__":
    unittest.main()
