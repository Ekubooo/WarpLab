"""Conservative finite-asset queries compared with the original unfiltered SDF."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
import warp as wp

from PBF import scene, boundary_model as boundary
from .query_variants import OUT, boundary_variant, load_module


QUERY_KERNEL = '''
import warp as wp
from PBF.boundary_model import BoundaryDeviceData
from tests.pbf import rigid_query_reference as reference
import {module} as candidate
wp.set_module_options({{"enable_backward": False}})

@wp.kernel
def compare(data: BoundaryDeviceData, instance: int, points: wp.array(dtype=wp.vec3),
            h: float, radius: float, flags: wp.array2d(dtype=int),
            values: wp.array2d(dtype=wp.vec3)):
    i = wp.tid()
    p = wp.transform_point(data.transforms[instance], points[i])
    old, distance, direction = reference.query_instance_boundary(data, instance, p, h)
    new, new_distance, new_direction = candidate.query_instance_boundary(data, instance, p, h)
    flags[i, 0] = int(old)
    flags[i, 1] = int(new)
    values[i, 0] = direction * distance
    values[i, 1] = new_direction * new_distance
    a, d, n = reference.query_instance_normal(data, instance, p)
    b, e, m = candidate.query_instance_normal(data, instance, p, radius)
    before = p
    after = p
    if a and d < radius:
        before += n * (radius - d)
    if b and e < radius:
        after += m * (radius - e)
    values[i, 2] = before
    values[i, 3] = after
    local = wp.transform_point(wp.transform_inverse(data.transforms[instance]), p)
    flags[i, 2] = int(candidate.inside_boundary_bounds(data, data.asset_ids[instance], local, h))
'''


class BoundaryCullingTests(unittest.TestCase):
    def test_published_assets_pressure_and_projection(self):
        with wp.ScopedDevice("cuda:0"):
            assets = scene.load_rigid_assets()
            with patch("trimesh.load", wraps=trimesh.load) as loader:
                boundaries = scene.load_rigid_boundaries(assets, .05, wp.get_device())
                self.assertEqual(loader.call_count, 2)  # two boxes share one mesh
                meshes = scene.load_render_meshes(boundaries)
                self.assertEqual(loader.call_count, 3)  # only static mesh remains
                self.assertIs(meshes[1], meshes[2])
            rng = np.random.default_rng(56)
            points = rng.uniform(-.35, .35, (100000, 3)).astype(np.float32)
            # Sample faces, corners, interiors, and either side of both reaches.
            critical = np.array([0., .15, .15 + .0125, .20], dtype=np.float32)
            critical = np.concatenate([critical, np.nextafter(critical, -np.inf),
                                       np.nextafter(critical, np.inf)])
            points = np.concatenate([points, np.array([[x, y, z] for x in critical
                for y in (-.15, 0., .15) for z in (-.15, 0., .15)], dtype=np.float32)])
            gpu_points = wp.array(points, dtype=wp.vec3)
            flags = wp.empty((len(points), 3), dtype=int)
            values = wp.empty((len(points), 4), dtype=wp.vec3)
            rotations = (wp.transform_identity(),
                wp.transform((1.3, -.2, .7), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1., 2., 3.)), .9)) *
                wp.transform((.13, -.07, .05), wp.quat_from_axis_angle(wp.vec3(0., 1., 0.), .4)))
            for mode in ("sphere", "box"):
                module = boundary_variant(mode)
                case = load_module("pbf_check_" + mode, QUERY_KERNEL.format(module=module.__name__))
                for transform in rotations:
                    boundaries.set_rigid_poses([tuple(transform)] * 3)
                    for instance in (1, 2, 3):
                        wp.launch(case.compare, len(points), inputs=[boundaries.device_data, instance,
                            gpu_points, .05, .0125, flags, values])
                        f, v = flags.numpy(), values.numpy()
                        np.testing.assert_array_equal(f[:, 0], f[:, 1], err_msg=mode)
                        np.testing.assert_array_equal(v[:, 0], v[:, 1], err_msg=mode)
                        np.testing.assert_array_equal(v[:, 2], v[:, 3], err_msg=mode)
                        self.assertGreater(np.count_nonzero(f[:, 0]), 100)
                        self.assertGreater(np.count_nonzero(f[:, 2] == 0), len(points) // 2)

    def test_offset_elongated_mesh_bounds(self):
        OUT.mkdir(parents=True, exist_ok=True)
        with wp.ScopedDevice("cuda:0"), tempfile.TemporaryDirectory(dir=OUT) as temp:
            mesh = trimesh.creation.box(extents=(.8, .12, .2))
            mesh.apply_translation((.2, -.3, .1))
            path = Path(temp) / "asset.obj"
            mesh.export(path)
            tiles = wp.array([[0, 0, 0]], dtype=wp.vec3i)
            volume = wp.Volume.allocate_by_tiles(tiles, .005, bg_value=1.)
            asset = boundary.BoundaryAsset(Path("synthetic.nvdb"), path, volume, volume)
            system = boundary.BoundarySystem([asset], [boundary.BoundaryInstance(0, 0)], "cuda:0")
            data = system.device_data
            np.testing.assert_allclose(data.bounds_center.numpy()[0], [.2, -.3, .1], atol=2e-8)
            np.testing.assert_allclose(data.bounds_half.numpy()[0], [.4, .06, .1], atol=2e-8)
            self.assertAlmostEqual(float(data.bounds_radius.numpy()[0]), np.linalg.norm([.4, .06, .1]), places=7)
            self.assertAlmostEqual(float(data.bounds_padding.numpy()[0]), np.sqrt(3) * .005, places=8)


if __name__ == "__main__":
    unittest.main()
