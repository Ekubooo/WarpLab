"""GPU boundary synchronization with reordered bodies, asset offsets and COMs."""

from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp

from PBF.boundary_model import BoundaryInstance, BoundarySystem


class BoundaryTransformTests(unittest.TestCase):
    def test_body_mapping_rotation_local_offset_and_com(self):
        with wp.ScopedDevice("cuda:0"):
            tiles = wp.array([[0, 0, 0]], dtype=wp.vec3i)
            volume = wp.Volume.allocate_by_tiles(tiles, voxel_size=.01, bg_value=1.)
            asset = SimpleNamespace(sdf=volume, volume_map=volume)
            boundaries = BoundarySystem([asset], [BoundaryInstance(0), BoundaryInstance(0, 0),
                                                  BoundaryInstance(0, 1)], wp.get_device())
            model = SimpleNamespace(device=wp.get_device(), body_count=2,
                body_com=wp.array([[.1, .2, .3], [-.1, .2, 0.]], dtype=wp.vec3))
            q = np.sqrt(.5)
            state = SimpleNamespace(body_q=wp.array([[1., 2., 3., 0., 0., q, q],
                                                     [-1., 4., 0., 0., 0., 0., 1.]], dtype=wp.transform))
            boundaries.bind_bodies(model, [1, 0], [[.2, 0., 0., 0., 0., 0., 1.],
                                                  [0., .3, 0., 0., 0., 0., 1.]])
            data = boundaries.device_data
            pointers = data.transforms.ptr, data.world_com.ptr, data.body_ids.ptr
            boundaries.sync_body_poses(state)
            np.testing.assert_allclose(data.transforms.numpy()[:, :3],
                                       [[0., 0., 0.], [-.8, 4., 0.], [.7, 2., 3.]], atol=2e-7)
            np.testing.assert_allclose(data.world_com.numpy(), [[-1.1, 4.2, 0.], [.8, 2.1, 3.3]], atol=3e-7)
            np.testing.assert_array_equal(data.asset_ids.numpy(), [0, 0, 0])
            np.testing.assert_array_equal(data.body_ids.numpy(), [1, 0])
            # Repeated substeps overwrite the same buffers and retain the static pose.
            boundaries.sync_body_poses(state)
            self.assertEqual((data.transforms.ptr, data.world_com.ptr, data.body_ids.ptr), pointers)
            np.testing.assert_array_equal(data.transforms.numpy()[0], [0., 0., 0., 0., 0., 0., 1.])


if __name__ == "__main__":
    unittest.main()
