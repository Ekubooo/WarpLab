"""Configuration ownership and volume-cache lifecycle after validation cleanup."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from PBF import boundary_model as boundary, config_io
from PBF.pbf_helper import PBFConfig


class ConfigurationTests(unittest.TestCase):
    def test_json_uses_the_same_parameter_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(asdict(PBFConfig())))
            self.assertEqual(config_io.load_config(path), PBFConfig())
            for values in ({"particle_radius": True}, {"substeps": 1.5},
                           {"gravity": [True, 0., 0.]}, {"gravity": ["0", 0, 0]},
                           {"viscosity": "1"}, {"pressure_iterations": 0}):
                with self.subTest(values=values):
                    with self.assertRaises(ValueError):
                        PBFConfig(**values)
                    path.write_text(json.dumps(values))
                    with self.assertRaises(ValueError):
                        config_io.load_config(path)
            path.write_text('{"unknown_parameter": 1}')
            with self.assertRaisesRegex(ValueError, "Unknown config fields"):
                config_io.load_config(path)


class VolumeCacheTests(unittest.TestCase):
    def test_rebuild_hit_support_limit_and_pair_layout(self):
        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cuda:0"):
            path = Path(directory) / "boundary.nvdb"
            tiles = wp.array([[0, 0, 0]], dtype=wp.vec3i)
            sdf = wp.Volume.allocate_by_tiles(tiles, voxel_size=.01, bg_value=1.)
            name = f"PBF_SDF|v=1|hmax={float(.05).hex()}"
            boundary._nvdb_io().save_volumes((sdf,), path, grid_names=(name,))
            with patch.object(boundary, "ASSET_DIR", Path(directory)):
                standalone = boundary.load_static_sdf(path.name)
                self.assertEqual(standalone.dtype, wp.float32)
            loaded, volume_map = boundary.load_boundary_volumes(path, .05, quadrature_points=2)
            np.testing.assert_array_equal(loaded.get_tiles().numpy(), volume_map.get_tiles().numpy())
            # Full tile verification belongs to asset acceptance, not startup.
            before = path.read_bytes()
            with patch.object(boundary, "_precompute_volume_map", side_effect=AssertionError("cache miss")):
                boundary.load_boundary_volumes(path, .05, quadrature_points=2)
            self.assertEqual(path.read_bytes(), before)
            boundary.load_boundary_volumes(path, .04, quadrature_points=2)
            with patch.object(boundary, "_precompute_volume_map", side_effect=AssertionError("cache miss")):
                boundary.load_boundary_volumes(path, .04, quadrature_points=2)
            with self.assertRaisesRegex(RuntimeError, "SDF supports"):
                boundary.load_boundary_volumes(path, .06, quadrature_points=2)
            for radius, points in ((0., 2), (.04, 0), (float("nan"), 2)):
                with self.assertRaises(ValueError):
                    boundary.load_boundary_volumes(path, radius, quadrature_points=points)
                with self.assertRaises(ValueError):
                    boundary.precompute_volumeMap(sdf, radius, quadrature_points=points)


if __name__ == "__main__":
    unittest.main()
