"""Archive destinations must remain isolated from the shared demo producer."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demos.sim_Ready import config as shared_config
from demos.sim_Ready import geometry as shared_geometry
from demos.sim_Ready import pipeline as shared_pipeline
from demos.sim_Ready import sdf as shared_sdf
from demos.sim_Ready import validation as shared_validation
from demos.PBF_Rigid.sim_ready import config, geometry, pipeline, sdf, validation


class ArchiveLayoutTests(unittest.TestCase):
    def test_common_algorithms_are_shared(self):
        self.assertIs(geometry.prepare_scene, shared_geometry.prepare_scene)
        self.assertIs(geometry.prepare_rigid, shared_geometry.prepare_rigid)
        self.assertIs(geometry.union, shared_geometry.union)
        self.assertIs(validation.validate_solid, shared_validation.validate_solid)
        self.assertIs(sdf.bake, shared_sdf.bake)
        self.assertIs(sdf.bake_rigid, shared_sdf.bake_rigid)

    def test_explicit_destinations_leave_shared_defaults_unchanged(self):
        before = (shared_config.ASSETS, shared_config.OUTPUT, shared_config.PBF_ASSETS,
                  shared_config.OUTPUT_DIRECTORIES)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.obj"
            source.write_text("# path resolution only\n")
            output, consumer = root / "output", root / "Assets"
            with patch.object(shared_pipeline, "resolve_asset_paths",
                              wraps=shared_pipeline.resolve_asset_paths) as resolve:
                paths = pipeline.resolve_asset_paths(source, "test", output=output,
                                                     destinations=(output, consumer))
            self.assertEqual(resolve.call_args.kwargs,
                             dict(output=output, destinations=(output, consumer)))
            self.assertEqual(paths.destinations, (output.resolve(), consumer.resolve()))
            self.assertFalse(output.exists())
        self.assertEqual(before, (shared_config.ASSETS, shared_config.OUTPUT,
                                 shared_config.PBF_ASSETS, shared_config.OUTPUT_DIRECTORIES))
        self.assertEqual(config.PBF_ASSETS, config.ROOT.parent / "Assets")
        self.assertNotEqual(config.ASSETS, shared_config.ASSETS)
        self.assertNotEqual(config.PBF_ASSETS, shared_config.PBF_ASSETS)


if __name__ == "__main__":
    unittest.main()
