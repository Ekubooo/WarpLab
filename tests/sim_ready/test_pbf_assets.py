"""Publication, cache boundaries and VM repair using private asset copies."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from PBF.pbf_helper import PBFConfig
from PBF import boundary_model
from Sim_Ready import publish
from Sim_Ready.asset_io import file_digest, sdf_digest
from Sim_Ready.pipelines.preset import pbf


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="sim-ready-tests-")
        cls.root = Path(cls.temporary.name)
        cls.assets = cls.root / "Assets"
        cls.cache = cls.root / "cache"
        with patch.object(pbf, "generate_assets", side_effect=AssertionError("Existing assets must be reused")):
            pbf.build_scene_assets("stairs-rigids", output_dir=cls.assets, cache_dir=cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_migration_preserves_every_attachment(self):
        for path in pbf.asset_files(self.assets, pbf.asset_names("stairs-rigids")):
            self.assertEqual(file_digest(path), file_digest(pbf.OUTPUT / path.relative_to(self.assets)))

    def test_cache_hit_skips_geometry_and_generation(self):
        with patch.object(pbf, "prepare_geometry", side_effect=AssertionError("cache hit")):
            pbf.build_scene_assets("stairs-rigids", output_dir=self.root / "second", cache_dir=self.cache)

    def test_corrupt_cache_does_not_replace_published_assets(self):
        directory = next(self.cache.glob("stairs-rigids-*"))
        mesh = directory / "rigid/stairs_box.obj"
        original = mesh.read_bytes()
        published = {str(p): file_digest(p) for p in pbf.asset_files(self.assets, pbf.asset_names("stairs-rigids"))}
        try:
            mesh.write_bytes(original + b"\n# modified\n")
            with self.assertRaisesRegex(ValueError, "Mesh digest differs"):
                pbf.build_scene_assets("stairs-rigids", output_dir=self.assets, cache_dir=self.cache)
        finally:
            mesh.write_bytes(original)
        self.assertEqual(published, {name: file_digest(name) for name in published})

    def test_static_publication_keeps_collision_and_dynamic_assets(self):
        before = {p.name: file_digest(p) for p in (self.assets / "rigid").iterdir()}
        pbf.build_scene_assets("static", output_dir=self.assets, cache_dir=self.cache)
        self.assertTrue((self.assets / "boundary/stairs_scene.collision.npz").is_file())
        self.assertEqual(before, {p.name: file_digest(p) for p in (self.assets / "rigid").iterdir()})

    def test_recipe_ignores_fluid_only_parameters(self):
        base = PBFConfig()
        changed = replace(base, frame_dt=base.frame_dt / 2, rest_density=900., pressure_iterations=4)
        self.assertEqual(pbf.build_recipe("stairs-rigids", base), pbf.build_recipe("stairs-rigids", changed))
        # Smaller support changes VM only; exceeding the baked band changes geometry cache.
        self.assertEqual(pbf.build_recipe("static", base),
                         pbf.build_recipe("static", replace(base, particle_radius=.01)))
        self.assertNotEqual(pbf.build_recipe("static", base),
                            pbf.build_recipe("static", replace(base, particle_radius=.013)))
        original = pbf.build_recipe("static", base)
        with patch.object(pbf, "file_digest", return_value="different-source"):
            self.assertNotEqual(original, pbf.build_recipe("static", base))

    def test_missing_attachment_rejected_offline_and_runtime(self):
        from PBF.scene import load_rigid_assets
        collision = self.assets / "boundary/stairs_scene.collision.npz"
        backup = collision.with_suffix(".saved")
        collision.rename(backup)
        try:
            with self.assertRaises(FileNotFoundError):
                load_rigid_assets(self.assets)
            with self.assertRaises(FileNotFoundError):
                pbf.validate_asset(self.assets, "boundary/stairs_scene", dict(kind="stairs", version=1), "cuda:0")
        finally:
            backup.rename(collision)

    def test_volume_map_repair_and_support_limit(self):
        import shutil
        path = self.root / "vm-test.nvdb"
        shutil.copy2(self.assets / "rigid/stairs_box.nvdb", path)
        sdf, _ = boundary_model.load_boundary_volumes(path, .05)
        identity = sdf_digest(sdf)
        sdf, _ = boundary_model.load_boundary_volumes(path, .04, quadrature_points=8)
        self.assertEqual(identity, sdf_digest(sdf))
        after = file_digest(path)
        with patch.object(boundary_model, "_precompute_volume_map", side_effect=AssertionError("expected VM hit")):
            boundary_model.load_boundary_volumes(path, .04, quadrature_points=8)
        self.assertEqual(after, file_digest(path))
        with self.assertRaisesRegex(RuntimeError, "supports h"):
            boundary_model.load_boundary_volumes(path, .06)
        self.assertEqual(after, file_digest(path))

    def test_runtime_never_imports_producers(self):
        script = """
import sys
from PBF.simulation import Simulation
sim = Simulation(scene='stairs-rigids')
sim.step()
assert all('Sim_Ready.' + name not in sys.modules for name in
           ('geometry', 'sdf', 'collision', 'validation', 'pipelines.preset.pbf', 'pipelines.container', 'pipelines.rigid'))
sim.close()
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_migrated_static_30_substeps(self):
        script = """
from pathlib import Path
import sys
from unittest.mock import patch
from PBF import boundary_model
from tests.pbf.reference import independent_pair, verify_pair
with patch.object(boundary_model, 'ASSET_DIR', Path(sys.argv[1])):
    with independent_pair() as (reference, simulation):
        result = verify_pair(reference, simulation, steps=30)
        assert result['bitwise_equal']
print(result)
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.assets)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_output_isolation(self):
        with self.assertRaisesRegex(ValueError, "Outputs must not overwrite"):
            pbf.build_scene_assets("static", output_dir=pbf.PROJECT / "demos/forbidden")


class PublicationTests(unittest.TestCase):
    def test_rigid_pipeline_only_stages_requested_products(self):
        from Sim_Ready import collision
        from Sim_Ready.config import BakeOptions
        from Sim_Ready.pipelines import rigid
        mesh = rigid.prepare(dict(kind="box", size=[.3] * 3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(publish, "publish_files", side_effect=AssertionError("preset owns publication")), \
                    patch.object(collision, "prepare_hulls", side_effect=AssertionError("native shape")):
                obj, nvdb, hulls = rigid.build(mesh, root / "box", bake_options=BakeOptions(voxel_size=.005))
            self.assertTrue(obj.is_file() and nvdb.is_file())
            self.assertEqual(hulls, ())
            self.assertEqual(sorted(p.suffix for p in root.iterdir()), [".nvdb", ".obj"])
            with patch.object(collision, "prepare_hulls", return_value=(mesh,)) as decompose:
                _, _, hulls = rigid.build(mesh, root / "proxy", collision_cache=root / "hulls")
            decompose.assert_called_once_with(mesh, "proxy", root / "hulls")
            self.assertEqual(hulls, (mesh,))

    def test_asset_ignore_rules_preserve_code_and_records(self):
        ignored = ("Sim_Ready/Assets/other.obj", "Sim_Ready/Assets/spot.obj",
                   "PBF/Assets/boundary/stairs_scene.json", "PBF/Assets/boundary/stairs_scene.collision.npz",
                   "PBF/Assets/boundary/stairs_scene.nvdb", "PBF/Assets/rigid/stairs_box.json")
        kept = ("Sim_Ready/Assets/bunny_flat_bottom.obj", "Sim_Ready/Assets/spot_triangulated_good.obj",
                "PBF/Assets/.gitignore", "PBF/config/default_para.json", "PBF/scene_data.py",
                "Sim_Ready/README.md", "Sim_Ready/PROVENANCE.md", "Sim_Ready/pipelines/preset/pbf.py")
        for expected, paths in ((0, ignored), (1, kept)):
            for path in paths:
                result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=pbf.PROJECT)
                self.assertEqual(result.returncode, expected, path)

    def test_rollback_after_partial_group_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"; source.write_bytes(b"new")
            first = root / "first"; first.write_bytes(b"old")
            second = root / "second"
            third = root / "third"
            original = publish.os.replace
            def replace_or_fail(a, b):
                if b == third:
                    raise OSError("injected publication failure")
                return original(a, b)
            with patch.object(publish.os, "replace", side_effect=replace_or_fail):
                with self.assertRaisesRegex(OSError, "injected"):
                    publish.publish_files([(source, first), (source, second), (source, third)])
            self.assertEqual(first.read_bytes(), b"old")
            self.assertFalse(second.exists())
            self.assertFalse(third.exists())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["first", "source"])

    def test_help_and_argument_conflict_do_not_initialize_cuda(self):
        for args in (("--help",), ("--scene", "static", "--config", "missing.json", "--particle-radius", ".01")):
            result = subprocess.run([sys.executable, "-m", "Sim_Ready.pipelines.preset.pbf", *args],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0 if args == ("--help",) else 2)
            self.assertNotIn("initialized", result.stdout)

    def test_independent_imports_and_newton_terrain(self):
        import ast
        for path in pbf.ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imports = ([node.module or ""] if isinstance(node, ast.ImportFrom) else
                           [n.name for n in node.names] if isinstance(node, ast.Import) else [])
                self.assertFalse(any(name.startswith("demos") for name in imports), str(path))
                if path.parent.name != "preset":
                    self.assertFalse(any(name.startswith("PBF") for name in imports), str(path))
        from Sim_Ready import geometry, sdf
        self.assertIn("Mesh.create_terrain(", (pbf.ROOT / "geometry.py").read_text())
        self.assertFalse(wp.get_module(sdf.__name__).options["enable_backward"])


if __name__ == "__main__":
    unittest.main()
