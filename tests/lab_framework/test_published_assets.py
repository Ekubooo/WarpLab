"""Published bundles are complete runtime inputs; production stays offline."""

import os
import json
import contextlib
from dataclasses import asdict
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from Lab_Framework.pbf.asset_io import ASSET_VERSION, build_command, file_digest, load_asset, sdf_digest
from Lab_Framework.pbf.config import PBFConfig
from Lab_Framework.pbf.scene_data import asset_geometry
from Lab_Framework.sim_ready import pipeline
from Lab_Framework.sim_ready import pbf_assets


class PublicationTests(unittest.TestCase):
    def test_cross_directory_publication_is_one_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            for i, relative in enumerate(("stairs_scene.json", "Rigid/spot.collision.npz", "Rigid/spot.json")):
                source = root / f"source_{i}"
                source.write_bytes(b"new")
                target = root / "consumer" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"old")
                pairs.append((source, target))
            replace = os.replace
            failed = False

            def fail_once(source, target):
                nonlocal failed
                if Path(target) == pairs[-1][1] and not failed:
                    failed = True
                    raise OSError("injected final attachment failure")
                return replace(source, target)

            with patch.object(pipeline.os, "replace", side_effect=fail_once), self.assertRaises(OSError):
                pipeline.publish_files(pairs)
            for _, target in pairs:
                self.assertEqual(target.read_bytes(), b"old")
            self.assertFalse(list((root / "consumer").rglob("*.asset.tmp")))

    def test_publication_rolls_back_all_four_attachments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir, target = root / "staging", root / "consumer"
            source_dir.mkdir()
            target.mkdir()
            names = ("spot.obj", "spot.nvdb", "spot.collision.npz", "spot.json")
            sources = [source_dir / name for name in names]
            for source in sources:
                source.write_bytes(b"new " + source.name.encode())
                (target / source.name).write_bytes(b"old " + source.name.encode())
            replace = os.replace
            failed = False

            def fail_metadata(source, destination):
                nonlocal failed
                if Path(destination) == target / "spot.json" and not failed:
                    failed = True
                    raise OSError("injected metadata publication failure")
                return replace(source, destination)

            with patch.object(pipeline.os, "replace", side_effect=fail_metadata):
                with self.assertRaisesRegex(OSError, "metadata publication"):
                    pipeline.publish_pair(sources, (target,))
            self.assertEqual({p.name for p in target.iterdir()}, set(names))
            for name in names:
                self.assertEqual((target / name).read_bytes(), b"old " + name.encode())
            pipeline.publish_pair(sources, (target,))
            for name in names:
                self.assertEqual((target / name).read_bytes(), b"new " + name.encode())

    def test_missing_asset_reports_offline_command_without_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = "python -m Lab_Framework.sim_ready.pbf_assets --scene stairs-rigids"
            with self.assertRaisesRegex((OSError, ValueError, RuntimeError), "sim_ready.pbf_assets"):
                load_asset(root / "absent", {}, command=command)
            self.assertEqual(list(root.iterdir()), [])


class PublishedMetadataTests(unittest.TestCase):
    def test_sdf_identity_ignores_pair_headers_without_mutating_host_data(self):
        raw = np.arange(128, dtype=np.uint8)
        original = raw.copy()
        volume = SimpleNamespace(device="cpu", array=lambda: SimpleNamespace(ptr=123),
                                 get_grid_info=lambda: SimpleNamespace(size_in_bytes=len(raw)))
        with patch("warp.array", return_value=SimpleNamespace(numpy=lambda: raw)):
            identity = sdf_digest(volume)
            np.testing.assert_array_equal(raw, original)
            raw[8:16], raw[20:32] = 200, 201
            self.assertEqual(sdf_digest(volume), identity)
            raw[-1] ^= 1
            self.assertNotEqual(sdf_digest(volume), identity)

    def make_bundle(self, directory):
        stem = Path(directory) / "spot"
        stem.with_suffix(".obj").write_text("v 0 0 0\n", encoding="utf-8")
        # These CPU tests exercise attachment metadata, not the NanoVDB reader.
        stem.with_suffix(".nvdb").write_bytes(b"test grid placeholder")
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
        np.savez(stem.with_suffix(".collision.npz"), count=1, vertices_0=vertices, faces_0=faces)
        metadata = dict(version=ASSET_VERSION, geometry={"kind": "test"},
            mesh_sha256=file_digest(stem.with_suffix(".obj")), sdf_sha256="0" * 64,
            collision_sha256=file_digest(stem.with_suffix(".collision.npz")),
            mass=dict(volume=1., com=[.1, .2, .3], inertia_unit_density=np.eye(3).tolist()))
        self.write_metadata(stem, metadata)
        return stem, metadata

    @staticmethod
    def write_metadata(stem, metadata):
        stem.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")

    def load(self, stem, geometry=None):
        return load_asset(stem, {"kind": "test"} if geometry is None else geometry,
                          command="python -m Lab_Framework.sim_ready.pbf_assets --scene stairs-rigids")

    def test_reads_arrays_and_mass_without_mesh_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            stem, metadata = self.make_bundle(directory)
            loaded = self.load(stem)
            self.assertEqual(loaded.metadata["mass"], metadata["mass"])
            self.assertEqual(len(loaded.hulls), 1)
            self.assertEqual(loaded.hulls[0].vertices.shape, (4, 3))
            self.assertEqual(loaded.hulls[0].faces.shape, (4, 3))
            self.assertFalse(hasattr(loaded, "mesh"))

    def test_rejects_missing_attachments_and_stale_geometry(self):
        for suffix in (".obj", ".nvdb", ".collision.npz"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                stem, _ = self.make_bundle(directory)
                stem.with_suffix(suffix).unlink()
                with self.assertRaisesRegex(RuntimeError, "Build it offline"):
                    self.load(stem)
        with tempfile.TemporaryDirectory() as directory:
            stem, _ = self.make_bundle(directory)
            with self.assertRaisesRegex(RuntimeError, "geometry"):
                self.load(stem, {"kind": "different"})
            stem.with_suffix(".obj").write_text("modified mesh", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "mesh has changed"):
                self.load(stem)

    def test_rejects_invalid_mass_properties_before_assembly(self):
        replacements = (
            ("volume", -1.), ("volume", float("nan")), ("com", [0., 0.]),
            ("com", [float("inf"), 0., 0.]),
            ("inertia_unit_density", np.diag([1., -1., 1.]).tolist()),
            ("inertia_unit_density", [[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]]),
        )
        for key, value in replacements:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                stem, metadata = self.make_bundle(directory)
                metadata["mass"][key] = value
                self.write_metadata(stem, metadata)
                with self.assertRaisesRegex(RuntimeError, "mass properties"):
                    self.load(stem)

    def test_spot_requires_mass_and_convex_proxy_metadata(self):
        for key in ("mass", "collision_sha256"):
            with self.subTest(missing=key), tempfile.TemporaryDirectory() as directory:
                stem, metadata = self.make_bundle(directory)
                metadata["geometry"] = {"kind": "spot"}
                del metadata[key]
                self.write_metadata(stem, metadata)
                with self.assertRaisesRegex(RuntimeError, "Missing published Spot"):
                    self.load(stem, {"kind": "spot"})

    def test_stairs_recipe_does_not_apply_unrelated_tank_constraints(self):
        from dataclasses import replace
        config = replace(PBFConfig(), container_size=(.5, .5, .5),
                         block_start=(.05, .05, .05), block_end=(.4, .4, .4))
        self.assertEqual(asset_geometry("stairs_scene", config)["kind"], "stairs")
        self.assertEqual(asset_geometry("stairs_box", config)["kind"], "box")
        with self.assertRaisesRegex(ValueError, "fit"):
            asset_geometry("tank_scene", config)

    def test_small_custom_container_hint_requires_matching_full_json(self):
        from dataclasses import replace
        config = replace(PBFConfig(), container_size=(1., 1., 1.),
                         block_start=(.1, .1, .1), block_end=(.5, .5, .5))
        command = build_command("static", config)
        self.assertIn("--config", command)
        self.assertNotIn("--container-size", command)

    def test_density_and_support_do_not_select_new_geometry(self):
        from dataclasses import replace
        from Lab_Framework.pbf.scene_data import tank_box_config
        original = tank_box_config()
        changed = replace(original, rest_density=900., particle_radius=.01)
        for name in ("tank_scene", "tank_box", "stairs_scene", "stairs_spot"):
            self.assertEqual(asset_geometry(name, original), asset_geometry(name, changed))


class OfflineArgumentsTests(unittest.TestCase):
    def test_help_imports_no_simulation_renderer_or_gpu_runtime(self):
        source = """
import json, sys
from Lab_Framework.sim_ready.pbf_assets import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
print(json.dumps([name for name in sys.modules if name == 'warp' or name == 'newton'
    or name in ('Lab_Framework.pbf.simulation', 'Lab_Framework.pbf.render_opengl')]))
"""
        result = subprocess.run([sys.executable, "-B", "-c", source],
            cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1]), [])

    def test_defaults_and_overrides_match_runtime_configuration_rules(self):
        from Lab_Framework.pbf.scene_data import tank_box_config
        for scene in ("static", "stairs-rigids"):
            self.assertEqual(pbf_assets.parse_args(["--scene", scene]).config, PBFConfig())
        self.assertEqual(pbf_assets.parse_args(["--scene", "tank-box"]).config, tank_box_config())
        args = pbf_assets.parse_args(["--scene", "tank-box", "--particle-radius", ".01",
                                     "--container-size", "1.2", "1.1", "1.3"])
        self.assertEqual(args.config.particle_radius, .01)
        self.assertEqual(args.config.container_size, (1.2, 1.1, 1.3))
        self.assertEqual(args.config.block_end, tank_box_config().block_end)

    def test_explicit_json_keeps_original_defaults_and_excludes_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fluid.json"
            path.write_text(json.dumps({"particle_radius": .02}), encoding="utf-8")
            args = pbf_assets.parse_args(["--scene", "tank-box", "--config", str(path)])
            expected = asdict(PBFConfig())
            expected["particle_radius"] = .02
            self.assertEqual(asdict(args.config), expected)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                pbf_assets.parse_args(["--scene", "tank-box", "--config", str(path), "--particle-radius", ".01"])
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
