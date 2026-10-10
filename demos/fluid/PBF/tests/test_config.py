"""Preset serialization, independent CLI sources and export without simulation."""

import contextlib
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from .. import config_io
from .. import PBF as solver
from .. import render_opengl as frontend
from ..pbf_helper import PBFConfig


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.preset = self.root / "preset.json"

    def write_preset(self, data):
        self.preset.write_text(json.dumps(data), encoding="utf-8")
        return str(self.preset)

    def test_full_round_trip_and_directory_creation(self):
        config = replace(PBFConfig(), particle_radius=0.01, gravity=(1.0, -5.0, 2.0),
                         viscosity=0.1, clamp_negative_pressure=False)
        with patch.object(config_io, "CONFIG_DIR", self.root / "new-config"):
            path = config_io.save_config(config, "water.json")
        self.assertEqual(config_io.load_config(path), config)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 17)
        self.assertEqual(set(data), set(asdict(config)))
        self.assertIsInstance(config_io.load_config(path).gravity, tuple)
        self.assertIn('\n  "particle_radius":', path.read_text(encoding="utf-8"))

    def test_partial_json_uses_defaults(self):
        path = self.write_preset({"particle_radius": 0.01, "clamp_negative_pressure": False})
        self.assertEqual(config_io.load_config(path),
                         replace(PBFConfig(), particle_radius=0.01, clamp_negative_pressure=False))
        self.assertEqual(config_io.load_config(self.write_preset({})), PBFConfig())

    def test_default_export_name(self):
        self.assertEqual(config_io.DEFAULT_CONFIG_NAME, "default_para.json")
        with patch.object(config_io, "CONFIG_DIR", self.root / "export"):
            self.assertEqual(config_io.save_config(PBFConfig()).name, "default_para.json")
            for entry in (frontend, solver):
                self.assertIsNone(entry.parse_args([]).export_config)
                self.assertEqual(entry.parse_args(["--export-config"]).export_config, "default_para.json")
                self.assertEqual(entry.parse_args(["--export-config", "water.json"]).export_config, "water.json")
                with patch.object(sys, "argv", [entry.__name__, "--export-config"]), \
                        patch.object(entry.wp, "ScopedDevice", side_effect=AssertionError("Device scope")), \
                        contextlib.redirect_stdout(io.StringIO()):
                    entry.main()
                self.assertEqual(config_io.load_config(self.root / "export" / "default_para.json"), PBFConfig())

    def test_invalid_json_and_values(self):
        for data, message in (
            ([], "object"), ({"unknown": 1}, "Unknown config fields: unknown"),
            ({"particle_radius": -1}, "particle_radius"),
            ({"particle_radius": "0.01"}, "particle_radius must be a number"),
            ({"rest_density": True}, "rest_density must be a number"),
            ({"substeps": 1.5}, "substeps must be an integer"),
            ({"substeps": True}, "substeps must be an integer"),
            ({"substeps": 0}, "substeps must be a positive integer"),
            ({"clamp_negative_pressure": 1}, "must be a bool"),
            ({"gravity": [0, 1]}, "three finite components"),
            ({"gravity": [0, "1", 0]}, "numeric JSON array"),
            ({"gravity": 0}, "numeric JSON array"),
            ({"frame_dt": float("nan")}, "finite and positive"),
            ({"gravity": [0, float("inf"), 0]}, "three finite components"),
            ({"artificial_pressure_q": 1}, "artificial_pressure_q"),
            ({"block_end": [4, 1, 1]}, "inside the container"),
        ):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, message):
                config_io.load_config(self.write_preset(data))
        self.preset.write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cannot load config"):
            config_io.load_config(self.preset)
        with self.assertRaisesRegex(ValueError, "Cannot load config"):
            config_io.load_config(self.root / "missing.json")

    def test_export_rejects_directory_names(self):
        for name in ("", ".", "..", "../water.json", "a/water.json", "a\\water.json",
                     "/water.json", "C:\\water.json", "C:water.json"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "filename"):
                config_io.save_config(PBFConfig(), name)

    def test_real_export_directory_does_not_follow_cwd(self):
        expected_dir = Path(config_io.__file__).resolve().parent / "config"
        self.assertEqual(config_io.CONFIG_DIR, expected_dir)
        name = f"test-{uuid4().hex}.json"
        output = expected_dir / name
        self.addCleanup(lambda: output.unlink(missing_ok=True))
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            path = config_io.save_config(PBFConfig(), name)
        finally:
            os.chdir(previous)
        self.assertEqual(path, output)
        self.assertTrue(path.is_file())
        self.assertFalse((self.root / name).exists())

    def test_json_and_cli_physics_are_mutually_exclusive(self):
        path = self.write_preset({})
        physics = ("--particle-radius 0.0125", "--rest-density 1000", "--frame-dt 0.01",
                   "--lambda-regularization 2", "--substeps 3", "--pressure-iterations 3",
                   "--container-size 3.1 8 1.6", "--block-start .05 0 .05",
                   "--block-end 1.55 1.5 1.55", "--clamp-negative-pressure",
                   "--no-clamp-negative-pressure")
        for entry, options in ((frontend, physics), (solver, physics[-2:])):
            for option in options:
                error = io.StringIO()
                with self.subTest(entry=entry.__name__, option=option), \
                        contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                    entry.parse_args(["--config", path, *option.split()])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("cannot be combined", error.getvalue())

    def test_runtime_options_do_not_change_imported_config(self):
        expected = replace(PBFConfig(), clamp_negative_pressure=False)
        path = self.write_preset({"clamp_negative_pressure": False})
        for entry, options in ((frontend, ["--speed-color-mid", "1", "--speed-color-max", "8"]),
                               (solver, ["--stage-path", "None"])):
            args = entry.parse_args(["--config", path, "--device", "cpu", "--num-frames", "1",
                                     "--verbose", *options])
            self.assertEqual(args.config, expected)
        self.assertEqual(solver.parse_args([]).config, PBFConfig())
        self.assertFalse(solver.parse_args(["--no-clamp-negative-pressure"]).config.clamp_negative_pressure)

    def test_export_exits_before_device_or_renderer_initialization(self):
        path = self.write_preset({"gravity": [1, -2, 3], "clamp_negative_pressure": False})
        for entry in (frontend, solver):
            for source in ([], ["--config", path], ["--no-clamp-negative-pressure"]):
                with self.subTest(entry=entry.__name__, source=source), \
                        patch.object(sys, "argv", [entry.__name__, *source, "--export-config", "water.json"]), \
                        patch.object(config_io, "CONFIG_DIR", self.root / "export"), \
                        patch.object(entry.wp, "init", side_effect=AssertionError("GPU initialization")), \
                        patch.object(entry.wp, "ScopedDevice", side_effect=AssertionError("Device scope")), \
                        contextlib.redirect_stdout(io.StringIO()):
                    entry.main()
                config = config_io.load_config(self.root / "export" / "water.json")
                expected = config_io.load_config(path) if "--config" in source else PBFConfig()
                if "--no-clamp-negative-pressure" in source:
                    expected = replace(expected, clamp_negative_pressure=False)
                self.assertEqual(config, expected)

    def test_entrypoints_report_invalid_config_and_export(self):
        path = self.write_preset({"bogus": 1})
        for entry in (frontend, solver):
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                entry.parse_args(["--config", path])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("Unknown config fields", error.getvalue())
            with patch.object(sys, "argv", [entry.__name__, "--export-config", "../escape.json"]), \
                    self.assertRaisesRegex(SystemExit, "Cannot export config.*filename"):
                entry.main()


if __name__ == "__main__":
    unittest.main()
