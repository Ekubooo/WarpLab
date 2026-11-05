"""Defaults and destinations shared by container and local-frame rigid flows."""

from demos.sim_Ready.config import BakeOptions, GeometryOptions
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "Assets"
OUTPUT = ROOT / "output"
PBF_ASSETS = ROOT.parent / "Assets"
# Filesystem destinations only: this package never imports its consumers.
OUTPUT_DIRECTORIES = (OUTPUT, PBF_ASSETS)
RIGID_OUTPUT = OUTPUT / "rigid"
RIGID_OUTPUT_DIRECTORIES = (RIGID_OUTPUT, PBF_ASSETS / "Rigid")
DEFAULT_NAME = "stairs_scene"

