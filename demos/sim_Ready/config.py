"""Defaults and destinations for the standalone, Y-up asset pipeline."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "Assets"
OUTPUT = ROOT / "output"
PBF_ASSETS = ROOT.parent / "fluid" / "PBF" / "Assets"
# Filesystem destinations only: this package never imports its consumers.
OUTPUT_DIRECTORIES = (OUTPUT, PBF_ASSETS)
DEFAULT_NAME = "stairs_scene"


@dataclass(frozen=True)
class GeometryOptions:
    size: tuple[float, float] = (3.1, 1.6)
    step_width: float = 0.15
    step_height: float = 0.1
    platform_width: float = 0.7
    model_fraction: float = 0.95
    model_sink: float = 0.003
    model_yaw: float = 0.0
    wall_thickness: float = 0.25
    overlap: float = 0.075
    ceiling: float = 4.5
    simplify_tolerance: float = 0.001


@dataclass(frozen=True)
class BakeOptions:
    voxel_size: float = 0.0075
    padding_voxels: int = 3
