"""Standalone PBF scene and physical configuration."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PBFConfig:
    """Inputs in meters, kilograms and seconds; initialization derives state and constants."""

    particle_radius: float = 0.0125
    rest_density: float = 1000.0
    frame_dt: float = 1.0 / 150.0
    substeps: int = 3
    pressure_iterations: int = 3
    lambda_regularization: float = 2.0
    gravity: tuple = (0.0, -9.81, 0.0)
    viscosity: float = 0.5
    vorticity_confinement: float = 0.5
    max_speed: float = 6.5
    artificial_pressure_strength: float = 0.0005
    artificial_pressure_q: float = 0.1
    clamp_negative_pressure: bool = True

    # Reference container/block translated by (1.55, 0, 0.8), so the existing
    # zero-origin position constraint can remain unchanged.
    container_size: tuple = (3.0, 5.0, 2.0)
    block_start: tuple = (0.1, 0.5, 0.1)
    block_end: tuple = (1.5, 2.5, 1.25)
    initial_velocity: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if not isinstance(self.clamp_negative_pressure, bool):
            raise ValueError("clamp_negative_pressure must be a bool")
        for name in (
            "particle_radius", "rest_density", "frame_dt", "lambda_regularization", "max_speed"
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("substeps", "pressure_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("viscosity", "vorticity_confinement", "artificial_pressure_strength"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.artificial_pressure_q) or not 0 < self.artificial_pressure_q < 1:
            raise ValueError("artificial_pressure_q must be finite and in (0, 1)")
        for name in ("container_size", "block_start", "block_end", "initial_velocity", "gravity"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain three finite components")
        size = np.asarray(self.container_size)
        start = np.asarray(self.block_start)
        end = np.asarray(self.block_end)
        if np.any(size <= 2.0 * self.particle_radius):
            raise ValueError("Container dimensions must exceed the particle diameter")
        if np.any(start < 0.0) or np.any(end > size) or np.any(end <= start):
            raise ValueError("Fluid block must have positive size and lie inside the container")

