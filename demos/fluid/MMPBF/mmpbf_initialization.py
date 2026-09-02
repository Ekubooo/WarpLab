"""Static scene, storage, and Grid initialization for MMPBF.

This module deliberately does not import :mod:`MMPBF`. The main solver passes
its simulation object in, preventing a circular dependency while keeping the
``Example`` class limited to its public construction and stepping methods.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp


def container_dimensions(
    particle_radius: float,
    fluid_width: int,
    fluid_depth: int,
    container_height: float,
) -> tuple[float, float, float]:
    """Return the exact dimensions defined in PBD ``FluidDemo/main.cpp``."""
    diameter = 2.0 * particle_radius
    width = (fluid_width + 1) * diameter * 5.0
    depth = (fluid_depth + 1) * diameter
    return width, container_height, depth


def create_fluid_particles(
    particle_radius: float,
    fluid_width: int,
    fluid_height: int,
    fluid_depth: int,
    container_size: tuple[float, float, float],
) -> np.ndarray:
    """Translate PBD ``createBreakingDam`` without changing its lattice."""
    diameter = 2.0 * particle_radius
    container_width, _, container_depth = container_size
    start = np.array(
        (
            -0.5 * container_width + diameter,
            diameter,
            -0.5 * container_depth + diameter,
        ),
        dtype=np.float64,
    )
    indices = np.stack(
        np.meshgrid(
            np.arange(fluid_width, dtype=np.float64),
            np.arange(fluid_height, dtype=np.float64),
            np.arange(fluid_depth, dtype=np.float64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    return np.ascontiguousarray(start + diameter * indices, dtype=np.float32)


def _add_wall(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    particle_distance: float,
) -> np.ndarray:
    """Translate PBD ``addWall`` including endpoints and duplicate policy."""
    minimum_array = np.asarray(minimum, dtype=np.float64)
    maximum_array = np.asarray(maximum, dtype=np.float64)
    difference = maximum_array - minimum_array
    steps = tuple(int(component / particle_distance) + 1 for component in difference)
    indices = np.stack(
        np.meshgrid(
            np.arange(steps[0], dtype=np.float64),
            np.arange(steps[1], dtype=np.float64),
            np.arange(steps[2], dtype=np.float64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    return minimum_array + particle_distance * indices


def create_boundary_particles(
    particle_radius: float,
    container_size: tuple[float, float, float],
) -> np.ndarray:
    """Translate ``initBoundaryData``; edge/corner duplicates are intentional."""
    width, height, depth = container_size
    x1, x2 = -0.5 * width, 0.5 * width
    y1, y2 = 0.0, height
    z1, z2 = -0.5 * depth, 0.5 * depth
    distance = 2.0 * particle_radius

    walls = (
        _add_wall((x1, y1, z1), (x2, y1, z2), distance),
        _add_wall((x1, y2, z1), (x2, y2, z2), distance),
        _add_wall((x1, y1, z1), (x1, y2, z2), distance),
        _add_wall((x2, y1, z1), (x2, y2, z2), distance),
        _add_wall((x1, y1, z1), (x2, y2, z1), distance),
        _add_wall((x1, y1, z2), (x2, y2, z2), distance),
    )
    return np.ascontiguousarray(np.concatenate(walls, axis=0), dtype=np.float32)


def create_scene(
    particle_radius: float,
    fluid_width: int,
    fluid_height: int,
    fluid_depth: int,
    container_height: float,
) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
    """Create all host-side position data for the official dam-break scene."""
    container_size = container_dimensions(
        particle_radius,
        fluid_width,
        fluid_depth,
        container_height,
    )
    fluid_positions = create_fluid_particles(
        particle_radius,
        fluid_width,
        fluid_height,
        fluid_depth,
        container_size,
    )
    boundary_positions = create_boundary_particles(particle_radius, container_size)
    return container_size, fluid_positions, boundary_positions


def initialize_static_data(simulation):
    """Derive parameters and allocate all storage and static Grid data."""
    config = simulation.config
    simulation.particle_radius = config.particle_radius
    simulation.particle_diameter = 2.0 * simulation.particle_radius
    simulation.support_radius = 4.0 * simulation.particle_radius
    simulation.smoothing_length = simulation.support_radius
    simulation.fluid_volume = 0.8 * simulation.particle_diameter**3
    simulation.particle_mass = simulation.fluid_volume * config.rest_density

    (
        simulation.container_size,
        fluid_positions,
        boundary_positions,
    ) = create_scene(
        simulation.particle_radius,
        config.fluid_width,
        config.fluid_height,
        config.fluid_depth,
        config.container_height,
    )
    width, height, depth = simulation.container_size
    simulation.boundary_min = wp.vec3(-0.5 * width, 0.0, -0.5 * depth)
    simulation.boundary_max = wp.vec3(0.5 * width, height, 0.5 * depth)

    simulation.n = len(fluid_positions)
    simulation.boundary_n = len(boundary_positions)
    device = simulation.device

    simulation.pos = wp.array(fluid_positions, dtype=wp.vec3, device=device)
    simulation.v = wp.zeros(simulation.n, dtype=wp.vec3, device=device)
    simulation.predicted_positions = wp.clone(simulation.pos)
    simulation.old_positions = wp.clone(simulation.pos)
    simulation.last_positions = wp.clone(simulation.pos)
    simulation.position_corrections = wp.zeros(
        simulation.n,
        dtype=wp.vec3,
        device=device,
    )
    simulation.lambdas = wp.zeros(simulation.n, dtype=float, device=device)
    simulation.normalized_densities = wp.zeros(simulation.n, dtype=float, device=device)
    simulation.densities = wp.zeros(simulation.n, dtype=float, device=device)
    simulation.xsph_velocities = wp.zeros(simulation.n, dtype=wp.vec3, device=device)

    simulation.boundary_positions = wp.array(
        boundary_positions,
        dtype=wp.vec3,
        device=device,
    )
    simulation.boundary_volumes = wp.zeros(
        simulation.boundary_n,
        dtype=float,
        device=device,
    )

    simulation.fluid_neighbor_counts = wp.zeros(simulation.n, dtype=int, device=device)
    simulation.fluid_neighbor_indices = wp.empty(
        simulation.n * config.max_fluid_neighbors,
        dtype=int,
        device=device,
    )
    simulation.boundary_neighbor_counts = wp.zeros(
        simulation.n,
        dtype=int,
        device=device,
    )
    simulation.boundary_neighbor_indices = wp.empty(
        simulation.n * config.max_boundary_neighbors,
        dtype=int,
        device=device,
    )
    simulation.neighbor_overflow = wp.zeros(1, dtype=int, device=device)
    simulation.density_error_sum = wp.zeros(1, dtype=float, device=device)
    simulation.max_speed_squared = wp.zeros(1, dtype=float, device=device)
    simulation.max_boundary_violation_buffer = wp.zeros(1, dtype=float, device=device)

    grid_shape = tuple(
        max(1, int(math.ceil(length / simulation.support_radius)))
        for length in simulation.container_size
    )
    simulation.fluid_grid = wp.HashGrid(*grid_shape, device=device)
    simulation.boundary_grid = wp.HashGrid(*grid_shape, device=device)
    simulation.boundary_grid.build(
        simulation.boundary_positions,
        simulation.support_radius,
    )


def finalize_initialization(simulation):
    """Validate boundary volumes and initialize all runtime scalar state."""
    boundary_volume_values = simulation.boundary_volumes.numpy()
    if not np.all(np.isfinite(boundary_volume_values)) or np.any(
        boundary_volume_values <= 0.0
    ):
        raise RuntimeError("official Akinci boundary initialization produced invalid volumes")

    simulation.sim_time = 0.0
    simulation.current_dt = simulation.config.initial_time_step
    simulation.last_dt = 0.0
    simulation.last_pressure_iterations = 0
    simulation.last_density_error_percent = math.inf
    simulation.last_boundary_violation = 0.0
