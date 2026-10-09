"""PBF configuration, regular fluid sampling, and device initialization.

Parameter relationships follow PBF2WayCoupling; particle state is initialized
by a Warp kernel on the active device. The legacy solver formulas in
PBF.py/pbf_functions.py have not yet been migrated.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass(frozen=True)
class PBFConfig:
    """Inputs in meters, kilograms and seconds; derived values live on sim."""

    particle_radius: float = 0.0125
    rest_density: float = 1000.0
    frame_dt: float = 1.0 / 90.0
    substeps: int = 3
    pressure_iterations: int = 3
    lambda_regularization: float = 2.0

    # Reference container/block translated by (1.55, 0, 0.8), so the existing
    # zero-origin position constraint can remain unchanged.
    container_size: tuple = (3.1, 8.0, 1.6)
    block_start: tuple = (0.05, 0.0, 0.05)
    block_end: tuple = (1.55, 1.5, 1.55)
    initial_velocity: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        for name in (
            "particle_radius", "rest_density", "frame_dt", "lambda_regularization"
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("substeps", "pressure_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("container_size", "block_start", "block_end", "initial_velocity"):
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


@wp.kernel
def _initialize_particles_kernel(
    positions: wp.array[wp.vec3],
    block_start: wp.vec3,
    spacing: float,
    count_y: int,
    count_z: int,
):
    tid = wp.tid()
    # One thread per particle; Z varies fastest, then Y, then X.
    z = tid % count_z
    y = (tid // count_z) % count_y
    x = tid // (count_y * count_z)
    offset = spacing * wp.vec3(float(x + 1), float(y + 1), float(z + 1))
    positions[tid] = block_start + offset


def compute_hash_grid_dims(container_min, container_max, cell_width):
    """Use the reference's support padding and strictly larger powers of two."""
    # Float32 matches Warp's cell mapping at integer cell boundaries.
    lower = np.asarray(container_min, dtype=np.float32)
    upper = np.asarray(container_max, dtype=np.float32)
    width = np.float32(cell_width)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not np.isfinite(width)
        or width <= 0
        or np.any(upper <= lower)
    ):
        raise ValueError("Hash grid bounds and cell width must be finite and valid")
   
    query_min = np.floor((lower - width) / width).astype(np.int64)
    query_max = np.floor((upper + width) / width).astype(np.int64)
    query_span = query_max - query_min + 1
    
    if np.any(query_span < 1):
        raise ValueError(f"Hash grid query span must be positive, got {tuple(query_span)}")
    
    dims = tuple(1 << int(span).bit_length() for span in query_span)
    cell_count = int(np.prod(dims, dtype=object))
    
    if cell_count > np.iinfo(np.int32).max:
        raise ValueError(
            "Automatic hash grid exceeds Warp's 32-bit cell index range: "
            f"span={tuple(query_span)}, dims={dims}, cells={cell_count}"
        )
    return dims


def init_parameters(sim):
    """Derive all particle scales and grid dimensions from the configuration."""
    config = sim.config
    sim.particle_radius = config.particle_radius
    sim.particle_spacing = 2.0 * sim.particle_radius
    sim.support_radius = 4.0 * sim.particle_radius
    # Compatibility name used by the existing kernel launch arguments.
    sim.smoothing_length = sim.support_radius

    sim.rest_density = config.rest_density
    sim.fluid_volume = 0.8 * sim.particle_spacing**3
    sim.fluid_mass = sim.fluid_volume * sim.rest_density
    # Prepared for the solver migration; the legacy kernel still uses Lamb_Eps.
    sim.lambda_epsilon = config.lambda_regularization / sim.support_radius**2

    sim.frame_dt = config.frame_dt
    sim.substep = config.substeps
    sim.iterations = config.pressure_iterations
    sim.dt = sim.frame_dt / sim.substep

    sim.width, sim.height, sim.length = config.container_size
    sim.boundary = wp.vec3(sim.width, sim.height, sim.length)
    sim.container_min = np.zeros(3, dtype=np.float32)
    sim.container_max = np.asarray(config.container_size, dtype=np.float32)

    # Only three axis counts are computed on the host; no host particle arrays.
    start = np.asarray(config.block_start, dtype=float)
    end = np.asarray(config.block_end, dtype=float)
    counts = np.floor((end - start) / sim.particle_spacing + 0.5).astype(int) - 1
    if np.any(counts < 1):
        raise ValueError("Fluid block must fit at least one particle on every axis")
    sim.block_counts = tuple(int(count) for count in counts)
    sim.n = int(np.prod(counts))

    # Validate dimensions before allocating any device storage.
    sim.hash_grid_dims = compute_hash_grid_dims(
        sim.container_min, sim.container_max, sim.support_radius
    )


def init_device_buffers(sim):
    """Allocate particle and work arrays before the initialization kernel runs."""
    sim.pos = wp.empty(sim.n, dtype=wp.vec3)

    sim.v = wp.zeros(sim.n, dtype=wp.vec3)
    sim.pre_Pos = wp.zeros(sim.n, dtype=wp.vec3)
    sim.pre_New = wp.zeros(sim.n, dtype=wp.vec3)
    sim.delta_Pos = wp.zeros(sim.n, dtype=wp.vec3)
    sim.lambda_Opt = wp.zeros(sim.n, dtype=float)

    sim.delta_Vel = wp.zeros(sim.n, dtype=wp.vec3)
    sim.curl = wp.zeros(sim.n, dtype=wp.vec4)

    sim.grid = wp.HashGrid(*sim.hash_grid_dims)

    # init device data
    wp.launch(
        kernel=_initialize_particles_kernel,
        dim=sim.n,
        inputs=[
            sim.pos,
            wp.vec3(*sim.config.block_start),
            sim.particle_spacing,
            sim.block_counts[1],
            sim.block_counts[2]
        ]
    )


def initialize(sim, config=None):
    """Derive counts, allocate device arrays, then initialize particle state."""
    sim.config = (
        config if config is not None 
        else PBFConfig()
    )

    init_parameters(sim)
    init_device_buffers(sim)
