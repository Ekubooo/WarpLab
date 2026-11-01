"""PBF configuration and device mathematics, independent of any application."""

from dataclasses import dataclass
import math

import numpy as np
import warp as wp


# =============================================================================
# Solver configuration and compile-time coefficients
# Radius and mass come from the Newton model, not a scene or renderer.
# =============================================================================

@dataclass(frozen=True)
class PBFConfig:
    rest_density: float = 1000.0
    pressure_iterations: int = 3
    lambda_regularization: float = 2.0
    viscosity: float = 0.5
    vorticity_confinement: float = 0.5
    max_speed: float = 6.5
    artificial_pressure_strength: float = 0.0005
    artificial_pressure_q: float = 0.1
    clamp_negative_pressure: bool = True
    neighbor_capacity: int = 192

    def __post_init__(self):
        for name in ("rest_density", "lambda_regularization", "max_speed"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("viscosity", "vorticity_confinement", "artificial_pressure_strength"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.artificial_pressure_q < 1:
            raise ValueError("artificial_pressure_q must be in (0, 1)")
        for name in ("pressure_iterations", "neighbor_capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.clamp_negative_pressure, bool):
            raise ValueError("clamp_negative_pressure must be a bool")


@dataclass(frozen=True)
class KernelParameters:
    config: PBFConfig
    particle_radius: float
    fluid_mass: float

    @property
    def support_radius(self):
        return 4.0 * self.particle_radius


# =============================================================================
# Neighbor storage and grid dimensions
# The cache keeps the original (capacity, particle_count) layout.
# =============================================================================

@wp.struct
class NeighborCache:
    NeighborCount: wp.array[int]
    CacheBuffer: wp.array2d[int]

    def allocateNCache(self, pNum, nCap):
        self.NeighborCount = wp.zeros(pNum, dtype=int)
        self.CacheBuffer = wp.zeros((nCap, pNum), dtype=int)
        # (nb_cache, p_num) shape is good for mem access.

        return self


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



# =============================================================================
# Device mathematics
# Keep the existing grid sorted-slot fast path and boundary expressions.
# =============================================================================

@wp.func_native("return query.cell_index - 1;")
def sorted_index(query: wp.HashGridQuery) -> int:
    """Read the sorted slot immediately after a successful hash_grid_query_next.

    Warp 1.18 increments its internal cell_index when yielding the input point
    index. Use the slot for particle state gathered in the same build's hash
    order. Recheck this internal field when upgrading Warp.
    """
    ...


@wp.func
def square(x: float):
    return x * x


@wp.func
def cube(x: float):
    return x * x * x


@wp.func
def fifth(x: float):
    return x * x * x * x * x


@wp.func
def query_sdf_normal(sdf_id: wp.uint64, position: wp.vec3):
    """Return validity, signed distance and a safe outward world-space normal.

    Validate before normalizing; accept negative distances for collision
    projection. Assumes axis-aligned, uniformly spaced voxels, as in the
    baked SDF. Invalid samples return zero distance and normal.
    """
    index = wp.volume_world_to_index(sdf_id, position)
    gradient = wp.vec3(0.0)
    distance = wp.volume_sample_grad_f(sdf_id, index, wp.Volume.LINEAR, gradient)
    if not wp.isfinite(distance):
        return False, 0.0, wp.vec3(0.0)

    # For this grid, the direction transform is simply gradient / voxel_size.
    world_gradient = wp.volume_world_to_index_dir(sdf_id, gradient)
    gradient_squared = wp.dot(world_gradient, world_gradient)
    if not wp.isfinite(gradient_squared) or gradient_squared <= 1.0e-12:
        return False, 0.0, wp.vec3(0.0)

    return True, distance, world_gradient / wp.sqrt(gradient_squared)


@wp.func
def apply_SDF(volume_id: wp.uint64, pos: wp.vec3, radius: wp.float32):
    valid, distance, normal = query_sdf_normal(volume_id, pos)
    if valid and distance < radius:
        pos += normal * (radius - distance)
    return pos


@wp.func
def query_volume_boundary(sdf_id: wp.uint64, position: wp.vec3, support_radius: float):
    """Return validity, protected pressure distance and direction toward the boundary."""
    valid, distance, normal = query_sdf_normal(sdf_id, position)
    if not valid or distance <= 0.0 or distance >= support_radius:
        return False, 0.0, wp.vec3(0.0)

    # PBF uses support_radius = 4 * particle_radius. Keep the equivalent
    # boundary point at least one particle diameter away to limit pressure.
    pressure_distance = wp.max(distance + 0.125 * support_radius, 0.5 * support_radius)
    return True, pressure_distance, -normal



# =============================================================================
# Per-instance fluid functions
# Closure constants remain compile-time values without mutable module globals.
# =============================================================================

def create_fluid_functions(parameters):
    Smoothing_Radius = wp.constant(parameters.support_radius)
    K_SPoly6 = wp.constant(315.0 / (
        64.0 * wp.pi * wp.pow(wp.abs(parameters.support_radius), 9.0)))
    K_DSPow3 = wp.constant(3.0 * (15.0 / (wp.pi * wp.pow(parameters.support_radius, 6.0))))

    @wp.func
    def Poly6(pos: wp.vec3, neighbor_pos: wp.vec3):
        O2N = neighbor_pos - pos
        sqrD2N = wp.dot(O2N, O2N)
        if sqrD2N < square(Smoothing_Radius):
            return cube(square(Smoothing_Radius) - sqrD2N) * K_SPoly6
        return 0.0

    @wp.func
    def GradPow3(pos: wp.vec3, neighbor_pos: wp.vec3):
        O2N = neighbor_pos - pos
        sqrD2N = wp.dot(O2N, O2N)
        if sqrD2N > 0.0 and sqrD2N < square(Smoothing_Radius):
            dst2N = wp.sqrt(sqrD2N)
            grad = -1.0 * square(Smoothing_Radius - dst2N) * K_DSPow3
            return (O2N/dst2N) * grad
        return wp.vec3(0,0,0)

    return Poly6, GradPow3
