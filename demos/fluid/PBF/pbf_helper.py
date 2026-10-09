"""PBF configuration, initialization, and device mathematics.

Parameter relationships follow PBF2WayCoupling; particle state is initialized
by a Warp kernel on the active device. The legacy solver formulas in
PBF.py/pbf_functions.py have not yet been migrated. Solver constants are bound
here before the first solver compilation, for one fixed configuration per process.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

# //////////////////////////////////////////////////////////////////////////////
# para initiative

@dataclass(frozen=True)
class PBFConfig:
    """Inputs in meters, kilograms and seconds; initialization derives state and constants."""

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
    block_counts: wp.vec3i
):
    tid = wp.tid()
    # One thread per particle; Z varies fastest, then Y, then X.
    z = tid % block_counts[2]
    y = (tid // block_counts[2]) % block_counts[1]
    x = tid // (block_counts[1] * block_counts[2])
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
    global S_corr_K, S_corr_N, Inv_Rho0, Lamb_Eps, visStrength, vorConfirm, MaxVel
    global K_SPow3, K_DSPow3, K_SPoly6, Fluid_Volume, Fluid_Mass
    global Boundary_Restitution, Boundary_Tangential_Retention
    global Boundary_Contact_Eps, Boundary_Disturbance

    config = sim.config
    sim.particle_radius = config.particle_radius
    sim.particle_spacing = 2.0 * sim.particle_radius
    sim.support_radius = 4.0 * sim.particle_radius
    # Compatibility name used by the existing kernel launch arguments.
    sim.smoothing_length = sim.support_radius

    rest_density = config.rest_density
    fluid_volume = 0.8 * sim.particle_spacing**3
    fluid_mass = fluid_volume * rest_density
    lambda_epsilon = config.lambda_regularization / sim.support_radius**2

    # Fixed solver settings; keep their existing values.
    ScorrK = 0.01
    ScorrN = 4.0
    viscosityStr = 0.025
    vorticityCon = 0.5
    VelLimit = 8.0
    boundaryRestitution = 0.98
    boundaryTangentialRetention = 1.0
    boundaryContactEpsScale = 1.0e-4
    boundaryDisturbance = 0.001

    # Preserve the original expressions, using the initialized support radius.
    paraPoly6 = 315.0 / (
        64.0 * wp.pi * wp.pow(wp.abs(sim.smoothing_length), 9.0)
    )
    paraPow3 = 15.0 / (wp.pi * wp.pow(sim.smoothing_length, 6.0))

    # Module globals are read directly by wp.func and by the solver via fn.
    Fluid_Volume = wp.constant(fluid_volume)
    Fluid_Mass = wp.constant(fluid_mass)
    S_corr_K = wp.constant(ScorrK)
    S_corr_N = wp.constant(ScorrN)
    Inv_Rho0 = wp.constant(1.0 / rest_density)
    Lamb_Eps = wp.constant(lambda_epsilon)
    visStrength = wp.constant(viscosityStr)
    vorConfirm = wp.constant(vorticityCon)
    MaxVel = wp.constant(VelLimit)
    K_SPow3 = wp.constant(paraPow3)
    K_DSPow3 = wp.constant(3.0 * paraPow3)
    K_SPoly6 = wp.constant(paraPoly6)
    Boundary_Restitution = wp.constant(boundaryRestitution)
    Boundary_Tangential_Retention = wp.constant(boundaryTangentialRetention)
    Boundary_Contact_Eps = wp.constant(
        boundaryContactEpsScale * sim.smoothing_length
    )
    Boundary_Disturbance = wp.constant(boundaryDisturbance)

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
    sim.block_counts = wp.vec3i(*(int(count) for count in counts))
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
            sim.block_counts
        ]
    )


# //////////////////////////////////////////////////////////////////////////////
# calculate function

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
def density_kernel(xyz: wp.vec3, smoothing_length: float):
    # calculate distance
    distance = wp.dot(xyz, xyz)

    return wp.max(cube(square(smoothing_length) - distance), 0.0)


@wp.func
def diff_pressure_kernel(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < smoothing_length:
        # calculate terms of kernel
        term_1 = -xyz / distance
        term_2 = (neighbor_pressure + pressure) / (2.0 * neighbor_rho)
        term_3 = square(smoothing_length - distance)
        return term_1 * term_2 * term_3
    else:
        return wp.vec3()


@wp.func
def diff_viscous_kernel(
    xyz: wp.vec3, v: wp.vec3, neighbor_v: wp.vec3, neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    # calculate terms of kernel
    if distance < smoothing_length:
        term_1 = (neighbor_v - v) / neighbor_rho
        term_2 = smoothing_length - distance
        return term_1 * term_2
    else:
        return wp.vec3()

@wp.func
def Poly6(dst: float, radius: float):
    if dst < radius: 
        # scale = 315.0/(64.0 * wp.pi * wp.pow(wp.abs(radius), 9.0))
        return cube(square(radius) - square(dst)) * K_SPoly6
    return 0.0

@wp.func
def DPow3(dst: float, radius: float):
    if dst < radius: 
        # SPow3Grad = 45.0/(wp.pi * wp.pow(radius, 6.0)) 
        return -1.0 * square(radius - dst) * K_DSPow3
    return 0.0

@wp.func
def Pow3(dst: float, radius: float):
    if dst < radius: 
        # SPow3 = 15.0 / (wp.pi * wp.pow(radius, 6.0)) 
        return cube(radius - dst) * K_SPow3
    return 0.0


@wp.func
def apply_boundary_collision_velocity(
    position: wp.vec3,
    boundary: wp.vec3,
    reconstructed_velocity: wp.vec3,
    incoming_velocity: wp.vec3,
):
    velocity = reconstructed_velocity
    contact_range = Boundary_Disturbance + Boundary_Contact_Eps

    for axis in range(3):
        normal = wp.vec3(0.0, 0.0, 0.0)
        if axis == 0:
            if position[0] <= contact_range:
                normal = wp.vec3(1.0, 0.0, 0.0)
            elif position[0] >= boundary[0] - contact_range:
                normal = wp.vec3(-1.0, 0.0, 0.0)
        elif axis == 1:
            if position[1] <= contact_range:
                normal = wp.vec3(0.0, 1.0, 0.0)
            elif position[1] >= boundary[1] - contact_range:
                normal = wp.vec3(0.0, -1.0, 0.0)
        else:
            if position[2] <= contact_range:
                normal = wp.vec3(0.0, 0.0, 1.0)
            elif position[2] >= boundary[2] - contact_range:
                normal = wp.vec3(0.0, 0.0, -1.0)

        incoming_normal_velocity = wp.dot(incoming_velocity, normal)
        if incoming_normal_velocity < 0.0:
            reconstructed_normal_velocity = wp.dot(velocity, normal)
            tangential_velocity = velocity - reconstructed_normal_velocity * normal
            target_normal_velocity = -Boundary_Restitution * incoming_normal_velocity
            final_normal_velocity = wp.max(reconstructed_normal_velocity, target_normal_velocity)
            velocity = (
                Boundary_Tangential_Retention * tangential_velocity
                + final_normal_velocity * normal
            )

    return velocity


@wp.func
def apply_boundary(pos: wp.vec3, boundary: wp.vec3, tid: int):

    width = boundary[0]
    high = boundary[1]
    length = boundary[2]

    state = wp.rand_init(123, tid)
    disturbance = Boundary_Disturbance * wp.vec3(
        wp.abs(wp.randf(state)),
        wp.abs(wp.randf(state)),
        wp.abs(wp.randf(state)),
    )

    # clamping
    # clamp pos left
    if pos[0] < 0.0:
        pos = wp.vec3(disturbance[0], pos[1], pos[2])

    # clamp x right
    if pos[0] > width:
        pos = wp.vec3(width - disturbance[0], pos[1], pos[2])

    # clamp y bot
    if pos[1] < 0.0:
        pos = wp.vec3(pos[0], disturbance[1], pos[2])

    # clamp y up
    if pos[1] > high:
        pos = wp.vec3(pos[0], high - disturbance[1], pos[2])

    # clamp z left
    if pos[2] < 0.0:
        pos = wp.vec3(pos[0], pos[1], disturbance[2])

    # clamp z right
    if pos[2] > length:
        pos = wp.vec3(pos[0], pos[1], length - disturbance[2])

    return pos
 
