"""Warp translation of PositionBasedDynamics' official ``FluidDemo``.

The main kernels, simulation state, and complete step order live together in
this module. Mathematical Warp functions and per-step Host result handling live
in :mod:`mmpbf_functions`; static setup lives in :mod:`mmpbf_initialization`.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

try:
    from . import mmpbf_functions as functions
    from . import mmpbf_initialization as initialization
except ImportError:
    import mmpbf_functions as functions
    import mmpbf_initialization as initialization


@wp.kernel
def compute_boundary_volumes(
    boundary_grid: wp.uint64,
    boundary_positions: wp.array(dtype=wp.vec3),
    support_radius: float,
    boundary_volumes: wp.array(dtype=float),
):
    """Compute Akinci pseudo-volumes exactly as ``FluidModel::initModel``."""
    i = wp.tid()
    xi = boundary_positions[i]
    kernel_sum = float(0.0)
    query = wp.hash_grid_query(boundary_grid, xi, support_radius)
    for j in query:
        distance = wp.length(xi - boundary_positions[j])
        if distance <= support_radius:
            kernel_sum += functions.cubic_kernel(distance, support_radius)

    volume = float(0.0)
    if kernel_sum > 0.0:
        volume = 1.0 / kernel_sum
    boundary_volumes[i] = volume


@wp.kernel
def predict_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    old_positions: wp.array(dtype=wp.vec3),
    last_positions: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
):
    """PBD history update followed by semi-implicit Euler integration."""
    i = wp.tid()
    last_positions[i] = old_positions[i]
    old_positions[i] = positions[i]
    velocity = velocities[i] + dt * gravity
    velocities[i] = velocity
    predicted_positions[i] = positions[i] + dt * velocity


@wp.kernel
def cache_neighbors(
    fluid_grid: wp.uint64,
    boundary_grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    boundary_positions: wp.array(dtype=wp.vec3),
    support_radius: float,
    max_fluid_neighbors: int,
    max_boundary_neighbors: int,
    fluid_neighbor_counts: wp.array(dtype=int),
    fluid_neighbor_indices: wp.array(dtype=int),
    boundary_neighbor_counts: wp.array(dtype=int),
    boundary_neighbor_indices: wp.array(dtype=int),
    overflow: wp.array(dtype=int),
):
    """Cache the single neighbor search reused by all five PBF iterations."""
    # Process nearby particles together while retaining original array indices.
    i = wp.hash_grid_point_id(fluid_grid, wp.tid())
    xi = positions[i]
    support_radius_squared = support_radius * support_radius

    fluid_count = int(0)
    fluid_query = wp.hash_grid_query(fluid_grid, xi, support_radius)
    for j in fluid_query:
        if j != i and wp.length_sq(xi - positions[j]) <= support_radius_squared:
            if fluid_count < max_fluid_neighbors:
                fluid_neighbor_indices[i * max_fluid_neighbors + fluid_count] = j
            else:
                wp.atomic_max(overflow, 0, 1)
            fluid_count += 1
    fluid_neighbor_counts[i] = wp.min(fluid_count, max_fluid_neighbors)

    boundary_count = int(0)
    boundary_query = wp.hash_grid_query(boundary_grid, xi, support_radius)
    for j in boundary_query:
        if wp.length_sq(xi - boundary_positions[j]) <= support_radius_squared:
            if boundary_count < max_boundary_neighbors:
                boundary_neighbor_indices[i * max_boundary_neighbors + boundary_count] = j
            else:
                wp.atomic_max(overflow, 0, 2)
            boundary_count += 1
    boundary_neighbor_counts[i] = wp.min(boundary_count, max_boundary_neighbors)


@wp.kernel
def compute_lambdas(
    positions: wp.array(dtype=wp.vec3),
    boundary_positions: wp.array(dtype=wp.vec3),
    boundary_volumes: wp.array(dtype=float),
    fluid_volume: float,
    support_radius: float,
    max_fluid_neighbors: int,
    max_boundary_neighbors: int,
    fluid_neighbor_counts: wp.array(dtype=int),
    fluid_neighbor_indices: wp.array(dtype=int),
    boundary_neighbor_counts: wp.array(dtype=int),
    boundary_neighbor_indices: wp.array(dtype=int),
    lambdas: wp.array(dtype=float),
    normalized_densities: wp.array(dtype=float),
    density_error_sum: wp.array(dtype=float),
):
    """Translate PBD density and Lagrange-multiplier evaluation literally."""
    i = wp.tid()
    xi = positions[i]
    density = fluid_volume * functions.cubic_kernel(0.0, support_radius)
    gradient_i = wp.vec3(0.0, 0.0, 0.0)
    squared_gradient_sum = float(0.0)

    for offset in range(fluid_neighbor_counts[i]):
        j = fluid_neighbor_indices[i * max_fluid_neighbors + offset]
        xij = xi - positions[j]
        density += fluid_volume * functions.cubic_kernel(wp.length(xij), support_radius)
        gradient_j = -fluid_volume * functions.cubic_kernel_gradient(xij, support_radius)
        squared_gradient_sum += wp.dot(gradient_j, gradient_j)
        gradient_i -= gradient_j

    for offset in range(boundary_neighbor_counts[i]):
        j = boundary_neighbor_indices[i * max_boundary_neighbors + offset]
        xij = xi - boundary_positions[j]
        boundary_volume = boundary_volumes[j]
        density += boundary_volume * functions.cubic_kernel(wp.length(xij), support_radius)
        gradient_j = -boundary_volume * functions.cubic_kernel_gradient(xij, support_radius)
        squared_gradient_sum += wp.dot(gradient_j, gradient_j)
        gradient_i -= gradient_j

    squared_gradient_sum += wp.dot(gradient_i, gradient_i)
    constraint = wp.max(density - 1.0, 0.0)
    lambda_value = float(0.0)
    if constraint != 0.0:
        lambda_value = -constraint / (squared_gradient_sum + 1.0e-6)

    lambdas[i] = lambda_value
    normalized_densities[i] = density
    wp.atomic_add(density_error_sum, 0, constraint)


@wp.kernel
def compute_position_corrections(
    positions: wp.array(dtype=wp.vec3),
    boundary_positions: wp.array(dtype=wp.vec3),
    boundary_volumes: wp.array(dtype=float),
    lambdas: wp.array(dtype=float),
    fluid_volume: float,
    support_radius: float,
    max_fluid_neighbors: int,
    max_boundary_neighbors: int,
    fluid_neighbor_counts: wp.array(dtype=int),
    fluid_neighbor_indices: wp.array(dtype=int),
    boundary_neighbor_counts: wp.array(dtype=int),
    boundary_neighbor_indices: wp.array(dtype=int),
    position_corrections: wp.array(dtype=wp.vec3),
):
    """Translate ``PositionBasedFluids::solveDensityConstraint``."""
    i = wp.tid()
    xi = positions[i]
    lambda_i = lambdas[i]
    correction = wp.vec3(0.0, 0.0, 0.0)

    for offset in range(fluid_neighbor_counts[i]):
        j = fluid_neighbor_indices[i * max_fluid_neighbors + offset]
        gradient_j = -fluid_volume * functions.cubic_kernel_gradient(
            xi - positions[j], support_radius
        )
        correction -= (lambda_i + lambdas[j]) * gradient_j

    for offset in range(boundary_neighbor_counts[i]):
        j = boundary_neighbor_indices[i * max_boundary_neighbors + offset]
        gradient_j = -boundary_volumes[j] * functions.cubic_kernel_gradient(
            xi - boundary_positions[j], support_radius
        )
        correction -= lambda_i * gradient_j

    position_corrections[i] = correction


@wp.kernel
def apply_position_corrections(
    position_corrections: wp.array(dtype=wp.vec3),
    positions: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    positions[i] += position_corrections[i]


@wp.kernel
def reconstruct_velocities(
    positions: wp.array(dtype=wp.vec3),
    old_positions: wp.array(dtype=wp.vec3),
    last_positions: wp.array(dtype=wp.vec3),
    dt: float,
    velocity_update_method: int,
    output_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    """PBD velocity methods: 0 is first order and 1 is second order."""
    i = wp.tid()
    position = positions[i]
    if velocity_update_method == 0:
        velocities[i] = (position - old_positions[i]) / dt
    else:
        velocities[i] = (
            1.5 * position - 2.0 * old_positions[i] + 0.5 * last_positions[i]
        ) / dt
    output_positions[i] = position


@wp.kernel
def scale_densities(
    normalized_densities: wp.array(dtype=float),
    rest_density: float,
    densities: wp.array(dtype=float),
):
    """Expose the physical density stored by PBD's ``FluidModel``."""
    i = wp.tid()
    densities[i] = rest_density * normalized_densities[i]


@wp.kernel
def compute_xsph_viscosity(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    normalized_densities: wp.array(dtype=float),
    fluid_volume: float,
    viscosity: float,
    max_particle_speed: float,
    support_radius: float,
    max_fluid_neighbors: int,
    fluid_neighbor_counts: wp.array(dtype=int),
    fluid_neighbor_indices: wp.array(dtype=int),
    output_velocities: wp.array(dtype=wp.vec3),
):
    """Apply FluidDemo XSPH, then clamp the final particle-speed magnitude."""
    i = wp.tid()
    xi = positions[i]
    vi = velocities[i]
    result = vi

    for offset in range(fluid_neighbor_counts[i]):
        j = fluid_neighbor_indices[i * max_fluid_neighbors + offset]
        density_j = wp.max(normalized_densities[j], 1.0e-12)
        result -= (
            viscosity
            * (fluid_volume / density_j)
            * (vi - velocities[j])
            * functions.cubic_kernel(wp.length(xi - positions[j]), support_radius)
        )

    speed_squared = wp.dot(result, result)
    max_speed_squared = max_particle_speed * max_particle_speed
    if speed_squared > max_speed_squared:
        result *= max_particle_speed / wp.sqrt(speed_squared)

    output_velocities[i] = result


@wp.kernel
def reduce_cfl_speed_squared(
    velocities: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    max_speed_squared: wp.array(dtype=float),
):
    """Reduce ``|v + a*h|^2`` for FluidDemo's beginning-of-step CFL update."""
    i = wp.tid()
    estimated_velocity = velocities[i] + dt * gravity
    wp.atomic_max(max_speed_squared, 0, wp.dot(estimated_velocity, estimated_velocity))


@wp.kernel
def reduce_boundary_violation(
    positions: wp.array(dtype=wp.vec3),
    boundary_min: wp.vec3,
    boundary_max: wp.vec3,
    max_violation: wp.array(dtype=float),
):
    """Diagnostic only; the official solver performs no explicit clamping."""
    i = wp.tid()
    position = positions[i]
    violation = wp.max(boundary_min[0] - position[0], position[0] - boundary_max[0])
    violation = wp.max(
        violation,
        wp.max(boundary_min[1] - position[1], position[1] - boundary_max[1]),
    )
    violation = wp.max(
        violation,
        wp.max(boundary_min[2] - position[2], position[2] - boundary_max[2]),
    )
    wp.atomic_max(max_violation, 0, wp.max(violation, 0.0))


@dataclass(frozen=True)
class MMPBFConfig:
    """Parameters from ``Demos/FluidDemo`` with test-size overrides."""

    particle_radius: float = 0.025
    fluid_width: int = 15
    fluid_height: int = 20
    fluid_depth: int = 15
    container_height: float = 5.0

    rest_density: float = 1000.0
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    initial_time_step: float = 0.0025
    cfl_min_time_step: float = 0.0001
    cfl_max_time_step: float = 0.005
    cfl_factor: float = 1.0

    pressure_iterations: int = 5
    velocity_update_method: int = 0
    xsph_viscosity: float = 0.02
    max_particle_speed: float = 4.0

    max_fluid_neighbors: int = 128
    max_boundary_neighbors: int = 128

    @classmethod
    def high_resolution(cls) -> "MMPBFConfig":
        """Return the 59,616-fluid-particle version of the official scene."""
        return cls(
            particle_radius=2.0 / 185.0,
            fluid_width=30,
            fluid_height=100,
            fluid_depth=30,
        )

    def __post_init__(self):
        if self.particle_radius <= 0.0:
            raise ValueError("particle_radius must be positive")
        if self.fluid_width < 1 or self.fluid_height < 1 or self.fluid_depth < 1:
            raise ValueError("fluid lattice dimensions must be positive")
        if self.container_height <= 0.0:
            raise ValueError("container_height must be positive")
        if self.rest_density <= 0.0:
            raise ValueError("rest_density must be positive")
        if self.initial_time_step <= 0.0 or self.cfl_min_time_step <= 0.0:
            raise ValueError("time steps must be positive")
        if self.cfl_max_time_step < self.cfl_min_time_step:
            raise ValueError("CFL maximum time step must not be smaller than its minimum")
        if self.cfl_factor <= 0.0:
            raise ValueError("cfl_factor must be positive")
        if self.pressure_iterations < 1:
            raise ValueError("pressure_iterations must be positive")
        if self.velocity_update_method not in (0, 1):
            raise ValueError("velocity_update_method must be 0 (first order) or 1 (second order)")
        if self.xsph_viscosity < 0.0:
            raise ValueError("xsph_viscosity must be non-negative")
        if self.max_particle_speed <= 0.0:
            raise ValueError("max_particle_speed must be positive")
        if self.max_fluid_neighbors < 1 or self.max_boundary_neighbors < 1:
            raise ValueError("neighbor capacities must be positive")


class Example:
    """Simulation object consumed by the existing billboard frontend."""

    def __init__(
        self,
        config: MMPBFConfig | None = None,
        verbose: bool = False,
        device=None,
    ):
        self.config = config or MMPBFConfig()
        self.verbose = verbose
        self.device = wp.get_device(device)
        self.renderer = None
        self.speed_color_mid = 2.0
        self.speed_color_max = 6.0

        initialization.initialize_static_data(self)
        wp.launch(
            compute_boundary_volumes,
            dim=self.boundary_n,
            inputs=[self.boundary_grid.id, self.boundary_positions, self.support_radius],
            outputs=[self.boundary_volumes],
            device=self.device,
        )
        initialization.finalize_initialization(self)

    def step(self):
        """Advance exactly one complete PBF physics step."""
        with wp.ScopedTimer("sub-step", synchronize=True):
            # Prologue
            with wp.ScopedTimer("Prologue", active=self.verbose):
                # FluidDemo stores h before its CFL update. The new value is
                # used by the next substep; this one continues with saved dt.
                dt = self.current_dt
                self.max_speed_squared.fill_(0.1)
                wp.launch(
                    reduce_cfl_speed_squared,
                    dim=self.n,
                    inputs=[self.v, wp.vec3(*self.config.gravity), dt],
                    outputs=[self.max_speed_squared],
                    device=self.device,
                )
                functions.update_cfl_from_reduction(self, dt)

                wp.launch(
                    predict_positions,
                    dim=self.n,
                    inputs=[self.pos, self.v, wp.vec3(*self.config.gravity), dt],
                    outputs=[
                        self.old_positions,
                        self.last_positions,
                        self.predicted_positions,
                    ],
                    device=self.device,
                )

                self.fluid_grid.build(self.predicted_positions, self.support_radius)
                self.neighbor_overflow.zero_()
                wp.launch(
                    cache_neighbors,
                    dim=self.n,
                    inputs=[
                        self.fluid_grid.id,
                        self.boundary_grid.id,
                        self.predicted_positions,
                        self.boundary_positions,
                        self.support_radius,
                        self.config.max_fluid_neighbors,
                        self.config.max_boundary_neighbors,
                    ],
                    outputs=[
                        self.fluid_neighbor_counts,
                        self.fluid_neighbor_indices,
                        self.boundary_neighbor_counts,
                        self.boundary_neighbor_indices,
                        self.neighbor_overflow,
                    ],
                    device=self.device,
                )
                functions.check_neighbor_overflow(self)

            # Solver
            with wp.ScopedTimer("Solver", active=self.verbose):
                for _ in range(self.config.pressure_iterations):
                    self.density_error_sum.zero_()
                    wp.launch(
                        compute_lambdas,
                        dim=self.n,
                        inputs=[
                            self.predicted_positions,
                            self.boundary_positions,
                            self.boundary_volumes,
                            self.fluid_volume,
                            self.support_radius,
                            self.config.max_fluid_neighbors,
                            self.config.max_boundary_neighbors,
                            self.fluid_neighbor_counts,
                            self.fluid_neighbor_indices,
                            self.boundary_neighbor_counts,
                            self.boundary_neighbor_indices,
                        ],
                        outputs=[
                            self.lambdas,
                            self.normalized_densities,
                            self.density_error_sum,
                        ],
                        device=self.device,
                    )
                    wp.launch(
                        compute_position_corrections,
                        dim=self.n,
                        inputs=[
                            self.predicted_positions,
                            self.boundary_positions,
                            self.boundary_volumes,
                            self.lambdas,
                            self.fluid_volume,
                            self.support_radius,
                            self.config.max_fluid_neighbors,
                            self.config.max_boundary_neighbors,
                            self.fluid_neighbor_counts,
                            self.fluid_neighbor_indices,
                            self.boundary_neighbor_counts,
                            self.boundary_neighbor_indices,
                        ],
                        outputs=[self.position_corrections],
                        device=self.device,
                    )
                    wp.launch(
                        apply_position_corrections,
                        dim=self.n,
                        inputs=[self.position_corrections],
                        outputs=[self.predicted_positions],
                        device=self.device,
                    )

                functions.update_pressure_diagnostics(self)
                wp.launch(
                    scale_densities,
                    dim=self.n,
                    inputs=[self.normalized_densities, self.config.rest_density],
                    outputs=[self.densities],
                    device=self.device,
                )

            # Epilogue
            with wp.ScopedTimer("Epilogue", active=self.verbose):
                wp.launch(
                    reconstruct_velocities,
                    dim=self.n,
                    inputs=[
                        self.predicted_positions,
                        self.old_positions,
                        self.last_positions,
                        dt,
                        self.config.velocity_update_method,
                    ],
                    outputs=[self.pos, self.v],
                    device=self.device,
                )
                wp.launch(
                    compute_xsph_viscosity,
                    dim=self.n,
                    inputs=[
                        self.pos,
                        self.v,
                        self.normalized_densities,
                        self.fluid_volume,
                        self.config.xsph_viscosity,
                        self.config.max_particle_speed,
                        self.support_radius,
                        self.config.max_fluid_neighbors,
                        self.fluid_neighbor_counts,
                        self.fluid_neighbor_indices,
                    ],
                    outputs=[self.xsph_velocities],
                    device=self.device,
                )
                wp.copy(self.v, self.xsph_velocities)

                self.max_boundary_violation_buffer.zero_()
                wp.launch(
                    reduce_boundary_violation,
                    dim=self.n,
                    inputs=[self.pos, self.boundary_min, self.boundary_max],
                    outputs=[self.max_boundary_violation_buffer],
                    device=self.device,
                )
                functions.finish_substep(self, dt)

    def render(self):
        """Render the current fluid state with the attached billboard renderer."""
        if self.renderer is None:
            return

        with wp.ScopedTimer("render",active=self.verbose):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_billboards(
                name="points",
                points=self.pos,
                velocities=self.v,
                radius=self.particle_radius,
                speed_color_mid=self.speed_color_mid,
                speed_color_max=self.speed_color_max,
            )
            self.renderer.end_frame()
