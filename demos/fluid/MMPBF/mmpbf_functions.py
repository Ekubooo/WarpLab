"""Warp functions and lightweight Host helpers for MMPBF."""

import math

import warp as wp


@wp.func
def cubic_kernel(distance: float, support_radius: float):
    """PBD ``CubicKernel::W`` in three dimensions."""
    value = float(0.0)
    q = distance / support_radius
    if q <= 1.0:
        h3 = support_radius * support_radius * support_radius
        coefficient = 8.0 / (wp.pi * h3)
        if q <= 0.5:
            q2 = q * q
            value = coefficient * (6.0 * q2 * q - 6.0 * q2 + 1.0)
        else:
            one_minus_q = 1.0 - q
            value = coefficient * 2.0 * one_minus_q * one_minus_q * one_minus_q
    return value


@wp.func
def cubic_kernel_gradient(displacement: wp.vec3, support_radius: float):
    """PBD ``CubicKernel::gradW`` in three dimensions."""
    gradient = wp.vec3(0.0, 0.0, 0.0)
    distance = wp.length(displacement)
    q = distance / support_radius
    if distance > 1.0e-6 and q <= 1.0:
        h3 = support_radius * support_radius * support_radius
        coefficient = 48.0 / (wp.pi * h3)
        grad_q = displacement / (distance * support_radius)
        if q <= 0.5:
            gradient = coefficient * q * (3.0 * q - 2.0) * grad_q
        else:
            one_minus_q = 1.0 - q
            gradient = -coefficient * one_minus_q * one_minus_q * grad_q
    return gradient


def update_cfl_from_reduction(simulation, dt: float):
    """Update the next substep size from the completed GPU speed reduction."""
    maximum_speed_squared = float(simulation.max_speed_squared.numpy()[0])
    next_dt = (
        simulation.config.cfl_factor
        * 0.4
        * simulation.particle_diameter
        / math.sqrt(maximum_speed_squared)
    )
    simulation.current_dt = min(
        max(next_dt, simulation.config.cfl_min_time_step),
        simulation.config.cfl_max_time_step,
    )
    simulation.last_dt = dt


def check_neighbor_overflow(simulation):
    """Raise instead of silently using a truncated neighbor list."""
    overflow = int(simulation.neighbor_overflow.numpy()[0])
    if overflow:
        neighbor_type = "boundary" if overflow == 2 else "fluid"
        raise RuntimeError(
            f"{neighbor_type} neighbor capacity exceeded; increase the corresponding "
            "MMPBFConfig capacity"
        )


def update_pressure_diagnostics(simulation):
    """Expose the error from the final fixed PBF iteration."""
    simulation.last_pressure_iterations = simulation.config.pressure_iterations
    simulation.last_density_error_percent = (
        100.0 * float(simulation.density_error_sum.numpy()[0]) / simulation.n
    )


def finish_substep(simulation, dt: float):
    """Read the final diagnostic scalar and advance recorded simulation time."""
    simulation.last_boundary_violation = float(
        simulation.max_boundary_violation_buffer.numpy()[0]
    )
    simulation.sim_time += dt
