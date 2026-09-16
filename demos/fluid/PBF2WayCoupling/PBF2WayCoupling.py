"""Warp PBF + Akinci particle boundaries and six-DOF mesh rigid bodies.

All simulation kernels and their launch order live here. Numerical reference:
SPlisHSPlasH f3f677140761db7637b5443beb54f19f1f835ed4 (MIT).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

try:
    from . import coupling_functions as fn
    from . import coupling_initialization as init
except ImportError:
    import coupling_functions as fn
    import coupling_initialization as init


HASH_GRID_DIMS = (128, 160, 128)
CONTACT_MANIFOLD_POINTS = 8
CONTACT_NORMAL_DUPLICATE_COS = 0.8660254  # cos(30 degrees)


# Device state: one row per rigid body, boundary sample, or fluid particle.


@wp.struct
class RigidState:
    """World-space pose/velocity with inverse inertia stored in body space."""

    position: wp.array(dtype=wp.vec3)
    rotation: wp.array(dtype=wp.quat)
    velocity: wp.array(dtype=wp.vec3)
    omega: wp.array(dtype=wp.vec3)
    inverse_mass: wp.array(dtype=float)
    inverse_inertia: wp.array(dtype=wp.mat33)
    force: wp.array(dtype=wp.vec3)
    torque: wp.array(dtype=wp.vec3)
    mesh: wp.array(dtype=wp.uint64)
    inverse_inertia_world: wp.array(dtype=wp.mat33)
    fault: wp.array(dtype=int)
    lower: wp.array(dtype=wp.vec3)
    upper: wp.array(dtype=wp.vec3)
    restitution: wp.array(dtype=float)
    friction: wp.array(dtype=float)
    wall: wp.array(dtype=int)


@wp.struct
class BoundaryState:
    """Body-local samples and their current world-space positions/velocities."""

    local: wp.array(dtype=wp.vec3)
    position: wp.array(dtype=wp.vec3)
    velocity: wp.array(dtype=wp.vec3)
    volume: wp.array(dtype=float)
    body: wp.array(dtype=int)


@wp.struct
class Neighbors:
    """Fixed-capacity neighbor lists; overflow slots identify fluid/boundary lists."""

    fault: wp.array(dtype=int)
    fluid_count: wp.array(dtype=int)
    boundary_count: wp.array(dtype=int)
    fluid: wp.array2d(dtype=int)
    boundary: wp.array2d(dtype=int)
    overflow: wp.array(dtype=int)


@wp.struct
class Contacts:
    """Raw contact candidates plus compact solver manifolds."""

    count: wp.array(dtype=int)
    candidate_count: wp.array(dtype=int)
    candidate_a: wp.array(dtype=int)
    candidate_b: wp.array(dtype=int)
    candidate_source: wp.array(dtype=int)
    candidate_point: wp.array(dtype=wp.vec3)
    candidate_normal: wp.array(dtype=wp.vec3)
    candidate_gap: wp.array(dtype=float)
    candidate_target: wp.array(dtype=float)
    selected_candidate: wp.array(dtype=int)
    selection_key: wp.array(dtype=wp.int64)
    a: wp.array(dtype=int)
    b: wp.array(dtype=int)
    point: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)
    gap: wp.array(dtype=float)
    target: wp.array(dtype=float)
    normal_impulse: wp.array(dtype=float)
    tangent_impulse: wp.array(dtype=wp.vec3)
    offset_a: wp.array(dtype=wp.vec3)
    offset_b: wp.array(dtype=wp.vec3)
    normal_mass: wp.array(dtype=float)
    overflow: wp.array(dtype=int)


# Boundary initialization and motion.


@wp.kernel
def update_boundary(rigid: RigidState, boundary: BoundaryState):
    """Transform each sample and evaluate v + omega cross world_offset."""
    if rigid.fault[0] != 0:
        return
    i = wp.tid()
    body_index = boundary.body[i]
    world_offset = wp.quat_rotate(rigid.rotation[body_index], boundary.local[i])
    boundary.position[i] = rigid.position[body_index] + world_offset
    boundary.velocity[i] = fn.point_velocity(
        rigid.velocity[body_index], rigid.omega[body_index], world_offset
    )


@wp.kernel
def boundary_volumes(grid: wp.uint64, h: float, rigid: RigidState, boundary: BoundaryState):
    """Compute Akinci pseudo-volumes within each permitted boundary group."""
    i = wp.tid()
    body_index = boundary.body[i]
    kernel_sum = float(0.0)
    for j in wp.hash_grid_query(grid, boundary.position[i], h):
        neighbor_body_index = boundary.body[j]
        if body_index == neighbor_body_index or (
            rigid.inverse_mass[body_index] == 0.0 and rigid.inverse_mass[neighbor_body_index] == 0.0
        ):
            kernel_sum += fn.poly6(wp.length(boundary.position[i] - boundary.position[j]), h)
    boundary.volume[i] = 1.0 / kernel_sum


@wp.kernel
def exclude_solid_particles(
    x: wp.array(dtype=wp.vec3), rigid: RigidState, radius: float, keep: wp.array(dtype=int)
):
    i = wp.tid()
    is_fluid = int(1)
    for body_index in range(rigid.position.shape[0]):
        if rigid.wall[body_index] == 0:
            local_position = wp.quat_rotate_inv(
                rigid.rotation[body_index], x[i] - rigid.position[body_index]
            )
            query = wp.mesh_query_point_sign_normal(rigid.mesh[body_index], local_position, 1000.0)
            if query.result:
                closest = wp.mesh_eval_position(
                    rigid.mesh[body_index], query.face, query.u, query.v
                )
                if query.sign * wp.length(local_position - closest) < radius:
                    is_fluid = 0
    keep[i] = is_fluid


# Fluid prediction, pressure projection and viscosity.


@wp.kernel
def begin_substep(rigid: RigidState, contacts: Contacts):
    if rigid.fault[0] != 0:
        return
    body = wp.tid()
    rigid.force[body] = wp.vec3(0.0)
    rigid.torque[body] = wp.vec3(0.0)
    if body == 0:
        contacts.count[0] = 0
        contacts.candidate_count[0] = 0


@wp.kernel
def predict(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    fault: wp.array(dtype=int),
):
    if fault[0] != 0:
        return
    i = wp.tid()
    old[i] = x[i]
    v[i] += dt * gravity
    x[i] += dt * v[i]


@wp.kernel
def reorder_fluid(
    fluid_grid: wp.uint64,
    source_positions: wp.array(dtype=wp.vec3),
    source_velocities: wp.array(dtype=wp.vec3),
    source_old_positions: wp.array(dtype=wp.vec3),
    source_particle_ids: wp.array(dtype=int),
    target_positions: wp.array(dtype=wp.vec3),
    target_velocities: wp.array(dtype=wp.vec3),
    target_old_positions: wp.array(dtype=wp.vec3),
    target_particle_ids: wp.array(dtype=int),
    old_to_sorted: wp.array(dtype=int),
    fault: wp.array(dtype=int),
):
    """Gather persistent fluid state into hash order and build its inverse map."""
    sorted_index = wp.tid()
    source_index = sorted_index
    if fault[0] == 0:
        source_index = wp.hash_grid_point_id(fluid_grid, sorted_index)
    target_positions[sorted_index] = source_positions[source_index]
    target_velocities[sorted_index] = source_velocities[source_index]
    target_old_positions[sorted_index] = source_old_positions[source_index]
    target_particle_ids[sorted_index] = source_particle_ids[source_index]
    old_to_sorted[source_index] = sorted_index


@wp.kernel
def cache_neighbors(
    fluid_grid: wp.uint64,
    boundary_grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    old_to_sorted: wp.array(dtype=int),
    boundary: BoundaryState,
    h: float,
    neighbors: Neighbors,
):
    if neighbors.fault[0] != 0:
        return
    i = wp.tid()
    fluid_count = int(0)
    boundary_count = int(0)
    for old_j in wp.hash_grid_query(fluid_grid, x[i], h):
        j = old_to_sorted[old_j]
        if j != i and wp.length_sq(x[i] - x[j]) < h * h:
            if fluid_count < neighbors.fluid.shape[0]:
                neighbors.fluid[fluid_count, i] = j
            else:
                wp.atomic_max(neighbors.overflow, 0, 1)
            fluid_count += 1
    for j in wp.hash_grid_query(boundary_grid, x[i], h):
        if wp.length_sq(x[i] - boundary.position[j]) < h * h:
            if boundary_count < neighbors.boundary.shape[0]:
                neighbors.boundary[boundary_count, i] = j
            else:
                wp.atomic_max(neighbors.overflow, 1, 1)
            boundary_count += 1
    neighbors.fluid_count[i] = wp.min(fluid_count, neighbors.fluid.shape[0])
    neighbors.boundary_count[i] = wp.min(boundary_count, neighbors.boundary.shape[0])


@wp.kernel
def density_lambda(
    x: wp.array(dtype=wp.vec3),
    boundary: BoundaryState,
    neighbors: Neighbors,
    volume: float,
    h: float,
    density: wp.array(dtype=float),
    lambdas: wp.array(dtype=float),
):
    """Evaluate normalized density and the PBF constraint multiplier."""
    if neighbors.fault[0] != 0:
        return
    i = wp.tid()
    normalized_density = volume * fn.poly6(0.0, h)
    gradient_i = wp.vec3(0.0)
    squared_gradient_sum = float(0.0)
    for k in range(neighbors.fluid_count[i]):
        j = neighbors.fluid[k, i]
        displacement = x[i] - x[j]
        normalized_density += volume * fn.poly6(wp.length(displacement), h)
        gradient_j = -volume * fn.spiky_gradient(displacement, h)
        squared_gradient_sum += wp.dot(gradient_j, gradient_j)
        gradient_i -= gradient_j
    for k in range(neighbors.boundary_count[i]):
        j = neighbors.boundary[k, i]
        displacement = x[i] - boundary.position[j]
        normalized_density += boundary.volume[j] * fn.poly6(wp.length(displacement), h)
        # SPlisHSPlasH excludes individual boundary gradient squares.
        gradient_i += boundary.volume[j] * fn.spiky_gradient(displacement, h)
    constraint = wp.max(normalized_density - 1.0, 0.0)
    density[i] = normalized_density
    lambdas[i] = -constraint / (squared_gradient_sum + wp.dot(gradient_i, gradient_i) + 1.0e-6)


@wp.kernel
def pressure_correction(
    x: wp.array(dtype=wp.vec3),
    boundary: BoundaryState,
    rigid: RigidState,
    neighbors: Neighbors,
    volume: float,
    mass: float,
    h: float,
    dt: float,
    two_way: int,
    lambdas: wp.array(dtype=float),
    correction: wp.array(dtype=wp.vec3),
):
    """Compute fluid displacement and accumulate its opposite rigid-body force."""
    if rigid.fault[0] != 0:
        return
    i = wp.tid()
    position_delta = wp.vec3(0.0)
    for k in range(neighbors.fluid_count[i]):
        j = neighbors.fluid[k, i]
        position_delta += (lambdas[i] + lambdas[j]) * volume * fn.spiky_gradient(x[i] - x[j], h)
    for k in range(neighbors.boundary_count[i]):
        j = neighbors.boundary[k, i]
        boundary_delta = (
            lambdas[i] * boundary.volume[j] * fn.spiky_gradient(x[i] - boundary.position[j], h)
        )
        position_delta += boundary_delta
        body_index = boundary.body[j]
        if two_way != 0 and rigid.inverse_mass[body_index] > 0.0:
            force = -mass * boundary_delta / (dt * dt)
            wp.atomic_add(rigid.force, body_index, force)
            wp.atomic_add(
                rigid.torque,
                body_index,
                wp.cross(boundary.position[j] - rigid.position[body_index], force),
            )
    correction[i] = position_delta


@wp.kernel
def apply_correction(
    x: wp.array(dtype=wp.vec3),
    correction: wp.array(dtype=wp.vec3),
    lower: wp.vec3,
    upper: wp.vec3,
    fault: wp.array(dtype=int),
):
    if fault[0] != 0:
        return
    i = wp.tid()
    x[i] = fn.clamp_to_container(x[i] + correction[i], lower, upper)


@wp.kernel
def reconstruct_velocity(
    x: wp.array(dtype=wp.vec3),
    old: wp.array(dtype=wp.vec3),
    dt: float,
    v: wp.array(dtype=wp.vec3),
    lower: wp.vec3,
    upper: wp.vec3,
    wall_damping: float,
    max_speed: float,
    fault: wp.array(dtype=int),
):
    if fault[0] != 0:
        return
    i = wp.tid()
    velocity = (x[i] - old[i]) / dt
    velocity = fn.reflect_wall_velocity(x[i], velocity, lower, upper, wall_damping)
    v[i] = fn.limit_speed(velocity, max_speed)


@wp.kernel
def viscosity(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    boundary: BoundaryState,
    rigid: RigidState,
    neighbors: Neighbors,
    density: wp.array(dtype=float),
    volume: float,
    mass: float,
    h: float,
    coefficient: float,
    boundary_coefficient: float,
    two_way: int,
    acceleration: wp.array(dtype=wp.vec3),
):
    if rigid.fault[0] != 0:
        return
    i = wp.tid()
    viscous_acceleration = wp.vec3(0.0)
    for k in range(neighbors.fluid_count[i]):
        j = neighbors.fluid[k, i]
        displacement = x[i] - x[j]
        viscosity_weight = 10.0 * coefficient * volume / density[j]
        viscous_acceleration += (
            viscosity_weight
            * wp.dot(v[i] - v[j], displacement)
            / (wp.length_sq(displacement) + 0.01 * h * h)
            * fn.spiky_gradient(displacement, h)
        )
    if boundary_coefficient > 0.0:
        for k in range(neighbors.boundary_count[i]):
            j = neighbors.boundary[k, i]
            displacement = x[i] - boundary.position[j]
            viscosity_weight = 10.0 * boundary_coefficient * boundary.volume[j] / density[i]
            boundary_acceleration = (
                viscosity_weight
                * wp.dot(v[i] - boundary.velocity[j], displacement)
                / (wp.length_sq(displacement) + 0.01 * h * h)
                * fn.spiky_gradient(displacement, h)
            )
            viscous_acceleration += boundary_acceleration
            body_index = boundary.body[j]
            if two_way != 0 and rigid.inverse_mass[body_index] > 0.0:
                force = -mass * boundary_acceleration
                wp.atomic_add(rigid.force, body_index, force)
                wp.atomic_add(
                    rigid.torque,
                    body_index,
                    wp.cross(boundary.position[j] - rigid.position[body_index], force),
                )
    acceleration[i] = viscous_acceleration


@wp.kernel
def apply_viscosity(
    v: wp.array(dtype=wp.vec3),
    acceleration: wp.array(dtype=wp.vec3),
    dt: float,
    x: wp.array(dtype=wp.vec3),
    lower: wp.vec3,
    upper: wp.vec3,
    wall_damping: float,
    max_speed: float,
    fault: wp.array(dtype=int),
):
    if fault[0] != 0:
        return
    i = wp.tid()
    velocity = v[i] + dt * acceleration[i]
    velocity = fn.reflect_wall_velocity(x[i], velocity, lower, upper, wall_damping)
    v[i] = fn.limit_speed(velocity, max_speed)


# Rigid-body integration and contact impulses.


@wp.kernel
def integrate_rigid(rigid: RigidState, gravity: wp.vec3, dt: float):
    if rigid.fault[0] != 0:
        return
    body_index = wp.tid()
    if rigid.inverse_mass[body_index] > 0.0:
        rotation = rigid.rotation[body_index]
        inverse_inertia_world = fn.world_inverse_inertia(
            rotation, rigid.inverse_inertia[body_index]
        )
        angular_velocity = wp.vec3(rigid.omega[body_index])
        # Euler's rigid-body equation, including the gyroscopic term.
        angular_velocity += dt * (
            inverse_inertia_world
            * (
                rigid.torque[body_index]
                - wp.cross(angular_velocity, wp.inverse(inverse_inertia_world) * angular_velocity)
            )
        )
        linear_velocity = rigid.velocity[body_index] + dt * (
            gravity + rigid.inverse_mass[body_index] * rigid.force[body_index]
        )
        rigid.position[body_index] += dt * linear_velocity
        rigid.rotation[body_index] = wp.normalize(
            rotation
            + 0.5
            * dt
            * wp.quat(angular_velocity[0], angular_velocity[1], angular_velocity[2], 0.0)
            * rotation
        )
        rigid.velocity[body_index] = linear_velocity
        rigid.omega[body_index] = angular_velocity


@wp.kernel
def detect_contacts(
    rigid: RigidState, boundary: BoundaryState, contacts: Contacts, tolerance: float, dt: float
):
    if rigid.fault[0] != 0:
        return
    i, body_b = wp.tid()
    body_a = boundary.body[i]
    if body_a == body_b or rigid.inverse_mass[body_a] == 0.0:
        return
    point = boundary.position[i]
    local_position = wp.quat_rotate_inv(rigid.rotation[body_b], point - rigid.position[body_b])
    gap = float(1.0e6)
    normal = wp.vec3(0.0)
    if rigid.wall[body_b] != 0:
        # Interior of the box is free space. Choose the nearest interior face.
        for axis in range(3):
            distance = local_position[axis] - rigid.lower[body_b][axis]
            if distance < gap:
                gap = distance
                normal = wp.vec3(0.0)
                normal[axis] = 1.0
            distance = rigid.upper[body_b][axis] - local_position[axis]
            if distance < gap:
                gap = distance
                normal = wp.vec3(0.0)
                normal[axis] = -1.0
    else:
        # Conservative local AABB cull before the mesh BVH query.
        bounds_min = rigid.lower[body_b] - wp.vec3(tolerance)
        bounds_max = rigid.upper[body_b] + wp.vec3(tolerance)
        if (
            local_position[0] < bounds_min[0]
            or local_position[1] < bounds_min[1]
            or local_position[2] < bounds_min[2]
            or local_position[0] > bounds_max[0]
            or local_position[1] > bounds_max[1]
            or local_position[2] > bounds_max[2]
        ):
            return
        query = wp.mesh_query_point_sign_normal(rigid.mesh[body_b], local_position, 1000.0)
        if query.result:
            closest = wp.mesh_eval_position(rigid.mesh[body_b], query.face, query.u, query.v)
            delta = local_position - closest
            distance = wp.length(delta)
            gap = query.sign * distance
            if distance > 1.0e-8:
                normal = query.sign * delta / distance
            else:
                normal = wp.mesh_eval_face_normal(rigid.mesh[body_b], query.face)
            normal = wp.quat_rotate(rigid.rotation[body_b], normal)
    if gap < tolerance:
        contact_index = wp.atomic_add(contacts.candidate_count, 0, 1)
        if contact_index >= contacts.candidate_a.shape[0]:
            wp.atomic_max(contacts.overflow, 0, 1)
            return
        offset_a, offset_b = point - rigid.position[body_a], point - rigid.position[body_b]
        relative_velocity = fn.point_velocity(
            rigid.velocity[body_a], rigid.omega[body_a], offset_a
        ) - fn.point_velocity(rigid.velocity[body_b], rigid.omega[body_b], offset_b)
        normal_velocity = wp.dot(relative_velocity, normal)
        target = -wp.max(gap, 0.0) / dt + 0.2 * wp.max(-gap, 0.0) / dt
        if gap <= 0.0 and normal_velocity < -0.5:
            target = wp.max(
                target,
                -wp.min(rigid.restitution[body_a], rigid.restitution[body_b]) * normal_velocity,
            )
        # Canonical body order makes reciprocal dynamic-body samples compete for
        # the same fixed manifold. Flip the normal to preserve its b-to-a sense.
        canonical_a = body_a
        canonical_b = body_b
        canonical_normal = normal
        if canonical_a > canonical_b:
            canonical_a = body_b
            canonical_b = body_a
            canonical_normal = -normal
        contacts.candidate_a[contact_index] = canonical_a
        contacts.candidate_b[contact_index] = canonical_b
        contacts.candidate_source[contact_index] = i
        contacts.candidate_point[contact_index] = point
        contacts.candidate_normal[contact_index] = canonical_normal
        contacts.candidate_gap[contact_index] = gap
        contacts.candidate_target[contact_index] = target


@wp.kernel
def initialize_contact_manifolds(rigid: RigidState, contacts: Contacts):
    """Clear fixed pair slots and their reduction keys entirely on the device."""
    if rigid.fault[0] != 0:
        return
    selection_index = wp.tid()
    contacts.selected_candidate[selection_index] = -1
    contacts.selection_key[selection_index] = wp.int64(0)


@wp.kernel
def score_contact_manifold_slot(
    rigid: RigidState,
    contacts: Contacts,
    body_count: int,
    particle_radius: float,
    tolerance: float,
    manifold_slot: int,
):
    """Score one candidate per thread and atomically retain the stable pair winner."""
    if rigid.fault[0] != 0:
        return
    candidate = wp.tid()
    if candidate >= wp.min(contacts.candidate_count[0], contacts.candidate_a.shape[0]):
        return
    body_a = contacts.candidate_a[candidate]
    body_b = contacts.candidate_b[candidate]
    if body_a >= body_b:
        return
    selection_index = (
        (body_a * body_count + body_b) * CONTACT_MANIFOLD_POINTS + manifold_slot
    )

    score = float(0.0)
    if manifold_slot == 0:
        # All detected candidates satisfy gap < tolerance, so this score is positive.
        score = tolerance - contacts.candidate_gap[candidate]
    else:
        point = contacts.candidate_point[candidate]
        normal = contacts.candidate_normal[candidate]
        duplicate_distance_sq = particle_radius * particle_radius
        normal_scale_sq = tolerance * tolerance
        minimum_score = float(1.0e30)
        for previous_slot in range(CONTACT_MANIFOLD_POINTS):
            if previous_slot >= manifold_slot:
                break
            selected = contacts.selected_candidate[selection_index - manifold_slot + previous_slot]
            if selected < 0:
                return
            distance_sq = wp.length_sq(point - contacts.candidate_point[selected])
            normal_dot = wp.clamp(
                wp.dot(normal, contacts.candidate_normal[selected]), -1.0, 1.0
            )
            if (
                distance_sq < duplicate_distance_sq
                and normal_dot > CONTACT_NORMAL_DUPLICATE_COS
            ):
                return
            minimum_score = wp.min(
                minimum_score, distance_sq + normal_scale_sq * (1.0 - normal_dot)
            )
        score = minimum_score

    # Pack coverage, penetration depth, and the stable boundary-sample id into a
    # signed 63-bit key. Atomic max applies the deterministic priority after
    # quantization: coverage max, gap min, source min. The fields cover this scene.
    score_quantized = wp.int64(
        wp.clamp(score * 1000000000.0, 1.0, 17179869183.0)
    )
    depth_quantized = wp.int64(
        wp.clamp(
            (tolerance - contacts.candidate_gap[candidate]) * 2000.0,
            1.0,
            1023.0,
        )
    )
    source_tie = wp.int64(
        524287 - wp.min(contacts.candidate_source[candidate], 524287)
    )
    key = (
        score_quantized * wp.int64(536870912)
        + depth_quantized * wp.int64(524288)
        + source_tie
    )
    wp.atomic_max(contacts.selection_key, selection_index, key)


@wp.kernel
def resolve_contact_manifold_slot(
    rigid: RigidState,
    contacts: Contacts,
    body_count: int,
    manifold_slot: int,
):
    """Resolve the atomic key back to its candidate using the stable sample id."""
    if rigid.fault[0] != 0:
        return
    candidate = wp.tid()
    if candidate >= wp.min(contacts.candidate_count[0], contacts.candidate_a.shape[0]):
        return
    body_a = contacts.candidate_a[candidate]
    body_b = contacts.candidate_b[candidate]
    if body_a >= body_b:
        return
    selection_index = (
        (body_a * body_count + body_b) * CONTACT_MANIFOLD_POINTS + manifold_slot
    )
    key = contacts.selection_key[selection_index]
    low_word = key - (key // wp.int64(524288)) * wp.int64(524288)
    winner_source = 524287 - int(low_word)
    if contacts.candidate_source[candidate] == winner_source:
        contacts.selected_candidate[selection_index] = candidate


@wp.kernel
def compact_contact_manifolds(
    rigid: RigidState,
    contacts: Contacts,
    body_count: int,
):
    """Copy fixed pair slots into the stable, pair-major sequential-solver array."""
    if rigid.fault[0] != 0:
        return
    output_count = int(0)
    for pair_index in range(body_count * body_count):
        body_a = pair_index // body_count
        body_b = pair_index - body_a * body_count
        if body_a >= body_b:
            continue
        slot_base = pair_index * CONTACT_MANIFOLD_POINTS
        for slot in range(CONTACT_MANIFOLD_POINTS):
            candidate = contacts.selected_candidate[slot_base + slot]
            if candidate < 0:
                continue
            contacts.a[output_count] = contacts.candidate_a[candidate]
            contacts.b[output_count] = contacts.candidate_b[candidate]
            contacts.point[output_count] = contacts.candidate_point[candidate]
            contacts.normal[output_count] = contacts.candidate_normal[candidate]
            contacts.gap[output_count] = contacts.candidate_gap[candidate]
            contacts.target[output_count] = contacts.candidate_target[candidate]
            contacts.normal_impulse[output_count] = 0.0
            contacts.tangent_impulse[output_count] = wp.vec3(0.0)
            output_count += 1
    contacts.count[0] = output_count


@wp.kernel
def prepare_rigid_contacts(rigid: RigidState):
    if rigid.fault[0] != 0:
        return
    body = wp.tid()
    rigid.inverse_inertia_world[body] = fn.world_inverse_inertia(
        rigid.rotation[body], rigid.inverse_inertia[body]
    )


@wp.kernel
def prepare_contacts(rigid: RigidState, contacts: Contacts):
    if rigid.fault[0] != 0:
        return
    i = wp.tid()
    if i >= wp.min(contacts.count[0], contacts.a.shape[0]):
        return
    a, b = contacts.a[i], contacts.b[i]
    offset_a = contacts.point[i] - rigid.position[a]
    offset_b = contacts.point[i] - rigid.position[b]
    normal = contacts.normal[i]
    contacts.offset_a[i] = offset_a
    contacts.offset_b[i] = offset_b
    contacts.normal_mass[i] = fn.effective_mass(
        rigid.inverse_mass[a], rigid.inverse_inertia_world[a], offset_a, normal
    ) + fn.effective_mass(rigid.inverse_mass[b], rigid.inverse_inertia_world[b], offset_b, normal)


@wp.kernel
def solve_contacts(rigid: RigidState, contacts: Contacts, iterations: int):
    # Sequential impulses: one device thread owns all body writes. The selected
    # scene has three dynamic bodies; this avoids racing Gauss-Seidel updates.
    if rigid.fault[0] != 0:
        return
    for iteration in range(iterations):
        for contact_index in range(wp.min(contacts.count[0], contacts.a.shape[0])):
            body_a, body_b = contacts.a[contact_index], contacts.b[contact_index]
            normal = contacts.normal[contact_index]
            offset_a = contacts.offset_a[contact_index]
            offset_b = contacts.offset_b[contact_index]
            inverse_inertia_a = rigid.inverse_inertia_world[body_a]
            inverse_inertia_b = rigid.inverse_inertia_world[body_b]
            velocity_a = wp.vec3(rigid.velocity[body_a])
            velocity_b = wp.vec3(rigid.velocity[body_b])
            angular_velocity_a = wp.vec3(rigid.omega[body_a])
            angular_velocity_b = wp.vec3(rigid.omega[body_b])
            # Normal impulse: meet the target separation speed without adhesion.
            relative_velocity = fn.point_velocity(
                velocity_a, angular_velocity_a, offset_a
            ) - fn.point_velocity(velocity_b, angular_velocity_b, offset_b)
            normal_inverse_effective_mass = contacts.normal_mass[contact_index]
            normal_impulse = wp.max(
                0.0,
                contacts.normal_impulse[contact_index]
                + (contacts.target[contact_index] - wp.dot(relative_velocity, normal))
                / normal_inverse_effective_mass,
            )
            normal_impulse_delta = (
                normal_impulse - contacts.normal_impulse[contact_index]
            ) * normal
            contacts.normal_impulse[contact_index] = normal_impulse
            velocity_a += rigid.inverse_mass[body_a] * normal_impulse_delta
            velocity_b -= rigid.inverse_mass[body_b] * normal_impulse_delta
            angular_velocity_a += inverse_inertia_a * wp.cross(offset_a, normal_impulse_delta)
            angular_velocity_b -= inverse_inertia_b * wp.cross(offset_b, normal_impulse_delta)
            # Tangential impulse: apply Coulomb friction after the normal update.
            relative_velocity = fn.point_velocity(
                velocity_a, angular_velocity_a, offset_a
            ) - fn.point_velocity(velocity_b, angular_velocity_b, offset_b)
            tangent = relative_velocity - wp.dot(relative_velocity, normal) * normal
            tangent_speed = wp.length(tangent)
            if tangent_speed > 1.0e-8:
                tangent /= tangent_speed
                tangent_inverse_effective_mass = fn.effective_mass(
                    rigid.inverse_mass[body_a], inverse_inertia_a, offset_a, tangent
                ) + fn.effective_mass(
                    rigid.inverse_mass[body_b], inverse_inertia_b, offset_b, tangent
                )
                tangent_impulse = (
                    contacts.tangent_impulse[contact_index]
                    - tangent_speed / tangent_inverse_effective_mass * tangent
                )
                friction_limit = (
                    wp.sqrt(rigid.friction[body_a] * rigid.friction[body_b]) * normal_impulse
                )
                if wp.length(tangent_impulse) > friction_limit:
                    tangent_impulse = wp.normalize(tangent_impulse) * friction_limit
                tangent_impulse_delta = tangent_impulse - contacts.tangent_impulse[contact_index]
                contacts.tangent_impulse[contact_index] = tangent_impulse
                velocity_a += rigid.inverse_mass[body_a] * tangent_impulse_delta
                velocity_b -= rigid.inverse_mass[body_b] * tangent_impulse_delta
                angular_velocity_a += inverse_inertia_a * wp.cross(offset_a, tangent_impulse_delta)
                angular_velocity_b -= inverse_inertia_b * wp.cross(offset_b, tangent_impulse_delta)
            rigid.velocity[body_a] = velocity_a
            rigid.velocity[body_b] = velocity_b
            rigid.omega[body_a] = angular_velocity_a
            rigid.omega[body_b] = angular_velocity_b


# Device diagnostics. Only explicit host diagnostics read these arrays.


@wp.kernel
def check_faults(neighbors: Neighbors, contacts: Contacts, invalid: wp.array(dtype=int)):
    code = neighbors.fault[0]
    if neighbors.overflow[0] != 0 or neighbors.overflow[1] != 0:
        code = code | 1
    if contacts.overflow[0] != 0:
        code = code | 2
    if invalid[0] != 0:
        code = code | 4
    neighbors.fault[0] = code
    if code != 0 and neighbors.fault[1] == 0:
        wp.printf(
            "PBF device fault %d: 1=neighbor overflow, 2=contact overflow, 4=non-finite state. Physics stopped; reset required.\n",
            code,
        )
        neighbors.fault[1] = 1


@wp.kernel
def audit_fluid(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    lower: wp.vec3,
    upper: wp.vec3,
    maxima: wp.array(dtype=float),
    invalid: wp.array(dtype=int),
):
    i = wp.tid()
    for axis in range(3):
        if not wp.isfinite(x[i][axis]) or not wp.isfinite(v[i][axis]):
            wp.atomic_max(invalid, 0, 1)
        violation = wp.max(lower[axis] - x[i][axis], x[i][axis] - upper[axis])
        wp.atomic_max(maxima, 0, wp.max(violation, 0.0))


@wp.kernel
def audit_rigid(
    rigid: RigidState,
    contacts: Contacts,
    maxima: wp.array(dtype=float),
    invalid: wp.array(dtype=int),
):
    i = wp.tid()
    if i < rigid.position.shape[0]:
        for axis in range(3):
            if (
                not wp.isfinite(rigid.position[i][axis])
                or not wp.isfinite(rigid.velocity[i][axis])
                or not wp.isfinite(rigid.omega[i][axis])
            ):
                wp.atomic_max(invalid, 0, 1)
        rotation_length = wp.length(rigid.rotation[i])
        if not wp.isfinite(rotation_length):
            wp.atomic_max(invalid, 0, 1)
        wp.atomic_max(maxima, 2, wp.abs(rotation_length - 1.0))
    if i < wp.min(contacts.count[0], contacts.gap.shape[0]):
        wp.atomic_max(maxima, 1, wp.max(-contacts.gap[i], 0.0))


@dataclass(frozen=True)
class PBF2WayCouplingConfig:
    """Scene selection and solver parameters in meters, kilograms and seconds."""

    # Scene and fluid material.
    scene: str = "dam-break-objects"
    particle_radius: float = 0.025
    rest_density: float = 1000.0
    gravity: tuple = (0.0, -9.81, 0.0)

    # One public step contains three fixed physical substeps.
    frame_dt: float = 1.0 / 90.0
    substeps: int = 3
    pressure_iterations: int = 3

    # Non-pressure forces and rigid-body feedback.
    viscosity: float = 0.01
    boundary_viscosity: float = 0.0
    two_way: bool = True
    wall_damping: float = 0.8
    max_speed: float = 6.0  # Fluid speed limit in meters per second.

    # Storage capacities and contact solver.
    max_fluid_neighbors: int = 256
    max_boundary_neighbors: int = 512
    max_contacts: int = 65536
    contact_tolerance: float = 0.06
    contact_iterations: int = 5

    def __post_init__(self):
        for name in (
            "particle_radius",
            "rest_density",
            "frame_dt",
            "contact_tolerance",
            "max_speed",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "substeps",
            "pressure_iterations",
            "max_fluid_neighbors",
            "max_boundary_neighbors",
            "max_contacts",
            "contact_iterations",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("viscosity", "boundary_viscosity"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if len(self.gravity) != 3 or not np.isfinite(self.gravity).all():
            raise ValueError("gravity must contain three finite components")
        if not np.isfinite(self.wall_damping) or not 0 <= self.wall_damping <= 1:
            raise ValueError("wall_damping must be finite and in [0, 1]")
        for name, fixed_value in (
            ("frame_dt", PBF2WayCouplingConfig.frame_dt),
            ("substeps", 3),
            ("contact_iterations", 5),
        ):
            if getattr(self, name) != fixed_value:
                raise ValueError(f"{name} is fixed at {fixed_value} in this demo")


class Example:
    """Own simulation state and expose construction, one fixed frame step, and rendering."""

    def __init__(self, config=None, device=None, verbose=False):
        # Runtime and host-side scene data.
        self.config = config or PBF2WayCouplingConfig()
        self.verbose = verbose
        wp.config.kernel_cache_dir = str(Path(__file__).resolve().parents[3] / ".warp_cache")
        wp.init()
        self.device = wp.get_device(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
        (
            self.body_models,
            initial_positions,
            initial_velocities,
            self.container_min,
            self.container_max,
        ) = init.load_scene(self.config)
        # Bounds apply to particle centers, keeping the full radius inside the box.
        self.fluid_lower = self.container_min + self.config.particle_radius
        self.fluid_upper = self.container_max - self.config.particle_radius
        if np.any(self.fluid_lower >= self.fluid_upper):
            raise ValueError("Container dimensions must exceed the particle diameter")

        # Time, diagnostics and fluid constants.
        self.renderer = None
        self.sim_time = 0.0
        self.frame_dt = self.config.frame_dt
        self.substep_dt = self.frame_dt / self.config.substeps
        self.current_dt = self.frame_dt
        self.last_dt = self.frame_dt
        self.iterations = self.config.pressure_iterations
        self.total_steps = 0
        self.total_substeps = 0
        self.support_radius = 4 * self.config.particle_radius
        query_min = np.floor((self.container_min - self.support_radius) / self.support_radius)
        query_max = np.floor((self.container_max + self.support_radius) / self.support_radius)
        query_cell_span = (query_max - query_min + 1).astype(int)
        if np.any(query_cell_span >= np.asarray(HASH_GRID_DIMS)):
            raise ValueError(
                "Hash grid dimensions must exceed the container query span: "
                f"dims={HASH_GRID_DIMS}, required>{tuple(query_cell_span)}"
            )
        self.fluid_volume = 0.8 * (2 * self.config.particle_radius) ** 3
        self.fluid_mass = self.fluid_volume * self.config.rest_density

        def to_device_array(data, dtype):
            return wp.array(np.asarray(data), dtype=dtype, device=self.device)

        def device_zeros(count, dtype):
            return wp.zeros(count, dtype=dtype, device=self.device)

        # Rigid state. Static bodies receive zero inverse mass and inertia.
        body_count = len(self.body_models)
        self.rigid = RigidState()
        self.rigid.fault = device_zeros(2, int)
        self.rigid.inverse_inertia_world = device_zeros(body_count, wp.mat33)
        self.rigid.position = to_device_array([body.position for body in self.body_models], wp.vec3)
        self.rigid.rotation = to_device_array([body.rotation for body in self.body_models], wp.quat)
        self.rigid.velocity = to_device_array(
            [body.velocity if body.mass else np.zeros(3) for body in self.body_models], wp.vec3
        )
        self.rigid.omega = to_device_array(
            [body.angular_velocity if body.mass else np.zeros(3) for body in self.body_models],
            wp.vec3,
        )
        self.rigid.inverse_mass = to_device_array(
            [1 / body.mass if body.mass else 0 for body in self.body_models], float
        )
        self.rigid.inverse_inertia = to_device_array(
            [
                np.linalg.inv(body.inertia) if body.mass else np.zeros((3, 3))
                for body in self.body_models
            ],
            wp.mat33,
        )
        self.rigid.lower = to_device_array(
            [body.vertices.min(axis=0) for body in self.body_models], wp.vec3
        )
        self.rigid.upper = to_device_array(
            [body.vertices.max(axis=0) for body in self.body_models], wp.vec3
        )
        self.rigid.restitution = to_device_array(
            [body.restitution for body in self.body_models], float
        )
        self.rigid.friction = to_device_array([body.friction for body in self.body_models], float)
        self.rigid.wall = to_device_array([int(body.wall) for body in self.body_models], int)
        self.rigid.force = device_zeros(body_count, wp.vec3)
        self.rigid.torque = device_zeros(body_count, wp.vec3)
        self.meshes = [
            wp.Mesh(
                to_device_array(body.vertices, wp.vec3), to_device_array(body.faces.flatten(), int)
            )
            for body in self.body_models
        ]
        self.rigid.mesh = to_device_array([mesh.id for mesh in self.meshes], wp.uint64)

        # Boundary samples and their invariant pseudo-volumes.
        self.boundary = BoundaryState()
        local_samples = np.concatenate([body.samples for body in self.body_models])
        self.boundary.local = to_device_array(local_samples, wp.vec3)
        self.boundary.body = to_device_array(
            np.repeat(np.arange(body_count), [len(body.samples) for body in self.body_models]), int
        )
        self.boundary.position = device_zeros(len(local_samples), wp.vec3)
        self.boundary.velocity = device_zeros(len(local_samples), wp.vec3)
        self.boundary.volume = device_zeros(len(local_samples), float)
        self.boundary_grid = wp.HashGrid(*HASH_GRID_DIMS, device=self.device)
        self.fluid_grid = wp.HashGrid(*HASH_GRID_DIMS, device=self.device)
        wp.launch(
            update_boundary, len(local_samples), [self.rigid, self.boundary], device=self.device
        )
        self.boundary_grid.build(self.boundary.position, self.support_radius)
        wp.launch(
            boundary_volumes,
            len(local_samples),
            [self.boundary_grid.id, self.support_radius, self.rigid, self.boundary],
            device=self.device,
        )
        # Remove initial fluid samples inside solid geometry, once at startup.
        keep = device_zeros(len(initial_positions), int)
        wp.launch(
            exclude_solid_particles,
            len(initial_positions),
            [
                to_device_array(initial_positions, wp.vec3),
                self.rigid,
                self.config.particle_radius,
                keep,
            ],
            device=self.device,
        )
        mask = keep.numpy().astype(bool)
        self.initial_excluded_particles = int((~mask).sum())
        self.positions = to_device_array(initial_positions[mask], wp.vec3)
        self.velocities = to_device_array(initial_velocities[mask], wp.vec3)
        self.num_particles = int(mask.sum())
        if not self.num_particles:
            raise ValueError("Scene contains no fluid outside rigid bodies")
        # Fluid work buffers, device audits and neighbor cache.
        particle_count = self.num_particles
        self.old_positions = device_zeros(particle_count, wp.vec3)
        self.particle_ids = to_device_array(np.arange(particle_count, dtype=np.int32), int)
        self.old_to_sorted = to_device_array(np.arange(particle_count, dtype=np.int32), int)
        self._reorder_positions = device_zeros(particle_count, wp.vec3)
        self._reorder_velocities = device_zeros(particle_count, wp.vec3)
        self._reorder_old_positions = device_zeros(particle_count, wp.vec3)
        self._reorder_particle_ids = device_zeros(particle_count, int)
        self.corrections = device_zeros(particle_count, wp.vec3)
        self.acceleration = device_zeros(particle_count, wp.vec3)
        self.densities = device_zeros(particle_count, float)
        self.lambdas = device_zeros(particle_count, float)
        self.audit_maxima = device_zeros(3, float)
        self.invalid_state = device_zeros(1, int)
        self.neighbors = Neighbors()
        self.neighbors.fault = self.rigid.fault
        self.neighbors.fluid_count = device_zeros(particle_count, int)
        self.neighbors.boundary_count = device_zeros(particle_count, int)
        self.neighbors.fluid = device_zeros((self.config.max_fluid_neighbors, particle_count), int)
        self.neighbors.boundary = device_zeros(
            (self.config.max_boundary_neighbors, particle_count), int
        )
        self.neighbors.overflow = device_zeros(2, int)
        # Contact geometry and per-contact accumulated impulses.
        self.contacts = Contacts()
        self.max_manifold_contacts = body_count * body_count * CONTACT_MANIFOLD_POINTS
        self.contacts.count = device_zeros(1, int)
        self.contacts.candidate_count = device_zeros(1, int)
        self.contacts.overflow = device_zeros(1, int)
        self.contacts.candidate_a = device_zeros(self.config.max_contacts, int)
        self.contacts.candidate_b = device_zeros(self.config.max_contacts, int)
        self.contacts.candidate_source = device_zeros(self.config.max_contacts, int)
        self.contacts.candidate_point = device_zeros(self.config.max_contacts, wp.vec3)
        self.contacts.candidate_normal = device_zeros(self.config.max_contacts, wp.vec3)
        self.contacts.candidate_gap = device_zeros(self.config.max_contacts, float)
        self.contacts.candidate_target = device_zeros(self.config.max_contacts, float)
        self.contacts.selected_candidate = device_zeros(self.max_manifold_contacts, int)
        self.contacts.selection_key = device_zeros(self.max_manifold_contacts, wp.int64)
        self.contacts.a = device_zeros(self.max_manifold_contacts, int)
        self.contacts.b = device_zeros(self.max_manifold_contacts, int)
        self.contacts.point = device_zeros(self.max_manifold_contacts, wp.vec3)
        self.contacts.normal = device_zeros(self.max_manifold_contacts, wp.vec3)
        self.contacts.gap = device_zeros(self.max_manifold_contacts, float)
        self.contacts.target = device_zeros(self.max_manifold_contacts, float)
        self.contacts.normal_impulse = device_zeros(self.max_manifold_contacts, float)
        self.contacts.tangent_impulse = device_zeros(self.max_manifold_contacts, wp.vec3)
        self.contacts.offset_a = device_zeros(self.max_manifold_contacts, wp.vec3)
        self.contacts.offset_b = device_zeros(self.max_manifold_contacts, wp.vec3)
        self.contacts.normal_mass = device_zeros(self.max_manifold_contacts, float)

    def _reorder_fluid(self):
        wp.launch(
            reorder_fluid,
            self.num_particles,
            [
                self.fluid_grid.id,
                self.positions,
                self.velocities,
                self.old_positions,
                self.particle_ids,
                self._reorder_positions,
                self._reorder_velocities,
                self._reorder_old_positions,
                self._reorder_particle_ids,
                self.old_to_sorted,
                self.rigid.fault,
            ],
            device=self.device,
        )
        self.positions, self._reorder_positions = self._reorder_positions, self.positions
        self.velocities, self._reorder_velocities = self._reorder_velocities, self.velocities
        self.old_positions, self._reorder_old_positions = (
            self._reorder_old_positions,
            self.old_positions,
        )
        self.particle_ids, self._reorder_particle_ids = (
            self._reorder_particle_ids,
            self.particle_ids,
        )

    def step(self):
        """Advance one fixed frame interval without reading device state on the CPU."""
        config, device = self.config, self.device
        self.iterations = config.pressure_iterations
        dt = self.substep_dt
        lower, upper = wp.vec3(*self.fluid_lower), wp.vec3(*self.fluid_upper)
        particle_count, boundary_count = self.num_particles, len(self.boundary.local)
        fault = self.rigid.fault

        for _ in range(config.substeps):
            # 1. Predict fluid motion. Fault flags are latched until a new Example is created.
            wp.launch(
                begin_substep, len(self.body_models), [self.rigid, self.contacts], device=device
            )
            wp.launch(
                predict,
                particle_count,
                [
                    self.positions,
                    self.velocities,
                    self.old_positions,
                    wp.vec3(*config.gravity),
                    dt,
                    fault,
                ],
                device=device,
            )

            # 2. Reorder persistent fluid state, then cache neighbors in the new index space.
            self.fluid_grid.build(self.positions, self.support_radius)
            self.boundary_grid.build(self.boundary.position, self.support_radius)
            self._reorder_fluid()
            wp.launch(
                cache_neighbors,
                particle_count,
                [
                    self.fluid_grid.id,
                    self.boundary_grid.id,
                    self.positions,
                    self.old_to_sorted,
                    self.boundary,
                    self.support_radius,
                    self.neighbors,
                ],
                device=device,
            )
            wp.launch(
                check_faults, 1, [self.neighbors, self.contacts, self.invalid_state], device=device
            )

            # 3. Fixed configured pressure count; every reaction is retained.
            for _ in range(config.pressure_iterations):
                wp.launch(
                    density_lambda,
                    particle_count,
                    [
                        self.positions,
                        self.boundary,
                        self.neighbors,
                        self.fluid_volume,
                        self.support_radius,
                        self.densities,
                        self.lambdas,
                    ],
                    device=device,
                )
                wp.launch(
                    pressure_correction,
                    particle_count,
                    [
                        self.positions,
                        self.boundary,
                        self.rigid,
                        self.neighbors,
                        self.fluid_volume,
                        self.fluid_mass,
                        self.support_radius,
                        dt,
                        int(config.two_way),
                        self.lambdas,
                        self.corrections,
                    ],
                    device=device,
                )
                wp.launch(
                    apply_correction,
                    particle_count,
                    [self.positions, self.corrections, lower, upper, fault],
                    device=device,
                )

            # 4. Reconstruct velocity and evaluate viscosity using the corrected density.
            wp.launch(
                reconstruct_velocity,
                particle_count,
                [
                    self.positions,
                    self.old_positions,
                    dt,
                    self.velocities,
                    lower,
                    upper,
                    config.wall_damping,
                    config.max_speed,
                    fault,
                ],
                device=device,
            )
            wp.launch(
                density_lambda,
                particle_count,
                [
                    self.positions,
                    self.boundary,
                    self.neighbors,
                    self.fluid_volume,
                    self.support_radius,
                    self.densities,
                    self.lambdas,
                ],
                device=device,
            )
            wp.launch(
                viscosity,
                particle_count,
                [
                    self.positions,
                    self.velocities,
                    self.boundary,
                    self.rigid,
                    self.neighbors,
                    self.densities,
                    self.fluid_volume,
                    self.fluid_mass,
                    self.support_radius,
                    config.viscosity,
                    config.boundary_viscosity,
                    int(config.two_way),
                    self.acceleration,
                ],
                device=device,
            )
            wp.launch(
                apply_viscosity,
                particle_count,
                [
                    self.velocities,
                    self.acceleration,
                    dt,
                    self.positions,
                    lower,
                    upper,
                    config.wall_damping,
                    config.max_speed,
                    fault,
                ],
                device=device,
            )

            # 5. Integrate rigid bodies, detect contacts, and prepare invariant contact data.
            wp.launch(
                integrate_rigid,
                len(self.body_models),
                [self.rigid, wp.vec3(*config.gravity), dt],
                device=device,
            )
            wp.launch(
                audit_fluid,
                particle_count,
                [
                    self.positions,
                    self.velocities,
                    lower,
                    upper,
                    self.audit_maxima,
                    self.invalid_state,
                ],
                device=device,
            )
            wp.launch(
                audit_rigid,
                len(self.body_models),
                [self.rigid, self.contacts, self.audit_maxima, self.invalid_state],
                device=device,
            )
            wp.launch(
                check_faults, 1, [self.neighbors, self.contacts, self.invalid_state], device=device
            )
            wp.launch(update_boundary, boundary_count, [self.rigid, self.boundary], device=device)
            wp.launch(
                detect_contacts,
                (boundary_count, len(self.body_models)),
                [self.rigid, self.boundary, self.contacts, config.contact_tolerance, dt],
                device=device,
            )
            wp.launch(
                check_faults, 1, [self.neighbors, self.contacts, self.invalid_state], device=device
            )
            wp.launch(
                initialize_contact_manifolds,
                self.max_manifold_contacts,
                [self.rigid, self.contacts],
                device=device,
            )
            for manifold_slot in range(CONTACT_MANIFOLD_POINTS):
                wp.launch(
                    score_contact_manifold_slot,
                    config.max_contacts,
                    [
                        self.rigid,
                        self.contacts,
                        len(self.body_models),
                        config.particle_radius,
                        config.contact_tolerance,
                        manifold_slot,
                    ],
                    device=device,
                )
                wp.launch(
                    resolve_contact_manifold_slot,
                    config.max_contacts,
                    [
                        self.rigid,
                        self.contacts,
                        len(self.body_models),
                        manifold_slot,
                    ],
                    device=device,
                )
            wp.launch(
                compact_contact_manifolds,
                1,
                [
                    self.rigid,
                    self.contacts,
                    len(self.body_models),
                ],
                device=device,
            )
            wp.launch(prepare_rigid_contacts, len(self.body_models), [self.rigid], device=device)
            wp.launch(
                prepare_contacts,
                self.max_manifold_contacts,
                [self.rigid, self.contacts],
                device=device,
            )
            wp.launch(
                solve_contacts,
                1,
                [self.rigid, self.contacts, config.contact_iterations],
                device=device,
            )
            wp.launch(update_boundary, boundary_count, [self.rigid, self.boundary], device=device)

            # 6. Device audits never synchronize the host or correct physical state.
            wp.launch(
                audit_rigid,
                max(self.max_manifold_contacts, len(self.body_models)),
                [self.rigid, self.contacts, self.audit_maxima, self.invalid_state],
                device=device,
            )
            wp.launch(
                check_faults, 1, [self.neighbors, self.contacts, self.invalid_state], device=device
            )

        self.total_steps += 1
        self.total_substeps += config.substeps
        self.sim_time = self.total_steps * self.frame_dt
        if self.verbose and self.total_steps % 100 == 0:
            print(
                f"submitted t={self.sim_time:.4f} steps={self.total_steps} substeps={self.total_substeps}"
            )

    def render(self):
        if self.renderer is None:
            raise RuntimeError("Attach a CouplingRenderer before calling render()")
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render_billboards(
            "fluid", self.positions, self.velocities, self.config.particle_radius, 2.0, 6.0
        )
        self.renderer.update_rigid_scene(self)
        self.renderer.end_frame()
