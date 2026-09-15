"""Original sequential contact sweep, retained only for equivalence tests/benchmarks."""

import warp as wp
from .PBF2WayCoupling import RigidState, Contacts
from . import coupling_functions as fn


@wp.kernel
def reference_contact_sweep(rigid: RigidState, contacts: Contacts):
    # Sequential impulses: one device thread owns all body writes. The selected
    # scene has three dynamic bodies; this avoids racing Gauss-Seidel updates.
    for contact_index in range(wp.min(contacts.count[0], contacts.a.shape[0])):
        body_a, body_b = contacts.a[contact_index], contacts.b[contact_index]
        normal = contacts.normal[contact_index]
        offset_a, offset_b = (
            contacts.point[contact_index] - rigid.position[body_a],
            contacts.point[contact_index] - rigid.position[body_b],
        )
        inverse_inertia_a = fn.world_inverse_inertia(
            rigid.rotation[body_a], rigid.inverse_inertia[body_a]
        )
        inverse_inertia_b = fn.world_inverse_inertia(
            rigid.rotation[body_b], rigid.inverse_inertia[body_b]
        )
        velocity_a = wp.vec3(rigid.velocity[body_a])
        velocity_b = wp.vec3(rigid.velocity[body_b])
        angular_velocity_a = wp.vec3(rigid.omega[body_a])
        angular_velocity_b = wp.vec3(rigid.omega[body_b])
        # Normal impulse: meet the target separation speed without adhesion.
        relative_velocity = fn.point_velocity(
            velocity_a, angular_velocity_a, offset_a
        ) - fn.point_velocity(velocity_b, angular_velocity_b, offset_b)
        normal_inverse_effective_mass = fn.effective_mass(
            rigid.inverse_mass[body_a], inverse_inertia_a, offset_a, normal
        ) + fn.effective_mass(rigid.inverse_mass[body_b], inverse_inertia_b, offset_b, normal)
        normal_impulse = wp.max(
            0.0,
            contacts.normal_impulse[contact_index]
            + (contacts.target[contact_index] - wp.dot(relative_velocity, normal))
            / normal_inverse_effective_mass,
        )
        normal_impulse_delta = (normal_impulse - contacts.normal_impulse[contact_index]) * normal
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
            ) + fn.effective_mass(rigid.inverse_mass[body_b], inverse_inertia_b, offset_b, tangent)
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
