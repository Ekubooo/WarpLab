"""Device mathematics for SPlisHSPlasH PBF/Akinci2012 coupling.

Reference: SPlisHSPlasH f3f677140761db7637b5443beb54f19f1f835ed4 (MIT).
The support radius h is four particle radii, not the particle diameter.
"""

import warp as wp


@wp.func
def clamp_to_container(position: wp.vec3, lower: wp.vec3, upper: wp.vec3):
    # Preserve invalid values so the device audit can report them instead of
    # silently converting an unstable state into a finite wall coordinate.
    result = position
    for axis in range(3):
        if wp.isfinite(position[axis]):
            result[axis] = wp.clamp(position[axis], lower[axis], upper[axis])
    return result


@wp.func
def reflect_wall_velocity(
    position: wp.vec3, velocity: wp.vec3, lower: wp.vec3, upper: wp.vec3, damping: float
):
    result = velocity
    for axis in range(3):
        if wp.isfinite(position[axis]) and wp.isfinite(velocity[axis]):
            outward = (position[axis] <= lower[axis] and velocity[axis] < 0.0) or (
                position[axis] >= upper[axis] and velocity[axis] > 0.0
            )
            if outward:
                result[axis] = -damping * velocity[axis]
    return result


@wp.func
def limit_speed(velocity: wp.vec3, max_speed: float):
    # Keep invalid input visible to the device audit. Scale before taking the
    # length so even very large finite velocities do not overflow the norm.
    for axis in range(3):
        if not wp.isfinite(velocity[axis]):
            return velocity
    scale = wp.max(wp.abs(velocity[0]), wp.max(wp.abs(velocity[1]), wp.abs(velocity[2])))
    if scale > 0.0:
        direction = velocity / scale
        allowed_scale = max_speed / wp.length(direction)
        if scale > allowed_scale:
            return direction * allowed_scale
    return velocity


@wp.func
def poly6(r: float, h: float):
    result = float(0.0)
    if r < h:
        kernel_factor = 1.0 - r * r / (h * h)
        result = 315.0 / (64.0 * wp.pi * h * h * h) * kernel_factor * kernel_factor * kernel_factor
    return result


@wp.func
def spiky_gradient(x: wp.vec3, h: float):
    distance = wp.length(x)
    result = wp.vec3(0.0)
    if distance > 1.0e-9 and distance < h:
        normalized_distance = 1.0 - distance / h
        result = (
            -45.0 / (wp.pi * h * h * h * h) * normalized_distance * normalized_distance / distance
        ) * x
    return result


@wp.func
def world_inverse_inertia(q: wp.quat, inertia: wp.mat33):
    rotation = wp.quat_to_matrix(q)
    return rotation * inertia * wp.transpose(rotation)


@wp.func
def point_velocity(v: wp.vec3, omega: wp.vec3, r: wp.vec3):
    return v + wp.cross(omega, r)


@wp.func
def effective_mass(inv_mass: float, inertia: wp.mat33, r: wp.vec3, n: wp.vec3):
    angular_gradient = wp.cross(r, n)
    return inv_mass + wp.dot(angular_gradient, inertia * angular_gradient)
