"""Device mathematics for SPlisHSPlasH PBF/Akinci2012 coupling.

Reference: SPlisHSPlasH f3f677140761db7637b5443beb54f19f1f835ed4 (MIT).
The support radius h is four particle radii, not the particle diameter.
"""

import warp as wp


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
