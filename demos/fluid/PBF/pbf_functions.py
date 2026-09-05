import warp as wp

rho0: float = 3.0
lambdaEps: float = 32.0
ScorrK: float = 0.01    # origin settting: 0
ScorrN: float = 4.0
viscosityStr: float = 0.025  # 0.01 or 0.99？
vorticityCon = 0.5
VelLimit = 50.0
boundaryRestitution = 0.98
boundaryTangentialRetention = 1.0
boundaryContactEpsScale = 1.0e-4
boundaryDisturbance = 0.001

smoothingLength = 0.8
paraPoly6 = 315.0/(64.0 * wp.pi * wp.pow(wp.abs(smoothingLength), 9.0))
paraPow3 = 15.0 / (wp.pi * wp.pow(smoothingLength, 6.0))

S_corr_K = wp.constant(ScorrK)
S_corr_N = wp.constant(ScorrN)
Inv_Rho0 = wp.constant(1.0/rho0)
Lamb_Eps = wp.constant(lambdaEps)
visStrength = wp.constant(viscosityStr)
vorConfirm = wp.constant(vorticityCon)
MaxVel = wp.constant(VelLimit)
K_SPow3 = wp.constant(paraPow3)
K_DSPow3 = wp.constant(3.0 * paraPow3)
K_SPoly6 = wp.constant(paraPoly6)
Boundary_Restitution = wp.constant(boundaryRestitution)
Boundary_Tangential_Retention = wp.constant(boundaryTangentialRetention)
Boundary_Contact_Eps = wp.constant(boundaryContactEpsScale * smoothingLength)
Boundary_Disturbance = wp.constant(boundaryDisturbance)


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
 
