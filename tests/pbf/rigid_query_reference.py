"""Pre-optimization dynamic queries, retained only for numerical/performance tests."""
import warp as wp
from PBF import pbf_helper as fn, boundary_model as boundary
from PBF.boundary_model import BoundaryDeviceData, SDFDeviceData
wp.set_module_options({"enable_backward": False})
SDF_id = wp.constant(wp.uint64(0))
VMAP_id = wp.constant(wp.uint64(0))
Rigid_Count = wp.constant(0)

@wp.func
def sample_boundary_sdf(boundary: SDFDeviceData, position: wp.vec3):
    """Return distance and world-space gradient within the baked grid domain.

    The baker uses axis-aligned, uniformly spaced voxels. Outside allocated
    voxels NanoVDB returns its background value, not a mesh distance query.
    """
    index = wp.volume_world_to_index(boundary.volume_id, position)
    gradient = wp.vec3(0.0)
    distance = wp.volume_sample_grad_f(boundary.volume_id, index, wp.Volume.LINEAR, gradient)
    world_gradient = wp.cw_mul(gradient, boundary.inverse_voxel_size)
    return distance, world_gradient


@wp.func
def sample_instance_sdf(data: BoundaryDeviceData, instance_id: int, position: wp.vec3):
    """Sample in asset coordinates; return distance and world-space gradient."""
    transform = data.transforms[instance_id]
    local = wp.transform_point(wp.transform_inverse(transform), position)
    asset_id = data.asset_ids[instance_id]
    boundary = SDFDeviceData()
    boundary.volume_id = data.sdf_ids[asset_id]
    boundary.inverse_voxel_size = data.inverse_voxel_sizes[asset_id]
    distance, gradient = sample_boundary_sdf(boundary, local)
    return distance, wp.transform_vector(transform, gradient)


@wp.func
def sample_instance_volume(data: BoundaryDeviceData, instance_id: int, position: wp.vec3):
    """Return the cached boundary volume [m^3] at a world-space position."""
    local = wp.transform_point(wp.transform_inverse(data.transforms[instance_id]), position)
    volume_id = data.volume_map_ids[data.asset_ids[instance_id]]
    index = wp.volume_world_to_index(volume_id, local)
    return wp.volume_sample_f(volume_id, index, wp.Volume.LINEAR)


@wp.func
def query_instance_normal(data: BoundaryDeviceData, instance_id: int, position: wp.vec3):
    distance, gradient = sample_instance_sdf(data, instance_id, position)
    length_squared = wp.dot(gradient, gradient)
    if not wp.isfinite(distance) or not wp.isfinite(length_squared) or length_squared <= 1.0e-12:
        return False, 0.0, wp.vec3(0.0)
    return True, distance, gradient / wp.sqrt(length_squared)


@wp.func
def query_instance_boundary(data: BoundaryDeviceData, instance_id: int,
                            position: wp.vec3, support_radius: float):
    valid, distance, normal = query_instance_normal(data, instance_id, position)
    if not valid or distance <= 0.0 or distance >= support_radius:
        return False, 0.0, wp.vec3(0.0)
    pressure_distance = wp.max(distance + 0.125 * support_radius, 0.5 * support_radius)
    return True, pressure_distance, -normal



@wp.kernel
def volume_query(
    pre_Pos: wp.array[wp.vec3],
    boundaries: boundary.BoundaryCouplingData,
    volume_Rho: wp.array[float],
    volume_Grad: wp.array[wp.vec3]
):
    i = wp.tid()
    currPos = pre_Pos[i]
    volumeRho = float(0.0)
    gradVMap = wp.vec3(0.0)

    # volume map query
    valid_boundary, projDst, dirVM = fn.query_volume_boundary(SDF_id, currPos, fn.Smoothing_Radius)
    if valid_boundary:
        vm_index = wp.volume_world_to_index(VMAP_id, currPos)
        vmapValue = wp.volume_sample_f(VMAP_id, vm_index, wp.Volume.LINEAR)
        # Evaluate the equivalent boundary point in local coordinates.
        boundaryOffset = dirVM * projDst
        volumeRho = vmapValue * fn.Rho_0 * fn.Poly6(wp.vec3(0.0), boundaryOffset)
        gradVMap = vmapValue * fn.GradPow3(wp.vec3(0.0), boundaryOffset)

    for rigid in range(Rigid_Count):
        instance = boundaries.instances.rigid_instance_ids[rigid]
        valid, distance, direction = query_instance_boundary(
            boundaries.instances, instance, currPos, fn.Smoothing_Radius)
        if valid:
            volume = sample_instance_volume(boundaries.instances, instance, currPos)
            offset = direction * distance
            volumeRho += volume * fn.Rho_0 * fn.Poly6(wp.vec3(0.0), offset)
            gradVMap += volume * fn.GradPow3(wp.vec3(0.0), offset)

    volume_Rho[i] = volumeRho
    volume_Grad[i] = gradVMap


@wp.kernel
def calc_deltaPos(
    pre_Pos: wp.array[wp.vec3],
    lambda_Opt: wp.array[float],
    Neighbor: fn.NeighborCache,
    volume_Grad: wp.array[wp.vec3],
    dt: float,
    boundaries: boundary.BoundaryCouplingData,
    delta_Pos: wp.array[wp.vec3]
):
    i = wp.tid()
    currPos = pre_Pos[i]

    # data
    delta_Q = fn.S_corr_Q * fn.Smoothing_Radius
    WDeltaQ = fn.Poly6(wp.vec3(0.0), wp.vec3(delta_Q,0,0))
    lambda_i = lambda_Opt[i]
    delta_Movement = wp.vec3(0,0,0)
    pressure_scale = fn.Fluid_Mass * fn.Inv_Rho0

    neighborCount = Neighbor.NeighborCount[i]
    if neighborCount > 0:
        nextIndex = Neighbor.CacheBuffer[0, i]
        nextPos = pre_Pos[nextIndex]
        nextNextIndex = Neighbor.CacheBuffer[wp.min(1, neighborCount - 1), i]

        for k in range(neighborCount):
            nIndex = nextIndex
            NPos = nextPos

            nextIndex = nextNextIndex
            nextPos = pre_Pos[nextIndex]
            nextNextIndex = Neighbor.CacheBuffer[wp.min(k + 2, neighborCount - 1), i]

            if nIndex == i:
                continue
            poly6 = fn.Poly6(currPos, NPos)

            x = poly6 * (1.0/WDeltaQ)
            x2 = x * x
            x4 = x2 * x2
            S_corr = -fn.S_corr_K * x4

            lambda_j = lambda_Opt[nIndex]
            lambda_Sum = lambda_i + lambda_j + S_corr
            currGrad = fn.GradPow3(currPos, NPos)
            delta_Movement -= lambda_Sum * currGrad

    delta_boundary = -lambda_i * volume_Grad[i]
    delta_Pos[i] = delta_Movement * pressure_scale + delta_boundary

    # Each pressure correction contributes -m*delta_x/dt^2 to its rigid body.
    # Requery each contribution; no particle-by-body cache is needed.
    for rigid in range(Rigid_Count):
        instance = boundaries.instances.rigid_instance_ids[rigid]
        valid, distance, direction = query_instance_boundary(
            boundaries.instances, instance, currPos, fn.Smoothing_Radius)
        if valid:
            volume = sample_instance_volume(boundaries.instances, instance, currPos)
            offset = direction * distance
            grad = volume * fn.GradPow3(wp.vec3(0.0), offset)
            force = (fn.Fluid_Mass * lambda_i / (dt * dt)) * grad
            arm = currPos + offset - boundaries.instances.world_com[rigid]
            wrench = wp.spatial_vector(force, wp.cross(arm, force))
            wp.atomic_add(boundaries.forces, boundaries.instances.body_ids[rigid], wrench)



@wp.kernel
def update_prePos(
    delta_Pos: wp.array[wp.vec3],
    boundaries: boundary.BoundaryCouplingData,
    pre_Pos: wp.array[wp.vec3]
):
    # can this kernel fusion into last kernel?
    tid = wp.tid()
    currPos = delta_Pos[tid] + pre_Pos[tid]
    currPos = fn.apply_SDF(SDF_id, currPos, fn.Particle_Radius)
    # Penetration projection remains a numerical correction without feedback.
    for rigid in range(Rigid_Count):
        instance = boundaries.instances.rigid_instance_ids[rigid]
        valid, distance, normal = query_instance_normal(boundaries.instances, instance, currPos)
        if valid and distance < fn.Particle_Radius:
            currPos += normal * (fn.Particle_Radius - distance)
    pre_Pos[tid] = currPos

