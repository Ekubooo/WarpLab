"""The original PBF GPU pipeline, specialized once per solver instance."""

from types import SimpleNamespace

import warp as wp
from . import pbf_helper as fn
from .boundary_model import (BoundaryCouplingData, query_instance_boundary,
                             query_instance_normal, sample_instance_volume)


def create_kernels(parameters, grid_id, sdf_id, volume_map_id, rigid_count=0):
    config = parameters.config
    h = parameters.support_radius
    GRID_id = wp.constant(wp.uint64(grid_id))
    SDF_id = wp.constant(wp.uint64(sdf_id))
    VMAP_id = wp.constant(wp.uint64(volume_map_id))
    Rigid_Count = wp.constant(rigid_count)
    Smoothing_Radius = wp.constant(h)
    Particle_Radius = wp.constant(parameters.particle_radius)
    Fluid_Mass = wp.constant(parameters.fluid_mass)
    Rho_0 = wp.constant(config.rest_density)
    Inv_Rho0 = wp.constant(1.0 / config.rest_density)
    Lamb_Eps = wp.constant(config.lambda_regularization / h**2)
    S_corr_K = wp.constant(config.artificial_pressure_strength * h**2)
    S_corr_Q = wp.constant(config.artificial_pressure_q)
    NP_clamping = wp.constant(int(config.clamp_negative_pressure))
    MaxVel = wp.constant(config.max_speed)
    visStrength = wp.constant(config.viscosity)
    vorConfirm = wp.constant(config.vorticity_confinement)
    Poly6, GradPow3 = fn.create_fluid_functions(parameters)
    query_volume_boundary = fn.query_volume_boundary
    apply_SDF = fn.apply_SDF
    sorted_index = fn.sorted_index
    square = fn.square

    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
    def reorder(pos: wp.array(dtype=wp.vec3), pre_Pos: wp.array(dtype=wp.vec3),
                pos_temp: wp.array(dtype=wp.vec3), pPos_temp: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        original = wp.hash_grid_point_id(GRID_id, i)
        pos_temp[i] = pos[original]
        pPos_temp[i] = pre_Pos[original]

    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
    def apply_predict(
        pos: wp.array[wp.vec3],
        dt: float,
        gravity: wp.vec3,
        prePos: wp.array[wp.vec3],
        vel: wp.array[wp.vec3]
    ):
        tid = wp.tid()

        v = vel[tid]
        v_new = v + gravity * dt
        # if external force: gravity+ext_force

        predicit = pos[tid] + dt * v_new
        prePos[tid] = predicit

    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
    def cache_neighbors(
        pos: wp.array(dtype=wp.vec3),
        NCache: fn.NeighborCache,
    ):
        i = wp.tid()
        currPos = pos[i]
        fluid_count = wp.int(0)
        R2 = Smoothing_Radius * Smoothing_Radius
        query = wp.hash_grid_query(GRID_id, currPos, Smoothing_Radius)
        loop_index = int(0)

        while wp.hash_grid_query_next(query, loop_index):
            j = sorted_index(query)
            cap = NCache.CacheBuffer.shape[0]
            if wp.length_sq(currPos - pos[j]) < R2:
                if fluid_count < cap :
                    NCache.CacheBuffer[fluid_count, i] = j
                fluid_count += 1

        NCache.NeighborCount[i] = wp.min(fluid_count, cap)

    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
    def volume_query(
        pre_Pos: wp.array[wp.vec3],
        boundary: BoundaryCouplingData,
        volume_Rho: wp.array[float],
        volume_Grad: wp.array[wp.vec3]
    ):
        i = wp.tid()
        currPos = pre_Pos[i]
        volumeRho = float(0.0)
        gradVMap = wp.vec3(0.0)

        # volume map query
        valid_boundary, projDst, dirVM = query_volume_boundary(SDF_id, currPos, Smoothing_Radius)
        if valid_boundary:
            vm_index = wp.volume_world_to_index(VMAP_id, currPos)
            vmapValue = wp.volume_sample_f(VMAP_id, vm_index, wp.Volume.LINEAR)
            # Evaluate the equivalent boundary point in local coordinates.
            boundaryOffset = dirVM * projDst
            volumeRho = vmapValue * Rho_0 * Poly6(wp.vec3(0.0), boundaryOffset)
            gradVMap = vmapValue * GradPow3(wp.vec3(0.0), boundaryOffset)

        for rigid in range(Rigid_Count):
            instance = boundary.instances.rigid_instance_ids[rigid]
            valid, distance, direction = query_instance_boundary(
                boundary.instances, instance, currPos, Smoothing_Radius)
            if valid:
                volume = sample_instance_volume(boundary.instances, instance, currPos)
                offset = direction * distance
                volumeRho += volume * Rho_0 * Poly6(wp.vec3(0.0), offset)
                gradVMap += volume * GradPow3(wp.vec3(0.0), offset)

        volume_Rho[i] = volumeRho
        volume_Grad[i] = gradVMap

    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
    def calc_lambda(
        pre_Pos: wp.array[wp.vec3],
        Neighbor: fn.NeighborCache,
        volume_Rho: wp.array[float],
        volume_Grad: wp.array[wp.vec3],
        lambda_Opt: wp.array[float]
    ):
        density = float(0.0)
        grad_j = float(0.0)
        grad_i = wp.vec3(0,0,0)
        pressure_scale = Fluid_Mass * Inv_Rho0

        i = wp.tid()
        currPos = pre_Pos[i]

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

                # Self density is included by the grid query; its zero gradient is harmless.
                density += Fluid_Mass * Poly6(currPos, NPos)
                currGrad = GradPow3(currPos, NPos) * pressure_scale

                grad_i += currGrad
                if nIndex!=i :
                    grad_j += wp.dot(currGrad, currGrad)

        volumeRho = volume_Rho[i]
        grad_i += volume_Grad[i]
        gradSum = grad_j + wp.dot(grad_i, grad_i)

        # constraint = density*Inv_Rho0 - 1.0
        constraint = (density + volumeRho) * Inv_Rho0 - 1.0
        if NP_clamping != 0:
            constraint = wp.max(constraint, 0.0)
        lambdaCurr = -constraint / (gradSum + Lamb_Eps)

        lambda_Opt[i] = lambdaCurr

    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
    def calc_deltaPos(
        pre_Pos: wp.array[wp.vec3],
        lambda_Opt: wp.array[float],
        Neighbor: fn.NeighborCache,
        volume_Grad: wp.array[wp.vec3],
        dt: float,
        boundary: BoundaryCouplingData,
        delta_Pos: wp.array[wp.vec3]
    ):
        i = wp.tid()
        currPos = pre_Pos[i]

        # data
        delta_Q = S_corr_Q * Smoothing_Radius
        WDeltaQ = Poly6(wp.vec3(0.0), wp.vec3(delta_Q,0,0))
        lambda_i = lambda_Opt[i]
        delta_Movement = wp.vec3(0,0,0)
        pressure_scale = Fluid_Mass * Inv_Rho0

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
                poly6 = Poly6(currPos, NPos)

                x = poly6 * (1.0/WDeltaQ)
                x2 = x * x
                x4 = x2 * x2
                S_corr = -S_corr_K * x4

                lambda_j = lambda_Opt[nIndex]
                lambda_Sum = lambda_i + lambda_j + S_corr
                currGrad = GradPow3(currPos, NPos)
                delta_Movement -= lambda_Sum * currGrad

        delta_boundary = -lambda_i * volume_Grad[i]
        delta_Pos[i] = delta_Movement * pressure_scale + delta_boundary

        # SPlisHSPlasH's weak coupling: each pressure correction contributes
        # -m*delta_x/dt^2 at its equivalent boundary point, across all iterations.
        for rigid in range(Rigid_Count):
            instance = boundary.instances.rigid_instance_ids[rigid]
            valid, distance, direction = query_instance_boundary(
                boundary.instances, instance, currPos, Smoothing_Radius)
            if valid:
                volume = sample_instance_volume(boundary.instances, instance, currPos)
                offset = direction * distance
                grad = volume * GradPow3(wp.vec3(0.0), offset)
                force = (Fluid_Mass * lambda_i / (dt * dt)) * grad
                arm = currPos + offset - boundary.instances.world_com[rigid]
                wrench = wp.spatial_vector(force, wp.cross(arm, force))
                wp.atomic_add(boundary.forces, boundary.instances.body_ids[rigid], wrench)


    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
    def update_prePos(
        delta_Pos: wp.array[wp.vec3],
        boundary: BoundaryCouplingData,
        pre_Pos: wp.array[wp.vec3]
    ):
        # can this kernel fusion into last kernel?
        tid = wp.tid()
        currPos = delta_Pos[tid] + pre_Pos[tid]
        currPos = apply_SDF(SDF_id, currPos, Particle_Radius)
        # Penetration projection is the existing numerical fallback, without
        # rigid feedback. It is deliberately separate from pressure forces.
        for rigid in range(Rigid_Count):
            instance = boundary.instances.rigid_instance_ids[rigid]
            valid, distance, normal = query_instance_normal(boundary.instances, instance, currPos)
            if valid and distance < Particle_Radius:
                currPos += normal * (Particle_Radius - distance)
        pre_Pos[tid] = currPos

    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
    def update_position(
        dt: float,
        pre_Pos: wp.array[wp.vec3],
        pos: wp.array[wp.vec3],
        vel: wp.array[wp.vec3]
    ):
        tid = wp.tid()
        pPos = pre_Pos[tid]
        currPos = pos[tid]
        reconstructedVel = (pPos - currPos) / dt

        # Limit reconstruction before it becomes input to XSPH/vorticity.
        speed_squared = wp.dot(reconstructedVel, reconstructedVel)
        if speed_squared > MaxVel * MaxVel:
            reconstructedVel *= MaxVel / wp.sqrt(speed_squared)

        # apply
        pos[tid] = pPos
        vel[tid] = reconstructedVel

    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
    def calc_curl(
        pre_Pos: wp.array[wp.vec3],
        vel: wp.array[wp.vec3],
        Neighbor: fn.NeighborCache,
        curl: wp.array[wp.vec4]
    ):
        i = wp.tid()
        currPos = pre_Pos[i]

        # data
        currVel = vel[i]
        omega_i = wp.vec3(0,0,0)
        volume_scale = Fluid_Mass * Inv_Rho0

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

                # curl
                NVel = vel[nIndex]
                Vel_ij = NVel - currVel
                omega_i += volume_scale * wp.cross(Vel_ij, GradPow3(currPos, NPos))

        curl[i] = wp.vec4(omega_i[0], omega_i[1], omega_i[2], wp.length(omega_i))

    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
    def calc_visvor(
        dt: float,
        pre_Pos: wp.array[wp.vec3],
        vel: wp.array[wp.vec3],
        curl: wp.array[wp.vec4],
        Neighbor: fn.NeighborCache,
        delta_Vel: wp.array[wp.vec3]
    ):
        i = wp.tid()
        currPos = pre_Pos[i]
        currVel = vel[i]
        currCurlMag = curl[i][3]

        impulse = wp.vec3(0,0,0)
        etaTotal = wp.vec3(0,0,0)
        vel_corr = wp.vec3(0,0,0)
        volume_scale = Fluid_Mass * Inv_Rho0

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
                O2N = NPos - currPos
                sqrD2N = wp.dot(O2N, O2N)
                if sqrD2N > square(Smoothing_Radius):
                    continue

                # calc
                NVel = vel[nIndex]
                Vel_ij = NVel - currVel
                NCurl = curl[nIndex]
                currGrad = GradPow3(currPos, NPos)

                # voricity
                etaTotal += -volume_scale * currGrad * (NCurl[3] - currCurlMag)

                # viscosity
                vel_corr += volume_scale * Vel_ij * Poly6(currPos, NPos)


        if wp.length(etaTotal) > 1e-8:
            epsilon = dt * vorConfirm
            currCurl = curl[i]
            N = wp.normalize(etaTotal)
            force = wp.cross(N, wp.vec3(currCurl[0],currCurl[1],currCurl[2]))
            impulse += epsilon * force

        # XSPH
        impulse += visStrength * vel_corr
        delta_Vel[i] = impulse

    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_backward": False})
    def update_velocity(
        delta_Vel: wp.array[wp.vec3],
        vel: wp.array[wp.vec3]
    ):
        tid = wp.tid()
        currDVel = vel[tid] + delta_Vel[tid]
        sqrDVel = wp.dot(currDVel,currDVel)
        if sqrDVel > MaxVel * MaxVel:
            currDVel *= MaxVel/wp.sqrt(sqrDVel)

        vel[tid] = currDVel

    return SimpleNamespace(
        reorder=reorder,
        apply_predict=apply_predict,
        cache_neighbors=cache_neighbors,
        volume_query=volume_query,
        calc_lambda=calc_lambda,
        calc_deltaPos=calc_deltaPos,
        update_prePos=update_prePos,
        update_position=update_position,
        calc_curl=calc_curl,
        calc_visvor=calc_visvor,
        update_velocity=update_velocity,
    )
