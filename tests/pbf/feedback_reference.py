"""Frozen fused delta/reaction kernel before feedback split.

Source PBF/PBF.py SHA256: 6ba5da56c9c57af6b03222c24c5d42f39d7fd2ae9907c77a49ba858ecc998cee
Import after Simulation initializes the single fixed configuration.
"""
import warp as wp
from PBF import pbf_helper as fn, boundary_model as boundary
from PBF.PBF import Rigid_Count
wp.set_module_options({"enable_backward": False})

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
    if wp.static(Rigid_Count > 0):
        for rigid in range(boundaries.instances.rigid_count):
            instance = boundaries.instances.rigid_instance_ids[rigid]
            valid, distance, direction = boundary.query_instance_boundary(
                boundaries.instances, instance, currPos, fn.Smoothing_Radius)
            if valid:
                volume = boundary.sample_instance_volume(boundaries.instances, instance, currPos)
                offset = direction * distance
                grad = volume * fn.GradPow3(wp.vec3(0.0), offset)
                force = (fn.Fluid_Mass * lambda_i / (dt * dt)) * grad
                arm = currPos + offset - boundaries.instances.world_com[rigid]
                wrench = wp.spatial_vector(force, wp.cross(arm, force))
                wp.atomic_add(boundaries.forces, boundaries.instances.body_ids[rigid], wrench)


