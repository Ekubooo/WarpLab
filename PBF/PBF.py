import warp as wp
import newton
from newton.solvers import SolverBase

wp.set_module_options({"enable_backward": False})

if __package__:
    from . import pbf_helper as fn
    from . import config_io
else:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from PBF import pbf_helper as fn
    from PBF import config_io

@wp.kernel
def apply_predict(
    pos: wp.array[wp.vec3],
    dt: float,
    gravity: wp.vec3,
    vel: wp.array[wp.vec3],
    prePos: wp.array[wp.vec3]
):
    tid = wp.tid()

    v = vel[tid]
    v_new = v + gravity * dt    
    # if external force: gravity+ext_force

    predicit = pos[tid] + dt * v_new
    prePos[tid] = predicit

@wp.kernel
def cache_neighbors(
    pos: wp.array(dtype=wp.vec3),
    NCache: fn.NeighborCache,
):
    i = wp.tid()
    currPos = pos[i]
    fluid_count = wp.int(0)
    R2 = fn.Smoothing_Radius * fn.Smoothing_Radius
    query = wp.hash_grid_query(fn.GRID_id, currPos, fn.Smoothing_Radius)
    loop_index = int(0)

    while wp.hash_grid_query_next(query, loop_index):
        j = fn.sorted_index(query)
        cap = NCache.CacheBuffer.shape[0]
        if wp.length_sq(currPos - pos[j]) < R2:
            if fluid_count < cap :
                NCache.CacheBuffer[fluid_count, i] = j
            fluid_count += 1

    NCache.NeighborCount[i] = wp.min(fluid_count, cap)
   
@wp.kernel
def volume_query(
    pre_Pos: wp.array[wp.vec3],
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

    volume_Rho[i] = volumeRho
    volume_Grad[i] = gradVMap

@wp.kernel
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
    pressure_scale = fn.Fluid_Mass * fn.Inv_Rho0

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
            density += fn.Fluid_Mass * fn.Poly6(currPos, NPos)
            currGrad = fn.GradPow3(currPos, NPos) * pressure_scale

            grad_i += currGrad
            if nIndex!=i :
                grad_j += wp.dot(currGrad, currGrad)

    volumeRho = volume_Rho[i]
    grad_i += volume_Grad[i]
    gradSum = grad_j + wp.dot(grad_i, grad_i)

    # constraint = density*fn.Inv_Rho0 - 1.0
    constraint = (density + volumeRho) * fn.Inv_Rho0 - 1.0
    if fn.NP_clamping != 0:
        constraint = wp.max(constraint, 0.0)
    lambdaCurr = -constraint / (gradSum + fn.Lamb_Eps)

    lambda_Opt[i] = lambdaCurr

@wp.kernel
def calc_deltaPos(
    pre_Pos: wp.array[wp.vec3],
    lambda_Opt: wp.array[float],
    Neighbor: fn.NeighborCache,
    volume_Grad: wp.array[wp.vec3],
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


@wp.kernel
def update_prePos(
    delta_Pos: wp.array[wp.vec3],
    pre_Pos: wp.array[wp.vec3]
):
    # can this kernel fusion into last kernel?
    tid = wp.tid()
    currPos = delta_Pos[tid] + pre_Pos[tid]
    pre_Pos[tid] = fn.apply_SDF(SDF_id, currPos, fn.Particle_Radius)

@wp.kernel
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
    if speed_squared > fn.MaxVel * fn.MaxVel:
        reconstructedVel *= fn.MaxVel / wp.sqrt(speed_squared)

    # apply
    pos[tid] = pPos
    vel[tid] = reconstructedVel

@wp.kernel
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
    volume_scale = fn.Fluid_Mass * fn.Inv_Rho0

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
            omega_i += volume_scale * wp.cross(Vel_ij, fn.GradPow3(currPos, NPos))

    curl[i] = wp.vec4(omega_i[0], omega_i[1], omega_i[2], wp.length(omega_i))

@wp.kernel
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
    volume_scale = fn.Fluid_Mass * fn.Inv_Rho0

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
            if sqrD2N > fn.square(fn.Smoothing_Radius):
                continue

            # calc
            NVel = vel[nIndex]
            Vel_ij = NVel - currVel
            NCurl = curl[nIndex]
            currGrad = fn.GradPow3(currPos, NPos)

            # voricity
            etaTotal += -volume_scale * currGrad * (NCurl[3] - currCurlMag)

            # viscosity
            vel_corr += volume_scale * Vel_ij * fn.Poly6(currPos, NPos)


    if wp.length(etaTotal) > 1e-8:
        epsilon = dt * fn.vorConfirm
        currCurl = curl[i]
        N = wp.normalize(etaTotal)
        force = wp.cross(N, wp.vec3(currCurl[0],currCurl[1],currCurl[2]))
        impulse += epsilon * force

    # XSPH
    impulse += fn.visStrength * vel_corr
    delta_Vel[i] = impulse

@wp.kernel
def update_velocity(
    delta_Vel: wp.array[wp.vec3],
    vel: wp.array[wp.vec3]
):
    tid = wp.tid()
    currDVel = vel[tid] + delta_Vel[tid] 
    sqrDVel = wp.dot(currDVel,currDVel)
    if sqrDVel > fn.MaxVel * fn.MaxVel:
        currDVel *= fn.MaxVel/wp.sqrt(sqrDVel)

    vel[tid] = currDVel


# //////////////////////////////////////////////////////////////////////////////
class PBF(SolverBase):
    """One fixed configuration/instance per process; one step is one substep."""

    def __init__(self, model, *, config, boundaries, verbose=False):
        super().__init__(model)
        if model.requires_grad:
            raise ValueError("PBF does not support automatic differentiation")
        self.verbose = verbose
        self.config = config
        self.n = model.particle_count
        self.gravity = wp.vec3(*model.gravity.numpy()[-1])

        with wp.ScopedDevice(model.device):
            fn.init_parameters(self)
            fn.init_device_buffers(self)

        self.BoundarySys = boundaries
        self.static_sdf = boundaries.assets[0].sdf
        self.volume_map = boundaries.assets[0].volume_map
        global SDF_id, VMAP_id
        SDF_id = wp.constant(wp.uint64(self.static_sdf.id))
        VMAP_id = wp.constant(wp.uint64(self.volume_map.id))

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        if state_in is state_out:
            raise ValueError("PBF requires distinct input/output States")
        if observables is not None:
            raise NotImplementedError("PBF observables are not supported")
        pos, vel = state_out.particle_q, state_out.particle_qd
        with wp.ScopedDevice(self.model.device):
            with wp.ScopedTimer("Prologue", active=self.verbose):
                # apply and predict
                wp.launch(
                    kernel=apply_predict,
                    dim=self.n,
                    inputs=[state_in.particle_q, dt, self.gravity, state_in.particle_qd],
                    outputs=[self.pre_Pos]
                )
            
                # hash grid, reorder and cache neighbor
                self.grid.build(self.pre_Pos, self.smoothing_length)
                fn.buffer_reorder(self, state_in.particle_q, pos)
                wp.launch(
                    kernel=cache_neighbors,
                    dim=self.n,
                    inputs=[self.pre_Pos],
                    outputs=[self.n_Cache]
                )

            with wp.ScopedTimer("Solver", active=self.verbose):
                # iterations per substep
                for i in range(self.iterations):
                    wp.launch(
                        kernel=volume_query,
                        dim=self.n,
                        inputs=[self.pre_Pos],
                        outputs=[self.volume_Rho, self.volume_Grad]
                    )
                    wp.launch(
                        kernel=calc_lambda,
                        dim=self.n,
                        inputs=[
                            self.pre_Pos,
                            self.n_Cache,
                            self.volume_Rho,
                            self.volume_Grad
                        ],
                        outputs=[self.lambda_Opt]
                    )
                    wp.launch(
                        kernel=calc_deltaPos, 
                        dim=self.n,
                        inputs=[
                            self.pre_Pos, 
                            self.lambda_Opt,
                            self.n_Cache,
                            self.volume_Grad
                        ],
                        outputs=[self.delta_Pos]
                    )
                    wp.launch(
                        kernel=update_prePos, 
                        dim=self.n,
                        inputs=[self.delta_Pos],
                        outputs=[self.pre_Pos]
                    )

            with wp.ScopedTimer("Epilogue", active=self.verbose):
                wp.launch(
                    kernel=update_position,
                    dim=self.n,
                    inputs=[dt, self.pre_Pos],
                    outputs=[pos, vel]
                )
                wp.launch(
                    kernel=calc_curl,
                    dim=self.n,
                    inputs=[
                        self.pre_Pos, 
                        vel,
                        self.n_Cache
                    ],
                    outputs=[self.curl]
                )
                wp.launch(
                    kernel=calc_visvor,
                    dim=self.n,
                    inputs=[
                        dt,
                        self.pre_Pos,
                        vel,
                        self.curl,
                        self.n_Cache
                    ],
                    outputs=[self.delta_Vel]
                )
                wp.launch(
                    kernel=update_velocity,
                    dim=self.n,
                    inputs=[self.delta_Vel],
                    outputs=[vel]
                )

    def reset(self, state, world_mask=None, flags=None):
        """Restore one complete particle state and clear solver work buffers."""
        if world_mask is not None or flags is not None:
            raise NotImplementedError("PBF currently supports complete single-scene reset only")
        with wp.ScopedDevice(self.model.device):
            wp.copy(state.particle_q, self.model.particle_q)
            wp.copy(state.particle_qd, self.model.particle_qd)
            fn.clear_work_buffers(self)

    def notify_model_changed(self, flags):
        if flags & newton.ModelFlags.MODEL_PROPERTIES:
            self.gravity = wp.vec3(*self.model.gravity.numpy()[-1])


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_pbf.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=480, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument(
        "--config", dest="config_path", metavar="PATH",
        help="Load JSON config instead of PBF configuration arguments.",
    )
    parser.add_argument(
        "--export-config", metavar="NAME", nargs="?", const=config_io.DEFAULT_CONFIG_NAME,
        help=f"Save config to the demo's config/NAME and exit (default name: {config_io.DEFAULT_CONFIG_NAME}).",
    )
    parser.add_argument(
        "--clamp-negative-pressure",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help=f"Clamp negative density constraints to zero (default: {fn.PBFConfig.clamp_negative_pressure}).",
    )

    args = parser.parse_args(argv)
    try:
        if args.config_path is not None:
            if hasattr(args, "clamp_negative_pressure"):
                raise ValueError("--config cannot be combined with PBF configuration arguments")
            args.config = config_io.load_config(args.config_path)
        else:
            overrides = (
                {"clamp_negative_pressure": args.clamp_negative_pressure}
                if hasattr(args, "clamp_negative_pressure") else {}
            )
            args.config = fn.PBFConfig(**overrides)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    from PBF.simulation import Simulation

    args = parse_args()
    if args.export_config is not None:
        try:
            print(config_io.save_config(args.config, args.export_config))
        except (OSError, ValueError) as error:
            raise SystemExit(f"Cannot export config: {error}") from error
        return

    with wp.ScopedDevice(args.device):
        example = Simulation(
            stage_path=args.stage_path, verbose=args.verbose,
            config=args.config,
        )

        for _ in range(args.num_frames):
            example.render()
            example.step()

        if example.renderer:
            example.renderer.save()


if __name__ == "__main__":
    main()
