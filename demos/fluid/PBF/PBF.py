import warp as wp
import warp.render

if __package__:
    from . import pbf_helper as fn
    from . import config_io
    from . import boundary_model
else:
    import pbf_helper as fn
    import config_io
    import boundary_model

# Select the offline asset here; OpenGL uses the same stem for its landscape OBJ.
SDF_FILENAME = "stairs_scene.nvdb"

@wp.kernel
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

    vel[tid] = v_new
    predicit = pos[tid] + dt * v_new
    prePos[tid] = predicit

@wp.kernel
def cache_neighbors(
    pos: wp.array(dtype=wp.vec3),
    NCache: fn.NeighborCache,
):
    i = wp.tid()
    fluid_count = wp.int(0)
    R2 = fn.Smoothing_Radius * fn.Smoothing_Radius
    query = wp.hash_grid_query(fn.GRID_id, pos[i], fn.Smoothing_Radius)
    loop_index = int(0)

    while wp.hash_grid_query_next(query, loop_index):
        j = fn.sorted_index(query)
        cap = NCache.NP_Array.shape[0]
        if wp.length_sq(pos[i] - pos[j]) < R2:
            if fluid_count < cap :
                NCache.NP_Array[fluid_count, i] = j
            fluid_count += 1

    NCache.NP_Count[i] = wp.min(fluid_count, cap)
   
@wp.kernel
def calc_lambda(
    pre_Pos: wp.array[wp.vec3],
    Neighbor: fn.NeighborCache,
    lambda_Opt: wp.array[float]
):
    density = float(0.0)
    volumeRho = float(0.0)
    grad_j = float(0.0)
    grad_i = wp.vec3(0,0,0)
    pressure_scale = fn.Fluid_Mass * fn.Inv_Rho0

    i = wp.tid()
    currPos = pre_Pos[i]   

    for k in range(Neighbor.NP_Count[i]):
        nIndex = Neighbor.NP_Array[k, i]
        NPos = pre_Pos[nIndex]
        O2N = NPos - currPos
        sqrD2N = wp.dot(O2N, O2N)
        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)
        # Self density is included by the grid query; its zero gradient is harmless.
        density += fn.Fluid_Mass * fn.Poly6(dst2N, fn.Smoothing_Radius)
        currGrad = dir2N * pressure_scale * fn.DPow3(dst2N, fn.Smoothing_Radius)

        grad_i += currGrad
        if nIndex!=i : 
            grad_j += wp.dot(currGrad, currGrad)

    # volume map query
    valid_boundary, projDst, dirVM = fn.query_volume_boundary(SDF_id, currPos, fn.Smoothing_Radius)
    if valid_boundary:
        vm_index = wp.volume_world_to_index(VMAP_id, currPos)
        vmapValue = wp.volume_sample_f(VMAP_id, vm_index, wp.Volume.LINEAR)
        volumeRho = vmapValue * fn.Rho_0 * fn.Poly6(projDst, fn.Smoothing_Radius)
        grad_i += vmapValue * dirVM * fn.DPow3(projDst, fn.Smoothing_Radius)
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
    delta_Pos: wp.array[wp.vec3]
):
    i = wp.tid()
    currPos = pre_Pos[i]

    # data
    delta_Q = fn.S_corr_Q * fn.Smoothing_Radius
    WDeltaQ = fn.Poly6(delta_Q, fn.Smoothing_Radius)
    lambda_i = lambda_Opt[i]
    delta_Movement = wp.vec3(0,0,0)
    pressure_scale = fn.Fluid_Mass * fn.Inv_Rho0

    for k in range(Neighbor.NP_Count[i]):
        nIndex = Neighbor.NP_Array[k, i]
        if nIndex == i: 
            continue
        NPos = pre_Pos[nIndex]
        O2N = NPos - currPos
        sqrD2N = wp.dot(O2N, O2N)

        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)
        poly6 = fn.Poly6(dst2N, fn.Smoothing_Radius)

        x = poly6 * (1.0/WDeltaQ)
        x2 = x * x
        x4 = x2 * x2 
        S_corr = -fn.S_corr_K * x4

        lambda_j = lambda_Opt[nIndex]
        lambda_Sum = lambda_i + lambda_j + S_corr
        currGrad = dir2N * fn.DPow3(dst2N, fn.Smoothing_Radius)
        delta_Movement -= lambda_Sum * currGrad

    # volume map query
    gradVMap = wp.vec3(0.0)
    valid_boundary, projDst, dirVM = fn.query_volume_boundary(SDF_id, currPos, fn.Smoothing_Radius)
    if valid_boundary:
        vm_index = wp.volume_world_to_index(VMAP_id, currPos)
        vmapValue = wp.volume_sample_f(VMAP_id, vm_index, wp.Volume.LINEAR)
        gradVMap = vmapValue * dirVM * fn.DPow3(projDst, fn.Smoothing_Radius)

    delta_boundary = -lambda_i * gradVMap
    delta_Pos[i] = delta_Movement * pressure_scale + delta_boundary

    # currPos = delta_Pos[tid] + pre_Pos[tid]
    # pre_Pos[tid] = fn.apply_boundary(currPos)

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
    incomingVel = vel[tid]
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

    for k in range(Neighbor.NP_Count[i]):
        nIndex = Neighbor.NP_Array[k, i]
        if nIndex == i: 
            continue
        NPos = pre_Pos[nIndex]
        O2N = NPos - currPos
        sqrD2N = wp.dot(O2N, O2N)
        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)

        # curl
        NVel = vel[nIndex]
        Vel_ij = NVel - currVel
        omega_i += volume_scale * wp.cross(Vel_ij, dir2N * fn.DPow3(dst2N, fn.Smoothing_Radius))

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

    # data
    currVel = vel[i]
    impulse = wp.vec3(0,0,0)
    etaTotal = wp.vec3(0,0,0)
    vel_corr = wp.vec3(0,0,0)
    volume_scale = fn.Fluid_Mass * fn.Inv_Rho0

    for k in range(Neighbor.NP_Count[i]):
        nIndex = Neighbor.NP_Array[k, i]
        if nIndex == i: 
            continue
        NPos = pre_Pos[nIndex]
        O2N = NPos - currPos
        sqrD2N = wp.dot(O2N, O2N)
        if sqrD2N > fn.square(fn.Smoothing_Radius):
            continue
        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)

        # calc
        NVel = vel[nIndex]
        Vel_ij = NVel - currVel
        NCurl = curl[nIndex]
        currGrad = fn.DPow3(dst2N, fn.Smoothing_Radius)

        # voricity
        etaTotal += -volume_scale * dir2N * currGrad * (NCurl[3] - curl[i][3])

        # viscosity
        vel_corr += volume_scale * Vel_ij * fn.Poly6(dst2N, fn.Smoothing_Radius)


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
class Example:
    def __init__(self, stage_path="example_pbf.usd", verbose=False, config=None):
        self.verbose = verbose
        self.sim_time = 0.0

        self.config = config if config is not None else fn.PBFConfig()
        self.gravity = wp.vec3(*self.config.gravity)

        fn.init_parameters(self)
        fn.init_device_buffers(self)
        # Keep the GPU Volume alive while the solver queries its SDF.
        self.static_sdf = boundary_model.load_static_sdf(SDF_FILENAME, device=wp.get_device())
        self.volume_map = boundary_model.precompute_volumeMap(self.static_sdf, self.smoothing_length)
        global SDF_id, VMAP_id
        SDF_id = wp.constant(wp.uint64(self.static_sdf.id))
        VMAP_id = wp.constant(wp.uint64(self.volume_map.id))

        self.renderer = (
            wp.render.UsdRenderer(stage_path)
            if stage_path else None
        )

    def step(self):
        with wp.ScopedTimer("sub-step", synchronize=True):
            # 3 substep: (or more)
            for _ in range(self.substep):
                with wp.ScopedTimer("Prologue", active=self.verbose):
                    # apply and predict
                    wp.launch(
                        kernel=apply_predict,
                        dim=self.n,
                        inputs=[self.pos, self.dt, self.gravity],
                        outputs= [self.pre_Pos, self.v]
                    )
                    
                    # hash grid, reorder and cache neighbor
                    self.grid.build(self.pre_Pos, self.smoothing_length)
                    fn.buffer_reorder(self)
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
                            kernel=calc_lambda,
                            dim=self.n,
                            inputs=[
                                self.pre_Pos,
                                self.n_Cache
                            ],
                            outputs=[self.lambda_Opt]
                        )
                        wp.launch(
                            kernel=calc_deltaPos, 
                            dim=self.n,
                            inputs=[
                                self.pre_Pos, 
                                self.lambda_Opt,
                                self.n_Cache
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
                    self.grid.build(self.pre_Pos, self.smoothing_length)
                    wp.launch(
                        kernel=update_position,
                        dim=self.n,
                        inputs=[self.dt, self.pre_Pos],
                        outputs=[self.pos, self.v]
                    )
                    wp.launch(
                        kernel=calc_curl,
                        dim=self.n,
                        inputs=[
                            self.pre_Pos, 
                            self.v,
                            self.n_Cache
                        ],
                        outputs=[self.curl]
                    )
                    wp.launch(
                        kernel=calc_visvor,
                        dim=self.n,
                        inputs=[
                            self.dt,
                            self.pre_Pos,
                            self.v,
                            self.curl,
                            self.n_Cache
                        ],
                        outputs=[self.delta_Vel]
                    )
                    wp.launch(
                        kernel=update_velocity,
                        dim=self.n,
                        inputs=[self.delta_Vel],
                        outputs=[self.v]
                    )


            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.pos.numpy(), radius=self.particle_radius, name="points", colors=(0.8, 0.3, 0.2)
            )
            self.renderer.end_frame()


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
    args = parse_args()
    if args.export_config is not None:
        try:
            print(config_io.save_config(args.config, args.export_config))
        except (OSError, ValueError) as error:
            raise SystemExit(f"Cannot export config: {error}") from error
        return

    with wp.ScopedDevice(args.device):
        example = Example(
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
