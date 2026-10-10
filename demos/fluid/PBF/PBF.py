import warp as wp
import warp.render

if __package__:
    from . import pbf_helper as fn
else:
    import pbf_helper as fn

@wp.kernel
def apply_predict(
    pos: wp.array[wp.vec3],
    dt: float,
    gravity: wp.vec3,
    prePos: wp.array[wp.vec3],
    preNew: wp.array[wp.vec3],
    vel: wp.array[wp.vec3]
):
    tid = wp.tid()

    v = vel[tid]
    v_new = v + gravity * dt    
    # if external force: gravity+ext_force

    vel[tid] = v_new
    predicit = pos[tid] + dt * v_new
    prePos[tid] = predicit
    preNew[tid] = predicit

@wp.kernel
def calc_lambda(
    grid: wp.uint64,
    smoothing_length: float,
    pre_Pos: wp.array[wp.vec3],
    pre_New: wp.array[wp.vec3],
    clamp_negative_pressure: int,
    lambda_Opt: wp.array[float]
):
    density = float(0.0)
    grad_j = float(0.0)
    grad_i = wp.vec3(0,0,0)
    pressure_scale = fn.Fluid_Mass * fn.Inv_Rho0

    # id of particle that order by hash/bucket/gridIndex(not "grid id")
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    qurryPos = pre_New[i]    # Snapshot
    currPos = pre_Pos[i]   
    neighbors = wp.hash_grid_query(grid, qurryPos, smoothing_length)

    for index in neighbors:
        NPos = pre_Pos[index]
        O2N = NPos - currPos
        sqrD2N = wp.dot(O2N, O2N)
        if sqrD2N > fn.square(smoothing_length):
            continue
        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)
        # Self density is included by the grid query; its zero gradient is harmless.
        density += fn.Fluid_Mass * fn.Poly6(dst2N, smoothing_length)
        currGrad = dir2N * pressure_scale * fn.DPow3(dst2N, smoothing_length)

        grad_i += currGrad
        if index!=i : 
            grad_j += wp.dot(currGrad, currGrad)
        # do something.
    
    gradSum = grad_j + wp.dot(grad_i, grad_i)
    constraint = density*fn.Inv_Rho0 - 1.0
    if clamp_negative_pressure != 0:
        constraint = wp.max(constraint, 0.0)
    lambdaCurr = -constraint / (gradSum + fn.Lamb_Eps)

    lambda_Opt[i] = lambdaCurr

@wp.kernel
def calc_deltaPos(
    grid: wp.uint64,
    smoothing_length: float,
    pre_Pos: wp.array[wp.vec3],
    pre_New: wp.array[wp.vec3],
    lambda_Opt: wp.array[float],
    delta_Pos: wp.array[wp.vec3]
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    currPos = pre_Pos[i]
    qurryPos = pre_New[i]    # Snapshot
    neighbors = wp.hash_grid_query(grid, qurryPos, smoothing_length)

    # data
    delta_Q = fn.S_corr_Q * smoothing_length
    WDeltaQ = fn.Poly6(delta_Q, smoothing_length)
    lambda_i = lambda_Opt[i]
    delta_Movement = wp.vec3(0,0,0)
    pressure_scale = fn.Fluid_Mass * fn.Inv_Rho0

    for index in neighbors:
        if index == i: 
            continue
        NPos = pre_Pos[index]
        O2N = NPos - currPos
        sqrD2N = wp.dot(O2N, O2N)
        if sqrD2N > fn.square(smoothing_length):
            continue

        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)
        poly6 = fn.Poly6(dst2N, smoothing_length)

        x = poly6 * (1.0/WDeltaQ)
        x2 = x * x
        x4 = x2 * x2 
        S_corr = -fn.S_corr_K * x4

        lambda_j = lambda_Opt[index]
        lambda_Sum = lambda_i + lambda_j + S_corr
        currGrad = dir2N * fn.DPow3(dst2N, smoothing_length)
        delta_Movement -= lambda_Sum * currGrad

    delta_Pos[i] = delta_Movement * pressure_scale

    # what if fusion？
    # currPos = delta_Pos[tid] + pre_Pos[tid]
    # pre_Pos[tid] = fn.apply_boundary(currPos)

@wp.kernel
def update_prePos(
    boundary: wp.vec3,
    delta_Pos: wp.array[wp.vec3],
    pre_Pos: wp.array[wp.vec3]
):
    # can this kernel fusion into last kernel?
    tid = wp.tid()
    currPos = delta_Pos[tid] + pre_Pos[tid]
    pre_Pos[tid] = fn.apply_boundary(currPos, boundary, tid)

@wp.kernel
def update_position(
    dt: float,
    boundary: wp.vec3,
    pre_Pos: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3]
):
    tid = wp.tid()
    pPos = pre_Pos[tid]
    currPos = pos[tid]
    incomingVel = vel[tid]
    reconstructedVel = (pPos - currPos) / dt
    reconstructedVel = fn.apply_boundary_collision_velocity(
        pPos, boundary, reconstructedVel, incomingVel
    )
    # Limit reconstruction before it becomes input to XSPH/vorticity.
    speed_squared = wp.dot(reconstructedVel, reconstructedVel)
    if speed_squared > fn.MaxVel * fn.MaxVel:
        reconstructedVel *= fn.MaxVel / wp.sqrt(speed_squared)

    # apply
    pos[tid] = pPos
    vel[tid] = reconstructedVel

@wp.kernel
def calc_curl(
    grid: wp.uint64,
    smoothing_length: float,
    pre_Pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    curl: wp.array[wp.vec4]
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    currPos = pre_Pos[i]
    neighbors = wp.hash_grid_query(grid, currPos, smoothing_length)

    # data
    currVel = vel[i]
    omega_i = wp.vec3(0,0,0)
    volume_scale = fn.Fluid_Mass * fn.Inv_Rho0

    for index in neighbors:
        if index == i: 
            continue
        NPos = pre_Pos[index]
        O2N = NPos - currPos
        sqrD2N = wp.dot(O2N, O2N)
        if sqrD2N > fn.square(smoothing_length):
            continue
        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)

        # curl
        NVel = vel[index]
        Vel_ij = NVel - currVel
        omega_i += volume_scale * wp.cross(Vel_ij, dir2N * fn.DPow3(dst2N, smoothing_length))

    curl[i] = wp.vec4(omega_i[0], omega_i[1], omega_i[2], wp.length(omega_i))

@wp.kernel
def calc_visvor(
    grid: wp.uint64,
    dt: float,
    smoothing_length: float,
    pre_Pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    curl: wp.array[wp.vec4],
    delta_Vel: wp.array[wp.vec3]
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    currPos = pre_Pos[i]
    neighbors = wp.hash_grid_query(grid, currPos, smoothing_length)

    # data
    currVel = vel[i]
    impulse = wp.vec3(0,0,0)
    etaTotal = wp.vec3(0,0,0)
    vel_corr = wp.vec3(0,0,0)
    volume_scale = fn.Fluid_Mass * fn.Inv_Rho0

    for index in neighbors:
        if index == i: 
            continue
        NPos = pre_Pos[index]
        O2N = NPos - currPos
        sqrD2N = wp.dot(O2N, O2N)
        if sqrD2N > fn.square(smoothing_length):
            continue
        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)

        # calc
        NVel = vel[index]
        Vel_ij = NVel - currVel
        NCurl = curl[index]
        currGrad = fn.DPow3(dst2N, smoothing_length)

        # voricity
        etaTotal += -volume_scale * dir2N * currGrad * (NCurl[3] - curl[i][3])

        # viscosity
        vel_corr += volume_scale * Vel_ij * fn.Poly6(dst2N, smoothing_length)


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
        self.renderer = (
            wp.render.UsdRenderer(stage_path)
            if stage_path else None
        )

    def reverse_gravity(self):
        """Reverse runtime gravity while preserving the startup configuration."""
        self.gravity = -self.gravity

    def reset(self):
        """Restore startup particles, gravity and time using the existing buffers."""
        fn.init_particles(self)
        for buffer in (
            self.v, self.pre_Pos, self.pre_New, self.delta_Pos,
            self.lambda_Opt, self.delta_Vel, self.curl,
        ):
            buffer.zero_()
        self.gravity = wp.vec3(*self.config.gravity)
        self.sim_time = 0.0
        self.grid.build(self.pos, self.smoothing_length)

    def step(self):
        with wp.ScopedTimer("sub-step", synchronize=True):
            # 3 substep: (or more)
            for _ in range(self.substep):
                # Prologue
                with wp.ScopedTimer("Init and SHGrid", active=self.verbose):
                    # apply and predict
                    wp.launch(
                        kernel=apply_predict,
                        dim=self.n,
                        inputs=[self.pos, self.dt, self.gravity],
                        outputs= [self.pre_Pos, self.pre_New, self.v]
                    )
                    self.grid.build(self.pre_New, self.smoothing_length)

                # Solver
                with wp.ScopedTimer("Core Iterations", active=self.verbose):
                    # iterations per substep
                    for i in range(self.iterations):
                        wp.launch(
                            kernel=calc_lambda,
                            dim=self.n,
                            inputs=[
                                self.grid.id, 
                                self.smoothing_length,
                                self.pre_Pos,
                                self.pre_New,
                                int(self.config.clamp_negative_pressure),
                            ],
                            outputs=[self.lambda_Opt]
                        )
                        wp.launch(
                            kernel=calc_deltaPos, 
                            dim=self.n,
                            inputs=[
                                self.grid.id, 
                                self.smoothing_length, 
                                self.pre_Pos, 
                                self.pre_New,
                                self.lambda_Opt
                            ],
                            outputs=[self.delta_Pos]
                        )
                        wp.launch(
                            kernel=update_prePos, 
                            dim=self.n,
                            inputs=[
                                self.boundary,
                                self.delta_Pos
                            ],
                            outputs=[self.pre_Pos]
                        )

                # Epilogue
                with wp.ScopedTimer("Update with effect", active=self.verbose):
                    self.grid.build(self.pre_Pos, self.smoothing_length)
                    wp.launch(
                        kernel=update_position,
                        dim=self.n,
                        inputs=[self.dt, self.boundary, self.pre_Pos],
                        outputs=[self.pos, self.v]
                    )
                    wp.launch(
                        kernel=calc_curl,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.smoothing_length,
                            self.pre_Pos,
                            self.v
                        ],
                        outputs=[self.curl]
                    )
                    wp.launch(
                        kernel=calc_visvor,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.dt,
                            self.smoothing_length,
                            self.pre_Pos,
                            self.v,
                            self.curl
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


if __name__ == "__main__":
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
        "--clamp-negative-pressure",
        action=argparse.BooleanOptionalAction,
        default=fn.PBFConfig.clamp_negative_pressure,
        help="Clamp negative density constraints to zero.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path, verbose=args.verbose,
            config=fn.PBFConfig(clamp_negative_pressure=args.clamp_negative_pressure),
        )

        for _ in range(args.num_frames):
            example.render()
            example.step()

        if example.renderer:
            example.renderer.save()
