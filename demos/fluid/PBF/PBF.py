import numpy as np

import warp as wp
import warp.render

if __package__:
    from .pbf_functions import *
else:
    from pbf_functions import *  # noqa: F403

@wp.kernel
def apply_predict(
    pos: wp.array[wp.vec3],
    dt: float,
    prePos: wp.array[wp.vec3],
    preNew: wp.array[wp.vec3],
    vel: wp.array[wp.vec3]
):
    tid = wp.tid()

    v = vel[tid]
    gravity = wp.vec3(0, -9.8, 0)
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
    lambda_Opt: wp.array[float]
):
    density = float(0.0)
    grad_j = float(0.0)
    grad_i = wp.vec3(0,0,0)

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
        if sqrD2N > square(smoothing_length):
            continue
        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)
        density += Poly6(dst2N, smoothing_length)
        currGrad = dir2N * Inv_Rho0 * DPow3(dst2N, smoothing_length)

        grad_i += currGrad
        if index!=i : 
            grad_j += wp.dot(currGrad, currGrad)
        # do something.
    
    gradSum = grad_j + wp.dot(grad_i, grad_i)
    constraint = wp.max(density*Inv_Rho0 - 1.0, 0.0)
    # constraint = density*Inv_Rho0 - 1.0
    lambdaCurr = -constraint / (gradSum + Lamb_Eps)

    # Density[tid] = density # density only for lambda.
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
    delta_Q = 0.3 * smoothing_length
    WDeltaQ = Poly6(delta_Q, smoothing_length)
    lambda_i = lambda_Opt[i]
    delta_Movement = wp.vec3(0,0,0)

    for index in neighbors:
        if index == i: 
            continue
        NPos = pre_Pos[index]
        O2N = NPos - currPos
        sqrD2N = wp.dot(O2N, O2N)
        if sqrD2N > square(smoothing_length):
            continue

        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)
        poly6 = Poly6(dst2N, smoothing_length)
        S_corr = -S_corr_K * wp.pow(wp.abs(poly6/WDeltaQ), S_corr_N)
        lambda_j = lambda_Opt[index]
        lambda_Sum = lambda_i + lambda_j + S_corr
        currGrad = dir2N * DPow3(dst2N, smoothing_length)
        delta_Movement -= lambda_Sum * currGrad

    delta_Pos[i] = delta_Movement * Inv_Rho0

    # what if fusion？
    # currPos = delta_Pos[tid] + pre_Pos[tid]
    # pre_Pos[tid] = apply_boundary(currPos)

@wp.kernel
def update_prePos(
    boundary: wp.vec3,
    delta_Pos: wp.array[wp.vec3],
    pre_Pos: wp.array[wp.vec3]
):
    # can this kernel fusion into last kernel?
    tid = wp.tid()
    currPos = delta_Pos[tid] + pre_Pos[tid]
    pre_Pos[tid] = apply_boundary(currPos, boundary, tid)

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
    reconstructedVel = apply_boundary_collision_velocity(
        pPos, boundary, reconstructedVel, incomingVel
    )

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

    for index in neighbors:
        if index == i: 
            continue
        NPos = pre_Pos[index]
        O2N = NPos - currPos
        sqrD2N = wp.dot(O2N, O2N)
        if sqrD2N > square(smoothing_length):
            continue
        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)

        # curl
        NVel = vel[index]
        Vel_ij = NVel - currVel
        omega_i += wp.cross(Vel_ij, dir2N * DPow3(dst2N, smoothing_length)) 

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
    currDensity = float(0.0)
    impulse = wp.vec3(0,0,0)
    etaTotal = wp.vec3(0,0,0)
    vel_corr = wp.vec3(0,0,0)

    for index in neighbors:
        if index == i: 
            continue
        NPos = pre_Pos[index]
        O2N = NPos - currPos
        sqrD2N = wp.dot(O2N, O2N)
        if sqrD2N > square(smoothing_length):
            continue
        dst2N = wp.sqrt(sqrD2N)
        dir2N = O2N/dst2N if dst2N > 0 else wp.vec3(0,0,0)

        # calc
        NVel = vel[index]
        Vel_ij = NVel - currVel
        NCurl = curl[index]
        currGrad = DPow3(dst2N, smoothing_length)

        # voricity
        etaTotal += -1.0 * dir2N * currGrad * NCurl[3]

        # viscosity
        vel_corr += Vel_ij * Poly6(dst2N, smoothing_length)
        # vel_corr += Vel_ij * Pow3(dst2N, smoothing_length) 

        # currDensity += Poly6(dst2N,smoothing_length)


    if wp.length(etaTotal) > 1e-4 and vorConfirm >0.0:
        epsilon = dt * vorConfirm
        currCurl = curl[i]
        N = wp.normalize(etaTotal)
        force = wp.cross(N, wp.vec3(currCurl[0],currCurl[1],currCurl[2]))
        impulse += epsilon * force

    # XSPH
    impulse += visStrength * vel_corr   # or vorCon?currDensity
    delta_Vel[i] = impulse

@wp.kernel
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


# ============================================

@wp.kernel
def compute_density(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_rho: wp.array[float],
    density_normalization: float,
    smoothing_length: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]

    # store density
    rho = float(0.0)

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    # loop through neighbors to compute density
    for index in neighbors:
        # compute distance
        distance = x - particle_x[index]

        # compute kernel derivative
        rho += density_kernel(distance, smoothing_length)

    # add external potential
    particle_rho[i] = density_normalization * rho


@wp.kernel
def get_acceleration(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_rho: wp.array[float],
    particle_a: wp.array[wp.vec3],
    isotropic_exp: float,
    base_density: float,
    gravity: float,
    pressure_normalization: float,
    viscous_normalization: float,
    smoothing_length: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]
    v = particle_v[i]
    rho = particle_rho[i]
    pressure = isotropic_exp * (rho - base_density)

    # store forces
    pressure_force = wp.vec3()
    viscous_force = wp.vec3()

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    # loop through neighbors to compute acceleration
    for index in neighbors:
        if index != i:
            # get neighbor velocity
            neighbor_v = particle_v[index]

            # get neighbor density and pressures
            neighbor_rho = particle_rho[index]
            neighbor_pressure = isotropic_exp * (neighbor_rho - base_density)

            # compute relative position
            relative_position = particle_x[index] - x

            # calculate pressure force
            pressure_force += diff_pressure_kernel(
                relative_position, pressure, neighbor_pressure, neighbor_rho, smoothing_length
            )

            # compute kernel derivative
            viscous_force += diff_viscous_kernel(relative_position, v, neighbor_v, neighbor_rho, smoothing_length)

    # sum all forces
    force = pressure_normalization * pressure_force + viscous_normalization * viscous_force

    # add external potential
    particle_a[i] = force / rho + wp.vec3(0.0, gravity, 0.0)


@wp.kernel
def apply_bounds(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    damping_coef: float,
    width: float, height: float, length: float,
):
    tid = wp.tid()

    # get pos and velocity
    x = particle_x[tid]
    v = particle_v[tid]

    # clamp x left
    if x[0] < 0.0:
        x = wp.vec3(0.0, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp x right
    if x[0] > width:
        x = wp.vec3(width, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp y bot
    if x[1] < 0.0:
        x = wp.vec3(x[0], 0.0, x[2])
        v = wp.vec3(v[0], v[1] * damping_coef, v[2])

    # clamp z left
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # clamp z right
    if x[2] > length:
        x = wp.vec3(x[0], x[1], length)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # apply clamps
    particle_x[tid] = x
    particle_v[tid] = v


@wp.kernel
def kick(particle_v: wp.array[wp.vec3], particle_a: wp.array[wp.vec3], dt: float):
    tid = wp.tid()
    v = particle_v[tid]
    particle_v[tid] = v + particle_a[tid] * dt


@wp.kernel
def drift(particle_x: wp.array[wp.vec3], particle_v: wp.array[wp.vec3], dt: float):
    tid = wp.tid()
    x = particle_x[tid]
    particle_x[tid] = x + particle_v[tid] * dt


@wp.kernel
def initialize_particles(
    particle_x: wp.array[wp.vec3],
    smoothing_length: float,
    width: float, height: float, length: float
):
    tid = wp.tid()

    # particle number per axis
    nr_x = int(width / 4.0 / smoothing_length)
    nr_y = int(height / smoothing_length)
    nr_z = int(length / 4.0 / smoothing_length)

    # calculate particle position
    z = float(tid % nr_z)
    y = float((tid // nr_z) % nr_y)
    x = float((tid // (nr_z * nr_y)) % nr_x)
    pos = smoothing_length * wp.vec3(x, y, z)

    # add small jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * smoothing_length * wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))

    # set position
    particle_x[tid] = pos


class Example:
    def __init__(self, stage_path="example_pbf.usd", verbose=False):
        self.verbose = verbose

        # render params
        fps = 90
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0

        # simulation params
        self.smoothing_length = smoothingLength  # NOTE change this to adjust number of particles
        self.width = 80.0  # x
        self.height = 80.0  # y
        self.length = 80.0  # z
        self.boundary = wp.vec3(self.width, self.height, self.length)
        self.isotropic_exp = 20
        self.base_density = 1.0
        self.particle_mass = 0.01 * self.smoothing_length**3  # reduce according to smoothing length
        # self.dt = 0.01 * self.smoothing_length  # decrease sim dt by smoothing length
        self.dt = self.frame_dt / 3.0
        self.dynamic_visc = 0.025
        self.damping_coef = -0.95
        self.gravity = -0.1
        self.n = int(
            self.height * (self.width / 4.0) * (self.height / 4.0) / (self.smoothing_length**3)
        )  # number particles (small box in corner)
        self.sim_step_to_frame_ratio = int(32 / self.smoothing_length)
        self.substep = 3
        self.iterations = 3

        # constants
        self.density_normalization = (315.0 * self.particle_mass) / (
            64.0 * np.pi * self.smoothing_length**9
        )  # integrate density kernel
        self.pressure_normalization = -(45.0 * self.particle_mass) / (np.pi * self.smoothing_length**6)
        self.viscous_normalization = (45.0 * self.dynamic_visc * self.particle_mass) / (
            np.pi * self.smoothing_length**6
        )

        # allocate arrays
        self.pos = wp.empty(self.n, dtype=wp.vec3)
        self.pre_Pos = wp.zeros(self.n, dtype=wp.vec3)
        self.pre_New = wp.zeros(self.n, dtype=wp.vec3)
        self.delta_Pos = wp.zeros(self.n, dtype=wp.vec3)
        self.v = wp.zeros(self.n, dtype=wp.vec3)

        self.delta_Vel = wp.zeros(self.n, wp.vec3)
        self.curl = wp.zeros(self.n, dtype=wp.vec4)

        self.lambda_Opt = wp.zeros(self.n, dtype=float)
        self.rho = wp.zeros(self.n, dtype=float)
        self.a = wp.zeros(self.n, dtype=wp.vec3)

        # set random positions
        wp.launch(
            kernel=initialize_particles,
            dim=self.n,
            inputs=[self.pos, self.smoothing_length, self.width, self.height, self.length],
        )  # initialize in small area

        # create hash array
        grid_size = int(self.height / (4.0 * self.smoothing_length))
        self.grid = wp.HashGrid(grid_size, grid_size, grid_size)

        # renderer
        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)

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
                        inputs=[self.pos, self.dt],
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
                                self.pre_New
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
            # self.sim_time += self.dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.pos.numpy(), radius=self.smoothing_length, name="points", colors=(0.8, 0.3, 0.2)
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

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, verbose=args.verbose)

        for _ in range(args.num_frames):
            example.render()
            example.step()

        if example.renderer:
            example.renderer.save()
