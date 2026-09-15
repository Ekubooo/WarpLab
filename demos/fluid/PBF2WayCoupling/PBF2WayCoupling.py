"""Warp PBF + Akinci particle boundaries and six-DOF mesh rigid bodies.

All simulation kernels and their launch order live here. Numerical reference:
SPlisHSPlasH f3f677140761db7637b5443beb54f19f1f835ed4 (MIT).
"""
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import warp as wp

try:
    from . import coupling_functions as fn
    from . import coupling_initialization as init
except ImportError:
    import coupling_functions as fn
    import coupling_initialization as init


@wp.struct
class RigidState:
    position: wp.array(dtype=wp.vec3)
    rotation: wp.array(dtype=wp.quat)
    velocity: wp.array(dtype=wp.vec3)
    omega: wp.array(dtype=wp.vec3)
    inverse_mass: wp.array(dtype=float)
    inverse_inertia: wp.array(dtype=wp.mat33)
    force: wp.array(dtype=wp.vec3)
    torque: wp.array(dtype=wp.vec3)
    mesh: wp.array(dtype=wp.uint64)
    lower: wp.array(dtype=wp.vec3)
    upper: wp.array(dtype=wp.vec3)
    restitution: wp.array(dtype=float)
    friction: wp.array(dtype=float)
    wall: wp.array(dtype=int)


@wp.struct
class BoundaryState:
    local: wp.array(dtype=wp.vec3)
    position: wp.array(dtype=wp.vec3)
    velocity: wp.array(dtype=wp.vec3)
    volume: wp.array(dtype=float)
    body: wp.array(dtype=int)


@wp.struct
class Neighbors:
    fluid_count: wp.array(dtype=int)
    boundary_count: wp.array(dtype=int)
    fluid: wp.array2d(dtype=int)
    boundary: wp.array2d(dtype=int)
    overflow: wp.array(dtype=int)


@wp.struct
class Contacts:
    count: wp.array(dtype=int)
    a: wp.array(dtype=int)
    b: wp.array(dtype=int)
    point: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)
    gap: wp.array(dtype=float)
    target: wp.array(dtype=float)
    normal_impulse: wp.array(dtype=float)
    tangent_impulse: wp.array(dtype=wp.vec3)
    overflow: wp.array(dtype=int)


@wp.kernel
def update_boundary(rigid: RigidState, boundary: BoundaryState):
    i = wp.tid()
    b = boundary.body[i]
    r = wp.quat_rotate(rigid.rotation[b], boundary.local[i])
    boundary.position[i] = rigid.position[b] + r
    boundary.velocity[i] = fn.point_velocity(rigid.velocity[b], rigid.omega[b], r)


@wp.kernel
def boundary_volumes(grid: wp.uint64, h: float, rigid: RigidState, boundary: BoundaryState):
    i = wp.tid()
    b = boundary.body[i]
    total = float(0.0)
    for j in wp.hash_grid_query(grid, boundary.position[i], h):
        c = boundary.body[j]
        if b == c or (rigid.inverse_mass[b] == 0.0 and rigid.inverse_mass[c] == 0.0):
            total += fn.poly6(wp.length(boundary.position[i] - boundary.position[j]), h)
    boundary.volume[i] = 1.0 / total


@wp.kernel
def exclude_solid_particles(x: wp.array(dtype=wp.vec3), rigid: RigidState, radius: float,
                            keep: wp.array(dtype=int)):
    i = wp.tid()
    valid = int(1)
    for b in range(rigid.position.shape[0]):
        if rigid.wall[b] == 0:
            local = wp.quat_rotate_inv(rigid.rotation[b], x[i] - rigid.position[b])
            query = wp.mesh_query_point_sign_normal(rigid.mesh[b], local, 1000.0)
            if query.result:
                closest = wp.mesh_eval_position(rigid.mesh[b], query.face, query.u, query.v)
                if query.sign * wp.length(local - closest) < radius:
                    valid = 0
    keep[i] = valid


@wp.kernel
def predict(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), old: wp.array(dtype=wp.vec3),
            gravity: wp.vec3, dt: float):
    i = wp.tid()
    old[i] = x[i]
    v[i] += dt * gravity
    x[i] += dt * v[i]


@wp.kernel
def cache_neighbors(fluid_grid: wp.uint64, boundary_grid: wp.uint64,
                    x: wp.array(dtype=wp.vec3), boundary: BoundaryState, h: float, neighbors: Neighbors):
    i = wp.hash_grid_point_id(fluid_grid, wp.tid())
    nf = int(0)
    nb = int(0)
    for j in wp.hash_grid_query(fluid_grid, x[i], h):
        if j != i and wp.length_sq(x[i] - x[j]) < h*h:
            if nf < neighbors.fluid.shape[1]:
                neighbors.fluid[i, nf] = j
            else:
                wp.atomic_max(neighbors.overflow, 0, 1)
            nf += 1
    for j in wp.hash_grid_query(boundary_grid, x[i], h):
        if wp.length_sq(x[i] - boundary.position[j]) < h*h:
            if nb < neighbors.boundary.shape[1]:
                neighbors.boundary[i, nb] = j
            else:
                wp.atomic_max(neighbors.overflow, 1, 1)
            nb += 1
    neighbors.fluid_count[i] = wp.min(nf, neighbors.fluid.shape[1])
    neighbors.boundary_count[i] = wp.min(nb, neighbors.boundary.shape[1])


@wp.kernel
def density_lambda(x: wp.array(dtype=wp.vec3), boundary: BoundaryState, neighbors: Neighbors,
                   volume: float, h: float, density: wp.array(dtype=float), lambdas: wp.array(dtype=float),
                   error: wp.array(dtype=float)):
    i = wp.tid()
    rho = volume * fn.poly6(0.0, h)
    gi = wp.vec3(0.0)
    norm = float(0.0)
    for k in range(neighbors.fluid_count[i]):
        j = neighbors.fluid[i, k]
        delta = x[i] - x[j]
        rho += volume * fn.poly6(wp.length(delta), h)
        gj = -volume * fn.spiky_gradient(delta, h)
        norm += wp.dot(gj, gj)
        gi -= gj
    for k in range(neighbors.boundary_count[i]):
        j = neighbors.boundary[i, k]
        delta = x[i] - boundary.position[j]
        rho += boundary.volume[j] * fn.poly6(wp.length(delta), h)
        # SPlisHSPlasH excludes individual boundary gradient squares.
        gi += boundary.volume[j] * fn.spiky_gradient(delta, h)
    constraint = wp.max(rho - 1.0, 0.0)
    density[i] = rho
    lambdas[i] = -constraint / (norm + wp.dot(gi, gi) + 1.0e-6)
    wp.atomic_add(error, 0, constraint)


@wp.kernel
def pressure_correction(x: wp.array(dtype=wp.vec3), boundary: BoundaryState, rigid: RigidState,
                        neighbors: Neighbors, volume: float, mass: float, h: float, dt: float,
                        two_way: int, lambdas: wp.array(dtype=float), correction: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dx = wp.vec3(0.0)
    for k in range(neighbors.fluid_count[i]):
        j = neighbors.fluid[i, k]
        dx += (lambdas[i] + lambdas[j]) * volume * fn.spiky_gradient(x[i] - x[j], h)
    for k in range(neighbors.boundary_count[i]):
        j = neighbors.boundary[i, k]
        delta = lambdas[i] * boundary.volume[j] * fn.spiky_gradient(x[i] - boundary.position[j], h)
        dx += delta
        b = boundary.body[j]
        if two_way != 0 and rigid.inverse_mass[b] > 0.0:
            force = -mass * delta / (dt * dt)
            wp.atomic_add(rigid.force, b, force)
            wp.atomic_add(rigid.torque, b, wp.cross(boundary.position[j] - rigid.position[b], force))
    correction[i] = dx


@wp.kernel
def apply_correction(x: wp.array(dtype=wp.vec3), correction: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    x[i] += correction[i]


@wp.kernel
def reconstruct_velocity(x: wp.array(dtype=wp.vec3), old: wp.array(dtype=wp.vec3),
                         dt: float, v: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    v[i] = (x[i] - old[i]) / dt


@wp.kernel
def viscosity(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), boundary: BoundaryState,
              rigid: RigidState, neighbors: Neighbors, density: wp.array(dtype=float), volume: float,
              mass: float, h: float, coefficient: float, boundary_coefficient: float, two_way: int,
              acceleration: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    acc = wp.vec3(0.0)
    for k in range(neighbors.fluid_count[i]):
        j = neighbors.fluid[i, k]
        delta = x[i] - x[j]
        factor = 10.0 * coefficient * volume / density[j]
        acc += factor * wp.dot(v[i] - v[j], delta) / (wp.length_sq(delta) + .01*h*h) * fn.spiky_gradient(delta, h)
    if boundary_coefficient > 0.0:
        for k in range(neighbors.boundary_count[i]):
            j = neighbors.boundary[i, k]
            delta = x[i] - boundary.position[j]
            factor = 10.0 * boundary_coefficient * boundary.volume[j] / density[i]
            a = factor * wp.dot(v[i] - boundary.velocity[j], delta) / (wp.length_sq(delta) + .01*h*h) * fn.spiky_gradient(delta, h)
            acc += a
            b = boundary.body[j]
            if two_way != 0 and rigid.inverse_mass[b] > 0.0:
                force = -mass * a
                wp.atomic_add(rigid.force, b, force)
                wp.atomic_add(rigid.torque, b, wp.cross(boundary.position[j] - rigid.position[b], force))
    acceleration[i] = acc


@wp.kernel
def apply_viscosity(v: wp.array(dtype=wp.vec3), acceleration: wp.array(dtype=wp.vec3), dt: float):
    i = wp.tid()
    v[i] += dt * acceleration[i]


@wp.kernel
def integrate_rigid(rigid: RigidState, gravity: wp.vec3, dt: float):
    b = wp.tid()
    if rigid.inverse_mass[b] > 0.0:
        q = rigid.rotation[b]
        iw = fn.world_inverse_inertia(q, rigid.inverse_inertia[b])
        w = wp.vec3(rigid.omega[b])
        # Euler's rigid-body equation, including the gyroscopic term.
        w += dt * (iw * (rigid.torque[b] - wp.cross(w, wp.inverse(iw) * w)))
        v = rigid.velocity[b] + dt * (gravity + rigid.inverse_mass[b] * rigid.force[b])
        rigid.position[b] += dt * v
        rigid.rotation[b] = wp.normalize(q + .5 * dt * wp.quat(w[0], w[1], w[2], 0.0) * q)
        rigid.velocity[b] = v
        rigid.omega[b] = w


@wp.kernel
def detect_contacts(rigid: RigidState, boundary: BoundaryState, contacts: Contacts,
                    tolerance: float, dt: float):
    i, b = wp.tid()
    a = boundary.body[i]
    if a == b or rigid.inverse_mass[a] == 0.0:
        return
    point = boundary.position[i]
    local = wp.quat_rotate_inv(rigid.rotation[b], point - rigid.position[b])
    gap = float(1.0e6)
    n = wp.vec3(0.0)
    if rigid.wall[b] != 0:
        # Interior of the box is free space. Choose the nearest interior face.
        for axis in range(3):
            distance = local[axis] - rigid.lower[b][axis]
            if distance < gap:
                gap = distance
                n = wp.vec3(0.0)
                n[axis] = 1.0
            distance = rigid.upper[b][axis] - local[axis]
            if distance < gap:
                gap = distance
                n = wp.vec3(0.0)
                n[axis] = -1.0
    else:
        # Conservative local AABB cull before the mesh BVH query.
        lo = rigid.lower[b] - wp.vec3(tolerance)
        hi = rigid.upper[b] + wp.vec3(tolerance)
        if local[0] < lo[0] or local[1] < lo[1] or local[2] < lo[2] or local[0] > hi[0] or local[1] > hi[1] or local[2] > hi[2]:
            return
        query = wp.mesh_query_point_sign_normal(rigid.mesh[b], local, 1000.0)
        if query.result:
            closest = wp.mesh_eval_position(rigid.mesh[b], query.face, query.u, query.v)
            delta = local - closest
            distance = wp.length(delta)
            gap = query.sign * distance
            if distance > 1.0e-8:
                n = query.sign * delta / distance
            else:
                n = wp.mesh_eval_face_normal(rigid.mesh[b], query.face)
            n = wp.quat_rotate(rigid.rotation[b], n)
    if gap < tolerance:
        k = wp.atomic_add(contacts.count, 0, 1)
        if k >= contacts.a.shape[0]:
            wp.atomic_max(contacts.overflow, 0, 1)
            return
        ra, rb = point - rigid.position[a], point - rigid.position[b]
        rel = fn.point_velocity(rigid.velocity[a], rigid.omega[a], ra) - fn.point_velocity(rigid.velocity[b], rigid.omega[b], rb)
        vn = wp.dot(rel, n)
        target = -wp.max(gap, 0.0)/dt + .2 * wp.max(-gap, 0.0)/dt
        if gap <= 0.0 and vn < -0.5:
            target = wp.max(target, -wp.min(rigid.restitution[a], rigid.restitution[b]) * vn)
        contacts.a[k] = a
        contacts.b[k] = b
        contacts.point[k] = point
        contacts.normal[k] = n
        contacts.gap[k] = gap
        contacts.target[k] = target
        contacts.normal_impulse[k] = 0.0
        contacts.tangent_impulse[k] = wp.vec3(0.0)


@wp.kernel
def solve_contacts(rigid: RigidState, contacts: Contacts):
    # Sequential impulses: one device thread owns all body writes. The selected
    # scene has three dynamic bodies; this avoids racing Gauss-Seidel updates.
    for k in range(wp.min(contacts.count[0], contacts.a.shape[0])):
        a, b = contacts.a[k], contacts.b[k]
        n = contacts.normal[k]
        ra, rb = contacts.point[k] - rigid.position[a], contacts.point[k] - rigid.position[b]
        ia = fn.world_inverse_inertia(rigid.rotation[a], rigid.inverse_inertia[a])
        ib = fn.world_inverse_inertia(rigid.rotation[b], rigid.inverse_inertia[b])
        va = wp.vec3(rigid.velocity[a])
        vb = wp.vec3(rigid.velocity[b])
        wa = wp.vec3(rigid.omega[a])
        wb = wp.vec3(rigid.omega[b])
        rel = fn.point_velocity(va, wa, ra) - fn.point_velocity(vb, wb, rb)
        denominator = fn.effective_mass(rigid.inverse_mass[a], ia, ra, n) + fn.effective_mass(rigid.inverse_mass[b], ib, rb, n)
        normal_impulse = wp.max(0.0, contacts.normal_impulse[k] + (contacts.target[k] - wp.dot(rel, n))/denominator)
        impulse = (normal_impulse - contacts.normal_impulse[k]) * n
        contacts.normal_impulse[k] = normal_impulse
        va += rigid.inverse_mass[a] * impulse
        vb -= rigid.inverse_mass[b] * impulse
        wa += ia * wp.cross(ra, impulse)
        wb -= ib * wp.cross(rb, impulse)
        rel = fn.point_velocity(va, wa, ra) - fn.point_velocity(vb, wb, rb)
        tangent = rel - wp.dot(rel, n)*n
        speed = wp.length(tangent)
        if speed > 1.0e-8:
            tangent /= speed
            denominator_t = fn.effective_mass(rigid.inverse_mass[a], ia, ra, tangent) + fn.effective_mass(rigid.inverse_mass[b], ib, rb, tangent)
            jt = contacts.tangent_impulse[k] - speed / denominator_t * tangent
            limit = wp.sqrt(rigid.friction[a] * rigid.friction[b]) * normal_impulse
            if wp.length(jt) > limit:
                jt = wp.normalize(jt) * limit
            change = jt - contacts.tangent_impulse[k]
            contacts.tangent_impulse[k] = jt
            va += rigid.inverse_mass[a] * change
            vb -= rigid.inverse_mass[b] * change
            wa += ia * wp.cross(ra, change)
            wb -= ib * wp.cross(rb, change)
        rigid.velocity[a] = va
        rigid.velocity[b] = vb
        rigid.omega[a] = wa
        rigid.omega[b] = wb


@wp.kernel
def reduce_speed(v: wp.array(dtype=wp.vec3), acceleration: wp.array(dtype=wp.vec3),
                 dt: float, speed: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_max(speed, 0, wp.length_sq(v[i] + dt * acceleration[i]))


@wp.kernel
def reduce_boundary_speed(boundary: BoundaryState, speed: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_max(speed, 0, wp.length_sq(boundary.velocity[i]))


@wp.kernel
def audit_fluid(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), lower: wp.vec3, upper: wp.vec3,
                maxima: wp.array(dtype=float), invalid: wp.array(dtype=int)):
    i = wp.tid()
    for axis in range(3):
        if not wp.isfinite(x[i][axis]) or not wp.isfinite(v[i][axis]):
            wp.atomic_max(invalid, 0, 1)
        violation = wp.max(lower[axis]-x[i][axis], x[i][axis]-upper[axis])
        wp.atomic_max(maxima, 0, wp.max(violation, 0.0))


@wp.kernel
def audit_rigid(rigid: RigidState, contacts: Contacts, maxima: wp.array(dtype=float), invalid: wp.array(dtype=int)):
    i = wp.tid()
    if i < rigid.position.shape[0]:
        for axis in range(3):
            if not wp.isfinite(rigid.position[i][axis]) or not wp.isfinite(rigid.velocity[i][axis]) or not wp.isfinite(rigid.omega[i][axis]):
                wp.atomic_max(invalid, 0, 1)
        norm = wp.length(rigid.rotation[i])
        if not wp.isfinite(norm):
            wp.atomic_max(invalid, 0, 1)
        wp.atomic_max(maxima, 2, wp.abs(norm - 1.0))
    if i < wp.min(contacts.count[0], contacts.gap.shape[0]):
        wp.atomic_max(maxima, 1, wp.max(-contacts.gap[i], 0.0))


@dataclass(frozen=True)
class PBF2WayCouplingConfig:
    scene: str = "dam-break-objects"
    particle_radius: float = .025
    rest_density: float = 1000.0
    gravity: tuple = (0.0, -9.81, 0.0)
    initial_time_step: float = .001
    min_time_step: float = .0001
    max_time_step: float = .005
    cfl_factor: float = .5
    min_iterations: int = 2
    max_iterations: int = 100
    max_density_error_percent: float = .01
    viscosity: float = .01
    boundary_viscosity: float = 0.0
    two_way: bool = True
    max_fluid_neighbors: int = 256
    max_boundary_neighbors: int = 512
    max_contacts: int = 65536
    contact_tolerance: float = .06
    contact_iterations: int = 5

    def __post_init__(self):
        for name in ("particle_radius", "rest_density", "initial_time_step", "min_time_step", "max_time_step", "cfl_factor", "contact_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_iterations", "max_iterations", "max_fluid_neighbors", "max_boundary_neighbors", "max_contacts", "contact_iterations"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("viscosity", "boundary_viscosity", "max_density_error_percent"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not self.min_time_step <= self.initial_time_step <= self.max_time_step:
            raise ValueError("Require min_time_step <= initial_time_step <= max_time_step")
        if self.min_iterations > self.max_iterations:
            raise ValueError("min_iterations must not exceed max_iterations")
        if len(self.gravity) != 3 or not np.isfinite(self.gravity).all():
            raise ValueError("gravity must contain three finite components")


class Example:
    def __init__(self, config=None, device=None, verbose=False):
        self.config = config or PBF2WayCouplingConfig()
        self.verbose = verbose
        wp.config.kernel_cache_dir = str(Path(__file__).resolve().parents[3] / ".warp_cache")
        wp.init()
        self.device = wp.get_device(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
        self.body_models, x, v, self.container_min, self.container_max = init.load_scene(self.config)
        self.renderer = None
        self.sim_time = 0.0
        self.current_dt = self.config.initial_time_step
        self.last_dt = self.current_dt
        self.iterations = 0
        self.density_error_percent = 0.0
        self.iteration_limit_streak = 0
        self.total_steps = 0
        self.support_radius = 4 * self.config.particle_radius
        self.fluid_volume = .8 * (2*self.config.particle_radius)**3
        self.fluid_mass = self.fluid_volume * self.config.rest_density
        nbody = len(self.body_models)
        array = lambda data, dtype: wp.array(np.asarray(data), dtype=dtype, device=self.device)
        zeros = lambda count, dtype: wp.zeros(count, dtype=dtype, device=self.device)
        self.rigid = RigidState()
        for field, data, dtype in (
            ("position", [b.position for b in self.body_models], wp.vec3),
            ("rotation", [b.rotation for b in self.body_models], wp.quat),
            ("velocity", [b.velocity if b.mass else np.zeros(3) for b in self.body_models], wp.vec3),
            ("omega", [b.angular_velocity if b.mass else np.zeros(3) for b in self.body_models], wp.vec3),
            ("inverse_mass", [1/b.mass if b.mass else 0 for b in self.body_models], float),
            ("inverse_inertia", [np.linalg.inv(b.inertia) if b.mass else np.zeros((3,3)) for b in self.body_models], wp.mat33),
            ("lower", [b.vertices.min(axis=0) for b in self.body_models], wp.vec3),
            ("upper", [b.vertices.max(axis=0) for b in self.body_models], wp.vec3),
            ("restitution", [b.restitution for b in self.body_models], float),
            ("friction", [b.friction for b in self.body_models], float),
            ("wall", [int(b.wall) for b in self.body_models], int),
        ):
            setattr(self.rigid, field, array(data, dtype))
        self.rigid.force, self.rigid.torque = zeros(nbody, wp.vec3), zeros(nbody, wp.vec3)
        self.meshes = [wp.Mesh(array(b.vertices, wp.vec3), array(b.faces.flatten(), int)) for b in self.body_models]
        self.rigid.mesh = array([m.id for m in self.meshes], wp.uint64)
        self.boundary = BoundaryState()
        local = np.concatenate([b.samples for b in self.body_models])
        self.boundary.local = array(local, wp.vec3)
        self.boundary.body = array(np.repeat(np.arange(nbody), [len(b.samples) for b in self.body_models]), int)
        self.boundary.position, self.boundary.velocity = zeros(len(local), wp.vec3), zeros(len(local), wp.vec3)
        self.boundary.volume = zeros(len(local), float)
        self.boundary_grid = wp.HashGrid(64, 64, 64, device=self.device)
        self.fluid_grid = wp.HashGrid(64, 64, 64, device=self.device)
        wp.launch(update_boundary, len(local), [self.rigid, self.boundary], device=self.device)
        self.boundary_grid.build(self.boundary.position, self.support_radius)
        wp.launch(boundary_volumes, len(local), [self.boundary_grid.id, self.support_radius, self.rigid, self.boundary], device=self.device)
        keep = zeros(len(x), int)
        wp.launch(exclude_solid_particles, len(x), [array(x, wp.vec3), self.rigid, self.config.particle_radius, keep], device=self.device)
        mask = keep.numpy().astype(bool)
        self.initial_excluded_particles = int((~mask).sum())
        self.positions, self.velocities = array(x[mask], wp.vec3), array(v[mask], wp.vec3)
        self.num_particles = int(mask.sum())
        if not self.num_particles:
            raise ValueError("Scene contains no fluid outside rigid bodies")
        n = self.num_particles
        self.old_positions, self.corrections, self.acceleration = (zeros(n, wp.vec3) for _ in range(3))
        self.densities, self.lambdas = zeros(n, float), zeros(n, float)
        self.error, self.max_speed = zeros(1, float), zeros(1, float)
        self.audit_maxima, self.invalid_state = zeros(3, float), zeros(1, int)
        self.neighbors = Neighbors()
        self.neighbors.fluid_count, self.neighbors.boundary_count = zeros(n, int), zeros(n, int)
        self.neighbors.fluid = zeros((n, self.config.max_fluid_neighbors), int)
        self.neighbors.boundary = zeros((n, self.config.max_boundary_neighbors), int)
        self.neighbors.overflow = zeros(2, int)
        self.contacts = Contacts()
        for field in ("count", "overflow"):
            setattr(self.contacts, field, zeros(1, int))
        for field, dtype in (("a",int),("b",int),("point",wp.vec3),("normal",wp.vec3),("gap",float),("target",float),("normal_impulse",float),("tangent_impulse",wp.vec3)):
            setattr(self.contacts, field, zeros(self.config.max_contacts, dtype))

    def step(self):
        c, d = self.config, self.device
        dt = self.current_dt  # One immutable dt for fluid, forces and rigid bodies.
        n, nb = self.num_particles, len(self.boundary.local)
        self.rigid.force.zero_()
        self.rigid.torque.zero_()
        self.neighbors.overflow.zero_()
        wp.launch(predict, n, [self.positions, self.velocities, self.old_positions, wp.vec3(*c.gravity), dt], device=d)
        self.fluid_grid.build(self.positions, self.support_radius)
        self.boundary_grid.build(self.boundary.position, self.support_radius)
        wp.launch(cache_neighbors, n, [self.fluid_grid.id, self.boundary_grid.id, self.positions, self.boundary, self.support_radius, self.neighbors], device=d)
        if self.neighbors.overflow.numpy().any():
            raise RuntimeError("Neighbor capacity exceeded; increase max_fluid_neighbors/max_boundary_neighbors")
        for iteration in range(c.max_iterations):
            self.error.zero_()
            wp.launch(density_lambda, n, [self.positions, self.boundary, self.neighbors, self.fluid_volume, self.support_radius, self.densities, self.lambdas, self.error], device=d)
            wp.launch(pressure_correction, n, [self.positions, self.boundary, self.rigid, self.neighbors, self.fluid_volume, self.fluid_mass, self.support_radius, dt, int(c.two_way), self.lambdas, self.corrections], device=d)
            wp.launch(apply_correction, n, [self.positions, self.corrections], device=d)
            self.iterations = iteration + 1
            self.density_error_percent = float(self.error.numpy()[0]) / n * 100
            if self.iterations >= c.min_iterations and self.density_error_percent <= c.max_density_error_percent:
                break
        at_limit = self.iterations == c.max_iterations and self.density_error_percent > c.max_density_error_percent
        self.iteration_limit_streak = self.iteration_limit_streak + 1 if at_limit else 0
        if self.iteration_limit_streak == 10 or (self.iteration_limit_streak > 10 and self.iteration_limit_streak % 100 == 0):
            warnings.warn(f"PBF pressure limit reached for {self.iteration_limit_streak} steps; error={self.density_error_percent:.5f}%", RuntimeWarning)
        wp.launch(reconstruct_velocity, n, [self.positions, self.old_positions, dt, self.velocities], device=d)
        self.error.zero_()
        wp.launch(density_lambda, n, [self.positions, self.boundary, self.neighbors, self.fluid_volume, self.support_radius, self.densities, self.lambdas, self.error], device=d)
        wp.launch(viscosity, n, [self.positions, self.velocities, self.boundary, self.rigid, self.neighbors, self.densities, self.fluid_volume, self.fluid_mass, self.support_radius, c.viscosity, c.boundary_viscosity, int(c.two_way), self.acceleration], device=d)
        wp.launch(apply_viscosity, n, [self.velocities, self.acceleration, dt], device=d)
        wp.launch(integrate_rigid, len(self.body_models), [self.rigid, wp.vec3(*c.gravity), dt], device=d)
        wp.launch(update_boundary, nb, [self.rigid, self.boundary], device=d)
        self.contacts.count.zero_()
        self.contacts.overflow.zero_()
        wp.launch(detect_contacts, (nb, len(self.body_models)), [self.rigid, self.boundary, self.contacts, c.contact_tolerance, dt], device=d)
        if self.contacts.overflow.numpy()[0]:
            raise RuntimeError("Contact capacity exceeded; increase max_contacts")
        for _ in range(c.contact_iterations):
            wp.launch(solve_contacts, 1, [self.rigid, self.contacts], device=d)
        wp.launch(update_boundary, nb, [self.rigid, self.boundary], device=d)
        self.max_speed.zero_()
        wp.launch(reduce_speed, n, [self.velocities, self.acceleration, dt, self.max_speed], device=d)
        wp.launch(reduce_boundary_speed, nb, [self.boundary, self.max_speed], device=d)
        wp.launch(audit_fluid, n, [self.positions, self.velocities, wp.vec3(*self.container_min), wp.vec3(*self.container_max), self.audit_maxima, self.invalid_state], device=d)
        wp.launch(audit_rigid, max(c.max_contacts, len(self.body_models)), [self.rigid, self.contacts, self.audit_maxima, self.invalid_state], device=d)
        speed_squared = float(self.max_speed.numpy()[0])
        if self.invalid_state.numpy()[0] or not np.isfinite(speed_squared) or not np.isfinite(self.density_error_percent):
            raise FloatingPointError("Non-finite simulation state")
        self.current_dt = float(np.clip(c.cfl_factor * .4 * (2*c.particle_radius) / np.sqrt(max(speed_squared, 1e-9)), c.min_time_step, c.max_time_step))
        self.last_dt = dt
        self.sim_time += dt
        self.total_steps += 1
        if self.verbose and self.total_steps % 100 == 0:
            print(f"t={self.sim_time:.4f} dt={dt:.6f} iterations={self.iterations} density_error={self.density_error_percent:.5f}%")

    def render(self):
        if self.renderer is None:
            raise RuntimeError("Attach a CouplingRenderer before calling render()")
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render_billboards("fluid", self.positions, self.velocities, self.config.particle_radius, 2.0, 6.0)
        self.renderer.update_rigid_scene(self)
        self.renderer.end_frame()
