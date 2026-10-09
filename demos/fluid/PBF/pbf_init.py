"""Parameter setup, device allocation, and initial particles for PBF.

The simulation owns the resulting attributes. This module does not import the
simulation class, so initialization can grow without a circular dependency.
"""

import warp as wp

if __package__:
    from .pbf_functions import smoothingLength
else:
    from pbf_functions import smoothingLength


def initialize(sim):
    """Prepare all simulation resources before the first step."""
    init_parameters(sim)
    init_device_buffers(sim)
    init_particles(sim)
    init_hash_grid(sim)


def init_parameters(sim):
    """Set frame timing, solver settings, and the current box scene."""
    fps = 60
    sim.substep = 3
    sim.iterations = 5
    sim.frame_dt = 1.0 / fps
    sim.dt = sim.frame_dt / sim.substep

    sim.width = 80.0
    sim.height = 80.0
    sim.length = 80.0
    sim.boundary = wp.vec3(sim.width, sim.height, sim.length)

    sim.smoothing_length = smoothingLength
    # Preserve the existing particle count while separating initialization.
    sim.n = int(
        sim.height * (sim.width / 2.0) * (sim.height / 2.0)
        / (sim.smoothing_length**3)
    )


def init_device_buffers(sim):
    """Allocate particle state and PBF working arrays."""
    sim.pos = wp.empty(sim.n, dtype=wp.vec3)
    sim.v = wp.zeros(sim.n, dtype=wp.vec3)

    # Predicted positions and the snapshot used for neighbor queries.
    sim.pre_Pos = wp.zeros(sim.n, dtype=wp.vec3)
    sim.pre_New = wp.zeros(sim.n, dtype=wp.vec3)
    sim.delta_Pos = wp.zeros(sim.n, dtype=wp.vec3)
    sim.lambda_Opt = wp.zeros(sim.n, dtype=float)

    # Velocity post-processing.
    sim.delta_Vel = wp.zeros(sim.n, dtype=wp.vec3)
    sim.curl = wp.zeros(sim.n, dtype=wp.vec4)


@wp.kernel
def _initialize_particles_kernel(
    particle_x: wp.array[wp.vec3],
    smoothing_length: float,
    width: float,
    height: float,
    length: float,
):
    tid = wp.tid()

    nr_x = int(width / 4.0 / smoothing_length)
    nr_y = int(height / smoothing_length)
    nr_z = int(length / 4.0 / smoothing_length)

    z = float(tid % nr_z)
    y = float((tid // nr_z) % nr_y)
    x = float((tid // (nr_z * nr_y)) % nr_x)
    pos = smoothing_length * wp.vec3(x, y, z)

    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * smoothing_length * wp.vec3(
        wp.randn(state), wp.randn(state), wp.randn(state)
    )
    particle_x[tid] = pos


def init_particles(sim):
    """Fill the position array with the current jittered particle block."""
    wp.launch(
        kernel=_initialize_particles_kernel,
        dim=sim.n,
        inputs=[sim.pos, sim.smoothing_length, sim.width, sim.height, sim.length],
    )


def init_hash_grid(sim):
    """Create the neighbor-search grid; step() builds its contents."""
    sim.grid = wp.HashGrid(128, 128, 128)
