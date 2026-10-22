"""Connect the PBF frontend to simulation creation, gravity and reset controls."""

import warp as wp

try:
    from .PBF import PBF as PBFSolver
    from . import pbf_helper as fn
except ImportError:
    from PBF import PBF as PBFSolver
    import pbf_helper as fn


def create_pbf_simulation(verbose: bool = False, config=None) -> PBFSolver:
    """Create PBF with optional configuration and no USD output."""
    return PBFSolver(stage_path=None, verbose=verbose, config=config)


def reverse_gravity(sim: PBFSolver) -> None:
    """Reverse runtime gravity without changing the startup configuration."""
    sim.gravity = -sim.gravity


def reset_simulation(sim: PBFSolver) -> None:
    """Restore the startup lattice and zero velocity without reallocating resources."""
    with wp.ScopedDevice(sim.pos.device):
        wp.launch(
            kernel=fn.init_particles,
            dim=sim.n,
            inputs=[
                wp.vec3(*sim.config.block_start),
                sim.particle_spacing,
                sim.block_counts,
            ],
            outputs=[sim.pos],
            device=sim.pos.device,
        )
        for buffer in (
            sim.v, sim.pre_Pos, sim.delta_Pos,
            sim.lambda_Opt, sim.delta_Vel, sim.curl,
        ):
            buffer.zero_()
        # Neighbor slots are ignored until the next step rebuilds their counts.
        sim.n_Cache.NeighborCount.zero_()
        sim.gravity = wp.vec3(*sim.config.gravity)
        sim.sim_time = 0.0
        sim.grid.build(sim.pos, sim.smoothing_length)
