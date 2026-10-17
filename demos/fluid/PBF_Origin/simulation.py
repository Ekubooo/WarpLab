"""Simulation creation and state control for the PBF_Origin demo."""

import warp as wp

if __package__:
    from .PBF import Example as PBFExample
    from . import pbf_helper as fn
else:
    from PBF import Example as PBFExample
    import pbf_helper as fn


def create_pbf_simulation(verbose: bool = False, config=None) -> PBFExample:
    """Create PBF with optional configuration and no USD output."""
    return PBFExample(stage_path=None, verbose=verbose, config=config)


def reverse_gravity(sim: PBFExample) -> None:
    """Reverse runtime gravity without changing the startup configuration."""
    sim.gravity = -sim.gravity


def reset_simulation(sim: PBFExample) -> None:
    """Restore startup state in place, preserving simulation and render resources."""
    with wp.ScopedDevice(sim.pos.device):
        fn.init_particles(sim)
        for buffer in (
            sim.v, sim.pre_Pos, sim.pre_New, sim.delta_Pos,
            sim.lambda_Opt, sim.delta_Vel, sim.curl,
        ):
            buffer.zero_()
        sim.gravity = wp.vec3(*sim.config.gravity)
        sim.sim_time = 0.0
        sim.grid.build(sim.pos, sim.smoothing_length)
