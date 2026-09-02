"""Simulation adapter around Warp's official SPH example."""

from warp.examples.core.example_sph import Example as SphExample


def create_sph_simulation(verbose: bool = False) -> SphExample:
    """Create the stock simulation with its USD renderer disabled."""
    return SphExample(stage_path=None, verbose=verbose)
