"""Simulation adapter for the Warp PBF example."""

try:
    from .PBF import Example as PBFExample
except ImportError:
    from PBF import Example as PBFExample


def create_pbf_simulation(verbose: bool = False, config=None) -> PBFExample:
    """Create PBF with optional configuration and no USD output."""
    return PBFExample(stage_path=None, verbose=verbose, config=config)
