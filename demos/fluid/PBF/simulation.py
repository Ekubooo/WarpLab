"""Simulation adapter for the Warp PBF example."""

try:
    from .PBF import Example as PBFExample
except ImportError:
    from PBF import Example as PBFExample


def create_pbf_simulation(verbose: bool = False) -> PBFExample:
    """Create PBF without the USD renderer used by the standalone example."""
    return PBFExample(stage_path=None, verbose=verbose)
