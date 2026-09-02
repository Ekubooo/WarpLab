"""Thin adapter around Warp's official DEM example."""

from warp.examples.core.example_dem import Example as DemExample


def create_dem_simulation() -> DemExample:
    """Create the stock DEM simulation with USD output disabled."""
    return DemExample(stage_path=None)
