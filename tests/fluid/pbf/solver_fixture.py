"""Bind individual test cases to the solver's compile-time configuration and resources."""

from types import SimpleNamespace

import warp as wp

from demos.fluid.PBF import PBF as solver
from demos.fluid.PBF import pbf_helper as fn


def bind_solver(config, grid, sdf, volume_map):
    """Replace the former per-launch arguments and rebuild test-only specializations.

    Production currently uses one fixed configuration and instance per process.
    Tests deliberately switch cases, so cached modules must not retain a previous
    case's coefficients or resource handles. This does not change solver equations.
    """
    fn.init_parameters(SimpleNamespace(config=config))
    fn.GRID_id = wp.constant(wp.uint64(grid.id))
    solver.SDF_id = wp.constant(wp.uint64(sdf.id))
    solver.VMAP_id = wp.constant(wp.uint64(volume_map.id))
    wp.get_module(fn.__name__).unload()
    wp.get_module(solver.__name__).unload()


def self_neighbor(device):
    """One particle's cached neighborhood, including its self-density contribution."""
    cache = fn.NeighborCache()
    cache.NeighborCount = wp.ones(1, dtype=int, device=device)
    cache.CacheBuffer = wp.zeros((1, 1), dtype=int, device=device)
    return cache
