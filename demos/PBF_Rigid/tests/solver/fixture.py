"""Small CUDA boundary assets shared by the new solver and framework tests."""

from pathlib import Path

import numpy as np
import warp as wp
import newton

from demos.PBF_Rigid.solver import SolverPBF, boundary_model as boundary
from demos.sim_Ready import nvdb_io


@wp.kernel
def _fill_plane(sdf: wp.uint64, volume_map: wp.uint64, tiles: wp.array(dtype=wp.vec3i)):
    i = wp.tid()
    local = i % 512
    cell = tiles[i // 512] + wp.vec3i(local // 64, (local // 8) % 8, local % 8)
    point = wp.volume_index_to_world(sdf, wp.vec3(float(cell[0]), float(cell[1]), float(cell[2])))
    wp.volume_store_f(sdf, cell[0], cell[1], cell[2], point[1])
    wp.volume_store_f(volume_map, cell[0], cell[1], cell[2], .00002)


def make_boundaries(directory, *, device="cuda:0", h=.05):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    tiles = wp.array([[x, y, z] for x in (0, 8) for y in (0, 8) for z in (0, 8)],
                     dtype=wp.vec3i, device=device)
    sdf = wp.Volume.allocate_by_tiles(tiles, .05, bg_value=1., translation=(-.2,) * 3, device=device)
    volume_map = wp.Volume.allocate_by_tiles(tiles, .05, bg_value=0., translation=(-.2,) * 3, device=device)
    wp.launch(_fill_plane, len(tiles)*512, inputs=[sdf.id, volume_map.id, tiles], device=device)
    path = directory / "stairs_scene.nvdb"
    nvdb_io.save_volumes((sdf, volume_map), path,
        grid_names=("SDF", boundary._volume_map_name(h, 16)))
    return boundary.BoundarySystem(
        (boundary.BoundaryAsset(path, path.with_suffix(".obj"), sdf, volume_map),),
        (boundary.BoundaryInstance(0),), device)


def make_solver(boundaries, *, radius=.0125, config=None):
    config = SolverPBF.Config() if config is None else config
    spacing = 2.0 * radius
    positions = .025 + spacing * (np.indices((5, 5, 5)).reshape(3, -1).T + 1)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    builder.gravity = (0., -9.81, 0.)
    builder.add_particles(pos=positions.tolist(), vel=[(0., 0., 0.)]*len(positions),
        mass=[.8 * spacing**3 * config.rest_density]*len(positions), radius=[radius]*len(positions))
    model = builder.finalize(device=boundaries.device)
    state = model.state()
    solver = SolverPBF(model, config=config, boundaries=boundaries,
        grid_bounds=((0., 0., 0.), (1., 1., 1.)))
    return solver, state
