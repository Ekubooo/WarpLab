"""Small coupling reference scene, separate from the sim-ready asset workflow.

The tank and box use the same dimensions for Newton collision, the rendered
mesh and the baked volume boundary. Generated assets belong to this scene's
cache; the existing stairs and imported rigid assets are never overwritten.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .config import PBFConfig


WALL_THICKNESS = .1
BOX_HALF_EXTENTS = (.12, .12, .12)
BOX_POSITION = (.5, .75, .5)
VOXEL_SIZE = .0075
ASSET_VERSION = 1
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "pbf" / "tank-box"


def tank_box_config():
    """Fluid defaults for the open, one-meter tank."""
    return replace(PBFConfig(), container_size=(1., 1., 1.),
                   block_start=(.05, .025, .05), block_end=(.95, .55, .95))


def tank_walls(config):
    """Return world-space (center, half extents) for the five solid walls."""
    x, y, z = config.container_size
    t = WALL_THICKNESS
    for center, half, size in zip(BOX_POSITION, BOX_HALF_EXTENTS, (x, y, z)):
        if center - half <= 0. or center + half >= size:
            raise ValueError("tank-box requires the initial box to fit inside container_size")
    return (
        ((x / 2., -t / 2., z / 2.), (x / 2. + t, t / 2., z / 2. + t)),
        ((-t / 2., y / 2., z / 2.), (t / 2., y / 2., z / 2. + t)),
        ((x + t / 2., y / 2., z / 2.), (t / 2., y / 2., z / 2. + t)),
        ((x / 2., y / 2., -t / 2.), (x / 2., y / 2., t / 2.)),
        ((x / 2., y / 2., z + t / 2.), (x / 2., y / 2., t / 2.)),
    )


def add_tank_box(builder, config):
    """Add static walls and one free body; shape density supplies its inertia."""
    import warp as wp

    walls = tank_walls(config)
    wall_cfg = builder.ShapeConfig(density=0., mu=.5, restitution=0.)
    for i, (center, half) in enumerate(walls):
        builder.add_shape_box(-1, xform=wp.transform(center, wp.quat_identity()),
            hx=half[0], hy=half[1], hz=half[2], cfg=wall_cfg, label=f"tank_wall_{i}")
    body_id = builder.add_body(
        xform=wp.transform(BOX_POSITION, wp.quat_identity()), label="floating_box")
    box_cfg = builder.ShapeConfig(density=.5 * config.rest_density, mu=.5, restitution=0.)
    hx, hy, hz = BOX_HALF_EXTENTS
    builder.add_shape_box(body_id, hx=hx, hy=hy, hz=hz, cfg=box_cfg, label="floating_box_shape")
    return body_id


def asset_cache_path(config, *, cache_dir=None, voxel_size=VOXEL_SIZE):
    """Geometry and sampling changes select an independent asset directory."""
    import math
    if not math.isfinite(voxel_size) or voxel_size <= 0.:
        raise ValueError("voxel_size must be finite and positive")
    description = dict(version=ASSET_VERSION, walls=tank_walls(config),
                       box_half=BOX_HALF_EXTENTS, voxel_size=voxel_size,
                       support_radius=4. * config.particle_radius)
    digest = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:20]
    return Path(CACHE_DIR if cache_dir is None else cache_dir) / digest


def _box_mesh(center, half):
    import trimesh
    mesh = trimesh.creation.box(extents=[2. * value for value in half])
    mesh.apply_translation(center)
    return mesh


def _prepare_assets(config, directory, voxel_size, device):
    from Lab_Framework.sim_ready import geometry, pipeline, sdf, validation
    from Lab_Framework.sim_ready.config import BakeOptions

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "Rigid").mkdir(exist_ok=True)
    options = BakeOptions(voxel_size=voxel_size, max_support_radius=4. * config.particle_radius)
    for name in ("tank", "Rigid/box"):
        mesh_path = directory / f"{name}.obj"
        sdf_path = directory / f"{name}.nvdb"
        if mesh_path.is_file() and sdf_path.is_file():
            continue
        if name == "tank":
            mesh = geometry.union([_box_mesh(center, half) for center, half in tank_walls(config)])
        else:
            mesh = _box_mesh((0., 0., 0.), BOX_HALF_EXTENTS)
        checked = validation.validate_solid(mesh)
        volume = sdf.bake_rigid(checked, options=options, device=device)
        pipeline.save_mesh_obj(mesh, mesh_path, digits=17)
        sdf.save_volume(volume, sdf_path)


def load_tank_box_boundaries(config, device, *, cache_dir=None, voxel_size=VOXEL_SIZE):
    """Bake on a cache miss, then load static tank and local-frame box volumes."""
    from Lab_Solver.pbf.boundary_model import BoundaryScene, BoundaryLoadOptions, load_boundaries

    directory = asset_cache_path(config, cache_dir=cache_dir, voxel_size=voxel_size)
    _prepare_assets(config, directory, voxel_size, device)
    return load_boundaries(
        BoundaryScene(directory, static_assets=("tank.nvdb",), default_rigid_count=1),
        BoundaryLoadOptions(4. * config.particle_radius, device=device),
    )
