"""Native tank assembly and loading of independently published boundaries."""

from pathlib import Path

from .asset_io import ASSET_DIR, load_asset, load_asset_volumes, build_command
from .scene_data import (WALL_THICKNESS, BOX_HALF_EXTENTS, BOX_POSITION,
                         tank_box_config, tank_walls, asset_geometry)


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


def load_tank_box_boundaries(config, device, *, asset_dir=None):
    """Load the published tank/box pair; only volume-map refresh is allowed."""
    from Lab_Solver.pbf.boundary_model import BoundaryAsset, BoundaryInstance, BoundarySystem

    root = ASSET_DIR if asset_dir is None else Path(asset_dir)
    command = build_command("tank-box", config)
    loaded = []
    for relative, name in (("tank_scene", "tank_scene"), ("Rigid/tank_box", "tank_box")):
        asset = load_asset(root / relative, asset_geometry(name, config), command=command)
        sdf, volume_map = load_asset_volumes(asset, 4. * config.particle_radius, device)
        loaded.append(BoundaryAsset(asset.sdf_path, asset.mesh_path, sdf, volume_map))
    return BoundarySystem(loaded, [BoundaryInstance(0), BoundaryInstance(1, 0)], device)
