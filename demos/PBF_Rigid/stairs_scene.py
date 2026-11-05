"""Original stairs and four explicitly assembled free rigid bodies."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from .asset_io import ASSET_DIR, PublishedAsset, load_asset, load_asset_volumes, build_command
from .scene_data import BOX_SIZE, SPHERE_RADIUS, SPOT_LENGTH, RIGIDS, asset_geometry


ROOT = Path(__file__).resolve().parent


@dataclass
class StairsAssets:
    static_boxes: tuple
    bunny_hulls: tuple
    spot_hulls: tuple
    boundaries: dict[str, PublishedAsset]
    static_mesh: Path
    static_sdf: Path
    static_asset: PublishedAsset


def load_stairs_assets(config, *, asset_dir=None):
    """Read published geometry and physical properties; no mesh processing."""
    root = ASSET_DIR if asset_dir is None else Path(asset_dir)
    command = build_command("stairs-rigids", config)
    static = load_asset(root / "stairs_scene", asset_geometry("stairs_scene", config), command=command)
    boundaries = {name: load_asset(root / "Rigid" / f"stairs_{name}",
        asset_geometry(f"stairs_{name}", config), command=command) for name in ("box", "sphere", "spot")}
    if not static.hulls or not boundaries["spot"].hulls or not boundaries["spot"].metadata.get("mass"):
        raise ValueError(f"Missing published convex collision or Spot mass properties. Build offline:\n{command}")
    boxes = np.asarray(static.metadata.get("static_boxes"), dtype=float)
    if boxes.shape != (24, 2, 3) or not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= 0):
        raise ValueError(f"Invalid published stair boxes. Build offline:\n{command}")
    return StairsAssets(tuple((tuple(c), tuple(h)) for c, h in boxes), static.hulls,
        boundaries["spot"].hulls, boundaries, static.mesh_path, static.sdf_path, static)


def _add_hulls(builder, body, hulls, label):
    import newton

    cfg = builder.ShapeConfig(density=0., mu=.5, restitution=0.)
    for i, hull in enumerate(hulls):
        mesh = newton.Mesh(np.asarray(hull.vertices, dtype=np.float32),
                           np.asarray(hull.faces, dtype=np.int32).reshape(-1), compute_inertia=False)
        builder.add_shape_convex_hull(body, mesh=mesh, cfg=cfg, label=f"{label}_{i}")


def add_stairs_rigids(builder, assets):
    """World collision first, then ordered bodies; never sum convex-piece mass."""
    static_cfg = builder.ShapeConfig(density=0., mu=.5, restitution=0.)
    for i, (center, half) in enumerate(assets.static_boxes):
        builder.add_shape_box(-1, xform=wp.transform(center, wp.quat_identity()),
            hx=half[0], hy=half[1], hz=half[2], cfg=static_cfg, label=f"stairs_solid_{i}")
    _add_hulls(builder, -1, assets.bunny_hulls, "static_bunny")
    body_ids = []
    for instance in RIGIDS:
        cfg = builder.ShapeConfig(density=instance.density, mu=.5, restitution=0.)
        if instance.asset == "spot":
            mass = assets.boundaries["spot"].metadata["mass"]
            com = np.asarray(mass["com"])
            body = builder.add_body(
                xform=wp.transform(np.asarray(instance.center) - com, wp.quat_identity()),
                mass=instance.density * mass["volume"], com=wp.vec3(*com),
                inertia=wp.mat33(*(instance.density * np.asarray(mass["inertia_unit_density"])).reshape(-1)),
                lock_inertia=True, label=instance.name)
            _add_hulls(builder, body, assets.spot_hulls, "spot_proxy")
        else:
            body = builder.add_body(xform=wp.transform(instance.center, wp.quat_identity()), label=instance.name)
            if instance.asset == "box":
                builder.add_shape_box(body, hx=BOX_SIZE / 2, hy=BOX_SIZE / 2, hz=BOX_SIZE / 2,
                                      cfg=cfg, label=instance.name)
            else:
                builder.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg, label=instance.name)
        body_ids.append(body)
    return body_ids


def load_stairs_boundaries(assets, config, device):
    """Load each volume once and map the explicit body order to four instances."""
    from demos.PBF_Rigid.solver.boundary_model import (
        BoundaryAsset, BoundaryInstance, BoundarySystem,
    )

    names = ("box", "sphere", "spot")
    sources = [assets.static_asset, *(assets.boundaries[name] for name in names)]
    loaded = []
    for asset in sources:
        sdf, volume_map = load_asset_volumes(asset, 4. * config.particle_radius, device)
        loaded.append(BoundaryAsset(asset.sdf_path, asset.mesh_path, sdf, volume_map))
    asset_ids = {name: i + 1 for i, name in enumerate(names)}
    instances = [BoundaryInstance(0)]
    instances += [BoundaryInstance(asset_ids[item.asset], i) for i, item in enumerate(RIGIDS)]
    return BoundarySystem(loaded, instances, device)
