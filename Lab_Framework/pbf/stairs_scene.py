"""Original stairs and four explicitly assembled free rigid bodies."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

from .rigid_assets import RigidAsset, prepare_boundary, prepare_hulls


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT.parents[1] / ".cache" / "pbf" / "stairs-rigids"
VOXEL_SIZE = .005
BOX_SIZE = .30
SPHERE_RADIUS = .15
SPOT_LENGTH = .35


@dataclass(frozen=True)
class RigidInstance:
    name: str
    asset: str
    density: float
    center: tuple[float, float, float]


RIGIDS = (
    RigidInstance("box_a", "box", 500., (2.10, 1.15, .55)),
    RigidInstance("box_b", "box", 1200., (2.45, 1.50, .55)),
    RigidInstance("sphere", "sphere", 750., (2.45, 1.15, 1.02)),
    RigidInstance("spot", "spot", 500., (2.10, 1.55, 1.02)),
)


@dataclass
class StairsAssets:
    static_boxes: tuple
    bunny_hulls: tuple
    spot_hulls: tuple
    boundaries: dict[str, RigidAsset]
    static_mesh: Path
    static_sdf: Path


def prepare_stairs_assets(config, device, *, cache_dir=None):
    """Prepare the fixed reference geometry before building a Newton Model."""
    from Lab_Framework.sim_ready import config as asset_config, geometry, validation

    root = CACHE_DIR if cache_dir is None else Path(cache_dir)
    options = asset_config.GeometryOptions()
    scene = geometry.prepare_scene(asset_config.ASSETS / "bunny_flat_bottom.obj", options=options)
    boxes = geometry.terrain_boxes(options)
    walls, _ = geometry.container_parts(scene.terrain, options)
    static_boxes = tuple((tuple(mesh.bounds.mean(axis=0)), tuple(mesh.extents / 2))
                         for mesh in [*boxes, *walls])
    source = geometry.prepare_rigid(asset_config.ASSETS / "spot_triangulated_good.obj")
    source.apply_scale(SPOT_LENGTH / source.extents.max())
    spot = geometry.union(validation.validate_rigid_parts(source))
    validation.validate_solid(spot)
    bunny_hulls = prepare_hulls(scene.model, "bunny", root)
    spot_hulls = prepare_hulls(spot, "spot", root)
    meshes = dict(box=trimesh.creation.box(extents=(BOX_SIZE,) * 3),
                  sphere=trimesh.creation.icosphere(subdivisions=3, radius=SPHERE_RADIUS), spot=spot)
    boundaries = {name: prepare_boundary(mesh, name, root,
        support_radius=max(.05, 4. * config.particle_radius), voxel_size=VOXEL_SIZE, device=device)
        for name, mesh in meshes.items()}
    return StairsAssets(static_boxes, bunny_hulls, spot_hulls, boundaries,
                        ROOT / "Assets" / "stairs_scene.obj", ROOT / "Assets" / "stairs_scene.nvdb")


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
            mesh = assets.boundaries["spot"].mesh
            com = mesh.center_mass
            body = builder.add_body(
                xform=wp.transform(np.asarray(instance.center) - com, wp.quat_identity()),
                mass=instance.density * mesh.volume, com=wp.vec3(*com),
                inertia=wp.mat33(*(instance.density * mesh.moment_inertia).reshape(-1)),
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
    from Lab_Solver.pbf.boundary_model import (
        BoundaryAsset, BoundaryInstance, BoundarySystem, load_boundary_volumes,
    )

    names = ("box", "sphere", "spot")
    paths = [(assets.static_sdf, assets.static_mesh)]
    paths += [(assets.boundaries[name].sdf_path, assets.boundaries[name].mesh_path) for name in names]
    loaded = []
    for sdf_path, mesh_path in paths:
        sdf, volume_map = load_boundary_volumes(sdf_path, 4. * config.particle_radius, device=device)
        loaded.append(BoundaryAsset(sdf_path, mesh_path, sdf, volume_map))
    asset_ids = {name: i + 1 for i, name in enumerate(names)}
    instances = [BoundaryInstance(0)]
    instances += [BoundaryInstance(asset_ids[item.asset], i) for i, item in enumerate(RIGIDS)]
    return BoundarySystem(loaded, instances, device)
