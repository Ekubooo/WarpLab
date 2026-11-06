"""Published stairs geometry and three explicitly assembled free rigid bodies."""

from dataclasses import dataclass
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp

from . import boundary_model


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
BOX_SIZE = .30
SPHERE_RADIUS = .15


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
)


@dataclass
class StairsAssets:
    root: Path
    static_boxes: np.ndarray
    bunny_hulls: tuple


def load_rigid_assets(asset_dir=None):
    """Read the published set once; geometry production stays offline."""
    root = ASSET_DIR if asset_dir is None else Path(asset_dir)
    recipes = (
        ("stairs_scene", dict(kind="stairs", version=1)),
        ("Rigid/stairs_box", dict(kind="box", size=[BOX_SIZE] * 3)),
        ("Rigid/stairs_sphere", dict(kind="sphere", radius=SPHERE_RADIUS, subdivisions=3)),
    )
    # Check the complete set before a volume-map cache can be updated.
    paths = [root / (name + suffix) for name, _ in recipes for suffix in (".obj", ".nvdb", ".json")]
    collision_path = root / "stairs_scene.collision.npz"
    for path in (*paths, collision_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing published stairs-rigids asset: {path}; restore the complete PBF/Assets set")
    metadata = {}
    for name, geometry in recipes:
        data = json.loads((root / (name + ".json")).read_text(encoding="utf-8"))
        if data["version"] != 1 or data["geometry"] != geometry or data["local_frame"] != "asset":
            raise ValueError(f"Published geometry or coordinate frame does not match {name}")
        metadata[name] = data
    boxes = np.asarray(metadata["stairs_scene"]["static_boxes"], dtype=np.float32)
    if boxes.shape != (24, 2, 3):
        raise ValueError("Published stairs collision must contain 24 aligned boxes")
    with np.load(collision_path, allow_pickle=False) as saved:
        count = int(saved["count"])
        if not 1 <= count <= 32:
            raise ValueError("Invalid published bunny convex-piece count")
        hulls = tuple((saved[f"vertices_{i}"].copy(), saved[f"faces_{i}"].copy()) for i in range(count))
    return StairsAssets(root, boxes, hulls)


def add_rigid_scene(builder, assets):
    """Create world collision, then ordered bodies with shape-derived inertia."""
    static_cfg = builder.ShapeConfig(density=0., mu=.5, restitution=0.)
    for i, (center, half) in enumerate(assets.static_boxes):
        builder.add_shape_box(-1, xform=wp.transform(center, wp.quat_identity()),
            hx=half[0], hy=half[1], hz=half[2], cfg=static_cfg, label=f"stairs_solid_{i}")
    for i, (vertices, faces) in enumerate(assets.bunny_hulls):
        mesh = newton.Mesh(np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.int32).reshape(-1), compute_inertia=False)
        builder.add_shape_convex_hull(-1, mesh=mesh, cfg=static_cfg, label=f"static_bunny_{i}")

    body_ids = []
    for instance in RIGIDS:
        body = builder.add_body(xform=wp.transform(instance.center, wp.quat_identity()), label=instance.name)
        cfg = builder.ShapeConfig(density=instance.density, mu=.5, restitution=0.)
        if instance.asset == "box":
            builder.add_shape_box(body, hx=BOX_SIZE / 2, hy=BOX_SIZE / 2, hz=BOX_SIZE / 2,
                cfg=cfg, label=instance.name)
        else:
            builder.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg, label=instance.name)
        body_ids.append(body)
    return body_ids


def load_rigid_boundaries(assets, support_radius, device):
    """Load each volume once; two boxes share one asset but distinct instances."""
    loaded = []
    for name in ("stairs_scene", "Rigid/stairs_box", "Rigid/stairs_sphere"):
        stem = assets.root / name
        sdf, volume_map = boundary_model.load_boundary_volumes(
            stem.with_suffix(".nvdb"), support_radius, device=device)
        loaded.append(boundary_model.BoundaryAsset(stem.with_suffix(".nvdb"), stem.with_suffix(".obj"), sdf, volume_map))
    instances = [boundary_model.BoundaryInstance(0)]
    asset_ids = {"box": 1, "sphere": 2}
    instances.extend(boundary_model.BoundaryInstance(asset_ids[body.asset], i) for i, body in enumerate(RIGIDS))
    return boundary_model.BoundarySystem(loaded, instances, device)


def load_render_meshes(boundaries):
    """Published meshes in instance order; repeated assets share host geometry."""
    return [boundaries.mesh_data(instance.asset_id) for instance in boundaries.instances]
