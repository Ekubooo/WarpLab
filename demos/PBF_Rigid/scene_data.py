"""Pure geometry and instance recipes shared by offline tools and assembly."""

from dataclasses import dataclass, replace
from .config import PBFConfig


WALL_THICKNESS = .1
BOX_HALF_EXTENTS = (.12, .12, .12)
BOX_POSITION = (.5, .75, .5)


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


def asset_geometry(name, config):
    """Only dimensions/geometry select published assets, never instance density."""
    if name == "tank_scene":
        return dict(kind="tank", walls=[[list(c), list(h)] for c, h in tank_walls(config)])
    recipes = {
        "stairs_scene": dict(kind="stairs", version=1),
        "stairs_box": dict(kind="box", size=[BOX_SIZE] * 3),
        "stairs_sphere": dict(kind="sphere", radius=SPHERE_RADIUS, subdivisions=3),
        "stairs_spot": dict(kind="spot", longest_edge=SPOT_LENGTH, frame="source"),
        "tank_box": dict(kind="box", size=[2 * v for v in BOX_HALF_EXTENTS]),
    }
    return recipes[name]


