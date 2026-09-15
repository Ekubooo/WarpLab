"""Deterministic host-side scene and mesh preparation; no solver imports."""
from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == "f":
            ids = [int(v.split("/")[0]) for v in fields[1:]]
            ids = [v - 1 if v > 0 else len(vertices) + v for v in ids]
            faces.extend((ids[0], ids[i], ids[i + 1]) for i in range(1, len(ids) - 1))
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3 or not len(f) or not np.isfinite(v).all():
        raise ValueError(f"Invalid triangle mesh: {path}")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError(f"Invalid OBJ vertex index: {path}")
    edges = np.concatenate((f[:, [0,1]], f[:, [1,2]], f[:, [2,0]]))
    _, inverse, counts = np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    orientation = np.bincount(inverse, weights=np.where(edges[:,0]<edges[:,1],1,-1))
    if np.any(counts != 2) or np.any(orientation != 0):
        raise ValueError(f"Mesh must be closed, welded, and consistently oriented: {path}")
    a,b,c = v[f].transpose(1,0,2)
    if np.einsum("ij,ij->i",a,np.cross(b,c)).sum() <= 0:
        raise ValueError(f"Mesh faces must have outward winding: {path}")
    return v, f


def mass_properties(vertices, faces, density):
    """Integrate signed tetrahedra (origin,a,b,c), then shift to the COM."""
    a, b, c = np.asarray(vertices, dtype=np.float64)[faces].transpose(1, 0, 2)
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    if abs(volume.sum()) < 1e-12:
        raise ValueError("Rigid mesh must enclose nonzero volume")
    volume *= np.sign(volume.sum())
    total = volume.sum()
    sums = a + b + c
    center = (volume[:, None] * sums).sum(axis=0) / (4.0 * total)
    second = sum(np.einsum("ni,nj->nij", x, x) for x in (a, b, c))
    second += np.einsum("ni,nj->nij", sums, sums)
    second = (volume[:, None, None] * second).sum(axis=0) / 20.0
    second -= total * np.outer(center, center)
    inertia = density * (np.trace(second) * np.eye(3) - second)
    if density <= 0 or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("Rigid density and inertia must be positive")
    return density * total, center, inertia


def sample_surface(vertices, faces, spacing):
    """Regular rows along each triangle's longest edge, shared points merged.

    Row height is at most spacing; row endpoints are included. This avoids
    oversampling long, thin triangles with a square barycentric grid.
    """
    points = []
    for tri in np.asarray(vertices)[faces]:
        edges = [np.linalg.norm(tri[(i + 1) % 3] - tri[i]) for i in range(3)]
        k = int(np.argmax(edges))
        a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
        length = edges[k]
        if length < 1e-12:
            continue
        height = np.linalg.norm(np.cross(b - a, c - a)) / length
        rows = max(1, math.ceil(height / spacing))
        for row in range(rows + 1):
            t = row / rows
            left, right = (1-t)*a+t*c, (1-t)*b+t*c
            count = max(1, math.ceil(np.linalg.norm(right-left) / spacing))
            points.append(np.linspace(left, right, count + 1))
    points = np.concatenate(points)
    _, ids = np.unique(np.round(points / (spacing * 1e-5)).astype(np.int64), axis=0, return_index=True)
    return np.ascontiguousarray(points[np.sort(ids)], dtype=np.float32)


def rotation_quaternion(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        if angle != 0:
            raise ValueError("Nonzero rotation needs a nonzero axis")
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return np.r_[axis / norm * np.sin(angle/2), np.cos(angle/2)].astype(np.float32)


def rotation_matrix(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@dataclass
class Body:
    vertices: np.ndarray
    faces: np.ndarray
    samples: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    mass: float
    inertia: np.ndarray
    color: tuple
    restitution: float
    friction: float
    wall: bool


def load_scene(config):
    path = Path(config.scene)
    if not path.is_file():
        path = ROOT / "scenes" / (config.scene + ".json")
    data = json.loads(path.read_text(encoding="utf-8"))
    bodies = []
    for item in data["RigidBodies"]:
        mesh_path = (path.parent / item["geometryFile"]).resolve()
        v, faces = load_obj(mesh_path)
        scale = np.asarray(item.get("scale", [1, 1, 1]), dtype=float)
        if np.any(scale <= 0):
            raise ValueError("Mesh scales must be positive")
        v *= scale
        mass, center, inertia = mass_properties(v, faces, item.get("density", 1000.0))
        q = rotation_quaternion(item.get("rotationAxis", [1, 0, 0]), item.get("rotationAngle", 0))
        pos = np.asarray(item.get("translation", [0, 0, 0])) + rotation_matrix(q) @ center
        v -= center
        dynamic = bool(item.get("isDynamic", False))
        wall = bool(item.get("isWall", False))
        restitution, friction = float(item.get("restitution",.6)), float(item.get("friction",.2))
        if not np.isfinite(restitution) or not 0 <= restitution <= 1 or not np.isfinite(friction) or friction < 0:
            raise ValueError("Restitution must be in [0,1]; friction must be finite and nonnegative")
        if wall and (dynamic or abs(q[3] - 1) > 1e-6):
            raise ValueError("Container must be a static axis-aligned box")
        bodies.append(Body(v.astype(np.float32), faces, sample_surface(v, faces, 1.5 * config.particle_radius),
                           pos.astype(np.float32), q, np.asarray(item.get("velocity", [0,0,0]), dtype=np.float32),
                           np.asarray(item.get("angularVelocity", [0,0,0]), dtype=np.float32),
                           mass if dynamic else 0.0, inertia, tuple(item.get("color", [.4,.6,.8])[:3]),
                           restitution, friction, wall))
    walls = [b for b in bodies if b.wall]
    if len(walls) != 1:
        raise ValueError("Scene requires exactly one static box container")
    lower = walls[0].vertices.min(axis=0) + walls[0].position
    upper = walls[0].vertices.max(axis=0) + walls[0].position
    fluids, velocities = [], []
    d = 2.0 * config.particle_radius
    for block in data["FluidBlocks"]:
        if block.get("denseMode", 0) != 0:
            raise ValueError("Only regular denseMode=0 fluid blocks are supported")
        start, end = np.asarray(block["start"], float), np.asarray(block["end"], float)
        scale = np.asarray(block.get("scale", [1,1,1]), float)
        start = start * scale + np.asarray(block.get("translation", [0,0,0]))
        end = end * scale + np.asarray(block.get("translation", [0,0,0]))
        # SimulatorBase::createFluidBlocks: round(diff/diam)-1, start=min+diam.
        count = np.floor((end-start)/d + .5).astype(int) - 1
        if np.any(count < 1):
            raise ValueError("Fluid block must fit at least one particle on every axis")
        grid = np.stack(np.meshgrid(*(np.arange(n) for n in count), indexing="ij"), axis=-1).reshape(-1,3)
        pts = start + d + d * grid
        fluids.append(pts)
        velocities.append(np.broadcast_to(block.get("initialVelocity", [0,0,0]), pts.shape))
    return bodies, np.concatenate(fluids).astype(np.float32), np.concatenate(velocities).astype(np.float32), lower, upper
