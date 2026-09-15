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
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    orientation = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    if np.any(counts != 2) or np.any(orientation != 0):
        raise ValueError(f"Mesh must be closed, welded, and consistently oriented: {path}")
    a, b, c = v[f].transpose(1, 0, 2)
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() <= 0:
        raise ValueError(f"Mesh faces must have outward winding: {path}")
    return v, f


def mass_properties(vertices, faces, density):
    """Integrate signed tetrahedra (origin,a,b,c), then shift to the COM."""
    a, b, c = np.asarray(vertices, dtype=np.float64)[faces].transpose(1, 0, 2)
    tetrahedron_volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    if abs(tetrahedron_volumes.sum()) < 1e-12:
        raise ValueError("Rigid mesh must enclose nonzero volume")
    tetrahedron_volumes *= np.sign(tetrahedron_volumes.sum())
    total_volume = tetrahedron_volumes.sum()
    vertex_sums = a + b + c
    center_of_mass = (tetrahedron_volumes[:, None] * vertex_sums).sum(axis=0) / (4.0 * total_volume)
    second_moment = sum(np.einsum("ni,nj->nij", x, x) for x in (a, b, c))
    second_moment += np.einsum("ni,nj->nij", vertex_sums, vertex_sums)
    second_moment = (tetrahedron_volumes[:, None, None] * second_moment).sum(axis=0) / 20.0
    second_moment -= total_volume * np.outer(center_of_mass, center_of_mass)
    inertia = density * (np.trace(second_moment) * np.eye(3) - second_moment)
    if density <= 0 or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("Rigid density and inertia must be positive")
    return density * total_volume, center_of_mass, inertia


def sample_surface(vertices, faces, spacing):
    """Regular rows along each triangle's longest edge, shared points merged.

    Row height is at most spacing; row endpoints are included. This avoids
    oversampling long, thin triangles with a square barycentric grid.
    """
    points = []
    for triangle in np.asarray(vertices)[faces]:
        edge_lengths = [np.linalg.norm(triangle[(i + 1) % 3] - triangle[i]) for i in range(3)]
        longest_edge = int(np.argmax(edge_lengths))
        a, b, c = (
            triangle[longest_edge],
            triangle[(longest_edge + 1) % 3],
            triangle[(longest_edge + 2) % 3],
        )
        length = edge_lengths[longest_edge]
        if length < 1e-12:
            continue
        height = np.linalg.norm(np.cross(b - a, c - a)) / length
        row_count = max(1, math.ceil(height / spacing))
        for row in range(row_count + 1):
            row_fraction = row / row_count
            left = (1 - row_fraction) * a + row_fraction * c
            right = (1 - row_fraction) * b + row_fraction * c
            count = max(1, math.ceil(np.linalg.norm(right - left) / spacing))
            points.append(np.linspace(left, right, count + 1))
    points = np.concatenate(points)
    _, unique_indices = np.unique(
        np.round(points / (spacing * 1e-5)).astype(np.int64), axis=0, return_index=True
    )
    return np.ascontiguousarray(points[np.sort(unique_indices)], dtype=np.float32)


def rotation_quaternion(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-12:
        if angle != 0:
            raise ValueError("Nonzero rotation needs a nonzero axis")
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return np.r_[axis / axis_length * np.sin(angle / 2), np.cos(angle / 2)].astype(np.float32)


def rotation_matrix(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


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
    """Build centered rigid meshes and regular fluid blocks from a scene JSON."""
    path = Path(config.scene)
    if not path.is_file():
        path = ROOT / "scenes" / (config.scene + ".json")
    data = json.loads(path.read_text(encoding="utf-8"))
    # Scale each mesh, then express its geometry relative to its center of mass.
    bodies = []
    for item in data["RigidBodies"]:
        mesh_path = (path.parent / item["geometryFile"]).resolve()
        vertices, faces = load_obj(mesh_path)
        scale = np.asarray(item.get("scale", [1, 1, 1]), dtype=float)
        if np.any(scale <= 0):
            raise ValueError("Mesh scales must be positive")
        vertices *= scale
        mass, center, inertia = mass_properties(vertices, faces, item.get("density", 1000.0))
        rotation = rotation_quaternion(
            item.get("rotationAxis", [1, 0, 0]), item.get("rotationAngle", 0)
        )
        position = (
            np.asarray(item.get("translation", [0, 0, 0])) + rotation_matrix(rotation) @ center
        )
        vertices -= center
        dynamic = bool(item.get("isDynamic", False))
        wall = bool(item.get("isWall", False))
        restitution = float(item.get("restitution", 0.6))
        friction = float(item.get("friction", 0.2))
        if (
            not np.isfinite(restitution)
            or not 0 <= restitution <= 1
            or not np.isfinite(friction)
            or friction < 0
        ):
            raise ValueError(
                "Restitution must be in [0,1]; friction must be finite and nonnegative"
            )
        if wall and (dynamic or abs(rotation[3] - 1) > 1e-6):
            raise ValueError("Container must be a static axis-aligned box")
        bodies.append(
            Body(
                vertices=vertices.astype(np.float32),
                faces=faces,
                samples=sample_surface(vertices, faces, 1.5 * config.particle_radius),
                position=position.astype(np.float32),
                rotation=rotation,
                velocity=np.asarray(item.get("velocity", [0, 0, 0]), dtype=np.float32),
                angular_velocity=np.asarray(
                    item.get("angularVelocity", [0, 0, 0]), dtype=np.float32
                ),
                mass=mass if dynamic else 0.0,
                inertia=inertia,
                color=tuple(item.get("color", [0.4, 0.6, 0.8])[:3]),
                restitution=restitution,
                friction=friction,
                wall=wall,
            )
        )
    # The supported container is one static, axis-aligned box.
    walls = [body for body in bodies if body.wall]
    if len(walls) != 1:
        raise ValueError("Scene requires exactly one static box container")
    lower = walls[0].vertices.min(axis=0) + walls[0].position
    upper = walls[0].vertices.max(axis=0) + walls[0].position
    # Match the reference scene's regular particle spacing and block offsets.
    fluids, velocities = [], []
    particle_diameter = 2.0 * config.particle_radius
    for block in data["FluidBlocks"]:
        if block.get("denseMode", 0) != 0:
            raise ValueError("Only regular denseMode=0 fluid blocks are supported")
        start, end = np.asarray(block["start"], float), np.asarray(block["end"], float)
        scale = np.asarray(block.get("scale", [1, 1, 1]), float)
        start = start * scale + np.asarray(block.get("translation", [0, 0, 0]))
        end = end * scale + np.asarray(block.get("translation", [0, 0, 0]))
        # SimulatorBase::createFluidBlocks: round(diff/diam)-1, start=min+diam.
        count = np.floor((end - start) / particle_diameter + 0.5).astype(int) - 1
        if np.any(count < 1):
            raise ValueError("Fluid block must fit at least one particle on every axis")
        grid = np.stack(
            np.meshgrid(*(np.arange(axis_count) for axis_count in count), indexing="ij"), axis=-1
        ).reshape(-1, 3)
        block_positions = start + particle_diameter + particle_diameter * grid
        fluids.append(block_positions)
        velocities.append(
            np.broadcast_to(block.get("initialVelocity", [0, 0, 0]), block_positions.shape)
        )
    return (
        bodies,
        np.concatenate(fluids).astype(np.float32),
        np.concatenate(velocities).astype(np.float32),
        lower,
        upper,
    )
