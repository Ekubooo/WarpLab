"""Shared geometry plus aligned boxes for native stair collisions."""

import numpy as np
import trimesh
from newton import Mesh

from demos.sim_Ready.geometry import (
    Scene, _check_options, _solid, container, container_parts, landscape,
    platform_bounds, prepare_rigid, prepare_scene, simplify, support_center, union,
)


def terrain_boxes(options):
    """Return generator boxes with shared float32 boundaries aligned once.

    Both the solid terrain and native rigid collisions use these coordinates.
    """
    _check_options(options)
    terrain = Mesh.create_terrain(
        grid_size=(1, 1), block_size=options.size, terrain_types="pyramid_stairs",
        compute_inertia=False,
        terrain_params={"pyramid_stairs": dict(step_width=options.step_width,
            step_height=options.step_height, platform_width=options.platform_width)},
    )
    source, indices = terrain.vertices, terrain.indices
    width = float(source[:, 1].max())
    points = np.column_stack((source[:, 0], source[:, 2], width - source[:, 1])).astype(np.float64)
    # Adjacent boxes can differ by one float32 ULP. Align only the generated
    # terrain, before union; applying this to an imported model would alter it.
    for axis in range(3):
        coordinates, inverse = np.unique(points[:, axis], return_inverse=True)
        epsilon = float(np.spacing(np.float32(max(1, np.abs(coordinates).max()))))
        anchors = coordinates.copy()
        start = 0
        for end in range(1, len(coordinates) + 1):
            if end == len(coordinates) or coordinates[end] - coordinates[start] > epsilon:
                anchors[start:end] = coordinates[start]
                start = end
        points[:, axis] = anchors[inverse]
    faces = indices.reshape(-1, 3)
    if not len(faces) or len(faces) % 12:
        raise ValueError("Expected twelve triangles per generated stair box")
    boxes = []
    for block in faces.reshape(-1, 12, 3):
        ids, local = np.unique(block, return_inverse=True)
        boxes.append(trimesh.Trimesh(points[ids], local.reshape(-1, 3), process=False))
    return boxes


def terrain_mesh(options):
    """Union the same aligned boxes used by the rigid collision scene."""
    return union(terrain_boxes(options))


