"""SDF 输入容器的简单检查；不修复网格，也不认证自交或壳的包含关系。"""

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import trimesh


@dataclass
class ValidatedContainer:
    vertices: np.ndarray  # float32 (N, 3)，与随后 GPU 使用的数据一致。
    faces: np.ndarray  # int32 (M, 3)
    bounds: np.ndarray  # (2, 3)
    signed_volumes: tuple[float, ...] = ()
    diagnostics: tuple[str, ...] = ()


def _geometry(vertices, faces, label):
    """仅精确合并同坐标点，拒绝退化和重复面，不删除或修补任何面。"""
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)):
        raise ValueError(f"{label}：容器必须包含非空顶点和三角面")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{label}：顶点包含非有限坐标")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"{label}：面索引必须是整数")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"{label}：面索引越界")
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[faces]
    sorted_faces = np.sort(faces, axis=1)
    if np.any(np.diff(sorted_faces, axis=1) == 0):
        raise ValueError(f"{label}：存在重复顶点索引或坐标塌缩的退化面")
    if len(np.unique(sorted_faces, axis=0)) != len(faces):
        raise ValueError(f"{label}：存在重复三角面（包括反向重复面）")
    triangles = vertices[faces].astype(np.float64)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(np.all(cross == 0, axis=1)) or not np.isfinite(cross).all():
        raise ValueError(f"{label}：存在零面积退化面或不可计算的三角面")
    # 未被面引用的顶点不参与边界，也不应扩大烘焙包围盒。
    referenced, inverse = np.unique(faces, return_inverse=True)
    return vertices[referenced], inverse.reshape(-1, 3)


def _topology(vertices, faces, label):
    edges = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1).reshape(-1, 2)
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    if np.any(counts != 2):
        raise ValueError(f"{label}：容器不闭合或存在非流形边；每条边必须恰有两个相邻面")
    order = np.argsort(inverse, kind="stable").reshape(-1, 2)
    if np.any(np.any(edges[order[:, 0]] != edges[order[:, 1], ::-1], axis=1)):
        raise ValueError(f"{label}：相邻三角面的绕序不一致")

    # 每个顶点的 link 必须是单个环；只检查每条边两面无法排除点接触的双扇区。
    links = [{} for _ in vertices]
    for a, b, c in faces:
        for center, first, second in ((a, b, c), (b, c, a), (c, a, b)):
            graph = links[center]
            graph.setdefault(first, set()).add(second)
            graph.setdefault(second, set()).add(first)
    for graph in links:
        if any(len(neighbors) != 2 for neighbors in graph.values()):
            raise ValueError(f"{label}：存在非流形顶点邻域")
        pending = [next(iter(graph))]
        reached = set(pending)
        while pending:
            for adjacent in graph[pending.pop()]:
                if adjacent not in reached:
                    reached.add(adjacent)
                    pending.append(adjacent)
        if len(reached) != len(graph):
            raise ValueError(f"{label}：存在多个不连通扇区的非流形顶点")

    # 按共享边标记表面分量；整个容器允许外壳和内壳不共享顶点。
    parents = np.arange(len(faces))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for first, second in order // 3:
        parents[root(second)] = root(first)
    roots = np.array([root(i) for i in range(len(faces))])
    volumes = []
    for component in np.unique(roots):
        triangles = vertices[faces[roots == component]].astype(np.float64)
        triangles -= (triangles.min(axis=(0, 1)) + triangles.max(axis=(0, 1))) / 2
        volumes.append(float(np.einsum("ij,ij->i", triangles[:, 0],
                                     np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6))
    if not np.isfinite(volumes).all() or sum(volumes) <= 0:
        raise ValueError(f"{label}：容器整体必须为有限的正体积实体")
    if (len(volumes) != 2 or sum(volume > 0 for volume in volumes) != 1
            or sum(volume < 0 for volume in volumes) != 1):
        raise ValueError(f"{label}：预期一个正体积外壳和一个负体积封闭流体空腔")
    return tuple(volumes)


def load_validate_container_obj(path) -> ValidatedContainer:
    """读取并验证容器 OBJ，返回独立、可上传 GPU 的数组。

    仅接受一个正外壳和一个反向内腔。不会自动翻转内腔法线。此简单检查
    不证明无自交或壳的几何包含关系；最小角小于 1° 仅告警，不拒绝。
    """
    path = Path(path)
    if path.suffix.lower() != ".obj":
        raise ValueError("SDF 容器输入必须是 .obj 文件")
    if not path.is_file():
        raise FileNotFoundError(f"找不到容器 OBJ：{path}")
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
    except (ValueError, TypeError, IndexError, KeyError) as error:
        raise ValueError(f"无法读取容器 OBJ：{path}：{error}") from error
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("容器 OBJ 必须包含三角网格")
    vertices, faces = _geometry(np.asarray(mesh.vertices), np.asarray(mesh.faces), "OBJ")
    _topology(vertices, faces, "OBJ")
    with np.errstate(over="ignore", invalid="ignore"):
        vertices = vertices.astype(np.float32)
    vertices, faces = _geometry(vertices, faces, "float32 GPU 网格")
    volumes = _topology(vertices, faces, "float32 GPU 网格")
    if len(vertices) > np.iinfo(np.int32).max:
        raise ValueError("容器顶点数超过 int32 索引范围")
    triangles = vertices[faces].astype(np.float64)
    angles = []
    for corner in range(3):
        first = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        second = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        angles.append(np.arctan2(np.linalg.norm(np.cross(first, second), axis=1),
                                 np.einsum("ij,ij->i", first, second)))
    minimum_angles = np.degrees(np.min(angles, axis=0))
    diagnostics = []
    if np.any(minimum_angles < 1):
        message = (f"容器存在瘦长三角形：最小角 {minimum_angles.min():.6g}°，"
                   f"{np.count_nonzero(minimum_angles < 1)} 面小于 1°；仅告警，不代表 SDF 不可烘焙")
        diagnostics.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return ValidatedContainer(
        vertices=np.ascontiguousarray(vertices, dtype=np.float32),
        faces=np.ascontiguousarray(faces, dtype=np.int32),
        bounds=np.array([vertices.min(axis=0), vertices.max(axis=0)]),
        signed_volumes=volumes, diagnostics=tuple(diagnostics),
    )
