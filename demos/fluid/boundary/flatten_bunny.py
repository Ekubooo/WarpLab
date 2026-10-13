"""从封闭兔子副本水平裁切并封口，保留原始资产；使用当前 Conda 环境运行。"""

from pathlib import Path

import numpy as np
import trimesh

if __package__:
    from .scene_creator import check_boolean_backend, prepare_solid
else:
    from scene_creator import check_boolean_backend, prepare_solid


ORIGIN_DIR = Path(__file__).resolve().parent / "Assets" / "origin"
SOURCE_PATH = ORIGIN_DIR / "bunny_repaired.obj"
OUTPUT_PATH = ORIGIN_DIR / "bunny_flat_bottom.obj"
# 输入是 Y-up。原补洞面最高 Y=0.0395292，裁切面略高于它以完整删除凹凸底面。
CUT_Y = 0.04


def flatten_bunny(source_path=SOURCE_PATH, output_path=OUTPUT_PATH, cut_y=CUT_Y):
    """保留 Y >= cut_y 的实体，在 cut_y 处生成平面封口，另存 OBJ。"""
    source_path, output_path = Path(source_path), Path(output_path)
    if (source_path.resolve() == output_path.resolve()
            or (output_path.exists() and source_path.samefile(output_path))):
        raise ValueError("裁平副本不能覆盖原始模型")
    source = prepare_solid(trimesh.load(source_path, force="mesh", process=False))
    if not source.is_volume:
        raise ValueError("源兔子必须是封闭、朝向一致的正体积实体")
    if not np.isfinite(cut_y) or not source.bounds[0, 1] < cut_y < source.bounds[1, 1]:
        raise ValueError("裁切高度必须在模型 Y 范围内")
    manifold3d = check_boolean_backend()
    solid = manifold3d.Manifold(manifold3d.Mesh64(
        vert_properties=np.asarray(source.vertices, dtype=np.float64),
        tri_verts=np.asarray(source.faces, dtype=np.uint64),
    ))
    if solid.status() != manifold3d.Error.NoError:
        raise ValueError(f"Manifold 拒绝源模型：{solid.status()}")
    clipped = solid.trim_by_plane((0, 1, 0), float(cut_y))
    if clipped.status() != manifold3d.Error.NoError:
        raise ValueError(f"水平裁切失败：{clipped.status()}")
    result = clipped.to_mesh64()
    mesh = trimesh.Trimesh(vertices=result.vert_properties[:, :3],
                           faces=result.tri_verts, process=False)
    caps = np.all(np.isclose(mesh.triangles[:, :, 1], cut_y, atol=1e-12, rtol=0), axis=1)
    if not mesh.is_volume or not caps.any():
        raise ValueError("裁切结果必须封闭且具有水平底面")
    if not np.allclose(mesh.face_normals[caps], [0, -1, 0], atol=1e-10, rtol=0):
        raise ValueError("封口底面必须水平且朝下")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 保留足够小数位；平面顶点统一导出为同一个 Y 值。
    output_path.write_text(trimesh.exchange.obj.export_obj(mesh, digits=12), encoding="utf-8")
    print(f"裁平副本：{output_path.resolve()}；Y={cut_y:g}，底面={caps.sum()} 个三角形，"
          f"封闭={mesh.is_watertight}，三角形={len(mesh.faces)}")
    return output_path


if __name__ == "__main__":
    flatten_bunny()
