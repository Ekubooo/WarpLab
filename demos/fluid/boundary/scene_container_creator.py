"""从 scene_creator.py 复制：输出简化景观和封闭容器，不烘焙 SDF。

直接运行或作为模块运行，无 CLI 参数。修改本文件顶部的配置，或调用 create_scene()。
所有构造采用 Z-up；输出默认统一转为 Y-up。原 scene_creator.py 保持独立。
"""

import sys
from pathlib import Path

import numpy as np
import trimesh

if __package__:
    from .terrain_generator import create_mesh_terrain
else:
    from terrain_generator import create_mesh_terrain


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
ORIGIN_DIR = ASSET_DIR / "origin"  # 导入模型。
OUTPUT_DIR = ASSET_DIR / "output"  # 合成场景及后续 SDF 结果。
DEFAULT_OBJ_PATH = ORIGIN_DIR / "bunny_flat_bottom.obj"  # 水平裁切并封口的兔子副本；原模型保持不变。
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "stairs_scene_bool_simplified.obj"
DEFAULT_CONTAINER_OUTPUT_PATH = OUTPUT_DIR / "stairs_container_bool_simplified.obj"
DEFAULT_BLOCK_SIZE = (3.1, 1.6)  # X 长、Y 宽，米；尺寸仅用于地形生成。
DEFAULT_STEP_WIDTH = 0.15  # 台阶宽度，米；墙体增厚不改变地形。
DEFAULT_STEP_HEIGHT = 0.1
DEFAULT_PLATFORM_WIDTH = 0.7
DEFAULT_CONTAINER_HEIGHT = 1.5  # 内顶面绝对 Z 坐标，米；不随场景高度变化。
DEFAULT_OVERLAP_DEPTH = 0.075  # 保持原内腔边界；增厚部分只向外扩展。None 仍可取墙厚的一半。
REFERENCE_PLATFORM_WIDTH = 1.0  # 尺寸标定基准，米；固定保留，不随当前地形参数改变。
DEFAULT_OBJ_SCALE = 0.95 / 0.15515922  # 当前兔子原始宽度为 0.15515922；在 1 米平台上标定为宽占 95%。
DEFAULT_OBJ_UP_AXIS = "Y"  # 当前兔子是 Y-up；OBJ 不提供统一 up 信息，不同模型需手动指定。
DEFAULT_OBJ_SINK = 0.003  # 平整底面下沉 3 毫米，完全进入平台内部，布尔并集删除整个兔子底面。
DEFAULT_SIMPLIFY_TOLERANCE = 0.001  # 两个分支始终简化，允许表面偏移，米。


def check_boolean_backend():
    """先检查依赖；不可用时停止，不退回普通数组拼接。"""
    try:
        import manifold3d
    except ImportError as error:
        raise ValueError("当前 Python 环境缺少可用的 Manifold；请运行 python -m pip install manifold3d") from error
    if "manifold" not in trimesh.boolean.engines_available:
        raise ValueError("Trimesh 未识别 Manifold，请确认 manifold3d 与 trimesh 安装在同一 Python 环境")
    return manifold3d


def generate_stairs(step_width=DEFAULT_STEP_WIDTH):
    """生成一块 Z-up 阶梯；在本函数中修改阶梯尺寸和参数。"""
    vertices, indices = create_mesh_terrain(
        grid_size=(1, 1),                   # (行数, 列数)，这里生成一块。
        block_size=DEFAULT_BLOCK_SIZE,
        terrain_types="pyramid_stairs",     # 四周低、中心高的同心阶梯。
        terrain_params={
            "pyramid_stairs": {
                "step_width": step_width,  # 仅控制台阶水平宽度。
                "step_height": DEFAULT_STEP_HEIGHT,
                "platform_width": DEFAULT_PLATFORM_WIDTH,
            }
        },
    )

    # vertices 是 (N, 3) 数组：N 个顶点，每一行是一个 [x, y, z]。
    # indices 是一维数组，如 [0, 1, 2, 2, 3, 0]：每三个数字表示一个三角形。
    # reshape(-1, 3) 将它变成 [[0, 1, 2], [2, 3, 0]]；-1 表示自动计算行数。
    faces = indices.reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def get_platform_bounds(terrain):
    """读取 Z-up 地形最高处向上的水平三角面，返回平台的 (2, 3) 包围盒。"""
    if (len(terrain.vertices) == 0 or len(terrain.faces) == 0
            or not np.isfinite(terrain.vertices).all()):
        raise ValueError("无法识别有效水平平台：地形缺少有限的顶点或三角面")
    top_z = terrain.bounds[1, 2]
    triangles = terrain.triangles
    top_faces = (np.all(np.isclose(triangles[:, :, 2], top_z, atol=1e-7, rtol=0), axis=1)
                 & (terrain.face_normals[:, 2] > 1.0 - 1e-7)
                 & (terrain.area_faces > 0))
    if not top_faces.any():
        raise ValueError("无法识别有效水平平台：地形最高处没有向上的水平面")
    points = triangles[top_faces].reshape(-1, 3)
    bounds = np.array([points.min(axis=0), points.max(axis=0)])
    if np.any(bounds[1, :2] - bounds[0, :2] <= 0):
        raise ValueError("无法识别有效水平平台：平台宽度和深度必须大于零")
    return bounds


def _effective_obj_scale(scale, platform_bounds=None):
    """平台为空时保留固定倍率；否则以平台短边相对标定基准等比缩放。"""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale 必须是大于零的有限数值")
    if platform_bounds is not None:
        if not np.isfinite(REFERENCE_PLATFORM_WIDTH) or REFERENCE_PLATFORM_WIDTH <= 0:
            raise ValueError("REFERENCE_PLATFORM_WIDTH 必须是大于零的有限数值，单位为米")
        width = np.min(platform_bounds[1, :2] - platform_bounds[0, :2])
        scale = scale * width / REFERENCE_PLATFORM_WIDTH
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("OBJ 实际倍率必须是大于零的有限数值")
    return float(scale)


def get_support_center(mesh, contact_depth=0.0):
    """返回 Z-up 支撑中心 [x, y, 最低Z]；优先使用平底的面积重心。

    无平底时，尝试最低点上方 contact_depth 处截面的面积重心；没有有效截面则取
    最低接触顶点的范围中心。计算不参考身体、耳朵等上部几何的投影包围盒。
    """
    if not np.isfinite(contact_depth) or contact_depth < 0:
        raise ValueError("contact_depth 必须是大于或等于零的有限数值")
    z_min = mesh.bounds[0, 2]
    bottom_faces = (np.all(np.isclose(mesh.triangles[:, :, 2], z_min, atol=1e-9, rtol=0), axis=1)
                    & (mesh.area_faces > 0))
    if bottom_faces.any():
        center = np.average(mesh.triangles_center[bottom_faces, :2], axis=0,
                            weights=mesh.area_faces[bottom_faces])
        return np.r_[center, z_min]

    if contact_depth > 0:
        solid = prepare_solid(mesh)
        if solid.is_volume:
            segments, face_ids = trimesh.intersections.mesh_plane(
                solid, [0, 0, 1], [0, 0, z_min + contact_depth], return_faces=True,
            )
            if len(segments):
                # 用外法线统一线段方向；外轮廓为逆时针，孔洞为顺时针，不依赖 Shapely。
                tangents = np.cross([0, 0, 1], solid.face_normals[face_ids])
                reverse = np.einsum("ij,ij->i", segments[:, 1] - segments[:, 0], tangents) < 0
                segments[reverse] = segments[reverse, ::-1]
                xy = segments[:, :, :2]
                origin = xy.reshape(-1, 2).mean(axis=0)
                a, b = xy[:, 0] - origin, xy[:, 1] - origin
                cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
                twice_area = cross.sum()
                tolerance = np.finfo(float).eps * max(float(np.max(solid.extents[:2])) ** 2, 1e-20) * 32
                if twice_area > tolerance:
                    center = ((a + b) * cross[:, None]).sum(axis=0) / (3 * twice_area) + origin
                    return np.r_[center, z_min]

    lowest = mesh.vertices[np.isclose(mesh.vertices[:, 2], z_min, atol=1e-9, rtol=0), :2]
    return np.r_[(lowest.min(axis=0) + lowest.max(axis=0)) / 2, z_min]


def load_and_place_obj(obj_path, terrain, scale=DEFAULT_OBJ_SCALE, yaw_deg=0.0,
                       obj_up_axis=DEFAULT_OBJ_UP_AXIS, position=None, sink=DEFAULT_OBJ_SINK,
                       *, scale_with_platform=True):
    """先统一 OBJ 坐标，再把它的底面支撑中心移到指定位置。

    scale：在 1 米基准平台上的标定倍率，当前兔子默认约 6.12274，宽占平台 95%。
    scale_with_platform：默认 True，实际倍率 = scale × 平台短边 / REFERENCE_PLATFORM_WIDTH。
        False 保留固定倍率；只改变 OBJ，绝不缩放或修改 terrain。
    yaw_deg：绕竖直 Z 轴旋转的角度，单位为度。
    obj_up_axis：原始 OBJ 的向上轴，Y 或 Z；当前兔子默认 Y。
    position：目标底面支撑中心 [x, y, z]，单位为米，采用 Z-up；None 放在平台中心。
    sink：自动放置时向下嵌入平台的深度，米；显式 position 不再叠加 sink。
    """
    obj_path = Path(obj_path)
    if not obj_path.is_file():
        raise FileNotFoundError(f"找不到 OBJ：{obj_path}；修改 DEFAULT_OBJ_PATH 或传入 obj_path。")
    if obj_path.suffix.lower() != ".obj":
        raise ValueError("请提供 .obj 文件")
    _effective_obj_scale(scale)  # 先检查用户倍率。
    if not np.isfinite(yaw_deg):
        raise ValueError("yaw_deg 必须是有限数值")
    if not np.isfinite(sink) or sink < 0:
        raise ValueError("sink 必须是大于或等于零的有限数值")
    obj_up_axis = obj_up_axis.upper()
    if obj_up_axis not in ("Y", "Z"):
        raise ValueError("obj_up_axis 必须是 Y 或 Z")

    platform_bounds = None
    if scale_with_platform or position is None:
        try:
            platform_bounds = get_platform_bounds(terrain)
        except ValueError:
            if scale_with_platform:
                raise
            # 关闭联动后，无水平平台的自定义地形仍可按旧逻辑在包围盒顶端放置。
    effective_scale = _effective_obj_scale(scale, platform_bounds if scale_with_platform else None)

    obj = trimesh.load(obj_path, force="mesh", process=False)
    if len(obj.vertices) == 0 or len(obj.faces) == 0 or not np.isfinite(obj.vertices).all():
        raise ValueError("OBJ 必须包含有效的顶点和三角面")

    # 整个放置过程统一使用 Z-up：X、Y 在地面上，Z 表示高度。
    # 按指定轴转换：如果原始 OBJ 为 Y-up，绕 X 轴 +90° 转为 Z-up。
    if obj_up_axis == "Y":
        obj.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    obj.apply_scale(effective_scale)
    obj.apply_transform(
        trimesh.transformations.rotation_matrix(np.deg2rad(yaw_deg), [0, 0, 1])
    )

    # 缩放和旋转之后按实际支撑面定位；按面积加权，不受顶点密度或耳朵伸出影响。
    # 显式 position 不隐含下沉：非平底模型此时按最低接触顶点定位。
    bottom_center = get_support_center(obj, contact_depth=sink if position is None else 0.0)

    if position is None:
        # 读取实际平台中心；当前约 [1.55, 0.8, 0.5]，下沉量仍为绝对米制值。
        placement_bounds = platform_bounds if platform_bounds is not None else terrain.bounds
        ground_min_x, ground_min_y, _ = placement_bounds[0]
        ground_max_x, ground_max_y, platform_z = placement_bounds[1]
        target = np.array([
            (ground_min_x + ground_max_x) / 2,
            (ground_min_y + ground_max_y) / 2,
            platform_z - sink,
        ])
    else:
        # 例如 [1.0, 2.0, 0.1]；自选位置时，Z 要按该处台阶高度填写。
        target = np.asarray(position, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("position 必须是三个有限数值 [x, y, z]")

    # 数组相减逐项进行：[目标X-当前X, 目标Y-当前Y, 目标Z-当前Z]。
    # 例如底部中心为 [0, 0, -0.5]，目标 [3, 2, 0.4]，移动量就是 [3, 2, 0.9]。
    translation = target - bottom_center
    obj.apply_translation(translation)  # 给物体的每一个顶点加上同一个移动量。
    return obj


def combine_meshes(terrain, obj):
    """手动拼接两套顶点和三角面，便于理解数组和索引的关系。"""
    # vstack 沿竖直方向追加数组的行，不改变任何顶点的坐标。
    # 地形有 N 个顶点、OBJ 有 M 个顶点，合并后就是 (N+M, 3)。
    vertices = np.vstack([terrain.vertices, obj.vertices])

    # 两个原始模型都从顶点编号 0 开始。
    # 追加 OBJ 后，它的顶点在合并数组中从编号 N 开始，所以面索引也要整体加 N。
    # 例如地形有 100 个顶点，OBJ 的面 [0, 1, 2] 就改成 [100, 101, 102]。
    vertex_offset = len(terrain.vertices)
    obj_faces = obj.faces + vertex_offset

    # 地形面索引保持原样，再追加已经调整编号的 OBJ 面索引。
    faces = np.vstack([terrain.faces, obj_faces])
    # 这里拼接几何，不执行实体布尔并集；各部分的内部面仍然存在。
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def prepare_solid(mesh):
    """仅焊接同位置顶点；不补洞或修复几何，在副本上操作。"""
    mesh = mesh.copy()
    # 箱体按面重复顶点，OBJ 也可能按 UV/法线拆点；布尔需要这些边真正共享顶点。
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def boolean_union_meshes(meshes, label):
    """将有效实体做并集，裁剪相交面，删除埋在并集内部的面。"""
    check_boolean_backend()
    if not meshes:
        raise ValueError(f"{label}：布尔输入不能为空")
    for index, mesh in enumerate(meshes):
        if not mesh.is_volume:
            raise ValueError(
                f"{label}：输入 {index + 1} 不是有效封闭实体。"
                "Manifold 要求每个输入单独封闭，开口埋入地形也不能绕过此要求。"
                "本脚本不补洞；请传入封闭 OBJ。"
            )
    result = trimesh.boolean.union(meshes, engine="manifold", check_volume=True)
    if not isinstance(result, trimesh.Trimesh) or not result.is_volume:
        raise ValueError(f"{label}：Manifold 没有返回有效封闭实体")
    return result


def validate_simplify_tolerance(tolerance):
    """容差控制允许的表面偏移，不是目标面数或平滑强度。"""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("simplify_tolerance 必须是大于零的有限数值，单位为米")


def simplify_mesh_manifold(mesh, tolerance=DEFAULT_SIMPLIFY_TOLERANCE):
    """对实体副本做小容差简化，返回独立 mesh；不平滑、补洞或重网格。

    tolerance：允许的表面容差，米，默认 0.001；不保证消除全部瘦长三角形。
    使用 Mesh64 避免将现有顶点额外转换为 float32；原 mesh 的顶点和面不改动。
    """
    validate_simplify_tolerance(tolerance)
    manifold3d = check_boolean_backend()
    if not hasattr(manifold3d, "Mesh64") or not hasattr(manifold3d.Manifold, "to_mesh64"):
        raise ValueError("当前 Manifold 缺少 Mesh64/to_mesh64，请使用支持双精度 mesh 的版本")
    source = mesh.copy()
    if not source.is_volume or not np.isfinite(source.vertices).all():
        raise ValueError("简化输入必须是坐标有限、封闭且朝向一致的正体积实体")
    # vert_properties 的每行前三列是 [x,y,z]；这里仅传几何，没有额外的法线或 UV 列。
    # tri_verts 每行是一个三角形的三个顶点编号，Mesh64 使用 uint64 索引。
    solid = manifold3d.Manifold(manifold3d.Mesh64(
        vert_properties=np.asarray(source.vertices, dtype=np.float64),
        tri_verts=np.asarray(source.faces, dtype=np.uint64),
    ))
    if solid.status() != manifold3d.Error.NoError:
        raise ValueError(f"Manifold 拒绝简化输入：{solid.status()}")
    minimum_tolerance = solid.get_tolerance()
    # simplify 会自动使用 max(请求容差, 实体固有容差)；不能悄悄突破用户要求。
    if tolerance < minimum_tolerance:
        raise ValueError(f"请求容差 {tolerance:g} m 小于 Manifold 实际最小容差 {minimum_tolerance:g} m")
    simplified = solid.simplify(tolerance)
    if simplified.status() != manifold3d.Error.NoError:
        raise ValueError(f"Manifold 简化失败：{simplified.status()}")
    result = simplified.to_mesh64()
    # [:, :3] 只取位置列；process=False 避免 Trimesh 自动改变已经得到的连接关系。
    output = trimesh.Trimesh(vertices=result.vert_properties[:, :3], faces=result.tri_verts, process=False)
    if (len(output.faces) == 0 or not np.isfinite(output.vertices).all()
            or not output.is_volume):
        raise ValueError("简化结果不是有效封闭实体，停止导出简化副本")
    return output


def report_simplification(original, simplified, tolerance):
    """比较连接、三角形形状和体积；表面容差由 Manifold 控制，体积变化不是距离误差。"""
    print(f"Manifold 简化（表面容差={tolerance:g} m）：", flush=True)
    for label, mesh in (("原布尔", original), ("简化副本", simplified)):
        # face_angles 是每个三角形的三个内角，弧度；先取每面最小角，再转成度。
        minimum_angles = np.degrees(mesh.face_angles.min(axis=1))
        components = len(mesh.split(only_watertight=False, repair=False))
        print(f"  {label}：三角形={len(mesh.faces)}，最小内角={minimum_angles.min():.6f}°，"
              f"小于1°的面={np.count_nonzero(minimum_angles < 1)}，封闭={mesh.is_watertight}，"
              f"朝向一致={mesh.is_winding_consistent}，表面分量={components}，体积={mesh.volume:.9f} m^3",
              flush=True)
    print(f"  体积变化={(simplified.volume / original.volume - 1) * 100:.6f}%", flush=True)


def validate_output_paths(obj_path, output_path, container_output_path):
    """导出前检查输入、景观输出和容器输出互不覆盖。"""
    paths = [Path(obj_path), Path(output_path), Path(container_output_path)]
    for path in paths[1:]:
        if path.suffix.lower() != ".obj":
            raise ValueError(f"输出路径必须以 .obj 结尾：{path}")
    # resolve 统一相对路径、.. 和已有符号链接；Windows Path 比较也会忽略大小写。
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("输入 OBJ、景观输出和容器输出必须是不同的文件，不能相互覆盖")
    # 不同路径还可能指向同一个硬链接文件，也应拒绝，避免写一个输出时改动另一份。
    for index, path in enumerate(paths):
        if path.exists():
            for other in paths[:index]:
                if other.exists() and path.samefile(other):
                    raise ValueError(f"路径指向同一个文件，不能相互覆盖：{path} 与 {other}")


def union_stair_boxes(terrain):
    """恢复当前 pyramid_stairs 的独立箱体，再做地形实体并集。"""
    # 当前生成器每个箱体输出 12 个连续三角形，按箱体依次拼接。
    # 必须先拆箱体再焊接；整个地形一起焊接会把相邻箱子的内部面也连起来。
    if len(terrain.faces) == 0 or len(terrain.faces) % 12 != 0:
        raise ValueError("当前阶梯不符合生成器每箱体 12 个三角形的结构")
    # 生成器输出 float32；相邻箱子的同一边界可能相差一个舍入单位，产生微小封闭空腔。
    # 仅在布尔输入副本上统一这些坐标，不改动原地形，也不对 OBJ 使用此对齐。
    vertices = np.array(terrain.vertices, dtype=np.float64, copy=True)
    for axis in range(3):
        values, inverse = np.unique(vertices[:, axis], return_inverse=True)
        tolerance = float(np.spacing(np.float32(max(1.0, np.max(np.abs(values))))))
        aligned = values.copy()
        start = 0
        for stop in range(1, len(values) + 1):
            if stop == len(values) or values[stop] - values[start] > tolerance:
                aligned[start:stop] = values[start]
                start = stop
        vertices[:, axis] = aligned[inverse]
    boxes = []
    for box_faces in terrain.faces.reshape(-1, 12, 3):
        # 挑出这个箱体使用的原顶点编号，再将它们重编号为 0、1、2……。
        vertex_ids, local_indices = np.unique(box_faces, return_inverse=True)
        box = trimesh.Trimesh(
            vertices=vertices[vertex_ids],
            faces=local_indices.reshape(-1, 3),
            process=False,
        )
        boxes.append(prepare_solid(box))
    result = boolean_union_meshes(boxes, "阶梯并集")
    print(f"阶梯并集：{len(boxes)} 个箱体 → {len(result.faces)} 个三角形")
    return result


def export_scene_mesh(mesh, output_path, up_axis):
    """对整个场景统一转换向上轴并导出，保留用于布尔计算的 Z-up 原件。"""
    mesh = mesh.copy()
    if up_axis == "Y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)


def resolve_overlap_depth(wall_thickness, overlap_depth=None):
    """默认 δ=w/2；必须保留外侧墙厚和顶盖厚度，避免穿透整块墙。"""
    if not np.isfinite(wall_thickness) or wall_thickness <= 0:
        raise ValueError("step_width / 墙厚必须是大于零的有限数值")
    depth = wall_thickness / 2 if overlap_depth is None else overlap_depth
    if not np.isfinite(depth) or not 0 < depth < wall_thickness:
        raise ValueError("overlap_depth 必须大于零且小于墙厚，单位为米")
    return float(depth)


def build_container_parts(terrain, wall_thickness=0.25,
                          container_height=DEFAULT_CONTAINER_HEIGHT, overlap_depth=DEFAULT_OVERLAP_DEPTH):
    """返回四面墙和顶盖的五个封闭箱体；没有另加平整底板。

    地形 X 范围为 L，Y 范围为 W；墙内侧进入地形 δ，外侧只增加 w-δ。
    统一外尺寸为 L+2(w-δ)、W+2(w-δ)，长墙横跨完整外长度。
    短墙目标长度 W+δ，但不得超过统一外宽度；相交余量留在外轮廓内。
    墙内侧进入地形 δ；顶盖底面 Z=container_height，墙顶比它高 δ。
    顶盖与四墙使用同一外边界，避免墙头和顶盖额外伸出。
    """
    depth = resolve_overlap_depth(wall_thickness, overlap_depth)
    if (len(terrain.vertices) == 0 or len(terrain.faces) == 0
            or not np.isfinite(terrain.vertices).all()):
        raise ValueError("构造容器需要具有有限坐标的地形")
    minimum, maximum = terrain.bounds
    length, width = maximum[:2] - minimum[:2]
    if min(length, width) <= 2 * depth:
        raise ValueError("地形水平尺寸必须大于两倍相交深度，以保留流体内部空间")
    if not np.isfinite(container_height) or container_height <= maximum[2]:
        raise ValueError("container_height 是内顶面绝对 Z 坐标，必须高于地形最高点")
    w = wall_thickness
    outer_min = minimum[:2] - (w - depth)
    outer_max = maximum[:2] + (w - depth)
    outer_size = outer_max - outer_min
    cx, cy = (outer_min + outer_max) / 2
    # 当 δ 较大时 W+δ 可能比完整外宽度还大，截到外边界；不要撑大容器。
    short_length = min(width + depth, outer_size[1])
    bottom = minimum[2]
    top = container_height + depth
    height, cz = top - bottom, (top + bottom) / 2

    def box(extents, center):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        return mesh

    walls = [
        box((outer_size[0], w, height),
            (cx, outer_min[1] + w / 2, cz)),
        box((outer_size[0], w, height),
            (cx, outer_max[1] - w / 2, cz)),
        box((w, short_length, height),
            (outer_min[0] + w / 2, cy, cz)),
        box((w, short_length, height),
            (outer_max[0] - w / 2, cy, cz)),
    ]
    lid = box((*tuple(outer_size), w),
              (*tuple((outer_min + outer_max) / 2), container_height + w / 2))
    return [*walls, lid]


def validate_container_mesh(mesh, label, *, outer_bounds=None, surface_tolerance=1e-6):
    """检查封闭实体及唯一封闭流体空腔，不修复或翻转内腔表面的法线。"""
    if not mesh.is_volume or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"{label}：容器不是有效封闭实体")
    surfaces = mesh.split(only_watertight=False, repair=False)
    if any(not surface.is_watertight or not surface.is_winding_consistent for surface in surfaces):
        raise ValueError(f"{label}：容器存在开口或不一致的表面朝向")
    # 外表面为正体积，封闭内腔表面为负体积；它们无需共享顶点。
    cavities = [surface for surface in surfaces if surface.volume < 0]
    if len(cavities) != 1:
        raise ValueError(f"{label}：预期一个封闭流体空腔，实际检测到 {len(cavities)} 个")
    if outer_bounds is not None:
        exterior = [surface for surface in surfaces if surface.volume > 0]
        if len(exterior) != 1:
            raise ValueError(f"{label}：预期一个长方体外表面，实际检测到 {len(exterior)} 个外表面")
        outer = exterior[0]
        if not np.allclose(outer.bounds, outer_bounds, rtol=0, atol=surface_tolerance):
            raise ValueError(f"{label}：外表面范围未对齐容器统一外边界")
        # 每个外表面三角形必须位于六个外平面之一；包围盒正确仍可能有凸檐或凹口。
        on_faces = np.any(np.all(np.isclose(
            outer.triangles[:, :, :, None], np.asarray(outer_bounds).T[None, None, :, :],
            rtol=0, atol=surface_tolerance), axis=1), axis=(1, 2))
        if not on_faces.all():
            raise ValueError(f"{label}：外表面不是完整长方体，存在伸出或凹进的面")
    print(f"{label}：表面分量={len(surfaces)}，流体空腔体积={-cavities[0].volume:.9f} m^3",
          flush=True)


def create_scene(obj_path=DEFAULT_OBJ_PATH, output_path=DEFAULT_OUTPUT_PATH,
                 scale=DEFAULT_OBJ_SCALE, yaw_deg=0.0,
                 obj_up_axis=DEFAULT_OBJ_UP_AXIS, position=None, up_axis="Y",
                 sink=DEFAULT_OBJ_SINK, simplify_tolerance=DEFAULT_SIMPLIFY_TOLERANCE,
                 *, scale_with_platform=True, container_output_path=DEFAULT_CONTAINER_OUTPUT_PATH,
                 step_width=DEFAULT_STEP_WIDTH, container_height=DEFAULT_CONTAINER_HEIGHT,
                 overlap_depth=DEFAULT_OVERLAP_DEPTH):
    """两个分支独立 Boolean 和简化，仅导出简化 OBJ，返回 (景观路径, 容器路径)。

    step_width 仅控制台阶宽；墙厚为 0.25 米，默认相交量固定为 0.075。
    显式 overlap_depth=None 使用墙厚的一半。
    container_height 是内部 Z-up 的内顶面绝对高度，不是场景包围盒高度。
    simplify_tolerance 是两份输出共同的简化容差；没有关闭简化的分支。
    """
    check_boolean_backend()
    up_axis = up_axis.upper()
    if up_axis not in ("Y", "Z"):
        raise ValueError("up_axis 必须是 Y 或 Z")
    output_path = Path(output_path)
    container_output_path = Path(container_output_path)
    validate_output_paths(obj_path, output_path, container_output_path)
    validate_simplify_tolerance(simplify_tolerance)
    wall_thickness = 0.25
    depth = resolve_overlap_depth(wall_thickness, overlap_depth)

    terrain = generate_stairs(step_width=step_width)
    obj = load_and_place_obj(
        obj_path=obj_path, terrain=terrain, scale=scale,
        yaw_deg=yaw_deg, obj_up_axis=obj_up_axis, position=position, sink=sink,
        scale_with_platform=scale_with_platform,
    )
    # generate_stairs 的最高水平面就是中央平台；显示实际尺寸，方便修改参数后核对。
    try:
        platform_bounds = get_platform_bounds(terrain)
        platform_size = platform_bounds[1, :2] - platform_bounds[0, :2]
        platform_text = f"{platform_size[0]:g}×{platform_size[1]:g} m"
    except ValueError:
        if scale_with_platform:
            raise
        platform_bounds = None
        platform_text = "未识别（使用固定倍率）"
    effective_scale = _effective_obj_scale(scale, platform_bounds if scale_with_platform else None)
    print(f"OBJ 缩放：平台={platform_text}，"
          f"指定倍率={scale:g}，实际倍率={effective_scale:g}，"
          f"平台联动={'开启' if scale_with_platform else '关闭'}", flush=True)
    obj = prepare_solid(obj)
    solid_terrain = union_stair_boxes(terrain)
    parts = build_container_parts(solid_terrain, wall_thickness, container_height, depth)
    lid_bounds = parts[-1].bounds
    # 相对原单墙厚 0.15 米，外底面也向外增加同样的距离；原地形表面保持不变。
    outward_growth = max(wall_thickness - 0.15, 0.0)
    outer_bounds = np.array([
        [*lid_bounds[0, :2], solid_terrain.bounds[0, 2] - outward_growth], lid_bounds[1],
    ])
    # 扩展段上端进入原底座，保证并集重叠；厚度由外移增量决定。
    base_extension = trimesh.creation.box(
        extents=(*tuple(lid_bounds[1, :2] - lid_bounds[0, :2]), outward_growth + depth))
    base_extension.apply_translation([
        *tuple(lid_bounds.mean(axis=0)[:2]), solid_terrain.bounds[0, 2] + (depth - outward_growth) / 2])
    if obj.bounds[1, 2] >= container_height:
        raise ValueError("内顶面必须高于放置后的 OBJ，避免模型与顶盖相交")
    print(f"容器：墙厚={wall_thickness:g} m，相交量={depth:g} m，内顶面 Z={container_height:g} m，"
          f"外尺寸={np.array2string(outer_bounds[1] - outer_bounds[0], precision=6)} m",
          flush=True)
    scene = boolean_union_meshes([solid_terrain, obj], "场景并集")
    # 直接从未简化资源求容器并集，不使用上面的景观结果或任何简化结果。
    container = boolean_union_meshes([solid_terrain, obj, *parts, base_extension], "容器并集")
    validate_container_mesh(container, "原容器并集", outer_bounds=outer_bounds)
    simplified_scene = simplify_mesh_manifold(scene, simplify_tolerance)
    simplified_container = simplify_mesh_manifold(container, simplify_tolerance)
    validate_container_mesh(simplified_container, "简化容器", outer_bounds=outer_bounds,
                            surface_tolerance=max(1e-6, simplify_tolerance))
    print("景观分支：", flush=True)
    report_simplification(scene, simplified_scene, simplify_tolerance)
    print("容器分支：", flush=True)
    report_simplification(container, simplified_container, simplify_tolerance)
    # 两个分支均计算并检查成功后才开始导出；不保存未简化中间 mesh。
    export_scene_mesh(simplified_scene, output_path, up_axis)
    export_scene_mesh(simplified_container, container_output_path, up_axis)
    return output_path, container_output_path


def main():
    if len(sys.argv) > 1:
        raise SystemExit("本脚本不接受 CLI 参数；请修改顶部配置或调用 create_scene()。")
    try:
        outputs = create_scene(
            obj_path=DEFAULT_OBJ_PATH, output_path=DEFAULT_OUTPUT_PATH,
            container_output_path=DEFAULT_CONTAINER_OUTPUT_PATH,
            scale=DEFAULT_OBJ_SCALE, obj_up_axis=DEFAULT_OBJ_UP_AXIS,
            simplify_tolerance=DEFAULT_SIMPLIFY_TOLERANCE,
            step_width=DEFAULT_STEP_WIDTH, container_height=DEFAULT_CONTAINER_HEIGHT,
            overlap_depth=DEFAULT_OVERLAP_DEPTH,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        raise SystemExit(f"生成失败：{error}") from error
    for label, output in zip(("简化景观", "简化容器"), outputs):
        print(f"{label}：{output.resolve()}（默认 Y-up，同名文件覆盖）")


if __name__ == "__main__":
    main()
