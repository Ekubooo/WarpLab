"""生成阶梯、放置 OBJ、导出布尔并集，可另存 Manifold 简化副本；不依赖 creator1.py。"""

import argparse
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
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "stairs_scene_bool.obj"  # 布尔结果；重复运行覆盖。
DEFAULT_RAW_OUTPUT_PATH = OUTPUT_DIR / "stairs_scene.obj"  # 同位置的原始拼接结果，方便对照。
REFERENCE_PLATFORM_WIDTH = 1.0  # 尺寸标定基准，米；固定保留，不随当前地形参数改变。
DEFAULT_OBJ_SCALE = 0.95 / 0.15515922  # 当前兔子原始宽度为 0.15515922；在 1 米平台上标定为宽占 95%。
DEFAULT_OBJ_UP_AXIS = "Y"  # 当前兔子是 Y-up；OBJ 不提供统一 up 信息，不同模型需手动指定。
DEFAULT_OBJ_SINK = 0.003  # 平整底面下沉 3 毫米，完全进入平台内部，布尔并集删除整个兔子底面。
DEFAULT_SIMPLIFY_TOLERANCE = 0.001  # 简化表面容差，米；默认 1 毫米，只在启用简化分支时使用。


def check_boolean_backend():
    """先检查依赖；不可用时停止，不退回普通数组拼接。"""
    try:
        import manifold3d
    except ImportError as error:
        raise ValueError("当前 Python 环境缺少可用的 Manifold；请运行 python -m pip install manifold3d") from error
    if "manifold" not in trimesh.boolean.engines_available:
        raise ValueError("Trimesh 未识别 Manifold，请确认 manifold3d 与 trimesh 安装在同一 Python 环境")
    return manifold3d


def generate_stairs():
    """生成一块 Z-up 阶梯；在本函数中修改阶梯尺寸和参数。"""
    vertices, indices = create_mesh_terrain(
        grid_size=(1, 1),                   # (行数, 列数)，这里生成一块。
        block_size=(3.1, 1.6),              # X 长度和 Y 长度，米。
        terrain_types="pyramid_stairs",     # 四周低、中心高的同心阶梯。
        terrain_params={
            "pyramid_stairs": {
                "step_width": 0.15,         # 每圈台阶的水平宽度，米。
                "step_height": 0.1,        # 每圈高度增量，米；当前源码应使用正值。
                "platform_width": 0.7,      # 中心正方形平台的边长，米。
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
        raise FileNotFoundError(f"找不到 OBJ：{obj_path}；默认读取 {DEFAULT_OBJ_PATH}，或使用 --obj 指定文件。")
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
                "本脚本不补洞；请用 --obj 指定封闭 OBJ 测试。"
            )
    result = trimesh.boolean.union(meshes, engine="manifold", check_volume=True)
    if not isinstance(result, trimesh.Trimesh) or not result.is_volume:
        raise ValueError(f"{label}：Manifold 没有返回有效封闭实体")
    return result


def validate_simplify_tolerance(tolerance):
    """容差控制允许的表面偏移，不是目标面数或平滑强度。"""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("simplify-tolerance 必须是大于零的有限数值，单位为米")


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
        components = len(mesh.split(only_watertight=False))
        print(f"  {label}：三角形={len(mesh.faces)}，最小内角={minimum_angles.min():.6f}°，"
              f"小于1°的面={np.count_nonzero(minimum_angles < 1)}，封闭={mesh.is_watertight}，"
              f"朝向一致={mesh.is_winding_consistent}，连通分量={components}，体积={mesh.volume:.9f} m^3",
              flush=True)
    print(f"  体积变化={(simplified.volume / original.volume - 1) * 100:.6f}%", flush=True)


def validate_output_paths(obj_path, output_path, raw_output_path, simplify, simplified_output_path):
    """在任何导出之前，检查所有输出互不覆盖；返回实际使用的简化路径或 None。"""
    if not simplify and simplified_output_path is not None:
        raise ValueError("--simplified-output 只能与 --simplify 一起使用")
    paths = [Path(obj_path), Path(output_path), Path(raw_output_path)]
    if simplify:
        # 与原布尔文件放在同一目录；自定义 --output 后也随它派生，不固定回默认目录。
        original = Path(output_path)
        simplified_output_path = (Path(simplified_output_path) if simplified_output_path is not None
                                  else original.with_name(f"{original.stem}_simplified.obj"))
        paths.append(simplified_output_path)
    for path in paths[1:]:
        if path.suffix.lower() != ".obj":
            raise ValueError(f"输出路径必须以 .obj 结尾：{path}")
    # resolve 统一相对路径、.. 和已有符号链接；Windows Path 比较也会忽略大小写。
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("输入 OBJ、原布尔输出、拼接对照和简化输出必须是不同的文件，不能相互覆盖")
    # 不同路径还可能指向同一个硬链接文件，也应拒绝，避免写一个输出时改动另一份。
    for index, path in enumerate(paths):
        if path.exists():
            for other in paths[:index]:
                if other.exists() and path.samefile(other):
                    raise ValueError(f"路径指向同一个文件，不能相互覆盖：{path} 与 {other}")
    return simplified_output_path


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


def create_scene(obj_path=DEFAULT_OBJ_PATH, output_path=DEFAULT_OUTPUT_PATH,
                 scale=DEFAULT_OBJ_SCALE, yaw_deg=0.0,
                 obj_up_axis=DEFAULT_OBJ_UP_AXIS, position=None, up_axis="Y",
                 sink=DEFAULT_OBJ_SINK, raw_output_path=DEFAULT_RAW_OUTPUT_PATH,
                 simplify=False, simplify_tolerance=DEFAULT_SIMPLIFY_TOLERANCE,
                 simplified_output_path=None, *, scale_with_platform=True):
    """导出拼接与原布尔结果；可选简化分支另存副本，仍返回原布尔路径。

    simplify：默认 False；True 才调用 Manifold 简化，不替换原布尔 mesh。
    simplify_tolerance：简化表面容差，米，仅在 simplify=True 时使用。
    simplified_output_path：简化 OBJ 的独立路径；None 为原布尔文件名加 _simplified。
    scale_with_platform：默认按中央平台相对 1 米基准调整 OBJ 倍率；地形不受影响。
    """
    check_boolean_backend()
    up_axis = up_axis.upper()
    if up_axis not in ("Y", "Z"):
        raise ValueError("up_axis 必须是 Y 或 Z")
    output_path = Path(output_path)
    raw_output_path = Path(raw_output_path)
    simplified_output_path = validate_output_paths(
        obj_path, output_path, raw_output_path, simplify, simplified_output_path,
    )
    if simplify:
        validate_simplify_tolerance(simplify_tolerance)

    terrain = generate_stairs()
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
    raw_scene = combine_meshes(terrain, obj)
    # 先保存摆放对照。即使 OBJ 不封闭、后续布尔停止，也可以查看下沉效果。
    export_scene_mesh(raw_scene, raw_output_path, up_axis)
    print(f"拼接对照：{raw_output_path.resolve()}", flush=True)
    solid_terrain = union_stair_boxes(terrain)
    scene = boolean_union_meshes([solid_terrain, obj], "场景并集")
    print(f"场景并集：封闭={scene.is_watertight}，体积={scene.volume:.6f} m^3，三角形={len(scene.faces)}")

    export_scene_mesh(scene, output_path, up_axis)
    if simplify:
        # 原并集已经独立保存；只有这一分支会简化副本，不修改 scene 或原输出。
        try:
            simplified = simplify_mesh_manifold(scene, simplify_tolerance)
            report_simplification(scene, simplified, simplify_tolerance)
            export_scene_mesh(simplified, simplified_output_path, up_axis)
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            raise ValueError(
                f"简化分支失败：{error}；原布尔结果已保留在 {output_path.resolve()}。"
                "本次简化输出未完成，已有同名简化文件可能是先前结果。"
            ) from error
        print(f"简化副本：{simplified_output_path.resolve()}（{up_axis}-up，同名文件覆盖）", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=DEFAULT_OBJ_PATH, help=f"输入 OBJ；默认 Assets/{DEFAULT_OBJ_PATH.relative_to(ASSET_DIR).as_posix()}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="布尔输出 OBJ；默认 Assets/output/stairs_scene_bool.obj")
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT_PATH, help="拼接对照 OBJ；默认 Assets/output/stairs_scene.obj")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=False,
                        help="另存 Manifold 简化副本；默认关闭，--no-simplify 显式关闭")
    parser.add_argument("--simplify-tolerance", type=float, default=DEFAULT_SIMPLIFY_TOLERANCE,
                        help="简化表面容差，米；默认 0.001，仅在 --simplify 时使用")
    parser.add_argument("--simplified-output", type=Path,
                        help="简化副本 OBJ 路径；默认由 --output 文件名追加 _simplified，不能覆盖其他输入/输出")
    parser.add_argument("--scale", type=float, default=DEFAULT_OBJ_SCALE,
                        help=f"OBJ 在 1 米基准平台上的标定倍率，默认 {DEFAULT_OBJ_SCALE:g}（兔子宽占平台95%%）；联动时再乘当前平台短边/1米，不改变地形")
    parser.add_argument("--scale-with-platform", action=argparse.BooleanOptionalAction, default=True,
                        help="默认按平台短边等比缩放 OBJ；--no-scale-with-platform 恢复固定 --scale 倍率")
    parser.add_argument("--yaw", type=float, default=0.0, help="模型绕 Z 轴旋转的角度，单位：度")
    parser.add_argument("--obj-up-axis", type=str.upper, choices=("Y", "Z"), default=DEFAULT_OBJ_UP_AXIS, help="输入 OBJ 的向上轴；当前兔子默认 Y")
    parser.add_argument("--up-axis", type=str.upper, choices=("Y", "Z"), default="Y", help="输出场景的向上轴")
    parser.add_argument("--position", type=float, nargs=3, metavar=("X", "Y", "Z"), help="OBJ 底面支撑中心的位置，Z-up、米；显式指定时不叠加 sink")
    parser.add_argument("--sink", type=float, default=DEFAULT_OBJ_SINK, help="自动放置时下沉深度，米；默认 0.003，让兔子底部与平台略微相交")
    args = parser.parse_args()
    try:
        output = create_scene(
            obj_path=args.obj,
            output_path=args.output,
            scale=args.scale,
            yaw_deg=args.yaw,
            obj_up_axis=args.obj_up_axis,
            position=args.position,
            up_axis=args.up_axis,
            sink=args.sink,
            raw_output_path=args.raw_output,
            simplify=args.simplify,
            simplify_tolerance=args.simplify_tolerance,
            simplified_output_path=args.simplified_output,
            scale_with_platform=args.scale_with_platform,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"已导出：{output.resolve()}（{args.up_axis}-up，同名文件覆盖）")


if __name__ == "__main__":
    main()
