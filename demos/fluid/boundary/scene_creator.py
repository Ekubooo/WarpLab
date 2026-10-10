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
DEFAULT_OBJ_PATH = ORIGIN_DIR / "spot_triangulated_good.obj"  # 当前奶牛；也可用 --obj 指定其他模型。
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "stairs_scene_bool.obj"  # 布尔结果；重复运行覆盖。
DEFAULT_RAW_OUTPUT_PATH = OUTPUT_DIR / "stairs_scene.obj"  # 同位置的原始拼接结果，方便对照。
DEFAULT_OBJ_SCALE = 0.5  # 奶牛缩小到一半：宽约 0.472、深 0.859、高 0.845 米，适合 1×1 米平台。
DEFAULT_OBJ_UP_AXIS = "Y"  # 当前奶牛是 Y-up；OBJ 不提供统一 up 信息，不同模型需手动指定。
DEFAULT_OBJ_SINK = 0.02  # 默认向平台下沉 2 厘米，让奶牛四只脚与平台略微相交。
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
        block_size=(6.0, 4.0),              # X 长度和 Y 长度，米。
        terrain_types="pyramid_stairs",     # 四周低、中心高的同心阶梯。
        terrain_params={
            "pyramid_stairs": {
                "step_width": 0.25,         # 每圈台阶的水平宽度，米。
                "step_height": 0.05,        # 每圈高度增量，米；当前源码应使用正值。
                "platform_width": 1.0,      # 中心正方形平台的边长，米。
            }
        },
    )

    # vertices 是 (N, 3) 数组：N 个顶点，每一行是一个 [x, y, z]。
    # indices 是一维数组，如 [0, 1, 2, 2, 3, 0]：每三个数字表示一个三角形。
    # reshape(-1, 3) 将它变成 [[0, 1, 2], [2, 3, 0]]；-1 表示自动计算行数。
    faces = indices.reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def load_and_place_obj(obj_path, terrain, scale=DEFAULT_OBJ_SCALE, yaw_deg=0.0,
                       obj_up_axis=DEFAULT_OBJ_UP_AXIS, position=None, sink=DEFAULT_OBJ_SINK):
    """先统一 OBJ 坐标，再把它的底部中心移到指定位置。

    scale：尺寸倍率，当前奶牛默认 0.5；例如厘米模型转米使用 0.01。
    yaw_deg：绕竖直 Z 轴旋转的角度，单位为度。
    obj_up_axis：原始 OBJ 的向上轴，Y 或 Z；当前奶牛默认 Y。
    position：目标底部中心 [x, y, z]，单位为米，采用 Z-up；None 放在阶梯顶端中心。
    sink：自动放置时向下嵌入平台的深度，米；显式 position 不再叠加 sink。
    """
    obj_path = Path(obj_path)
    if not obj_path.is_file():
        raise FileNotFoundError(f"找不到 OBJ：{obj_path}；默认读取 {DEFAULT_OBJ_PATH}，或使用 --obj 指定文件。")
    if obj_path.suffix.lower() != ".obj":
        raise ValueError("请提供 .obj 文件")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale 必须是大于零的有限数值")
    if not np.isfinite(yaw_deg):
        raise ValueError("yaw_deg 必须是有限数值")
    if not np.isfinite(sink) or sink < 0:
        raise ValueError("sink 必须是大于或等于零的有限数值")
    obj_up_axis = obj_up_axis.upper()
    if obj_up_axis not in ("Y", "Z"):
        raise ValueError("obj_up_axis 必须是 Y 或 Z")

    obj = trimesh.load(obj_path, force="mesh", process=False)
    if len(obj.vertices) == 0 or len(obj.faces) == 0 or not np.isfinite(obj.vertices).all():
        raise ValueError("OBJ 必须包含有效的顶点和三角面")

    # 整个放置过程统一使用 Z-up：X、Y 在地面上，Z 表示高度。
    # 按指定轴转换：如果原始 OBJ 为 Y-up，绕 X 轴 +90° 转为 Z-up。
    if obj_up_axis == "Y":
        obj.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    obj.apply_scale(scale)
    obj.apply_transform(
        trimesh.transformations.rotation_matrix(np.deg2rad(yaw_deg), [0, 0, 1])
    )

    # bounds 是 (2, 3) 数组：第一行 [最小X, 最小Y, 最小Z]，第二行是各轴最大值。
    # 这些数已经包含刚才的旋转和缩放效果。
    x_min, y_min, z_min = obj.bounds[0]
    x_max, y_max, _ = obj.bounds[1]  # 下划线表示这里不需要最大 Z。

    # 用包围盒定义“底部中心”：X、Y 取中点，Z 取最低点。
    # 模型原点不一定在脚底，因此直接移动模型原点可能造成悬空或埋入地面。
    bottom_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])

    if position is None:
        # 当前阶梯的最高位置就是中央平台。
        # 平台顶面约 0.4 米；默认下沉 0.02 米，脚底最低处变成 [3, 2, 约0.38]。
        ground_min_x, ground_min_y, _ = terrain.bounds[0]
        ground_max_x, ground_max_y, platform_z = terrain.bounds[1]
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
    boxes = []
    for box_faces in terrain.faces.reshape(-1, 12, 3):
        # 挑出这个箱体使用的原顶点编号，再将它们重编号为 0、1、2……。
        vertex_ids, local_indices = np.unique(box_faces, return_inverse=True)
        box = trimesh.Trimesh(
            vertices=terrain.vertices[vertex_ids],
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
                 simplified_output_path=None):
    """导出拼接与原布尔结果；可选简化分支另存副本，仍返回原布尔路径。

    simplify：默认 False；True 才调用 Manifold 简化，不替换原布尔 mesh。
    simplify_tolerance：简化表面容差，米，仅在 simplify=True 时使用。
    simplified_output_path：简化 OBJ 的独立路径；None 为原布尔文件名加 _simplified。
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
    )
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
    parser.add_argument("--scale", type=float, default=DEFAULT_OBJ_SCALE, help="模型尺寸倍率；当前奶牛默认 0.5，厘米转米用 0.01")
    parser.add_argument("--yaw", type=float, default=0.0, help="模型绕 Z 轴旋转的角度，单位：度")
    parser.add_argument("--obj-up-axis", type=str.upper, choices=("Y", "Z"), default=DEFAULT_OBJ_UP_AXIS, help="输入 OBJ 的向上轴；当前奶牛默认 Y")
    parser.add_argument("--up-axis", type=str.upper, choices=("Y", "Z"), default="Y", help="输出场景的向上轴")
    parser.add_argument("--position", type=float, nargs=3, metavar=("X", "Y", "Z"), help="OBJ 底部中心的位置，Z-up、米；显式指定时不叠加 sink")
    parser.add_argument("--sink", type=float, default=DEFAULT_OBJ_SINK, help="自动放置时下沉深度，米；默认 0.02，让奶牛脚部与平台相交")
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
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"已导出：{output.resolve()}（{args.up_axis}-up，同名文件覆盖）")


if __name__ == "__main__":
    main()
