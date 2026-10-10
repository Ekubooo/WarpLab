"""通过 CLI 生成地形，保存到 Assets/procedural/<地形类型>.obj，同名文件覆盖。"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

if __package__:
    from .terrain_generator import create_mesh_terrain
else:
    from terrain_generator import create_mesh_terrain


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
OUTPUT_DIR = ASSET_DIR / "procedural"  # 纯程序化地形资产。


def generate_flat():
    """平面：height 为地表高度，没有厚度。"""
    return create_mesh_terrain(
        grid_size=(1, 1),       # (行数, 列数)：行沿 Y、列沿 X；这里仅生成一块。
        block_size=(6.0, 4.0),  # 每块在 X、Y 方向的长度，单位：米。
        terrain_types="flat",   # 选择无厚度的平整地表。
        terrain_params={        # 按地形类型名称传入该类型的专用参数。
            "flat": {
                "height": 0.0,  # 整个平面的 Z 高度，单位：米；不控制厚度。
            }
        },
    )


def generate_pyramid_stairs():
    """台阶：宽度、高度增量、中心平台宽度，单位均为米。"""
    return create_mesh_terrain(
        grid_size=(1, 1),                   # (行数, 列数)：行沿 Y、列沿 X；这里仅生成一块。
        block_size=(6.0, 4.0),              # 每块在 X、Y 方向的长度，单位：米。
        terrain_types="pyramid_stairs",     # 从外围向中心逐圈升高的台阶。
        terrain_params={            # 参数字典的键必须与所选地形名称一致。
            "pyramid_stairs": {
                "step_width": 0.25,          # 每圈台阶的水平宽度，米；越小可容纳的圈数越多。
                "step_height": 0.05,         # 每圈台阶的高度增量，米；中心平台另由源码计算。
                "platform_width": 1.0,      # 中心正方形平台的边长，米；应小于块的短边。
            }
        },
    )


def generate_random_grid(seed=None):
    """随机格子；seed 为整数种子，固定值可复现布局，None 表示不固定。"""
    return create_mesh_terrain(
        grid_size=(1, 1),               # (行数, 列数)：行沿 Y、列沿 X；这里仅生成一块。
        block_size=(6.0, 4.0),          # 每块在 X、Y 方向的长度，单位：米。
        terrain_types="random_grid",    # 用不同随机高度的小方格组成地面。
        terrain_params={  # 参数字典的键必须与所选地形名称一致。
            "random_grid": {
                "grid_width": 0.5,                      # 小方格的边长，米；不能整除的边缘余量会被舍弃。
                "grid_height_range": (-0.15, 0.15),     # 顶面相对 Z=0 的均匀随机高度范围，米。
                "platform_width": None,                 # 预留的平台宽度参数；当前源码未使用，设置无效。
            }
        },
        seed=seed,  # 主入口随机种子；为每块 random_grid 分配可复现的子种子。
    )


def generate_wave():
    """波浪：幅度单位为米，频率为半周期数，resolution 为每轴采样点数。"""
    return create_mesh_terrain(
        grid_size=(1, 1),       # (行数, 列数)：行沿 Y、列沿 X；这里仅生成一块。
        block_size=(6.0, 4.0),  # 每块在 X、Y 方向的长度，单位：米。
        terrain_types="wave",   # 连续的正弦波地表；不带底面或侧壁。
        terrain_params={  # 参数字典的键必须与所选地形名称一致。
            "wave": {
                "wave_amplitude": 0.75,     # 起伏幅度 A，米；理论高度范围为 [-A, A]。
                "wave_frequency": 2.0,      # 每轴跨越的半周期数；2 为一个完整周期，越大起伏越密。
                "resolution": 30,           # 每轴采样点数，整数且至少为 2；越大网格越细、文件越大。
            }
        },
    )


def generate_box():
    """中央平台：平台顶面高度和平台宽度，单位为米。"""
    return create_mesh_terrain(
        grid_size=(1, 1),       # (行数, 列数)：行沿 Y、列沿 X；这里仅生成一块。
        block_size=(6.0, 4.0),  # 每块在 X、Y 方向的长度，单位：米。
        terrain_types="box",    # 平地上叠加一个中心正方形凸起平台。
        terrain_params={        # 参数字典的键必须与所选地形名称一致。
            "box": {
                "box_height": 0.5,      # 平台顶面相对 Z=0 地面的高度，米；建议为正。
                "platform_width": 1.5,  # 中心平台边长，米；应小于块的短边。
            }
        },
    )


def generate_gap():
    """环绕沟壑：间隔宽度和中心平台宽度，单位为米。"""
    return create_mesh_terrain(
        grid_size=(1, 1),       # (行数, 列数)：行沿 Y、列沿 X；这里仅生成一块。
        block_size=(6.0, 4.0),  # 每块在 X、Y 方向的长度，单位：米。
        terrain_types="gap",    # 中心平台与外围地面之间留出环绕间隔，没有沟底。
        terrain_params={        # 参数字典的键必须与所选地形名称一致。
            "gap": {
                "gap_width": 0.8,       # 平台与外围地面之间的间隔宽度，米。
                "platform_width": 1.2,  # 中心平台边长，米；边长+2×间隔应小于块的短边。
            }
        },
    )


def generate_heightfield(heightfield_path=None, ground_z=None):
    """高度图：提供地表、底面和侧壁。

    heightfield_path：二维 .npy 高度数组路径；第一维对应 X，第二维对应 Y，值为米。
                     None 时生成带隆起和凹坑的示例高度图。
    ground_z：轴转换前的底面 Z 高度，米；None 时取最低地表高度减 0.3 米。
    """
    if heightfield_path is not None:
        height = np.asarray(np.load(heightfield_path, allow_pickle=False), dtype=np.float32)
    else:
        # 示例高度图：修改以下参数可直接观察隆起、凹陷及网格精细程度的变化。
        resolution_x = 41  # X 方向采样点数；越大网格越细，至少为 2。
        resolution_y = 31  # Y 方向采样点数；可与 X 不同，至少为 2。
        hill_height = 1.0  # 隆起强度，米；边缘衰减后实际峰值会略低于此值。
        pit_depth = 0.6  # 凹坑强度，米；越大坑越深。
        hill_radius = 0.18  # 隆起宽度，相对于块长宽的比例；越大隆起越宽。
        pit_radius = 0.16  # 凹坑宽度，相对于块长宽的比例；越大凹坑越宽。
        hill_center = (0.35, 0.45)  # 隆起中心占块 X、Y 长度的比例，范围为 [0, 1]。
        pit_center = (0.70, 0.60)  # 凹坑中心占块 X、Y 长度的比例，范围为 [0, 1]。

        x = np.linspace(0.0, 1.0, resolution_x)  # X 的归一化采样坐标；实际长度由 block_size 决定。
        y = np.linspace(0.0, 1.0, resolution_y)  # Y 的归一化采样坐标。
        X, Y = np.meshgrid(x, y, indexing="ij")  # 第一维为 X，第二维为 Y。
        hill = hill_height * np.exp(
            -((X - hill_center[0]) ** 2 + (Y - hill_center[1]) ** 2) / (2 * hill_radius**2)
        )
        pit = pit_depth * np.exp(
            -((X - pit_center[0]) ** 2 + (Y - pit_center[1]) ** 2) / (2 * pit_radius**2)
        )
        edge_fade = np.sin(np.pi * X) ** 2 * np.sin(np.pi * Y) ** 2  # 四周高度平滑衰减到 0，便于块间衔接。
        height = ((hill - pit) * edge_fade).astype(np.float32)  # 每个采样点的 Z 高度，包含正隆起和负凹陷。
    if height.ndim != 2 or min(height.shape) < 2 or not np.isfinite(height).all():
        raise ValueError("高度图必须是两个方向至少各有 2 个点的二维有限数值数组")
    ground_z = float(height.min()) - 0.3 if ground_z is None else ground_z
    if not np.isfinite(ground_z) or ground_z >= float(height.min()):
        raise ValueError("ground_z 必须是有限数值，且严格低于最低地表高度")
    return create_mesh_terrain(
        grid_size=(6, 6),               # (行数, 列数)：行沿 Y、列沿 X；生成 36 块相同高度图。
        block_size=(6.0, 4.0),          # 高度图覆盖的 X、Y 长度，米；与数组采样点数无关。
        terrain_types="heightfield",    # 将二维高度数组转为带底面和侧壁的网格。
        terrain_params={  # 参数字典的键必须与所选地形名称一致。
            "heightfield": {
                "heightfield": height,  # 二维浮点数组：height[i,j] 是 (x_i,y_j) 的 Z 高度，米。
                "center_x": None,       # 本块中心 X，米；None 取 X 长度的一半，使块从 X=0 开始。
                "center_y": None,       # 本块中心 Y，米；None 取 Y 长度的一半，使块从 Y=0 开始。
                "ground_z": ground_z,   # 底面 Z 高度，米；必须严格低于所有地表高度。
            }
        },
    )


TERRAIN_GENERATORS = {
    "flat": generate_flat,
    "pyramid_stairs": generate_pyramid_stairs,
    "random_grid": generate_random_grid,
    "wave": generate_wave,
    "box": generate_box,
    "gap": generate_gap,
    "heightfield": generate_heightfield,
}


def select_generator(terrain_type):
    """按七种类型名称或 all 返回函数；选择时不生成或写文件。"""
    if terrain_type == "all":
        return generate_all
    try:
        return TERRAIN_GENERATORS[terrain_type]
    except KeyError:
        raise ValueError(f"未知地形类型：{terrain_type}") from None


def export_mesh(terrain_type, vertices, indices, up_axis="Y"):
    """统一转换坐标并导出，覆盖 Assets/procedural/<类型名>.obj，返回文件路径。

    terrain_type：决定输出文件名，如 wave 对应 wave.obj。
    vertices：生成器返回的 (N, 3) 顶点数组，采用 Z-up 坐标。
    indices：生成器返回的展平三角形索引数组，三个索引组成一个面。
    up_axis：导出向上轴；Y 绕 X 轴旋转 -90°，Z 保留原始坐标。
    """
    up_axis = up_axis.upper()
    if up_axis not in ("Y", "Z"):
        raise ValueError("up_axis 必须是 Y 或 Z")
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=indices.reshape(-1, 3), process=False,
    )
    if up_axis == "Y":
        # (x, y, z) -> (x, z, -y)，保持右手坐标系。
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{terrain_type}.obj"
    mesh.export(output_path)
    return output_path


def generate_all(seed=None, heightfield_path=None, ground_z=None, up_axis="Y"):
    """显式调用七种独立生成函数并分别导出，返回七个输出路径。

    seed：random_grid 的随机种子；None 表示不固定。
    heightfield_path：heightfield 的二维 .npy 路径；None 使用默认隆起与凹坑示例。
    ground_z：heightfield 底面 Z 高度，米；None 为最低地表高度减 0.3 米。
    up_axis：全部资产的导出向上轴，Y 或 Z。
    """
    # 先读取并验证高度图，避免输入有误时已覆盖其他地形资产。
    heightfield_vertices, heightfield_indices = generate_heightfield(
        heightfield_path=heightfield_path,
        ground_z=ground_z,
    )
    # 每种地形独立调用；其参数设置仍在对应 generate_* 函数中。
    return (
        export_mesh("flat", *generate_flat(), up_axis=up_axis),
        export_mesh("pyramid_stairs", *generate_pyramid_stairs(), up_axis=up_axis),
        export_mesh("random_grid", *generate_random_grid(seed=seed), up_axis=up_axis),
        export_mesh("wave", *generate_wave(), up_axis=up_axis),
        export_mesh("box", *generate_box(), up_axis=up_axis),
        export_mesh("gap", *generate_gap(), up_axis=up_axis),
        export_mesh("heightfield", heightfield_vertices, heightfield_indices, up_axis=up_axis),
    )


def create_parser():
    """创建可选择一种地形或 all 的 CLI 选择器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terrain", choices=(*TERRAIN_GENERATORS, "all"), default="wave",
        help="本次生成的地形类型，默认 wave；all 分别导出七种类型",
    )
    parser.add_argument(
        "--up-axis", type=str.upper, choices=("Y", "Z"), default="Y",
        help="导出向上轴，默认 Y；与 Newton 场景一致时使用 Z",
    )
    parser.add_argument("--seed", type=int, help="random_grid 的随机种子")
    parser.add_argument(
        "--heightfield", type=Path, metavar="FILE.npy",
        help="heightfield 使用的二维高度数组；省略时生成带隆起和凹坑的封闭地形",
    )
    parser.add_argument(
        "--ground-z", type=float,
        help="heightfield 在轴转换前的底面高度；默认比最低地表低 0.3 米",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.terrain not in ("heightfield", "all") and (
        args.heightfield is not None or args.ground_z is not None
    ):
        parser.error("--heightfield 和 --ground-z 仅用于 --terrain heightfield 或 all")

    generator = select_generator(args.terrain)
    try:
        if args.terrain == "all":
            output_paths = generator(
                seed=args.seed,
                heightfield_path=args.heightfield,
                ground_z=args.ground_z,
                up_axis=args.up_axis,
            )
            print(f"已导出全部地形（{args.up_axis}-up，同名文件覆盖）：", *output_paths, sep="\n")
            return
        elif args.terrain == "random_grid":
            vertices, indices = generator(seed=args.seed)
        elif args.terrain == "heightfield":
            vertices, indices = generator(
                heightfield_path=args.heightfield, ground_z=args.ground_z,
            )
        else:
            vertices, indices = generator()
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    output_path = export_mesh(args.terrain, vertices, indices, args.up_axis)
    print(f"已导出：{output_path}（{args.up_axis}-up，同名文件覆盖）")


if __name__ == "__main__":
    main()
