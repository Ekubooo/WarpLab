# sim_Ready 离线资产管线

包含两条独立流程：默认容器流程将模型放置在程序化阶梯中央；刚体流程处理保留局部原点的独立实体。两条流程均输出同名 OBJ 和单网格 SDF NVDB；容器发布到资产根目录，刚体发布到 `Rigid` 子目录。几何生成与仿真通过资产文件衔接；PBF 只复用 SimReady 独立的 NVDB 写入工具，不启动几何生成管线。

容器流程的 `NAME.obj` 是无墙景观，`NAME.nvdb` 是含四墙与顶盖的碰撞距离场，使用 Y-up 世界坐标和米制单位。刚体流程的两份文件表示同一独立实体，使用一致的资产局部坐标。

## 使用

在仓库根目录、配置了依赖的 Python 环境中运行：

```powershell
python -m demos.sim_Ready
python -m demos.sim_Ready --obj bunny_flat_bottom.obj --name stairs_scene --device cuda:0
python -m demos.sim_Ready --obj spot_triangulated_good.obj --name stairs_spot
python demos/sim_Ready/__main__.py --help
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--obj PATH` | `bunny_flat_bottom.obj` | 裸文件名从本目录的 `Assets` 读取；含目录的相对路径按工作目录解释，支持绝对路径 |
| `--name NAME` | `stairs_scene` | 两种输出共用的文件名主体，不含扩展名、目录或盘符 |
| `--obj-up-axis Y\|Z` | `Y` | 输入模型的向上轴，OBJ 自身通常不记录该信息 |
| `--sdf-voxel-size` | `0.0075` | 体素间距，米 |
| `--sdf-max-support-radius` | `0.05` | 此 SDF 支持的最大流体支撑半径，米；应不小于 PBF 的 `4 × particle_radius` |
| `--device` | `cuda:0` | 烘焙 CUDA 设备 |

路径根据脚本位置确定；从其他工作目录运行模块时，需要让仓库根目录处于 Python 模块搜索路径。帮助与参数解析不加载模型或初始化 CUDA。

依赖：`numpy`、`trimesh`、支持 `Mesh64` 的 `manifold3d` 和支持 CUDA NanoVDB 的 `warp`。运行测试另需 `scipy`。输入模型必须是封闭、朝向一致的正体积实体；不会自动补洞或裁平模型。

## 独立刚体流程

从仓库根目录使用独立入口，容器入口 `python -m demos.sim_Ready` 保持原行为：

```powershell
conda run -n WarpSim --no-capture-output python -m demos.sim_Ready.rigid_pipeline --obj bunny_flat_bottom.obj --name rigid_bunny
# 直接脚本运行同样支持
python demos/sim_Ready/rigid_pipeline.py --obj path/to/part.obj --name rigid_part
```

`--obj` 和 `--name` 必填，名称和路径解析规则与容器一致。新增参数及差异如下，其余烘焙参数和设备参数沿用上表：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--scale` | `1.0` | 显式等比缩放，必须为有限正数；输入单位按米解释，可用于单位换算 |
| `--obj-up-axis Y\|Z` | 不指定 | 默认保留模型局部轴；仅显式指定 Z 时绕原点旋转为 Y-up |

不自动居中、按平台缩放、放置到地形或简化模型。输出为 `output/rigid/NAME.obj` 与 `output/rigid/NAME.nvdb`，由 `config.RIGID_OUTPUT` 和 `config.RIGID_OUTPUT_DIRECTORIES` 控制，默认发布到 `PBF/Assets/Rigid`。mesh、SDF 和后续体积图始终按同一名称配对；不为每份资产增加子目录或单独的体积图文件。旧根目录下的刚体资产不会自动搬移，可重新烘焙或手动成对移动。

```text
rigid_pipeline.build_rigid_asset()
├─ 解析路径与源文件覆盖保护
├─ geometry.prepare_rigid()：读取模型、显式缩放／轴转换
├─ validation.validate_rigid_parts()：校验各个输入部件
├─ geometry.union()：合并重叠实体，移除内部重复表面
├─ validation.validate_solid()：校验结果及 float32 转换
├─ pipeline.save_mesh_obj()：保存局部坐标 OBJ
├─ sdf.bake_rigid() → sdf.bake()：计算外扩范围，复用 GPU 烘焙
└─ 保存单网格 NVDB → pipeline.publish_pair()
```

同一个资产可以包含多个固定在一起的正向封闭部件。支持重叠、面接触和彼此分离的实体；布尔并集后只在点／边连接而形成非流形的结果会被拒绝。输入中的开放面、反向部件、退化／重复面和 float32 转换后坍缩都会报错；只合并完全重合的顶点，不自动补洞、修复法线或推断嵌套空腔。带反向内壳的空腔模型不在本流程支持范围。拓扑校验不构成通用自交证明。

刚体每侧采样域至少外扩 `ceil((2*h_max + 2*sqrt(3)*dx)/dx)` 个体素，并取不小于 `BakeOptions.padding_voxels` 的值；默认外扩 17 个体素（0.1275 m），随后按原有 tile 规则对齐。该范围覆盖当前 PBF 体积图的外侧积分和插值邻域；不改变容器默认 3 个体素的外扩规则。SDF 距离符号、正侧截断背景及 `hmax` 元数据格式仍与容器一致。

Python API：

```python
from demos.sim_Ready.config import BakeOptions
from demos.sim_Ready.rigid_pipeline import build_rigid_asset

result = build_rigid_asset(
    "part.obj", name="rigid_part", scale=1.0, obj_up_axis=None,
    bake_options=BakeOptions(voxel_size=0.0075, max_support_radius=0.05),
    device="cuda:0",
)
print(result.mesh, result.sdf, result.destinations)
```

本流程只生产几何与 SDF，不导入 Newton／MimicKit，不处理关节、质量惯量或动态位姿。PBF 扫描 `Assets/Rigid` 下的 OBJ，并加载启用资产的同名 NVDB，生成并缓存体积图到同一文件。重新运行刚体流程会重新发布单网格 SDF，下次 PBF 加载时重新生成体积图。当前 PBF 只完成动态资产与实例管理；动态实例尚不参与流体压力、碰撞投影或渲染。

## 容器阶段与职责

```text
__main__.py：CLI 参数
└─ pipeline.py：流程与发布
   ├─ config.py：配置与静态目的地
   ├─ geometry.py：地形、模型定位、布尔和简化
   │  └─ terrain_generator.py：原样复制的地形源码
   ├─ validation.py：内存网格检查
   ├─ sdf.py：正侧窄带 CUDA 烘焙
   └─ nvdb_io.py：NVDB 网格组装与原子保存

生成地形 → 统一 Y-up → 读取/缩放/放置模型
├─ 地形＋模型 → 布尔并集 → 简化 → output/NAME.obj
└─ 地形＋模型＋墙/盖/底座扩展 → 独立布尔并集 → 简化
   └─ 检查源网格及 float32 网格 → CUDA 烘焙 → output/NAME.nvdb
      └─ 发布同名 OBJ/NVDB 到其他配置目的地
```

景观先输出，随后构造和检查容器。两条布尔分支都使用未简化的基础几何；容器不使用简化景观作为输入。容器仅在内存中存在，不输出容器 OBJ、拼接对照或未简化副本。

## 几何规则

全部参数集中在 `config.GeometryOptions`，默认值为：

- 地形长宽 `3.1×1.6 m`，台阶宽 `0.15 m`、高度增量 `0.1 m`，中央平台 `0.7×0.7 m`。
- 底层地形为 Z-up，返回后以 `(x,y,z) → (x,z,W-y)` 转换；`W` 是原地形 Y 最大值。后续全部使用 Y-up。
- 模型先转换输入轴、绕 Y 旋转，再按 `0.95 × 平台短边 / 模型水平最大跨度` 等比缩放。默认 yaw 为零。支撑中心对齐平台中央，下沉 `0.003 m`；有平底时使用面积重心，否则使用下沉截面重心，最后退回最低接触点范围中心。
- 墙厚 `w=0.25 m`、内嵌 `δ=0.075 m`，内顶面 `Y=4.5 m`。长墙覆盖完整外长度，短墙长度为 `min(地形宽+δ, 外宽)`；墙顶进入顶盖 `δ`，墙、盖、底座扩展共用外轮廓。底座向下扩展 `max(w-0.15,0)`，上部进入原底座。
- 两条分支始终使用 Manifold 简化，容差 `0.001 m`，不平滑或重网格。

默认外轮廓约为 `(-0.175,-0.2,-0.175)` 至 `(3.275,4.75,1.775)`，空腔极值约为 `(0.075,0,0.075)` 至 `(3.025,4.5,1.525)`；空腔范围内部仍有台阶和模型。

检查包含有限坐标、面索引、退化与重复面、闭合边、顶点流形性、绕序、一个正向外壳和一个反向内腔，以及长方体统一外轮廓；float32 转换后再次检查。检查不构成通用自交证明，瘦长但非退化三角形仅告警，不自动调整内腔法线。

## SDF 存储规则

- 固体内部为负距离，全部保留真实采样值；空腔和外部属于正距离一侧。
- 设最大支撑半径为 `h_max`、体素间距为 `dx`，实际正侧截断距离及背景值为 `B = 2*h_max + 2*sqrt(3)*dx`。默认 `B ≈ 0.12598 m`。两倍支撑半径覆盖体积图非零区域，额外余量保护 SDF 和体积图的插值邻域。
- 烘焙保存 `min(distance, B)`，远离表面的正侧区域使用常量正背景。每个候选 `8³` tile 先查询中心距离，并加上 tile 采样点包围半径作保守筛选，再分配和填充保留的 tile；不先分配完整稠密 Volume。边界 tile 中可能保留少量背景值采样，这是 tile 粒度造成的正常开销。
- 外侧范围沿用原版 `padding_voxels`，默认每侧 3 个体素（0.0225 m），不随 h_max 或 B 扩大；原点为 `mesh.bounds[0] - padding_voxels*dx`，候选区域继续按 `8³` 叶块对齐。默认场景网格外包络恢复为长 X × 宽 Z × 高 Y = `3.54×2.04×5.04 m`。体积图复制 SDF 的稀疏 tile 和坐标变换。
- 正侧有效带及体积图正确性针对封闭容器的空腔使用区域；外表面邻近网格边缘处可能采到背景，不保证与充分扩展的全域参考一致。空腔内的距离、法线及体积图需与覆盖完整积分和插值邻域的稠密参考一致；不能把该范围约束直接推广到流体接触模型外表面的场景。
- SDF 网格名保存 `PBF_SDF|v=1|hmax=<十六进制浮点数>`。PBF 支撑半径超过 h_max 时需要使用更大的 `--sdf-max-support-radius` 重新烘焙，不能仅重算体积图。

## 输出与失败处理

`config.OUTPUT_DIRECTORIES` 默认包含本目录 `output` 和静态路径 `PBF_ASSETS`。以后新增目的地，直接在该配置元组中追加路径；不需要新增 CLI 参数。默认根输出目录必须保留在元组中。

刚体流程独立使用 `RIGID_OUTPUT`（默认 `output/rigid`）和 `RIGID_OUTPUT_DIRECTORIES`（默认包含本地目录与 `PBF_ASSETS/Rigid`）；新增刚体发布目的地时修改该元组。两条流程共用路径保护、原子保存和发布回滚工具。

- 每次运行重新生成、检查和烘焙，同名覆盖；SimReady 自身没有文件存在即复用的缓存逻辑，输出为单网格 SDF。
- 本地景观在首阶段原子替换。后续检查或烘焙失败时，这份新景观保留，既有 NVDB 不变，消费者目录不发布新文件。
- NVDB 完整写入同目录临时文件后原子替换；生成成功才准备消费者目录的 OBJ/NVDB 副本与备份。
- 发布时发生可处理的 I/O 错误则回滚已替换文件。若回滚本身失败，保留恢复副本并报告位置。多个文件与目录之间无法保证同时可见，应在仿真停止时发布。
- 拒绝输出覆盖输入模型，包括符号链接或硬链接指向同一文件的情况。临时文件正常结束时清理。

Python API 支持独立配置对象：

```python
from demos.sim_Ready.config import GeometryOptions, BakeOptions
from demos.sim_Ready.pipeline import build_assets

result = build_assets(
    "bunny_flat_bottom.obj", name="stairs_scene",
    geometry_options=GeometryOptions(),
    bake_options=BakeOptions(voxel_size=0.0075), device="cuda:0",
)
```

## PBF 与归档

```powershell
python -m demos.fluid.PBF.render_opengl --device cuda:0
# 使用 Spot 时，先在 boundary_model.py 中设置 SDF_FILENAME = "stairs_spot.nvdb"
python -m demos.fluid.PBF.render_opengl --device cuda:0
```

PBF 从自身 `Assets` 根目录读取静态场景，由 `boundary_model.py` 模块级常量 `SDF_FILENAME = "stairs_scene.nvdb"` 选择；OpenGL 从 `BoundarySys` 的资产记录取得同名 OBJ 路径。动态资产从 `Assets/Rigid` 发现，实例数量由同一模块中的 `DEFAULT_RIGID_COUNT` 和 `RIGID_COUNTS` 控制。PBF 启动 CLI 不接受 SDF 文件名，仿真构造函数只向默认加载入口传入平滑半径，结果保存为 `self.BoundarySys`。静态 USD 输出只要求 NVDB；动态资产发现要求存在配对 OBJ，但本阶段只记录其路径，不解析 mesh。现有静态渲染和流体物理配置保持原状。

SimReady 已自动发布文件到 PBF 的资产目录，无需手动移动。PBF 首次读取单网格 SDF 时预计算体积图，并把两份网格原子写回它读取的同一 NVDB；之后参数匹配时直接读取缓存。该缓存写回不会反向同步到 `sim_Ready/output`。重新运行 SimReady 会再次发布单网格 SDF，下次 PBF 启动自动补齐体积图。

旧 Boundary 目录和 PBF 的 `Assets/Boundary`、`Assets/Container` 保留为归档，不回退读取。新管线不导入任何归档模块；只有 `terrain_generator.py` 原样复制，保留其许可证与版权信息。

运行测试：

```powershell
python -m unittest discover -s tests/sim_ready -t . -v
python -m unittest discover -s tests/fluid/pbf -t . -v
```
