## PBFConfig
a set of good para for the water effect

run:
```
python -m demos.fluid.PBF.render_opengl --device cuda:0 --particle-radius 0.01  --substeps 3 --pressure-iterations 3
--container-size 3.1 8 1.6 --block-start 0.05 0 0.05 --block-end 1.55 1.5 1.55
```

| para | value | note |
| --- | --- | --- |
| `particle_radius` | `0.0125` | 粒子半径，m |
| `rest_density` | `1000.0` | 静止密度，kg/m³ |
| `frame_dt` | `1.0 / 90.0` | 每个 step 的时间，s |
| `substeps` | `3` | 每个 step 的子步数 |
| `pressure_iterations` | `3` | 每个子步的压力迭代次数 |
| `lambda_regularization` | `2.0` | 正则化系数，ε = α / h² |
| `gravity` | `(0.0, -9.81, 0.0)` | 初始重力加速度，m/s² |
| `viscosity` | `0.25 or 0.5` | XSPH 速度修正系数 |
| `vorticity_confinement` | `0.5` | 涡量补偿系数 |
| `max_speed` | `7.5 or 6.5` | 速度上限，m/s |
| `artificial_pressure_strength` | `0.0005` | 人工压力强度，0 关闭 |
| `artificial_pressure_q` | `0.1` | 人工压力参考距离与 h 的比例 |
| `clamp_negative_pressure` | `True` | 将负密度约束钳制为 0 |
| `container_size` | `(3.1, 8.0, 1.6)` | 容器 XYZ 尺寸，m |
| `block_start` | `(0.05, 0.0, 0.05)` | 初始粒子块最小坐标，m |
| `block_end` | `(1.55, 1.5, 1.55)` | 初始粒子块最大坐标，m |
| `initial_velocity` | `(0.0, 0.0, 0.0)` | 预留字段；当前初始化速度固定为零 |

## OpenGL CLI：render_opengl.py

启动时从 `PBF/Assets` 根目录读取所选 NVDB 的同名景观 OBJ：默认 `stairs_scene.nvdb` 对应 `stairs_scene.obj`，绘制底座、台阶和模型，围墙与顶盖保持不可见。资产由独立的 `sim_Ready` 管线生成，统一采用 Y-up 世界坐标。路径根据脚本位置确定，不依赖工作目录；缺失或损坏时报告完整路径。

景观仅在启动时加载、注册一次，使用 Warp 默认网格着色器、灰色 `(0.55, 0.55, 0.55)` 和 `smooth_shading=False`；暂停或重置不会重新加载。粒子仍使用原有按速度着色的 billboard。`--help` 和 `--export-config` 不加载景观。

| para | value | note |
| --- | --- | --- |
| `-h / --help` | — | 显示帮助并退出 |
| `--device` | `None` | Warp 设备，如 `cuda:0`、`cpu` |
| `--num-frames` | `0` | 最大显示帧数，0 不限 |
| `--speed-color-mid` | `2.0` | 速度配色中点，m/s |
| `--speed-color-max` | `7.5` | 速度配色上限，须大于中点 |
| `--verbose` | `False` | 输出各 kernel 耗时 |
| `--config PATH` | 无 | 导入 JSON，缺失字段用默认值；与 CLI 仿真参数互斥 |
| `--export-config [NAME]` | `default_para.json` | 指定选项时导出到 PBF 的 `config/` 后退出；NAME 可省略，仅接受文件名 |
| `--particle-radius` | `0.0125` | 粒子半径，m |
| `--rest-density` | `1000.0` | 静止密度，kg/m³ |
| `--frame-dt` | `1.0 / 90.0` | 每个 step 的时间，s |
| `--lambda-regularization` | `2.0` | 正则化系数 |
| `--substeps` | `3` | 每个 step 的子步数 |
| `--pressure-iterations` | `3` | 每个子步的压力迭代次数 |
| `--container-size X Y Z` | `3.1 8.0 1.6` | 容器尺寸，m |
| `--block-start X Y Z` | `0.05 0.0 0.05` | 初始粒子块最小坐标，m |
| `--block-end X Y Z` | `1.55 1.5 1.55` | 初始粒子块最大坐标，m |
| `--clamp-negative-pressure / --no-clamp-negative-pressure` | `True` | 开启／关闭负压钳制 |

## 离线/USD CLI：PBF.py

| 参数 | 默认值 | 简短说明 |
| --- | --- | --- |
| `-h / --help` | — | 显示帮助并退出 |
| `--device` | `None` | Warp 设备 |
| `--stage-path` | `example_pbf.usd` | USD 输出路径，`None` 禁用输出 |
| `--num-frames` | `480` | 仿真帧数 |
| `--verbose` | `False` | 输出各 kernel 耗时 |
| `--config PATH` | 无 | 导入 JSON，缺失字段用默认值；与 CLI 仿真参数互斥 |
| `--export-config [NAME]` | `default_para.json` | 指定选项时导出到 PBF 的 `config/` 后退出；NAME 可省略，仅接受文件名 |
| `--clamp-negative-pressure / --no-clamp-negative-pressure` | `True` | 开启／关闭负压钳制 |

## 命令目录（仓库根目录）

| operation | cmd |
| --- | --- |
| 默认启动 | `conda run -n warplab --no-capture-output python -m demos.fluid.PBF.render_opengl --device cuda:0` |
| 导入 `default_para.json` | `conda run -n warplab --no-capture-output python -m demos.fluid.PBF.render_opengl --device cuda:0 --config demos/fluid/PBF/config/default_para.json` |
| 默认导出 `default_para.json` | `conda run -n warplab --no-capture-output python -m demos.fluid.PBF.render_opengl --export-config` |
| 指定导出 `water.json` | `conda run -n warplab --no-capture-output python -m demos.fluid.PBF.render_opengl --export-config water.json` |
| 导入 `water.json` | `conda run -n warplab --no-capture-output python -m demos.fluid.PBF.render_opengl --device cuda:0 --config demos/fluid/PBF/config/water.json` |

## 启动时读取静态 SDF

资产选择由 `boundary_model.py` 的模块级常量控制；固定目录、文件名校验、路径解析和 GPU 读取由 `boundary_model.py` 管理，不随工作目录变化：

```python
# boundary_model.py：修改这里即可切换碰撞与配对景观
SDF_FILENAME = "stairs_scene.nvdb"
```

两个入口均不提供 SDF 文件选择 CLI。选定 CUDA 设备后创建仿真实例，由 `PBF.__init__()` 调用 `boundary_model.load_default_boundaries(self.smoothing_length)`，取得 `self.BoundarySys`。默认入口只接收平滑半径；设备沿用当前 Warp 设备，积分点数等加载默认值由边界模块管理。资产表统一持有 SDF／体积图的 `wp.Volume`，实例表及 GPU 数据保存映射与位姿。默认静态资产仍通过 `self.static_sdf`、`self.volume_map` 提供兼容引用。文件名不进入构造或创建函数的参数列表，物理 JSON 不包含资产选择。OpenGL 从已加载系统的首个静态实例取得配对 OBJ 路径；静态 USD 输出只要求 NVDB。重置不会重新加载资源；文件缺失、损坏或类型错误会报告完整路径，不自动烘焙几何。

### 动态资产与实例管理

SimReady 刚体流程发布同名文件到 `Assets/rigid/NAME.obj` 和 `NAME.nvdb`。加载器非递归扫描其中的 OBJ，以同名 NVDB 配对，按文件名排序；不读取 OBJ 内容，也不导入 Newton。无刚体目录或目录为空时，动态实例数为零。代码中的数量配置位于 `boundary_model.py`，例如：

```python
DEFAULT_RIGID_COUNT = 1
RIGID_COUNTS = {"rigid_box": 3, "rigid_bunny": 0}
```

未覆盖的资产默认生成 1 个实例；数量为 0 时不加载该资产的 Volume。数量必须为非负整数，覆盖键必须是扫描到的资产名称（不含扩展名），避免误拼写。多个实例只加载一份共享 SDF／体积图。内部 `rigid_id` 只是当前实例编号，不是 Newton `body_id`。

需要自定义场景、设备或积分点数时，仍可独立使用底层加载接口；PBF 默认入口无需创建这些配置对象：

```python
from demos.fluid.PBF import boundary_model as boundary

scene = boundary.BoundaryScene(
    static_assets=("stairs_scene.nvdb",),
    default_rigid_count=1,
    rigid_counts={"rigid_box": 3},
)
system = boundary.load_boundaries(
    scene, boundary.BoundaryLoadOptions(support_radius=0.05, device="cuda:0"),
)
```

`system.assets` 记录 Volume 资源和配对 mesh 路径；`system.instances` 记录资产编号与静态／动态关系；`system.device_data` 是供 Warp kernel 使用的批量数据。所有实例初始变换为单位变换，不自动布置多个模型。`system.set_rigid_poses(poses)` 按动态实例顺序更新资产局部到世界的变换：CPU 输入为 `N×7` 的 XYZ＋XYZW 数组，四元数在副本上归一化；GPU 输入为有限、单位四元数的 `wp.transform` 数组，不回读 CPU。静态实例不受该方法影响。

`system.reset()` 原位恢复初始单位变换，PBF 的重置操作也会调用它；资源与设备缓冲不重建。`sample_instance_sdf()` 返回世界坐标下的距离与梯度，`sample_instance_volume()` 返回边界体积，供独立验证和后续求解接入使用。

**本阶段的动态实例尚不参与 PBF 压力、SDF 碰撞投影或渲染，也不推进刚体动力学。** 现有 kernel 继续查询默认静态容器。运行中增删、Newton 后端和反作用力属于后续扩展。旧根目录下的刚体资产不自动搬移；请重新烘焙或将同名 OBJ／NVDB 成对移入 `rigid`。

首次读取 SimReady 发布的单网格 SDF 时，boundary_model 调用现有体积图预计算，然后调用 SimReady 的 `nvdb_io.save_volumes()` 原子写回当前 NVDB。文件使用一个 segment，依次保存 SDF 和体积图两个 float32 grid；写回后释放临时资源，再通过统一读取路径返回两个 Volume。此过程不安装额外 NanoVDB 库、不修改压力或碰撞算法，也不手动复制资产。再次启动若缓存有效则直接读取，不再积分。仅 PBF 正在读取的资产副本会补齐缓存，SimReady 的 output 副本仍可保持单网格。

体积图 grid 名称保存 `h`、每轴高斯积分点数（默认 16）和 `VOLUME_MAP_ALGORITHM_VERSION`（初始 1）；任一不匹配或缺少缓存标记都会重新计算并替换第二网格。修改会影响预计算结果的公式或规则时递增该版本。SimReady 的正侧窄带 SDF 名称记录 `max_support_radius`，运行时要求 `h <= max_support_radius`；超出时须用更大上限重新烘焙，不能依靠重算体积图恢复已省略的距离。旧全域 SDF 无此标记，仍然兼容。

在仓库根目录、Conda `warplab` 环境中启动，无需 SDF CLI 参数：

```powershell
conda run -n warplab --no-capture-output python -m demos.fluid.PBF.render_opengl --device cuda:0
```

仅验证读取、不启动窗口或导出 USD：

```powershell
conda run -n warplab --no-capture-output python -m demos.fluid.PBF.PBF --device cuda:0 --stage-path None --num-frames 0
```

直接脚本运行也支持。仿真入口不接受 `--boundary-obj`、`--sdf-voxel-size`、`--sdf-force-rebuild`，新资产由独立的 sim_Ready 管线准备。要换文件，修改 `boundary_model.py` 的 `SDF_FILENAME`；`--help` 和 `--export-config` 仍直接退出，不读 SDF 或初始化设备。

`PBF(stage_path="example_pbf.usd", verbose=False, config=None)` 和 `create_pbf_simulation(verbose=False, config=None)` 无需接收资产参数。每次创建实例读取所选 NVDB 并分配 GPU Volume；缓存缺失时额外预计算、保存和重读。**每轮压力迭代使用 SDF 投影粒子位置；SDF 与 PBF 均为 Y-up，直接使用世界坐标，不再有运行时 Z 偏移。** 原箱体位置限制及碰撞速度反弹未启用；初始化仍为长方体点阵，未按 SDF 筛选。OpenGL 入口单独渲染无墙景观；完整 SDF 容器的围墙及顶盖不显示。

默认资产体素间距为 0.0075 米，墙厚 0.25 米、坑底最薄处 0.20 米、内顶面 Y=4.5 米。空腔坐标极值为 `(0.075,0,0.075)` 至 `(3.025,4.5,1.525)`，内部仍有台阶和模型。新烘焙保留全部固体内部负距离与正侧保留带，远场使用正背景并省略 tile；体积图复用相同 tile 和坐标变换、背景为零。两种背景值不同是预期行为。缓存首次写回有 CPU 打包开销，加载时也有临时缓冲。NanoVDB 边界加载需要 CUDA；直接通过 Python API 创建实例也会加载，因此同样需要 CUDA 和有效的 SDF 文件。保存或校验失败会报错，并保留替换前的文件；不支持多 segment 或超过两个 grid 的输入。

选择其他已烘焙文件时，在 `boundary_model.py` 中设置 `SDF_FILENAME = "other.nvdb"`；OpenGL 将读取 `Assets/other.obj`。

原 `Assets/Boundary` 与 `Assets/Container` 完整保留为归档，不参与加载，也不作为缺失文件的回退路径。OpenGL 按 NVDB 文件名选择同名 OBJ，USD 入口只要求 NVDB。运行期仅调用 SimReady 的 NVDB 写入工具，不启动几何资产生成管线；生成程序不导入 PBF。

## 独立资产准备

从仓库根目录运行，生成、检查、烘焙并发布到 `PBF/Assets`：

```powershell
python -m demos.sim_Ready --obj bunny_flat_bottom.obj --name stairs_scene --device cuda:0
python -m demos.sim_Ready --obj spot_triangulated_good.obj --name stairs_spot --device cuda:0
python -m demos.fluid.PBF.render_opengl --device cuda:0
```

上例生成 Spot 后，若要在 PBF 使用它，先将 `boundary_model.py` 的 `SDF_FILENAME` 改为 `"stairs_spot.nvdb"`。

裸 OBJ 文件名从 `sim_Ready/Assets` 读取；新管线每次重新计算，不使用旧缓存。默认体素间距 `0.0075 m`、每侧 padding 三个体素，外侧包围盒沿用原版范围，不随支撑半径扩大；实体为负、空腔和外部为正。正侧有效带及体积图正确性保证针对封闭容器的空腔使用区域，外表面邻近网格边缘处不保证与扩展全域参考一致。输出同名的无墙景观 OBJ 与封闭容器 NVDB，不保存容器 OBJ。完整说明见 [sim_Ready](../../sim_Ready/README.md)。Boundary 的旧生成和烘焙工具仍完整保留，但属于归档流程，不用于当前资源发布。

## 测试

PBF 测试位于仓库根目录 `tests/fluid/pbf/`，新资产管线测试位于 `tests/sim_ready/`；旧烘焙测试仍位于 `tests/fluid/boundary/`，完整命令参见 [测试说明](../../../tests/README.md)。在仓库根目录运行：

```powershell
conda run -n warplab --no-capture-output python -m unittest discover -s tests/fluid/pbf -t . -v
```
