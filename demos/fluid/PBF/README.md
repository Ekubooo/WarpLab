## PBFConfig
a set of good para for the water effect

| para | value | note |
| --- | --- | --- |
| `particle_radius` | `0.0125` | 粒子半径，m |
| `rest_density` | `1000.0` | 静止密度，kg/m³ |
| `frame_dt` | `1.0 / 90.0` | 每个 step 的时间，s |
| `substeps` | `3` | 每个 step 的子步数 |
| `pressure_iterations` | `3` | 每个子步的压力迭代次数 |
| `lambda_regularization` | `2.0` | 正则化系数，ε = α / h² |
| `gravity` | `(0.0, -9.81, 0.0)` | 初始重力加速度，m/s² |
| `viscosity` | `0.25` | XSPH 速度修正系数 |
| `vorticity_confinement` | `0.5` | 涡量补偿系数 |
| `max_speed` | `7.5` | 速度上限，m/s |
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
| `--sdf-file NAME.nvdb` | `stairs_scene.nvdb` | 仅文件名，从 `PBF/Assets` 根目录读取；OpenGL 同时选择同名 OBJ |
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
| `--sdf-file NAME.nvdb` | `stairs_scene.nvdb` | 仅文件名，从 `PBF/Assets` 根目录读取；OpenGL 同时选择同名 OBJ |
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

SDF 的目录、默认文件名、CLI 文件名校验和 GPU 读取全部由 `boundary_model.py` 管理，不随工作目录变化：

```python
ASSET_DIR = Path(__file__).resolve().parent / "Assets"
DEFAULT_SDF_FILENAME = "stairs_scene.nvdb"
```

两个入口均支持 `--sdf-file NAME.nvdb`，只接受带 `.nvdb` 扩展名的裸文件名，不接受绝对路径、目录、盘符或路径穿越。该选项与 `--config` 可同时使用，不写入物理参数 JSON。选定 CUDA 设备后创建仿真实例，由 `Example.__init__()` 调用 `load_static_sdf(sdf_filename, device=...)`，读取已有的 float32 NanoVDB。返回的完整 `wp.Volume` 由 `self.static_sdf` 持有，随仿真实例存活；`reset()` 不重新加载。入口不再单独加载。SDF 加载不读取 OBJ、不自动烘焙，也不比较源模型或烘焙参数；文件缺失、损坏或类型错误会报告完整路径。

在仓库根目录、Conda `warplab` 环境中启动，无需 SDF CLI 参数：

```powershell
conda run -n warplab --no-capture-output python -m demos.fluid.PBF.render_opengl --device cuda:0
```

仅验证读取、不启动窗口或导出 USD：

```powershell
conda run -n warplab --no-capture-output python -m demos.fluid.PBF.PBF --device cuda:0 --stage-path None --num-frames 0
```

直接脚本运行也支持。仿真入口不接受 `--boundary-obj`、`--sdf-voxel-size`、`--sdf-force-rebuild`，新资产由独立的 sim_Ready 管线准备。要换文件，通过 `--sdf-file` 指定；`--help` 和 `--export-config` 仍直接退出，不读 SDF 或初始化设备。

`Example(stage_path="example_pbf.usd", verbose=False, config=None, *, sdf_filename=None)` 和 `create_pbf_simulation(verbose=False, config=None, *, sdf_filename=None)` 仅增加一个可选关键字参数，不指定则加载默认文件。每次创建实例都会读取所选 SDF 并分配 GPU Volume。**每轮压力迭代使用 SDF 投影粒子位置；SDF 与 PBF 均为 Y-up，直接使用世界坐标，不再有运行时 Z 偏移。** 原箱体位置限制及碰撞速度反弹未启用；初始化仍为长方体点阵，未按 SDF 筛选。OpenGL 入口单独渲染无墙景观；完整 SDF 容器的围墙及顶盖不显示。

当前文件的体素间距为 0.0075 米，墙厚 0.25 米、坑底最薄处 0.20 米、内顶面 Y=4.5 米。空腔坐标极值为 `(0.075,0,0.075)` 至 `(3.025,4.5,1.525)`，内部仍有台阶和模型。全覆盖体素场的显存和文件大小随高度增长，加载时另有临时缓冲。NanoVDB 读取需要 CUDA；直接通过 Python API 创建实例也会加载，因此当前创建仿真实例同样需要 CUDA 和有效的 SDF 文件。

选择其他已烘焙文件：

```powershell
python -m demos.fluid.PBF.render_opengl --device cuda:0 --sdf-file other.nvdb
```

原 `Assets/Boundary` 与 `Assets/Container` 完整保留为归档，不参与加载，也不作为缺失文件的回退路径。OpenGL 按 NVDB 文件名选择同名 OBJ，USD 入口只要求 NVDB。PBF 不导入资产生成程序，生成程序也不导入 PBF。

## 独立资产准备

从仓库根目录运行，生成、检查、烘焙并发布到 `PBF/Assets`：

```powershell
python -m demos.sim_Ready --obj bunny_flat_bottom.obj --name stairs_scene --device cuda:0
python -m demos.sim_Ready --obj spot_triangulated_good.obj --name stairs_spot --device cuda:0
python -m demos.fluid.PBF.render_opengl --device cuda:0 --sdf-file stairs_spot.nvdb
```

裸 OBJ 文件名从 `sim_Ready/Assets` 读取；新管线每次重新计算，不使用旧缓存。默认体素间距 `0.0075 m`、padding 三个体素，实体为负、空腔和外部为正。输出同名的无墙景观 OBJ 与封闭容器 NVDB，不保存容器 OBJ。完整说明见 [sim_Ready](../../sim_Ready/README.md)。Boundary 的旧生成和烘焙工具仍完整保留，但属于归档流程，不用于当前资源发布。

## 测试

PBF 测试位于仓库根目录 `tests/fluid/pbf/`，新资产管线测试位于 `tests/sim_ready/`；旧烘焙测试仍位于 `tests/fluid/boundary/`，完整命令参见 [测试说明](../../../tests/README.md)。在仓库根目录运行：

```powershell
conda run -n warplab --no-capture-output python -m unittest discover -s tests/fluid/pbf -t . -v
```
