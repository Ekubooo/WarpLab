# PBF_Origin 阶段基线 demo

后续开发继续在 `PBF` 目录进行，本目录保留当前阶段的求解器、配置及资产快照。运行时使用本目录的代码与资源，不导入开发版 `demos.fluid.PBF`。共享依赖为 `demos/fluid/common` 的 billboard 渲染器，以及 Warp、NumPy、Trimesh、Pyglet；它是仓库内独立 demo，并非可脱离仓库的单文件程序。

从仓库根目录启动：

```powershell
python -m demos.fluid.PBF_Origin.render_opengl --device cuda:0
python -m demos.fluid.PBF_Origin.PBF --device cuda:0 --stage-path None --num-frames 1
```

直接脚本运行也支持：`python demos/fluid/PBF_Origin/render_opengl.py --device cuda:0`。模块入口从其他工作目录运行时，需要将仓库根目录加入 `PYTHONPATH`；资产和配置导出目录本身不依赖工作目录。

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

启动时从 `PBF_Origin/Assets` 根目录读取所选 NVDB 的同名景观 OBJ：默认 `stairs_scene.nvdb` 对应 `stairs_scene.obj`，绘制底座、台阶和模型，围墙与顶盖保持不可见。资产由独立的 `sim_Ready` 管线生成，统一采用 Y-up 世界坐标。路径根据脚本位置确定，不依赖工作目录；缺失或损坏时报告完整路径。

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
| `--export-config [NAME]` | `default_para.json` | 指定选项时导出到 PBF_Origin 的 `config/` 后退出；NAME 可省略，仅接受文件名 |
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
| `--export-config [NAME]` | `default_para.json` | 指定选项时导出到 PBF_Origin 的 `config/` 后退出；NAME 可省略，仅接受文件名 |
| `--clamp-negative-pressure / --no-clamp-negative-pressure` | `True` | 开启／关闭负压钳制 |

## 命令目录（仓库根目录）

| operation | cmd |
| --- | --- |
| 默认启动 | `conda run -n warplab --no-capture-output python -m demos.fluid.PBF_Origin.render_opengl --device cuda:0` |
| 导入 `default_para.json` | `conda run -n warplab --no-capture-output python -m demos.fluid.PBF_Origin.render_opengl --device cuda:0 --config demos/fluid/PBF_Origin/config/default_para.json` |
| 默认导出 `default_para.json` | `conda run -n warplab --no-capture-output python -m demos.fluid.PBF_Origin.render_opengl --export-config` |
| 指定导出 `water.json` | `conda run -n warplab --no-capture-output python -m demos.fluid.PBF_Origin.render_opengl --export-config water.json` |
| 导入 `water.json` | `conda run -n warplab --no-capture-output python -m demos.fluid.PBF_Origin.render_opengl --device cuda:0 --config demos/fluid/PBF_Origin/config/water.json` |

## 启动时读取静态 SDF

资产选择由 `PBF.py` 的模块级常量控制；固定目录、文件名校验、路径解析和 GPU 读取由 `boundary_model.py` 管理，不随工作目录变化：

```python
# PBF.py：修改这里即可切换碰撞与配对景观
SDF_FILENAME = "stairs_scene.nvdb"
```

两个入口均不提供 SDF 文件选择 CLI。选定 CUDA 设备后创建仿真实例，由 `Example.__init__()` 调用 `boundary_model.load_static_sdf(SDF_FILENAME, device=...)`，读取已有的 float32 NanoVDB。加载逻辑仍在构造函数中，文件名不进入构造或创建函数的参数列表。OpenGL 从同一 `PBF.SDF_FILENAME` 解析配对 OBJ；USD 只读取 NVDB。物理 JSON 不包含资产选择。返回的完整 `wp.Volume` 由 `self.static_sdf` 持有；重置不会重新加载。文件缺失、损坏或类型错误会报告完整路径，不自动烘焙。

在仓库根目录、Conda `warplab` 环境中启动，无需 SDF CLI 参数：

```powershell
conda run -n warplab --no-capture-output python -m demos.fluid.PBF_Origin.render_opengl --device cuda:0
```

仅验证读取、不启动窗口或导出 USD：

```powershell
conda run -n warplab --no-capture-output python -m demos.fluid.PBF_Origin.PBF --device cuda:0 --stage-path None --num-frames 0
```

直接脚本运行也支持。仿真入口不接受 `--boundary-obj`、`--sdf-voxel-size`、`--sdf-force-rebuild`，新资产由独立的 sim_Ready 管线准备。要换文件，修改 `PBF.py` 的 `SDF_FILENAME`；`--help` 和 `--export-config` 仍直接退出，不读 SDF 或初始化设备。

`Example(stage_path="example_pbf.usd", verbose=False, config=None)` 和 `create_pbf_simulation(verbose=False, config=None)` 无需接收资产参数。每次创建实例都会读取所选 SDF 并分配 GPU Volume。**每轮压力迭代使用 SDF 投影粒子位置；SDF 与 PBF 均为 Y-up，直接使用世界坐标，不再有运行时 Z 偏移。** 原箱体位置限制及碰撞速度反弹未启用；初始化仍为长方体点阵，未按 SDF 筛选。OpenGL 入口单独渲染无墙景观；完整 SDF 容器的围墙及顶盖不显示。

当前文件的体素间距为 0.0075 米，墙厚 0.25 米、坑底最薄处 0.20 米、内顶面 Y=4.5 米。空腔坐标极值为 `(0.075,0,0.075)` 至 `(3.025,4.5,1.525)`，内部仍有台阶和模型。全覆盖体素场的显存和文件大小随高度增长，加载时另有临时缓冲。NanoVDB 读取需要 CUDA；直接通过 Python API 创建实例也会加载，因此当前创建仿真实例同样需要 CUDA 和有效的 SDF 文件。

选择其他已烘焙文件时，在 `PBF.py` 中设置 `SDF_FILENAME = "other.nvdb"`；OpenGL 将读取 `Assets/other.obj`。

所有运行资源仅从本 demo 的 `Assets` 根目录读取，不回退到其他目录。OpenGL 按 NVDB 文件名选择同名 OBJ，USD 入口只要求 NVDB。PBF 不导入资产生成程序，生成程序也不导入 PBF。

## 独立资产快照

本目录保留独立的 OBJ/NVDB 副本。`sim_Ready` 默认只发布到开发版 `PBF/Assets`，不会更新这里；这可避免后续重新烘焙覆盖基线 demo 的资源。需要主动更新本 demo 时，先在 `sim_Ready` 生成，再将同名 OBJ/NVDB 成对复制到 `PBF_Origin/Assets`。缺失资源不会回退读取开发版。

OBJ/NVDB 受现有 Git 忽略规则影响，不会随普通提交保存。完整留档时需同时备份本目录的 `Assets`。

## 测试

本 demo 的独立入口与资源测试：

```powershell
python -m unittest discover -s tests/fluid/pbf_origin -t . -v
```

物理算法仍保留当前基线实现；开发版 PBF 的完整测试位于 `tests/fluid/pbf`。
