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

启动时会读取 `boundary/Assets/output/stairs_scene_bool_simplified.obj`，绘制底座、台阶和兔子，围墙与顶盖保持不可见。该景观由 `scene_container_creator.py` 生成，采用与 SDF 相同的 Y-up 世界坐标，无需位移或缩放。路径根据脚本位置确定，不依赖工作目录；缺失或无法读取时会报告完整路径。更新边界时应同步重新生成景观文件。

景观仅在启动时加载、注册一次，使用 Warp 默认网格着色器、灰色 `(0.55, 0.55, 0.55)` 和 `smooth_shading=False`；暂停或重置不会重新加载。粒子仍使用原有按速度着色的 billboard。`--help` 和 `--export-config` 不加载景观。

| para | value | note |
| --- | --- | --- |
| `-h / --help` | — | 显示帮助并退出 |
| `--device` | `None` | Warp 设备，如 `cuda:0`、`cpu` |
| `--sdf-file NAME.nvdb` | `stairs_container_bool_simplified.nvdb` | 仅文件名，从 `PBF/Assets/Container` 读取离线 SDF |
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
| `--sdf-file NAME.nvdb` | `stairs_container_bool_simplified.nvdb` | 仅文件名，从 `PBF/Assets/Container` 读取离线 SDF |
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
CONTAINER_DIR = ASSET_DIR / "Container"
DEFAULT_SDF_FILENAME = "stairs_container_bool_simplified.nvdb"
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

直接脚本运行也支持。仿真入口不接受 `--boundary-obj`、`--sdf-voxel-size`、`--sdf-force-rebuild`，这些参数由 boundary 的独立烘焙命令提供。要换文件，通过 `--sdf-file` 指定；`--help` 和 `--export-config` 仍直接退出，不读 SDF 或初始化设备。

`Example(stage_path="example_pbf.usd", verbose=False, config=None, *, sdf_filename=None)` 和 `create_pbf_simulation(verbose=False, config=None, *, sdf_filename=None)` 仅增加一个可选关键字参数，不指定则加载默认文件。每次创建实例都会读取所选 SDF 并分配 GPU Volume。**每轮压力迭代使用 SDF 投影粒子位置；SDF 与 PBF 均为 Y-up，直接使用世界坐标，不再有运行时 Z 偏移。** 原箱体位置限制及碰撞速度反弹未启用；初始化仍为长方体点阵，未按 SDF 筛选。OpenGL 入口单独渲染无墙景观；完整 SDF 容器的围墙及顶盖不显示。

当前文件的体素间距为 0.0075 米，墙厚 0.25 米、坑底最薄处 0.20 米、内顶面 Y=4.5 米。空腔坐标极值为 `(0.075,0,0.075)` 至 `(3.025,4.5,1.525)`，内部仍有台阶和模型。全覆盖体素场的显存和文件大小随高度增长，加载时另有临时缓冲。NanoVDB 读取需要 CUDA；直接通过 Python API 创建实例也会加载，因此当前创建仿真实例同样需要 CUDA 和有效的 SDF 文件。

选择其他已烘焙文件：

```powershell
python -m demos.fluid.PBF.render_opengl --device cuda:0 --sdf-file other.nvdb
```

原 `Assets/Boundary` 中的文件暂留作备份，不参与加载，也不会作为缺失文件的回退路径。切换 NVDB 不自动切换无墙景观 OBJ；景观仍由 OpenGL 入口按原固定路径读取。

## 独立烘焙命令

在仓库根目录、Conda `warplab` 环境中运行，成功后输出结果路径并退出，不创建仿真：

```powershell
conda run -n warplab --no-capture-output python -m demos.fluid.boundary.bake_sdf --boundary-obj stairs_container_bool_simplified.obj
```

直接脚本运行也支持：

```powershell
conda run -n warplab --no-capture-output python demos/fluid/boundary/bake_sdf.py --boundary-obj stairs_container_bool_simplified.obj
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--boundary-obj PATH` | 必填 | 容器 OBJ；裸文件名从 `boundary/Assets/output` 解析 |
| `--sdf-voxel-size` | `0.0075` | 烘焙体素间距，米，仅在烘焙时使用 |
| `--sdf-force-rebuild` | 关闭 | 忽略同名缓存，重新检查与烘焙，成功后替换 |
| `--device` | `cuda:0` | 仅支持 CUDA |

包含目录的相对路径（包括 `./room.obj`）按工作目录解析，绝对路径直接使用。输出固定为 `PBF/Assets/Container/<OBJ文件名去扩展名>.nvdb`，默认目录根据脚本位置确定。同名缓存存在则直接读取，跳过 OBJ 检查与参数比较；损坏缓存明确报错，不静默重建。修改 OBJ 或体素间距后，显式添加 `--sdf-force-rebuild`：

```powershell
conda run -n warplab --no-capture-output python -m demos.fluid.boundary.bake_sdf --boundary-obj stairs_container_bool_simplified.obj --sdf-force-rebuild
```

烘焙工具不导入 PBF、仿真或渲染模块。默认同名文件烘焙后可直接启动仿真；其他文件名通过 `--sdf-file NAME.nvdb` 选择。烘焙失败不会覆盖已有有效缓存。

## 离线烘焙与缓存 API

网格检查和烘焙继续由 `boundary/mesh_validation.py`、`boundary/sdf_baker.py` 提供，独立 CLI 复用该 API，与启动加载分开。也可通过 Python API 更新文件，再启动 PBF：

```python
from pathlib import Path
from demos.fluid.boundary.sdf_baker import SDFOptions, load_or_bake_sdf

# 示例路径相对于仓库根目录；OBJ 来自生成器输出。
assets = Path("demos/fluid/PBF/Assets")
result = load_or_bake_sdf(
    Path("demos/fluid/boundary/Assets/output/stairs_container_bool_simplified.obj"),
    cache_dir=assets / "Container",
    options=SDFOptions(voxel_size=0.0075, force_rebuild=True),
    device="cuda:0",
)
```

离线 API 保留原缓存逻辑：同名 `.nvdb` 存在且未强制重建时直接读取，不解析 OBJ、不核对参数/哈希；缺失或强制重建才检查、烘焙并原子保存。烘焙保持 OBJ 坐标、float32 全覆盖场、每侧 padding 3 体素；实体负、空腔和外部正。网格检查不自动修补、翻转或认证全局自交；瘦长但非退化三角形仅告警。

## 测试

PBF 测试位于仓库根目录 `tests/fluid/pbf/`，独立烘焙 CLI 测试位于 `tests/fluid/boundary/test_bake_sdf.py`，完整命令参见 [测试说明](../../../tests/README.md)。在仓库根目录运行：

```powershell
conda run -n warplab --no-capture-output python -m unittest discover -s tests/fluid/pbf -t . -v
```
