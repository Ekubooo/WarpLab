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

## 静态容器 SDF：烘焙、缓存与读取

两个入口均支持以下参数，可与 `--config` 一起使用；它们不写入 PBF 物理参数 JSON。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--boundary-obj PATH` | 无 | 指定封闭容器 OBJ；不指定时按原流程运行。 |
| `--sdf-voxel-size METERS` | `0.0075` | 首次烘焙的体素间距，米。 |
| `--sdf-force-rebuild` | 关闭 | 重新检查 OBJ、烘焙并替换同名缓存。 |

将 `boundary/scene_container_creator.py` 生成的容器 OBJ 放入本目录的 `Assets/Container/`，然后在仓库根目录运行：

```powershell
conda run -n warplab --no-capture-output python -m demos.fluid.PBF.render_opengl --device cuda:0 --boundary-obj stairs_container_bool_simplified.obj
```

仅验证烘焙与读入、不启动窗口或导出 USD：

```powershell
conda run -n warplab --no-capture-output python -m demos.fluid.PBF.PBF --device cuda:0 --stage-path None --num-frames 0 --boundary-obj stairs_container_bool_simplified.obj
```

也支持直接执行 `python demos/fluid/PBF/PBF.py ...` 或 `python demos/fluid/PBF/render_opengl.py ...`。裸文件名从 `Assets/Container/` 读取；含目录的相对路径相对于当前工作目录，绝对路径保持原意。默认目录依据代码位置确定。

缓存统一位于 `Assets/Boundary/<OBJ文件名去扩展名>.nvdb`。**先检查缓存，命中后直接加载，不读取或检查源 OBJ，也不核对源内容和体素参数。** 修改网格或间距后须显式加 `--sdf-force-rebuild`；不同路径的同名 OBJ 共用同一缓存名。坏缓存会报错，不自动覆盖。缓存缺失时才检查、烘焙并原子保存；首次烘焙的 GPU Volume 直接保留使用。无需额外元数据文件。

本轮要求 CUDA。SDF 为 float32 全覆盖 NanoVDB，每侧预留 3 个体素；保持 OBJ 的原始坐标和单位，不做缩放或向上轴转换。实体内为负，流体空腔及容器外为正。当前容器在 0.0075 米间距下，Volume 本体约 **110.81 MiB 显存**，未压缩缓存约 **116.19 MB**；加载期间网格缓冲临时约 **221.62 MiB**，不含 PBF、CUDA 和烘焙临时资源。

代码职责：`boundary/mesh_validation.py` 检查输入，`boundary/sdf_baker.py` 负责 Warp DSL 烘焙与缓存；`boundary_model.py` 管理 PBF 路径、参数及 Volume 生命周期。`Example.static_boundary` 持有 Volume 与设备数据，`sample_boundary_sdf(data, position)` 可在 Warp kernel 中采样距离及世界坐标梯度。超出已分配网格时返回 NanoVDB 背景值，不代表真实表面距离。

**本轮仅加载 GPU SDF，没有接入碰撞、位置投影或场景渲染；现有箱体边界继续生效。** SDF 与粒子区域的尺寸、位置对齐仍需后续配置。

输入简单检查要求：一个闭合正向外壳、一个闭合反向空腔、总体积为正；拒绝非有限坐标、非法索引、重复面、退化面、非流形及绕序错误，也检查 float32 转换后的塌缩。仅在副本中合并完全相同的坐标，不自动修补或翻转内腔。细长但非退化的三角形仅告警；此次不认证全局自交、壳包含关系或薄特征能否被网格解析。

## 测试

测试脚本统一位于仓库根目录 `tests/fluid/pbf/`，完整命令参见 [测试说明](../../../tests/README.md)。在仓库根目录运行：

```powershell
conda run -n warplab --no-capture-output python -m unittest discover -s tests/fluid/pbf -t . -v
```
