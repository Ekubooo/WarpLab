# Warp PBF 流固双向耦合

独立的三维 Warp DSL 演示：PBF 流体、Akinci2012 粒子边界、可平移和旋转的网格刚体、刚体碰撞，以及 billboard 流体和带阴影的实体渲染。

## 运行

在仓库根目录使用现有环境（Python 3.10、`warp-lang==1.15.0`、NumPy、Pyglet）：

```powershell
# 默认溃坝：球、圆柱、环面
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/render_opengl.py

# 静水中的轻球上浮、重球下沉
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/render_opengl.py --scene floating-equilibrium

# 10 秒无窗口验证，保存 JSON 诊断和 NPZ 最终状态
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/simulation.py --seconds 10 --output outputs/pbf2way/run

# CPU 小分辨率运行；渲染前端本身需要 CUDA/OpenGL
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/render_opengl.py --headless --device cpu --particle-radius 0.05 --seconds 0.1

# 隐藏窗口截图；--warmup 指定截图前的仿真秒数
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/render_opengl.py --hidden --num-frames 2 --warmup 1 --screenshot outputs/pbf2way/frame.png
```

默认自动选择 CUDA，否则无窗口仿真使用 CPU。首次启动会编译 kernels，缓存写入仓库 `.warp_cache`。所有网格随代码提供，运行无需联网。

交互：`Space` 暂停/继续，`R` 重置并暂停，`G` 反转重力，`Q/E` 绕 Z 轴旋转重力 10°；鼠标和 `W/A/S/D` 控制相机。重置恢复初始物理状态及重力，保留相机。默认播放速度为真实时间的 `0.5`，可用 `--playback-speed` 调整；每帧最多处理 16 个物理子步。`--num-frames` 限制帧数，`--one-way` 关闭流体对刚体的反作用力。

渲染保留粒子方形 billboard 和速度配色。刚体使用实体三角网格、方向光、2048² 深度阴影贴图及 3×3 PCF。地板可见，容器侧壁和顶面只参与物理，以免遮挡内部。阴影覆盖刚体和地板，不覆盖流体；完整的 8 米高容器可通过相机移动查看。

## 官方参考与改编

主参考为 [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/tree/f3f677140761db7637b5443beb54f19f1f835ed4)，固定提交 `f3f677140761db7637b5443beb54f19f1f835ed4`。仓库另一个参考 [PositionBasedDynamics](https://github.com/InteractiveComputerGraphics/PositionBasedDynamics) 的独立 FluidDemo 提供静态边界 PBF；SPlisHSPlasH 的 `TimeStepPBF` 则已有压力反作用力，并通过 PBD wrapper 驱动动态刚体，因此选择后者作为耦合基线。

| 本实现 | 官方来源 |
| --- | --- |
| Poly6、Spiky 核 | `SPlisHSPlasH/SPHKernels.h` |
| 密度、lambda、修正、压力受力、停止判据 | `SPlisHSPlasH/PBF/TimeStepPBF.cpp` |
| Akinci 伪体积与静态/动态邻域分组 | `BoundaryModel_Akinci2012.cpp`、`Simulation::updateBoundaryVolume` |
| `F`、`(x_boundary−COM)×F`、边界点速度 | `SPlisHSPlasH/BoundaryModel.h` |
| Standard viscosity 及反作用力 | `SPlisHSPlasH/Viscosity/Viscosity_Standard.cpp` 的标量 Akinci 路径 |
| CFL | `Simulation::updateTimeStepSizeCFL` |
| 规则流体块 | `Simulator/SimulatorBase::createFluidBlocks` |
| 刚体受力/速度更新的接口语义 | `Simulator/PositionBasedDynamicsWrapper/PBDRigidBody.h` |

当前所查官方 53 个场景 JSON 都没有默认选择 PBF，但 PBF 求解器本身支持双向耦合。`scenes/DamBreakWithObjects.source.json` 原样保留官方场景；默认演示保留其几何、缩放、位置、朝向、密度、摩擦及恢复系数，将原 DFSPH + 体积边界改为 PBF + 粒子边界。可执行场景精简为几何/刚体/流体块字段；求解器参数统一由 `PBF2WayCouplingConfig` 设置，不静默读取原场景中的 DFSPH、软体或关节参数。

四份官方 OBJ 原样保存，SHA-256 和来源在 `assets/sources.json`，官方 MIT 许可证在 `assets/LICENSE.SPlisHSPlasH`。参考公式的 Warp 转写也保留该来源与许可。无需安装或编译 SPlisHSPlasH/PBD。

## 数值模型

- 默认粒子半径 `r=0.025`、支撑半径 `h=4r=0.1`、静止密度 `1000`、粒子体积 `0.8(2r)³`、粒子质量 `0.1`。
- 密度使用 Poly6，约束梯度使用 Spiky；压缩约束为 `max(ρ/ρ₀−1,0)`，lambda 正则项 `1e-6`。边界梯度累加到中心粒子的梯度，但不单独计入邻居梯度平方和。这一点区别于现有 MMPBF。
- 压力至少迭代 2 次，最多 100 次；停止条件是平均压缩误差不超过 `0.01%`。`density_error_percent` 是最后一轮修正前的停止判据值，`densities` 是修正后的归一化密度。
- 边界引起的流体位移为 `Δx`，刚体反作用力为 `−mΔx/dt²`，力矩使用该边界样本相对质心的力臂。每一轮压力修正都累积贡献。开启边界粘度时，同样累积 `−m a_boundary`。
- 边界点世界坐标为 `COM+R x_local`，速度为 `v+ω×(R x_local)`。静态边界共同计算伪体积；运动刚体各自在自身样本内计算，之后不随位姿改变。
- 子步次序：清空受力 → 流体重力预测 → 缓存一次邻域 → PBF 迭代/反作用力 → 一阶速度重建 → 重算密度及 Standard viscosity → 刚体积分 → 接触速度约束 → 更新边界 → 计算下一步 CFL。
- 当前子步固定使用同一 `dt` 完成流体、刚体和受力积分；下一步 CFL 使用流体速度及粘度加速度、边界点速度。初始步长 `0.001`，上下限 `[0.0001,0.005]`，系数 `0.5`。Standard viscosity 默认 `0.01`，边界粘度默认 `0`。
- 不加入人工浮力、粒子限速、流体位置夹取、涡量约束或人工压力。邻域/接触容量不足会报错；连续 10 步达到压力上限会警告，持续时每 100 步再报告。

## 网格、接口与实现范围

`PBF2WayCoupling.py` 放置全部仿真 kernels、状态类型、配置和 `Example`；`step()` 显式展示调用顺序。`coupling_functions.py` 放数学函数，`coupling_initialization.py` 放资产加载、场景采样及质量惯量计算；`simulation.py` 提供工厂与无窗口入口。

```python
from demos.fluid.PBF2WayCoupling.simulation import (
    PBF2WayCouplingConfig, create_pbf2way_simulation,
)

config = PBF2WayCouplingConfig(
    scene="floating-equilibrium",
    boundary_viscosity=0.02,
)
simulation = create_pbf2way_simulation(config=config, device="cuda:0")
simulation.step()  # 一个完整自适应物理子步
```

公开数据：`positions`、`velocities`、`densities`、`rigid.position/rotation/velocity/omega/force/torque`、`boundary.position/velocity/volume/body`、`sim_time`、`current_dt`、`last_dt`、`iterations`、`density_error_percent`。数组驻留所选 Warp 设备；`.numpy()` 只用于初始化、显式诊断、测试或每帧少量刚体位姿传输。

场景支持 `RigidBodies` 和 `FluidBlocks`。刚体支持 OBJ 路径、正缩放、平移、轴角旋转、密度、初速度/角速度、颜色、静态/动态状态、摩擦和恢复系数；流体块支持 `start/end`、缩放、平移、`denseMode=0` 和 `initialVelocity`。容器必须是一个静态轴对齐盒体。OBJ 需封闭、顶点焊接、绕序一致且朝外；不支持软体、关节、发射器、任意官方场景格式或其他流体算法。

### 相对官方路径的明确差异

1. 网格体积积分得到质量、质心和完整惯量；本实现不转换到官方 PBD 的主惯量坐标系，而是在质心坐标系保存完整矩阵。初始化只剔除与刚体重叠的流体粒子，运行中不删粒子。
2. 表面使用确定性规则三角形行采样，间距不超过 `1.5r`，合并重复点；不复现官方 Poisson-disk 随机样本和 Partio 缓存。边界伪体积公式保持不变。
3. 刚体碰撞采用局部 AABB 粗筛、Warp mesh BVH 有符号距离、双向表面样本接触；容器用内向盒体距离。没有移植 Discregrid 或原场景的球/圆柱/环面解析碰撞代理。
4. 接触使用 5 轮设备端顺序冲量：恢复系数取两者最小值，摩擦取几何平均，穿透偏置系数 `0.2`，容差 `0.06`。正间隙采用预测接触，已穿透接触允许小量速度级纠正。适用于少量刚体，没有连续碰撞检测，不保证任意高速、薄壁、深度初始穿透的处理。
5. 刚体角速度用包含陀螺项的 Euler 方程，四元数显式积分后归一化。GPU 原子累积与接触生成次序不是逐位确定的，长时轨迹允许分歧。
6. 粒子边界的有限支撑宽度会改变有效排水体积。`floating-equilibrium` 用半径 `0.2` 的球、密度 `200/3000`、初始高度 `0.5` 和 1 米高水块做明确的上浮/下沉对照；它不是近中性浮力的精确密度阈值标定。近中性物体需分辨率收敛研究。

## 测试与实际结果

```powershell
# 14 项数值测试；CPU 公式检查，有 CUDA 时增加短程对照
.venv/Scripts/python.exe -m unittest demos.fluid.PBF2WayCoupling.test_coupling -v

# 独立隐藏窗口 OpenGL 测试，需要 NVIDIA CUDA/OpenGL
.venv/Scripts/python.exe -m unittest demos.fluid.PBF2WayCoupling.test_rendering -v

# 三个场景各运行至少 10 秒，输出逐秒诊断、最终状态及汇总
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.validate
```

2026-09-15，本机 RTX 5070 12 GiB、Warp 1.15.0 实测：14 项数值测试及 1 项 OpenGL 测试通过。公式对照使用独立 NumPy 计算；尚未编译官方 C++ 程序做整体轨迹对照。

| 场景 | 流体/边界粒子 | 仿真时间 | 子步数 | 平均子步耗时¹ | 全程最大接触穿透 | 最终密度误差 |
| --- | --- | --- | --- | --- | --- | --- |
| 默认溃坝 | 24,389 / 64,176 | 10.001 s | 4,869 | 5.18 ms | 0.00378 m | 0.009991% |
| 静水双向 | 31,747 / 63,336 | 10.005 s | 2,991 | 11.56 ms | 0.00138 m | 0.009845% |
| 静水单向 | 31,747 / 63,336 | 10.004 s | 3,113 | 10.94 ms | 0.00123 m | 0.009989% |

¹ 含 Python 调度、同步和诊断，不含初始化、编译及渲染；不是帧率保证。

三组均保持粒子数量不变；逐子步累计的流体容器越界为零，最大四元数长度误差约 `1.2e-7`，结束时无持续压力上限警告。默认场景最终球、圆柱、环面质心高度约 `0.708/0.602/0.696 m`，都有明显平移和转动。静水轻球从 `0.5 m` 上浮至 `0.857 m`，重球落至 `0.198 m`；关闭反作用力后，轻球也落至 `0.198 m`。

OpenGL 检查包含阴影开/关的像素差、billboard 与实体共用深度缓冲、GL 错误、暂停/重置请求和重力控制回调；没有依赖人工点击。截图和详细运行日志写入忽略的 `outputs/pbf2way/`，重新运行验证脚本可生成。
