# Warp PBF 流固双向耦合

独立的三维 Warp DSL 演示：PBF 流体、Akinci2012 粒子边界、可平移和旋转的网格刚体、刚体碰撞，以及 billboard 流体和带阴影的实体渲染。

## 运行

在仓库根目录使用现有环境（本机 Python 3.12.3、`warp-lang==1.15.0`、NumPy、Pyglet）：

```powershell
# 默认溃坝：球、圆柱、环面
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/render_opengl.py

# 静水中的轻球上浮、重球下沉
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/render_opengl.py --scene floating-equilibrium

# 10 秒无窗口运行，正常循环不回读
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/simulation.py --seconds 10

# 主动验证：每秒回读诊断，并导出最终状态
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/simulation.py --seconds 10 --diagnostic-interval 1 --output outputs/pbf2way/run

# CPU 小分辨率运行；渲染前端本身需要 CUDA/OpenGL
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/render_opengl.py --headless --device cpu --particle-radius 0.05 --seconds 0.1

# 隐藏窗口截图；--warmup 指定截图前的仿真秒数
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/render_opengl.py --hidden --num-frames 2 --warmup 1 --screenshot outputs/pbf2way/frame.png
```

默认自动选择 CUDA，否则无窗口仿真使用 CPU。首次启动会编译 kernels，缓存写入仓库 `.warp_cache`。所有网格随代码提供，运行无需联网。

交互：`Space` 暂停/继续，`R` 重置并暂停，`G` 反转重力，`Q/E` 绕 Z 轴旋转重力 10°；鼠标和 `W/A/S/D` 控制相机。重置恢复初始物理状态及重力，保留相机。`--num-frames` 限制帧数，`--one-way` 关闭流体对刚体的反作用力。

每次 `step()` 推进 `1/90 s`，紧接着调用一次 `render()`。暂停时只刷新画面。播放速度仅决定帧间等待：默认 `--playback-speed 0.5` 的目标是 45 帧/秒，设为 `1` 则是 90 帧/秒。算不完时自然变慢，不追赶、不跳步、不跳帧；已关闭 vsync 和 Tab 跳过渲染快捷键。`--warmup` 仅用于显式截图/测试预热，会在创建渲染器前推进指定时间。

固定推进时间在 `PBF2WayCoupling.py` 的 `PBF2WayCouplingConfig.frame_dt` 修改，目前为 `1.0 / 90.0`。子步时间自动计算为 `frame_dt / substeps = 1/270 s`，播放节拍和无窗口步数也由此推导。

压力迭代次数在同一配置类的 `pressure_iterations` 修改，当前默认 **3 次/子步**；也可在构造配置时传入 `PBF2WayCouplingConfig(pressure_iterations=3)`。此前校验中写死的“必须为 5”已删除，现在接受正整数，并在每个子步执行指定次数，不根据误差提前退出。三个子步合计 **9 次压力修正/step**。`contact_iterations=5` 是另一项刚体接触配置，不是流体压力迭代次数。

渲染保留粒子方形 billboard 和速度配色。刚体使用实体三角网格、方向光、2048² 深度阴影贴图及 3×3 PCF。地板可见，容器侧壁和顶面只参与物理，以免遮挡内部。阴影覆盖刚体和地板，不覆盖流体；完整的 8 米高容器可通过相机移动查看。

## 调整流体粒子数量

没有独立的粒子总数配置项。`coupling_initialization.py` 根据场景 JSON 的 `FluidBlocks.start/end/scale` 和粒子半径生成规则网格：每轴数量为 `floor(轴长 / (2r) + 0.5) - 1`，总数是三个轴数量的乘积；多个水块相加，初始化再剔除与固体重叠的粒子。

- 保持水块尺寸、提高分辨率：减小命令行 `--particle-radius`，或设置 `PBF2WayCouplingConfig(particle_radius=...)`。三维中总数大致与半径的三次方成反比；支撑半径、粒子质量和边界采样间距自动随之调整。
- 保持粒子分辨率、增加水量：扩大场景的 `FluidBlocks.start/end` 范围或增加水块，并确保它们位于容器内且互不重叠。
- 不要直接修改 `num_particles`；它是初始化后的实际计数，相关设备数组和缓存都按此分配。

默认溃坝水块边长为 1.5 m。`r=0.025` 时为 `29³=24,389` 个；改为 `r=0.015625` 时为 `47³=103,823` 个，约 10 万。规则立方网格的数量是离散变化的，不会恰好等于 100,000。默认半径保持 0.025，以下命令启用约 10 万粒子：

```powershell
# 交互渲染：约 10 万流体粒子，一次 step 一次 render
.venv/Scripts/python.exe demos/fluid/PBF2WayCoupling/render_opengl.py --device cuda:0 --particle-radius 0.015625 --playback-speed 1

# 无窗口十秒，显式诊断和导出
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.simulation --device cuda:0 --particle-radius 0.015625 --seconds 10 --diagnostic-interval 1 --output outputs/pbf2way/100k/dam-break
```

2026-09-16，RTX 5070 实跑得到 **103,823 个流体粒子 / 163,732 个边界样本**，完成 600 step / 1800 子步 / 10 秒仿真。粒子数量不变，流体内缩边界累计越界为零，无非有限数值和缓存溢出。含逐秒诊断、排除初始化及渲染，平均 **22.74 ms/step**。隐藏窗口渲染和截图通过。最终平均压缩误差为 **0.4858%**；圆柱沉至约 0.092 m，球和环面位于约 0.620/0.656 m，因此不能将更高分辨率视为保持原轨迹或已通过原浮沉验收。固定三子步、五轮压力迭代没有改变，超过帧预算时播放自然变慢。

## 官方参考与改编


主参考为 [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/tree/f3f677140761db7637b5443beb54f19f1f835ed4)，固定提交 `f3f677140761db7637b5443beb54f19f1f835ed4`。仓库另一个参考 [PositionBasedDynamics](https://github.com/InteractiveComputerGraphics/PositionBasedDynamics) 的独立 FluidDemo 提供静态边界 PBF；SPlisHSPlasH 的 `TimeStepPBF` 则已有压力反作用力，并通过 PBD wrapper 驱动动态刚体，因此选择后者作为耦合基线。

| 本实现 | 官方来源 |
| --- | --- |
| Poly6、Spiky 核 | `SPlisHSPlasH/SPHKernels.h` |
| 密度、lambda、修正、压力受力 | `SPlisHSPlasH/PBF/TimeStepPBF.cpp` |
| Akinci 伪体积与静态/动态邻域分组 | `BoundaryModel_Akinci2012.cpp`、`Simulation::updateBoundaryVolume` |
| `F`、`(x_boundary−COM)×F`、边界点速度 | `SPlisHSPlasH/BoundaryModel.h` |
| Standard viscosity 及反作用力 | `SPlisHSPlasH/Viscosity/Viscosity_Standard.cpp` 的标量 Akinci 路径 |
| 规则流体块 | `Simulator/SimulatorBase::createFluidBlocks` |
| 刚体受力/速度更新的接口语义 | `Simulator/PositionBasedDynamicsWrapper/PBDRigidBody.h` |

当前所查官方 53 个场景 JSON 都没有默认选择 PBF，但 PBF 求解器本身支持双向耦合。`scenes/DamBreakWithObjects.source.json` 原样保留官方场景；默认演示保留其几何、缩放、位置、朝向、密度、摩擦及恢复系数，将原 DFSPH + 体积边界改为 PBF + 粒子边界。可执行场景精简为几何/刚体/流体块字段；求解器参数统一由 `PBF2WayCouplingConfig` 设置，不静默读取原场景中的 DFSPH、软体或关节参数。

四份官方 OBJ 原样保存，SHA-256 和来源在 `assets/sources.json`，官方 MIT 许可证在 `assets/LICENSE.SPlisHSPlasH`。参考公式的 Warp 转写也保留该来源与许可。无需安装或编译 SPlisHSPlasH/PBD。

## 数值模型

- 默认粒子半径 `r=0.025`、支撑半径 `h=4r=0.1`、静止密度 `1000`、粒子体积 `0.8(2r)³`、粒子质量 `0.1`。
- 密度使用 Poly6，约束梯度使用 Spiky；压缩约束为 `max(ρ/ρ₀−1,0)`，lambda 正则项 `1e-6`。边界梯度累加到中心粒子的梯度，但不单独计入邻居梯度平方和。这一点区别于现有 MMPBF。
- 每个 `step()` 固定包含 3 个 `dt=1/270 s` 子步，默认每子步 3 轮压力迭代和 5 轮接触迭代。压力轮数可设为任意正整数，运行时按配置固定执行；没有 CFL、误差提前退出或自适应迭代。`densities` 是修正后的归一化密度，显式诊断按其计算平均压缩误差，不要求达到旧的 `0.01%`。
- 边界引起的流体位移为 `Δx`，刚体反作用力为 `−mΔx/dt²`，力矩使用该边界样本相对质心的力臂。每一轮压力修正都累积贡献。开启边界粘度时，同样累积 `−m a_boundary`。
- 边界点世界坐标为 `COM+R x_local`，速度为 `v+ω×(R x_local)`。静态边界共同计算伪体积；运动刚体各自在自身样本内计算，之后不随位姿改变。
- 子步次序：清空受力 → 流体重力预测 → 缓存一次邻域 → PBF 迭代/反作用力 → 一阶速度重建 → 重算密度及 Standard viscosity → 刚体积分 → 接触检测与速度约束 → 更新边界。阶段之间保留设备端溢出和非有限数值检查。
- 流体、反作用力及刚体积分统一使用 `substep_dt=1/270`，不能使用完整步的 `current_dt` 计算子步冲量。Standard viscosity 默认 `0.01`，边界粘度默认 `0`。
- 容器保底：每轮应用 `Δp` 时执行 `x = clamp(x + Δp, container_min + r, container_max - r)`，将粒子中心限制在向内缩一个半径的盒体内。动态刚体仍走原有粒子边界耦合；额外位置修正视为静态容器约束，不修改压力反作用力公式。
- 速度重建和粘度速度更新后均检查墙面：仅将贴墙且向外的速度分量改为 `−wall_damping × v`，默认 `wall_damping=0.8`，配置范围 `[0,1]`。保留切向及向内速度，棱角逐轴处理，已反向的速度不会重复衰减。这里使用当前更新后的速度，并非另行保存碰撞前的入射速度。该措施引入碰壁耗散，不添加全局速度上限。
- 不加入人工浮力、涡量约束或人工压力。设备故障标志保持到重置，检测到邻域/接触溢出或非有限状态后停止后续物理写入，并通过设备 `printf` 输出一次错误；保底钳制保留 NaN/Inf，避免掩盖异常。正常循环不读故障标志；主机提交计数仍继续增长，因此故障后的 `sim_time` 只代表提交的时间，不代表成功推进。显式 `diagnostics(simulation)` 会回读并抛出异常。

## 网格、接口与实现范围

`PBF2WayCoupling.py` 放置全部仿真 kernels、状态类型、配置和 `Example`；`step()` 显式展示调用顺序。`coupling_functions.py` 放数学函数，`coupling_initialization.py` 放资产加载、场景采样及质量惯量计算；`simulation.py` 提供工厂与无窗口入口。

```python
from demos.fluid.PBF2WayCoupling.simulation import (
    PBF2WayCouplingConfig, create_pbf2way_simulation,
)

config = PBF2WayCouplingConfig(
    scene="floating-equilibrium",
    boundary_viscosity=0.02,
    wall_damping=0.8,
    pressure_iterations=3,
)
simulation = create_pbf2way_simulation(config=config, device="cuda:0")
simulation.step()  # 推进 1/90 秒：3 个子步，每子步 3 轮压力迭代
```

公开数据：`positions`、`velocities`、`densities`、`rigid.position/rotation/velocity/omega/force/torque`、`boundary.position/velocity/volume/body`、`rigid.fault`、`sim_time`、`frame_dt`、`substep_dt`、`current_dt`、`last_dt`、`iterations`、`total_steps`、`total_substeps`。`current_dt/last_dt/frame_dt` 均为完整步的 `1/90 s`，`iterations` 表示每子步的压力轮数，默认为 3。密度误差由主动调用 `diagnostics()` 获取，不维护逐步回读的 CPU 属性。

`fluid_lower/fluid_upper` 为内缩后的粒子中心范围。诊断中的 `fluid_boundary_violation` 检查此范围，`max_fluid_violation_ever` 累计每个子步结束时对此范围的越界；`fluid_container_violation` 仍检查原始容器几何范围。预测阶段可能暂时越界，随后由压力迭代的位置更新钳回，渲染读取完整 step 完成后的状态。

仿真数组驻留所选 Warp 设备；正常 `step()/render()` 不调用 `.numpy()`。刚体模型矩阵由 Warp 直接写入 CUDA/OpenGL 共享缓冲区，实体和阴影共用；粒子首次渲染分配也避免回读。初始化、显式诊断、测试、截图和导出允许读回。渲染要求真正的 CUDA/OpenGL 互操作，注册失败直接报错，不回退到 CPU 拷贝；必要的资源 map/unmap 同步仍保留。

场景支持 `RigidBodies` 和 `FluidBlocks`。刚体支持 OBJ 路径、正缩放、平移、轴角旋转、密度、初速度/角速度、颜色、静态/动态状态、摩擦和恢复系数；流体块支持 `start/end`、缩放、平移、`denseMode=0` 和 `initialVelocity`。容器必须是一个静态轴对齐盒体。OBJ 需封闭、顶点焊接、绕序一致且朝外；不支持软体、关节、发射器、任意官方场景格式或其他流体算法。

### 相对官方路径的明确差异

1. 网格体积积分得到质量、质心和完整惯量；本实现不转换到官方 PBD 的主惯量坐标系，而是在质心坐标系保存完整矩阵。初始化只剔除与刚体重叠的流体粒子，运行中不删粒子。
2. 表面使用确定性规则三角形行采样，间距不超过 `1.5r`，合并重复点；不复现官方 Poisson-disk 随机样本和 Partio 缓存。边界伪体积公式保持不变。
3. 刚体碰撞采用局部 AABB 粗筛、Warp mesh BVH 有符号距离、双向表面样本接触；容器用内向盒体距离。没有移植 Discregrid 或原场景的球/圆柱/环面解析碰撞代理。
4. 接触使用 5 轮设备端顺序冲量：恢复系数取两者最小值，摩擦取几何平均，穿透偏置系数 `0.2`，容差 `0.06`。每子步预计算世界逆惯量、接触力臂、法向有效质量的分母，五轮在同一 kernel 内按原接触顺序执行；切向分母随当前切向重算。正间隙采用预测接触，已穿透接触允许小量速度级纠正。适用于少量刚体，没有连续碰撞检测，不保证任意高速、薄壁、深度初始穿透的处理。
5. 刚体角速度用包含陀螺项的 Euler 方程，四元数显式积分后归一化。GPU 原子累积与接触生成次序不是逐位确定的，长时轨迹允许分歧。
6. 粒子边界的有限支撑宽度会改变有效排水体积。`floating-equilibrium` 用半径 `0.2` 的球、密度 `200/3000`、初始高度 `0.5` 和 1 米高水块做明确的上浮/下沉对照；它不是近中性浮力的精确密度阈值标定。近中性物体需分辨率收敛研究。

## 测试与实际结果

2026-09-16 将完整步改为 `1/90 s`、默认压力迭代改为 3 后，26 项数值/GL/调度测试通过，验证每步 9 次压力修正、90 次 step 推进一秒，以及其他正整数压力轮数实际生效。以下十秒场景、十万粒子及性能数据均为先前 `1/60 s`、5 轮压力配置的历史实测；新的十秒运行对应 900 step / 2700 子步，尚未重新进行完整十秒验收。验证旧 nsys 记录时，给 `verify_profile` 添加 `--pressure-iterations 5`；省略时按当前配置校验。

性能收益、nsys 零回读验证、ncu 权限限制及复现命令见 [性能分析](PERFORMANCE.md)。`profile_simulation.py --steps N` 现在表示 N 次完整 `step()`（3N 个子步），加 `--render` 后每步渲染一次，计时不包含播放等待。

```powershell
# 数值测试；CPU 公式检查，有 CUDA 时增加短程对照
.venv/Scripts/python.exe -m unittest demos.fluid.PBF2WayCoupling.test_coupling -v

# 容器六面/棱角、反弹衰减、粘度后再次检查及非有限值测试
.venv/Scripts/python.exe -m unittest demos.fluid.PBF2WayCoupling.test_wall_clamp -v

# 独立隐藏窗口 OpenGL 测试，需要 NVIDIA CUDA/OpenGL
.venv/Scripts/python.exe -m unittest demos.fluid.PBF2WayCoupling.test_rendering -v

# 三个场景各运行至少 10 秒，输出逐秒诊断、最终状态及汇总
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.validate

# 另加两个十秒重力反转/旋转交互压力测试
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.validate --interactions --output outputs/pbf2way/wall-clamp/acceptance
```

2026-09-15，加入容器钳制后，本机 RTX 5070 12 GiB、Warp 1.15.0：22 项数值测试及 3 项 OpenGL/调度测试通过。覆盖六面与棱角、大幅越界、0/0.8/1 反弹系数、粘度后的向外速度、非有限值保留，以及固定步计数、故障锁存、CPU/CUDA 对照、暂停/重置和零回读。公式对照使用独立 NumPy 计算；尚未编译官方 C++ 程序做整体轨迹对照。

| 场景 | 流体/边界粒子 | 仿真时间 | step/子步数 | 平均 step 耗时¹ | 全程最大接触穿透 | 最终密度误差 |
| --- | --- | --- | --- | --- | --- | --- |
| 默认溃坝 | 24,389 / 64,176 | 10.000 s | 600 / 1,800 | 5.69 ms | 0.01209 m | 0.1821% |
| 静水双向 | 31,747 / 63,336 | 10.000 s | 600 / 1,800 | 6.25 ms | 0.00175 m | 0.2328% |
| 静水单向 | 31,747 / 63,336 | 10.000 s | 600 / 1,800 | 7.96 ms | 0.00151 m | 0.2325% |

¹ 含 Python 调度、同步和诊断，不含初始化、编译及渲染；不是帧率保证。

三组保持有限数值和粒子数量，无缓存溢出；逐子步累计流体内缩边界越界为零，最大四元数长度误差约 `1.2e-7`。默认场景最终球、圆柱、环面质心高度约 `0.680/0.548/0.673 m`，都有明显平移和转动。静水轻球从 `0.5 m` 上浮至 `0.855 m`，重球落至 `0.198 m`；关闭反作用力后，轻球也落至 `0.198 m`。

两个交互压力测试分别在 2/6 秒反转重力、4 秒旋转 90°、8 秒再旋转 45°，各推进十秒。流体在每子步结束时均位于内缩范围内，保持有限数值、粒子数和无故障；但刚体最大接触穿透分别达到 **0.1383 m / 0.0446 m**，超出普通场景验收界限。本次仅给流体增加容器保底，未修复强交互下的刚体接触问题，也不保证任意场景长期稳定。最新结果与性能记录见 [性能分析](PERFORMANCE.md)，原始数据在 `outputs/pbf2way/wall-clamp/`。

OpenGL 检查包含阴影开/关的像素差、billboard 与实体共用深度缓冲、GL 错误、暂停/重置请求和重力控制回调；没有依赖人工点击。截图和详细运行日志写入忽略的 `outputs/pbf2way/`，重新运行验证脚本可生成。
