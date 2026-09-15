# PBF 双向耦合性能分析

## 固定哈希扩容与流体属性重排（2026-09-16）

两张 HashGrid 现统一固定为 `128×160×128`，其中 Y 维由 96 增至 160，以覆盖约 10 万粒子配置所需的 `(52,131,28)` 查询单元跨度。初始化会按 `container ± support_radius` 计算每轴查询单元跨度，并要求严格小于哈希维度，以避免周期取模令远距离单元产生别名。每个子步在流体建表后执行一次 gather：将 `positions`、`velocities`、`old_positions` 和 `particle_ids` 永久切换到哈希顺序，同时生成原下标到新下标的 `old_to_sorted` 逆映射。流体邻居查询返回的建表前编号经该映射转换；边界点不重排。密度、lambda、修正量、加速度和邻居缓存随后完整覆盖，没有随排序搬运。

### 正确性与行为

- 22 项 `test_coupling` CPU/CUDA 测试通过，包含四个持久属性与稳定 ID 的逐项对应、逆映射、稳定 ID 邻域集合与暴力搜索一致、哈希跨度拒绝、CPU/CUDA 轨迹按 ID 对齐，以及锁存故障后的恒等复制与冻结。
- 完整数值/墙面/隐藏窗口集合共 32 项，其中 31 项通过；唯一失败是工作树既有的默认速度断言仍期望 `10 m/s`，而按要求保留的当前配置是 `6 m/s`，与本次哈希及重排无关。
- 溃坝、静水双向和静水单向各运行 `10 s = 900 step = 2700 substep`，无非有限状态、邻居溢出、粒子越界或故障锁存。平均 step 分别为 `4.06 / 5.33 / 7.35 ms`；最终 NPZ 均含 `particle_ids`，抽查为完整排列。

### 无 profiler 与 nsys

RTX 5070 / Warp 1.15.0，默认溃坝预热 3 秒后各测 270 step，四次无 profiler 平均 step 为 `3.363 / 3.681 / 4.073 / 3.157 ms`，中位数 **3.522 ms**、范围 **3.157–4.073 ms**。运行间接触状态不同，不能由该波动推断重排的净收益。

新的 nsys 在预热后采集 90 step：GPU→CPU 复制为 **0**；270 个子步恰好有 270 次 `reorder_fluid`。重排累计 **0.609 ms**，平均 **2.254 μs/次**，约占 kernel 时间 **0.3%**。主要 kernel 仍为：邻居缓存 **53.54 ms / 24.0%**、压力修正 **50.19 ms / 22.5%**、密度/lambda **39.84 ms / 17.9%**。HashGrid 的 CUB radix sort 主过程累计 **20.03 ms / 9.0%**；全部 CUDA memset 合计 5.72 ms，但其中不只有哈希表清零，不能全归因于扩容。该结果表明重排自身很便宜，但扩容和重排没有消除邻居遍历、压力及密度 kernel 的主要成本；本次按要求保留实现，不以性能作为撤销条件。

原始结果位于 `outputs/pbf2way/hash-reorder/`。复现命令：

```powershell
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.validate --output outputs/pbf2way/hash-reorder/acceptance
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=outputs/pbf2way/hash-reorder/headless --export=sqlite .venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.profile_simulation --device cuda:0 --warmup-seconds 3 --steps 90 --capture --output outputs/pbf2way/hash-reorder/profile.json
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.verify_profile outputs/pbf2way/hash-reorder/headless.sqlite --steps 90
```

ncu 在当前非管理员进程中仍返回 `ERR_NVGPUCTRPERM`。启用 NVIDIA performance counter 权限后，在管理员 PowerShell 中执行下列命令可采集重排、邻居、密度和压力的前四次匹配调用：

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.1\target\windows-desktop-win7-x64\ncu.exe' --profile-from-start off --kernel-name-base function --kernel-name 'regex:^(reorder_fluid|cache_neighbors|density_lambda|pressure_correction).*' --launch-count 4 --set full --clock-control none --force-overwrite --export outputs/pbf2way/hash-reorder/core-kernels .venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.profile_simulation --device cuda:0 --warmup-seconds 0.1 --steps 1 --capture --output outputs/pbf2way/hash-reorder/ncu-profile.json
```

**速度上限更新（2026-09-16）：** 当前流体默认 `max_speed=10 m/s`，在速度重建、粘度更新的墙面处理后按模长限速，方向不变；复用原有 kernel，保留设备故障检测。28 项数值/GL/调度测试通过。下方性能与十秒行为数据均采集于添加全局限速之前。

**当前配置（2026-09-16）：** 完整 step 已改为 `1/90 s`，三个子步各为 `1/270 s`；默认压力迭代为 3 次/子步，接触迭代仍为 5 次。26 项数值/GL/调度测试通过，包括每步 9 次压力修正和 90 步推进一秒。下方保留的是调整前 `1/60 s`、5 轮压力配置的性能与行为数据，不能直接作为新配置的实测结果。验证下方旧 nsys 数据时，给 `verify_profile` 命令添加 `--pressure-iterations 5`。

## 约十万流体粒子试运行（2026-09-16）

保持默认溃坝几何，将运行参数改为 `--particle-radius 0.015625`；实际流体数 **103,823**（47³），边界样本数 **163,732**。仅通过现有参数提高分辨率，默认配置仍为半径 0.025；求解器保持三个子步、每子步五轮压力/接触迭代。

RTX 5070、Warp 1.15.0，十秒仿真共 600 step / 1800 子步，耗时 **13.64 s**、平均 **22.74 ms/step**，包括主动逐秒诊断，排除初始化、编译、渲染和最终 NPZ 导出。这不是纯 GPU kernel 时间，也不是实际渲染帧率。相比此前 24,389 粒子试跑约 5.69 ms/step，约为四倍，但场景轨迹和接触数量不同，不能当成严格的规模扩展基准。

无非有限状态、粒子数变化、缓存溢出或流体内缩边界越界；全程最大刚体接触穿透 **0.01347 m**。最终平均压缩误差 **0.4858%**；圆柱沉至约 **0.092 m**，球和环面高度为 **0.620/0.656 m**。此结果说明十万级配置可以运行，但并未保持较粗分辨率下的浮沉行为，不能宣称更高分辨率提高了本次固定五轮求解的物理精度。未增加子步、迭代或限速。

另外完成三秒预热后的隐藏窗口绘制与截图。运行命令见 README 的“调整流体粒子数量”；原始诊断和截图在 `outputs/pbf2way/100k/dam-break.json`、`dam-break.npz`、`frame.png`。

## 最新变更：容器位置钳制与法向速度反弹

2026-09-15，在固定步版本 `f51fe2b` 上添加保底。每轮应用压力位移时将粒子中心限制在容器向内缩一个半径的范围；速度重建和粘度更新后，只将向外的法向速度乘以 `−0.8`。切向/向内速度、动态刚体压力受力公式和固定求解轮数保持原样。NaN/Inf 保留给设备异常检测。此措施会引入容器碰撞约束和耗散。

### 验证结果

- 25 项 CPU/CUDA 数值、调度及隐藏窗口测试通过。新增测试覆盖六面、棱角、多轮位置钳制、大幅越界、0/0.8/1 衰减、重复速度检查、粘度重新产生向外速度，以及非有限值和故障冻结。
- 溃坝、静水双向和静水单向各运行十秒：600 step / 1800 子步，粒子数及有限数值保持正常，无设备故障，内缩范围的逐子步累计越界为零。平均 step 为 5.69 / 6.25 / 7.96 ms，包含主动逐秒诊断，不含初始化和渲染。
- 静水轻球最终高度 0.85494 m，重球 0.19772 m；关闭反馈后轻球 0.19793 m。默认两场景最终密度误差 0.1821% / 0.2328%。
- 额外对两个场景各执行十秒重力交互：2 秒反转，4 秒绕 Z 轴旋转 90°，6 秒反转，8 秒旋转 45°。流体内缩范围累计越界仍为零，无缓存溢出或非有限数值，未设置全局限速。结束时最大粒子速率分别为 48.66 / 20.72 m/s。
- **限制：** 上述强交互中的刚体最大接触穿透为 0.1383 / 0.0446 m，超过普通场景的粒子半径验收界限；静水交互结束时密度误差为 1.2632%。因此交互测试确认的是流体保底有效及状态有限，不代表刚体高速碰撞或整体物理精度达标。本次未改变刚体接触方案。

### 零回读与性能

在初始化和预热之后，分别采集 60 次无窗口 step，以及 60 次 step 后各渲染一帧：两次 nsys SQLite 导出均为 **0 次 GPU→CPU 复制**。调用数均为 180 次预测、900 次压力修正、900 次位置更新、180 次速度重建、180 次粘度速度更新和 180 次接触求解；渲染组另有 60 次刚体模型矩阵更新。没有增加独立 kernel 调用，仍保留必要的 CUDA/OpenGL 同步。

无 profiler、相同默认溃坝、预热一个 step 后测量 600 step：改动前为 **5.455 ms/step**，改动后为 **4.795 ms/step**。这是两次独立运行，最终接触数分别为 119 和 0；钳制与原子顺序会改变轨迹，不能把这个时间下降归因为钳制带来的加速，也不能用差值估算新增分支的成本。

新无窗口 trace 中，三个含钳制的更新 kernel 平均分别为 **0.942 / 1.173 / 1.108 μs/次**（位置更新 / 速度重建 / 粘度更新）。按每 step 的 15/3/3 次调用折算，合计约 **0.021 ms/step**，包含原有更新算术，并非钳制的净增量。其总 GPU 成本在本次场景中很小；整程序仍主要消耗在压力、邻域及接触计算。

复现：

```powershell
.venv/Scripts/python.exe -m unittest demos.fluid.PBF2WayCoupling.test_wall_clamp demos.fluid.PBF2WayCoupling.test_coupling demos.fluid.PBF2WayCoupling.test_rendering -q
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.validate --interactions --output outputs/pbf2way/wall-clamp/acceptance
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.profile_simulation --device cuda:0 --warmup-seconds 0 --steps 600 --output outputs/pbf2way/wall-clamp/after.json

nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=outputs/pbf2way/wall-clamp/rendered --export=sqlite .venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.profile_simulation --device cuda:0 --warmup-seconds 3 --steps 60 --render --capture --output outputs/pbf2way/wall-clamp/rendered_profile.json
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.verify_profile outputs/pbf2way/wall-clamp/rendered.sqlite --steps 60 --render
```

无窗口 trace 去掉两个命令中的 `--render`，并将输出名改为 `headless`。原始日志、JSON、nsys 和 SQLite 保存在忽略目录 `outputs/pbf2way/wall-clamp/`。下方保留钳制加入前的测量作为历史对照。

## 固定步版本：零回读与接触优化（钳制加入前）

2026-09-15，Windows / RTX 5070 12 GiB / Warp 1.15.0。当前 `step()` 固定推进 `1/60 s`，含三个 `1/180 s` 子步，每子步五轮压力和五轮接触迭代。每次未暂停的 `step()` 后只绘制一帧；播放速度只控制帧间等待。以下为本次实现和实测，下方“历史分析”保留旧自适应版本的瓶颈证据。

### 改动与范围

- 移除 CFL、误差归约、压力提前退出及正常循环的数组回读。设备端故障锁存、非有限数值检查和穿透统计仍运行；溢出在消费截断列表前阻断物理更新，设备错误只输出一次，重置后才恢复。
- 刚体模型矩阵和粒子 billboard 直接使用 CUDA/OpenGL 共享缓冲区。修复共享 billboard 首次分配中的隐式回读；新增分配钩子，其他演示保留原有行为。互操作失败不回退到 CPU 拷贝。
- 每子步预计算世界逆惯量、接触力臂和法向有效质量分母，将原来的五次接触求解调用合并成一次 kernel 内的五轮顺序遍历。切向有效质量随当前切向重新计算。没有接触裁减、Jacobi 或改变摩擦/恢复系数。
- SPlisHSPlasH 的官方参考通过 `BoundarySimulator_PBD` / `PBDWrapper` 在 **CPU** 上求解刚体接触，流体力在 CPU 线程间归约并交给刚体；该路径不存在 GPU→CPU 回读。本实现继续使用 GPU，避免 CPU 松耦合所需的力与位姿传输。参考：[BoundarySimulator_PBD.cpp](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/f3f677140761db7637b5443beb54f19f1f835ed4/Simulator/BoundarySimulator_PBD.cpp)、[PBDWrapper.cpp](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/f3f677140761db7637b5443beb54f19f1f835ed4/Simulator/PositionBasedDynamicsWrapper/PBDWrapper.cpp)。

### 分阶段收益

未开启 profiler；默认溃坝、预热到约 3 秒，测量随后 1 秒仿真；各阶段独立初始化，串行运行 3 次取中位数。统一报告推进 `1/60 s` 所需的等效墙钟时间，旧自适应版本按实际推进量折算，不能把它的一次子步当成新版本的一次 step。

| 阶段 | 等效 ms / 1⁄60 s | 三次测量范围 |
| --- | ---: | ---: |
| 原自适应版本 | 44.11 | 41.96–59.39 |
| 固定三个子步、每子步五轮；保留旧回读/归约/接触 | 9.41 | 8.30–12.21 |
| 去掉回读及无用归约，仍使用原五次接触 kernel | 6.35 | 6.16–7.37 |
| 完整优化版本 | 6.01 | 4.86–6.60 |

相邻中位数下降约 **79% / 33% / 5%**。第一项主要来自改变求解工作量及误差要求，不能称为保持数值精度的实现加速；第二项同时包含移除归约、重置和循环调度的变化。各独立运行的接触数和轨迹不同，三次样本也不足以给出置信区间，尤其不能将最后的 5% 当成精确的接触收益。诊断脚本还包含替换 kernel 的 Python 包装开销。该表用于观察整程序量级，接触收益使用下面的固定状态实验。

本机分阶段脚本及旧代码快照在忽略目录 `outputs/pbf2way/fixed-step/measure_stages.py`、`before/`，原始数据为 `measure_stages.json`；旧基线为提交 `cb581ae` 的求解器。它们不影响正式入口。

### 同一接触状态的对照

从当前溃坝 3 秒状态保存 **50 个接触**。每次恢复相同线速度、角速度、法向/切向累计冲量；接触顺序、目标速度、刚体姿态完全相同。使用 CUDA graph 连续执行 100 组，以 CUDA events 计时，7 次取中位数。两侧均包含相同的设备内恢复拷贝；优化侧包含两次预计算 kernel，排除 Python 逐次提交开销。

| 实现 | 每组五轮接触耗时 |
| --- | ---: |
| 原始五次顺序 kernel | 257.22 μs |
| 预计算 + 单次 kernel 内五轮 | 188.00 μs |

下降 **26.9%**。此状态下线速度、角速度、法向和切向累计冲量的最大绝对差均为 **0**。另有 CPU/CUDA 六接触非对称测试及双动态刚体碰撞测试。整体长时轨迹不要求逐位一致，因为接触生成和压力反作用力依赖原子累积顺序。接触仍由一个有效 GPU 线程顺序求解，刚体/接触数增加后仍可能成为瓶颈。

### nsys：正常运行零 GPU→CPU 复制

两次独立采集均在初始化、JIT、3 秒仿真预热及 GL 首次分配之后开始，在显式结果回读之前结束；场景区间为 3–4 秒。渲染组使用隐藏 1280×800 窗口、2048² 阴影，采集不等待播放节拍。

| 60 次 step 的采集区间 | GPU→CPU 复制 | predict | 压力修正 | 合并接触 kernel | 刚体模型矩阵 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无窗口 | **0** | 180 | 900 | 180 | — |
| 每步后渲染 | **0** | 180 | 900 | 180 | 60 |

两个 SQLite 导出均只记录到 360 次 CPU→GPU 小拷贝，共 25,920 字节，来自 HashGrid 构建的主机参数上传。必要的 GL map/unmap 和资源同步仍存在。禁止 `wp.array.numpy()` 的独立测试覆盖正常 step、首次绘制及重置后绘制，补充验证未被预热掩盖的首次分配路径。

nsys 采集期间分别为 5.24 ms/step、8.11 ms/step+render；无 profiler 的一次相同渲染窗口为 **8.98 ms/step+render**。这些运行的接触状态不同、桌面负载和驱动调度也有影响，不能用它们相减推断渲染纯 GPU 耗时或 profiler 开销。可见窗口仍受显示环境影响，不承诺帧率。

无窗口 trace 中主要 kernel 总时间为：压力修正 83.88 ms、密度/lambda 54.81 ms、邻域缓存 41.04 ms、接触求解 17.97 ms（均累计 180 个子步）。移除往返后，压力及邻域计算仍是主要后续优化方向。没有新增 ncu 硬件计数器结论：此前 `ERR_NVGPUCTRPERM` 权限限制仍适用。

### 数值与交互验收

20 项单元/GL 测试通过；60 次 step 恰好推进 1 秒。故障测试实际注入非有限刚体速度和邻域/接触溢出，验证设备锁存、后续状态冻结和显式诊断异常。暂停只绘制；重置恢复初始状态并暂停；播放不足一帧预算时才等待，不积累任务。Warp 基类原有 `end_frame()` 暂停内部循环已由本演示覆盖，避免额外绘制及阻塞重置。

默认溃坝、静水双向和静水单向分别运行 **10 秒 = 600 step = 1800 子步**。含主动逐秒诊断、不含初始化/渲染，平均 step 为 **5.37 / 6.55 / 8.37 ms**。无非有限数值、粒子数量变化或缓存溢出，累计流体容器越界为零。最大接触穿透为 **0.01032 / 0.00175 / 0.00151 m**，均小于粒子半径 0.025 m。

最终平均压缩误差为 **0.1819% / 0.2324% / 0.2353%**，明显高于旧 0.01% 阈值；溃坝穿透也更大。轻球升至 0.855 m，重球降至 0.198 m，关闭反馈后轻球也降至 0.198 m。未发现本次十秒测试中的持续穿墙或发散；没有通过增加子步、迭代或限速来掩盖固定配置的误差。详细结果见 `outputs/pbf2way/fixed-step/acceptance/summary.json`。

### 当前复现命令

```powershell
# 无 profiler；--steps 是完整 step 数，--render 可省略
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.profile_simulation --device cuda:0 --warmup-seconds 3 --steps 60 --render --output outputs/pbf2way/fixed-step/rendered_baseline.json

# nsys：正常仿真 + 每步一帧；无窗口采集时去掉 --render 并更换输出名
nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=outputs/pbf2way/fixed-step/rendered --export=sqlite .venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.profile_simulation --device cuda:0 --warmup-seconds 3 --steps 60 --render --capture --output outputs/pbf2way/fixed-step/rendered_profile.json

# 核对复制与调用数；无窗口验证去掉 --render
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.verify_profile outputs/pbf2way/fixed-step/rendered.sqlite --steps 60 --render

# 同一状态比较原接触公式与优化结果、计时
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.benchmark_contacts --output outputs/pbf2way/fixed-step/contact_benchmark.json

# 数值/调度/GL，以及三个十秒场景
.venv/Scripts/python.exe -m unittest demos.fluid.PBF2WayCoupling.test_coupling demos.fluid.PBF2WayCoupling.test_rendering -q
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.validate --output outputs/pbf2way/fixed-step/acceptance
```

原始 `.nsys-rep`、SQLite、`*.verified.json`、测试日志及阶段计时均在忽略目录 `outputs/pbf2way/fixed-step/`。复现会因原子顺序产生不同接触数量，应以重新采集的结果为准。

---

## 历史分析：原自适应版本，非当前运行行为

以下采集同为 2026-09-15，基于提交 `cb581ae` 的原始代码，**发生在上述实现之前**。旧命令和输出脚本仅适用于旧版本；当前使用上方复现命令，`--steps` 的单位也已由子步改成完整 step。以下的“优先方向”是当时的分析建议，已由本次明确授权的固定配置方案替代。

## 结论

主要成本是 **PBF 多轮迭代的 GPU 计算与 CPU/GPU 往返调度**，其次是 **单线程刚体接触求解**。反作用力累积和邻居缓存布局有可测量的优化空间。实际前端的低帧率主要来自每帧执行多个物理子步。

不能将这些结论简单解释为显卡算力不足：nsys 记录中存在大量 CUDA 工作之间的空隙；接触求解又只使用一个有效线程。GPU 活动持续时间与 SM 利用率是不同概念。

## 环境和测量范围

- Windows，RTX 5070 12 GiB / GB205 / 48 SM，驱动 595.79。
- Python 3.12.3，Warp 1.15.0，CUDA Toolkit 12.9。
- Nsight Systems 2026.3.1；Nsight Compute 2025.1.1。
- 默认粒子半径 0.025，原始压力阈值、CFL、5 轮接触迭代不变。
- 每个无窗口窗口测量 200 个完整子步。初始化、JIT、首轮分配、场景预热在采集范围外；测量中不写逐步日志、不导出全量粒子数组。
- 各次运行串行使用 GPU。桌面和驱动调度仍可能引入干扰；未锁定时钟。
- 接触生成及反作用力采用原子操作，后期独立运行的轨迹、步长和压力迭代数会有差异。因此后期基线与 nsys 的时间比值不能作为严格的 profiler 开销估计。

### 未开启 profiler 的实测

| 场景 / 预热仿真时间 | 流体 / 边界粒子 | 平均子步 | P95 子步 | 平均压力轮数 | 平均 dt | 仿真秒 / 墙钟秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 溃坝 / 0.1 s | 24,389 / 64,176 | 2.956 ms | 4.086 ms | 10.555 | 1.481 ms | 0.501 |
| 溃坝 / 3 s | 24,389 / 64,176 | 4.987 ms | 7.233 ms | 10.925 | 1.726 ms | 0.346 |
| 静水 / 3 s | 31,747 / 63,336 | 17.789 ms | 19.476 ms | 57.830 | 5.000 ms | 0.281 |

这是指定时间窗口的一次实测，不能代表整个 10 秒过程。静水采集窗口约为仿真 3–4 秒；溃坝后期窗口约为 3–3.35 秒。

nsys 采集时平均子步分别为 4.279、7.166、35.940 ms。尤其是大量短 kernel 的工作负载，trace 本身会明显影响运行时间；上表才是实际性能基线。

采集诊断中存在“部分事件可能未收集”和未使用 NVTX 的提示，venv 启动进程本身也没有 CUDA 事件。已交叉核对实际 Python 子进程的关键调用数：每组均为 200 次预测/邻域搜索、1,000 次接触求解，压力调用数等于日志中迭代总数，密度调用数再多 200 次，GPU→CPU 复制数也吻合 `迭代总数 + 800`。这些核对支持下述热点统计，但不保证所有驱动/图形事件完整。

## 1. 压力循环：计算和同步共同成为主瓶颈

下表百分比的分母为 **所有 CUDA kernel 时间之和**，不是应用墙钟时间。括号是每子步累计 GPU 时间；它已包含该子步的多次调用。

| kernel | 溃坝早期 | 溃坝后期 | 静水 |
| --- | ---: | ---: | ---: |
| `density_lambda` | 42.6%（0.568 ms） | 27.8%（1.027 ms） | 42.4%（4.738 ms） |
| `pressure_correction` | 25.1%（0.335 ms） | 32.0%（1.182 ms） | 45.1%（5.035 ms） |
| `solve_contacts` | 0.2%（0.003 ms） | 26.1%（0.967 ms） | 7.8%（0.875 ms） |
| `cache_neighbors` | 19.5%（0.261 ms） | 6.4%（0.235 ms） | 2.2%（0.248 ms） |

静水中密度与压力修正合计占 **87.5%** 的 kernel 时间。nsys 记录 11,546 轮压力迭代 / 200 步，即每步 57.73 轮，最大 59 轮；并非触发 100 轮上限。每轮都有三次 kernel launch、误差清零和一次 CPU 读回。减小单轮成本会被迭代次数放大。

相关代码：`PBF2WayCoupling.py` 的 `density_lambda`、`pressure_correction` 及 `Example.step()` 压力循环。

### 4–8 字节读回的主要代价是等待，不是传输带宽

`Example.step()` 每步有 **压力轮数 + 4** 次 `.numpy()`：每轮的压缩误差，以及邻域溢出、接触溢出、最大速度、无效状态。Warp 的 `array.numpy()` 明确执行同步的 GPU→CPU 复制，保证之前的工作已经完成。

| 200 步窗口 | GPU→CPU 次数 | 总字节数 | 复制的 GPU 活动时间 | 对应复制 API 的 CPU 时间 |
| --- | ---: | ---: | ---: | ---: |
| 溃坝早期 | 2,911 | 12,444 B | 0.865 ms | 295.335 ms |
| 溃坝后期 | 3,399 | 14,396 B | 1.047 ms | 774.727 ms |
| 静水 | 12,346 | 50,184 B | 3.604 ms | 2,423.117 ms |

API 时间包含等待前面的 GPU 计算完成，**不能与 kernel 时间相加，也不能全部视为可消除的同步开销**。但数据量极小、调用频繁且阻断流水，足以确定这是重要的调度问题。

nsys 中 CUDA kernel/复制/memset 的活动区间取并集，再除以首末 CUDA 活动跨度，早期/后期/静水分别为 **31.6% / 51.9% / 31.2%**。余下区间没有本进程的这些 CUDA 活动，可能包含 Python 提交、驱动等待、调度及 profiler 开销；这不是全卡硬件利用率，也不是可直接套用的加速上限。

### 优先方向

1. 减少每轮 host 往返：评估设备端停止条件、条件 CUDA graph 或保留逐轮检查的其他设备调度方式。首先保持当前误差定义与停止语义。
2. 将同一同步点的多个诊断标量合并读回；溢出检查仍必须在后续依赖结果的计算之前生效。
3. 评估误差、最大速度等全局标量的分层归约。目前每粒子向同一地址执行原子操作；其精确收益仍需专项实验或 ncu 验证。
4. 不把提高密度阈值、强制减小迭代上限作为保行为的优化方案。压力收敛需求应与实现成本分开处理。

## 2. 反作用力累积与邻居布局：固定状态实验

在同一个溃坝约 3 秒状态上直接重复原有 kernel。用 CUDA graph 连续提交 100 次，以 CUDA events 计时，重复 7 组取中位数。该实验排除了 Python 每次提交成本，不推进仿真。

| 密度 / 压力调用 | 原始行布局 | 改为列布局 |
| --- | ---: | ---: |
| `density_lambda` | 73.045 μs | 66.323 μs |
| 压力修正，保留反作用力 | 100.492 μs | 87.112 μs |
| 压力修正，跳过反作用力 | 62.853 μs | 50.663 μs |

**反作用力路径有显著成本。** 原布局下跳过该路径降低约 37.5% kernel 时间；列布局下降约 41.8%。该路径包含力/力矩计算及向少数刚体的原子累积；实验不能进一步把算术成本与原子竞争分离。可以优先评估粒子/线程块内汇总后再向刚体归约。关闭双向作用只用于诊断，不是功能完整的优化方案。

**布局成本也得到实测支持。** 原缓存逻辑形状为 `[粒子数, 最大邻居数]`，流体/边界相邻粒子的物理步幅是 1024/2048 字节。多个相邻线程读取各自第 k 个邻居时地址分散。实验保持同一逻辑形状、邻居索引和遍历顺序，仅改变底层 strides，使相邻粒子对应项相差 4 字节。

密度、lambda 和流体位置修正输出在这个状态上逐位一致。密度 kernel 下降约 9.2%，保留反作用力的压力 kernel 下降约 13.3%。这支持改善缓存布局，但尚未测量邻域缓存写入成本和完整子步收益；构建转置缓存的成本未计入该实验。没有 ncu 数据，因此不声称具体的缓存命中率、内存吞吐率或 stall 类型。

实验脚本与原始数据：`outputs/pbf2way/profiling/kernel_experiments.py` 和同名 `.json`。它们只作用于独立诊断进程，正式入口未采用任何实验变化。

## 3. 接触求解只使用一个有效 GPU 线程

`solve_contacts` 以 `dim=1` 启动，一个有效线程遍历所有接触，每步重复 5 次。nsys 记录一个 block；Warp 启动的 block 大小虽然是 256，但逻辑任务数只有 1。该线程持续进行刚体状态读写、惯量矩阵变换和顺序冲量计算。

溃坝后期采集起止接触数为 290/151；每次接触 kernel 平均 193.3 μs，5 轮累计 0.967 ms/步，占 kernel 总时间 26.1%。早期无接触时每次仅约 0.585 μs。静水约 156–158 个接触，平均每次 175.0 μs。

这说明“只有三个刚体”并不意味着接触成本小：开销与表面样本生成的接触数量和顺序处理次数相关。

优先评估每子步预计算世界逆惯量及不变的接触几何项。接触流形裁减、图着色或 Jacobi 并行求解可能进一步改善并行度，但会改变约束集或求解顺序，需要单独的行为验收，不能直接用无保护的逐接触并行写替换当前顺序求解。

## 4. 邻域搜索和其他成本

`cache_neighbors` 每步约 0.24–0.26 ms，在早期占比显著，静水中则远小于多轮压力成本。两个 HashGrid 每步都重建，包含多个排序 kernel；可以评估静态容器与动态边界分开维护，避免静态部分反复排序。

溃坝后期 `detect_contacts` 约 0.095 ms/步，比后续顺序接触求解小得多。`viscosity`、边界更新、诊断 kernel 都不是当前第一优先级。不要仅按线程数、数组容量或函数长度判断热点。

## 5. 实际前端

调用原始 `render_opengl.main()`，隐藏窗口、1280×800、阴影开启、默认 vsync、播放速度 0.5。先预热 3 秒物理和 5 帧，再测量 120 帧；仅用包装函数记录原始 scheduler 和 render 调用的耗时。

未开启 profiler：

- 16.798 秒完成 120 帧，约 **7.14 FPS**。
- 平均物理调度 **137.942 ms/帧**，render 调用 **2.031 ms/帧**。
- 共 1,255 个子步，平均 **10.46 子步/帧**，5 帧达到 16 子步上限。
- 仿真约从 3.108 s 到 9.077 s，实际推进速度 **0.355 仿真秒/墙钟秒**，低于请求的 0.5。

render 调用是 CPU 范围计时，可能包含提交、同步与呈现；它不是纯 shader GPU 时间。隐藏窗口结果也不能替代所有显示器环境下的可见窗口测试。但这次测量足以说明，单纯减少阴影分辨率无法解决占据绝大部分帧时间的物理步骤。

前端会在绘制前同步执行多个完整子步；`PlaybackScheduler` 将上一帧墙钟差值限幅到 0.1 秒，并在达到子步上限时丢弃完整的积压步数。物理处理速度不足会同时影响帧率与播放速度。需要分别记录 FPS 和仿真秒/墙钟秒，避免只改善其中一个指标。

另采集了 120 帧的 CUDA/OpenGL 联合 trace（`frontend.nsys-rep`）。该次采集耗时 139.234 秒，远高于基线；`cuLaunchKernel` 的 CPU API 时间升至约 122 秒，并有 OpenGL/CUDA 事件可能不完整的诊断。因此不使用其 0.86 FPS 作为实际性能，也不从中推断完整的 shader 占比。其主要计算调用数仍能与 1,317 个子步、48,739 轮压力迭代核对，说明整个较长播放区间的平均压力轮数约 37，明显高于 3 秒附近短窗口的 11–13 轮。

补充的 20 帧短窗口只开启 CUDA trace（`frontend_short.nsys-rep`）：281 子步、6,336 轮压力迭代，平均 22.55 轮/步。密度与压力修正共占 84.9% 的 CUDA kernel 时间；物理调度平均 176.268 ms/帧、render 调用 2.675 ms/帧。它再次支持计算热点定位；采集中的 5.58 FPS 同样不替代上面的无 profiler 基线。

## ncu 状态与限制

已经用 ncu 对 `cache_neighbors`、`density_lambda`、`pressure_correction` 尝试 full collection；沙箱内外都返回 `ERR_NVGPUCTRPERM`，未得到硬件计数器报告。失败过程的耗时不能作为性能数据。

因此本报告不提供 achieved occupancy、缓存命中率、带宽饱和度、warp stall 等未经测量的数据。Nsight Systems 已提供执行时间、launch 参数及 CPU/GPU 时间线；反作用力和布局通过固定状态实验补充验证。

后续可从 Windows **管理员终端**运行 ncu，或由管理员按 [NVIDIA 官方说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)开放 GPU 性能计数器权限，再执行下方命令。此次未改动驱动权限设置。

## 复现命令与产物

从仓库根目录运行。`profile_simulation.py` 的 `--warmup-seconds` 是仿真时间，`--steps` 是完整子步数；`--capture` 只控制 profiler 起止 API。

```powershell
# 不开 profiler 的基线
.venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.profile_simulation --device cuda:0 --warmup-seconds 3 --steps 200 --output outputs/pbf2way/profiling/late_baseline.json

# nsys：预热完毕后才采集
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=outputs/pbf2way/profiling/late --export=sqlite .venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.profile_simulation --device cuda:0 --warmup-seconds 3 --steps 200 --capture --output outputs/pbf2way/profiling/late_profile.json

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum --format csv --output outputs/pbf2way/profiling/late_stats outputs/pbf2way/profiling/late.sqlite
```

早期改为 `--warmup-seconds 0.1`；静水增加 `--scene floating-equilibrium`，并修改输出前缀。

本机额外诊断脚本的复现命令（脚本和数据保存在忽略的输出目录）：

```powershell
.venv/Scripts/python.exe outputs/pbf2way/profiling/kernel_experiments.py
.venv/Scripts/python.exe outputs/pbf2way/profiling/frontend_runner.py --output outputs/pbf2way/profiling/frontend_baseline.json
nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=outputs/pbf2way/profiling/frontend_short --export=sqlite .venv/Scripts/python.exe outputs/pbf2way/profiling/frontend_runner.py --frames 20 --capture --output outputs/pbf2way/profiling/frontend_short_profile.json
.venv/Scripts/python.exe outputs/pbf2way/profiling/analyze_trace.py early late floating frontend_short
```

```powershell
# 管理员终端；直接调用 exe，避免 ncu.bat 将正则的 | 当作 shell 管道。
$ncuExe = 'C:/Program Files/NVIDIA Corporation/Nsight Compute 2025.1.1/target/windows-desktop-win7-x64/ncu.exe'
& $ncuExe --profile-from-start off --kernel-name 'regex:^(cache_neighbors|density_lambda|pressure_correction).*' --launch-count 3 --set full --clock-control none --force-overwrite --export outputs/pbf2way/profiling/early_kernels .venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.profile_simulation --device cuda:0 --warmup-seconds 0.1 --steps 1 --capture --output outputs/pbf2way/profiling/early_ncu.json

# 接触较多的状态，单独选择接触求解 kernel
& $ncuExe --profile-from-start off --kernel-name 'regex:^solve_contacts.*' --launch-count 1 --set full --clock-control none --force-overwrite --export outputs/pbf2way/profiling/contact_kernel .venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.profile_simulation --device cuda:0 --warmup-seconds 3 --steps 1 --capture --output outputs/pbf2way/profiling/contact_ncu.json
```

`--clock-control none` 不更改 GPU 时钟；比较硬件指标时仍需留意时钟与桌面负载。ncu replay 会引入大量采集开销，应用计时不能用于推断正常运行速度，见 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。

本机原始产物保存在忽略的 `outputs/pbf2way/profiling/`：

- `early.nsys-rep`、`late.nsys-rep`、`floating.nsys-rep` 及 SQLite 导出。
- `*_baseline.json`、`*_profile.json`：工作量和逐子步时间。
- `*_stats_*.csv`、`*_analysis.json`：kernel/API 汇总与 CUDA 活动并集统计。
- `kernel_experiments.py/.json`：同一状态的布局与反作用力对照。
- `frontend_runner.py`、`frontend_baseline.json`：原始前端包装计时；`frontend_short.nsys-rep` 为补充的短窗口 CUDA trace，`frontend.nsys-rep` 为扰动较大的 CUDA/OpenGL 联合采集。
- `early_ncu.log`：性能计数器权限失败的原始日志。

建议按“减少压力循环往返 → 汇总反作用力与改善邻居布局 → 优化顺序接触成本 → 减少重复邻域构建”的顺序做后续实现。每项改动都需重新采集同一工作量，并验证数值和行为；本次没有宣称整程序的预期加速倍数。
