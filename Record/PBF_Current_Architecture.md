# PBF 当前架构与协作标准

更新日期：2026-11-07。适用范围：根目录 `PBF/`、`Sim_Ready/` 及对应根目录测试。

本文是供新对话、新 agent、新电脑和其他参与者使用的工程交接文档，不是 skill，也不承担自动执行或授权功能。只描述当前实现、设计约束和工作方法。执行具体任务时，以用户当次要求和实际代码为准；先读相关实现，再参考说明，不能用文档替代代码核对。

阅读顺序：理解运行先看第 2～4 节；修改代码先看第 9 节；处理资产看第 8 节；换电脑运行和验收看第 10～11 节。

## 1. 项目定位与基本约定

当前 PBF 是使用 NVIDIA Warp 编写的流体求解器，接入 Newton 的 Model、State 和 SolverBase，支持基于体积图的动态刚体压力双向弱耦合。刚体推进通过 Newton 的 SolverMuJoCo，默认使用 MuJoCo Warp GPU 后端。

这里的“Newton 应用层”指管理 Newton 物理资源、调度与显示的应用运行层，不代表使用了 Newton Viewer 或 UI。当前显示为独立 OpenGL billboard 渲染，也支持 USD 输出；没有接入 MimicKit。

当前范围与能力：

- 场景为 `static` 和 `stairs-rigids`，默认 CLI 场景仍是 `static`。
- `static` 是阶梯、兔子与容器边界中的流体；`stairs-rigids` 在相同流体和地形上增加两个自由盒子、一个自由球。
- 流体使用 SDF/VM 边界，刚体之间及刚体与场景之间使用 MuJoCo 原生接触。
- 流固交换的是压力修正产生的反力和力矩。SDF 穿透投影不反馈力，限速、涡量和黏性处理不额外补偿动量。
- 当前为单场景、单套固定流体配置、单进程一套 PBF 实例的实现；模块级编译常量是这一约束的一部分。
- 不支持自动微分、多 world 选择性重置、流体 observables 或通用粒子外力接口。不能仅因继承 SolverBase 就宣称覆盖 Newton 全部能力。

## 2. 目录与职责

```text
PBF/
├─ app.py                  CLI、应用装配、相机初值、主循环、演示交互语义
├─ simulation.py           Newton 应用运行层：资源、帧推进、Graph、重置、renderer
├─ PBF.py                  主要 kernels 与 PBF(SolverBase)，一步是一个物理子步
├─ pbf_helper.py           PBFConfig、初始化、工作缓冲、排序及数学辅助函数
├─ boundary_model.py       SDF/VM、边界资产与实例、局部变换、粗筛、GPU 位姿同步
├─ coupling.py             PBF 与 MuJoCo 的组合求解器、后端代码选择
├─ scene.py                已发布资产装配、静态碰撞与刚体实例、边界绑定
├─ scene_data.py           基础几何尺寸，生产和运行装配共用的轻量数据
├─ config_io.py            显式配置读写、JSON 解析与入口规则
├─ renderers/
│  ├─ opengl.py            billboard、速度配色、网格实例、窗口与原始事件
│  └─ usd.py               USD 帧输出与关闭保存
├─ config/                 流体参数 JSON 副本／记录，默认不读取
└─ Assets/
   ├─ boundary/            静态容器发布资产
   └─ rigid/               动态边界发布资产

Sim_Ready/
├─ pipelines/
│  ├─ container.py         容器／地形边界的 prepare、build
│  ├─ rigid.py             局部实体边界的 prepare、build
│  └─ preset/pbf.py        PBF 场景配方、CLI、缓存、VM、元数据、整组发布
├─ geometry.py             通用几何处理、Newton 地形调用与坐标对齐
├─ validation.py           实体、空腔等几何校验
├─ sdf.py                  离线 SDF 烘焙
├─ collision.py            CoACD、凸块校验、缓存与近似误差记录
├─ nvdb_io.py              NVDB 读写
├─ asset_io.py             内容摘要、SDF 身份计算
├─ publish.py              暂存、发布和失败回滚
├─ config.py               几何与烘焙参数
└─ Assets/                 源模型，不是运行时发布目录

tests/pbf/                 流体、耦合、Graph、显示和 benchmark
tests/sim_ready/           资产生产、缓存、发布、隔离验收
Record/                   工程说明与记录
outputs/pbf-newton/        本地日志、报告、图像、USD、性能采样
```

不要把文件按“类越少越好”或“层数越多越规范”机械合并、拆分。主要流程应能直接读懂；辅助计算、资源处理和显示细节放在职责明确的模块中。

### 2.1 应用、运行层、渲染器的调用关系

```text
app.main() / app.main_usd()
  → 解析 CLI、选择配置
  → 创建 Simulation
      → 装配场景、Model、BoundarySystem、solver
      → GPU 初始化粒子，创建两份 State
      → 创建 Control（耦合场景）
  → Simulation.set_renderer(renderer)
      → renderer.initialize(simulation)
  → 注册应用键盘行为（OpenGL）
  → 主循环
      → 处理待执行的 reset
      → Simulation.render() → renderer.render(simulation)
      → 未暂停时 Simulation.step() 或 launch_graph()
  → Simulation.close() → renderer.close()
```

应用决定何时推进、暂停或退出。Simulation 持有物理资源和 renderer；renderer 不创建 Simulation、不推进物理。`PBF.py` 的主要内容是 solver，末尾仅保留 USD 启动转发到 `app.main_usd()`。

renderer 的轻量约定是 `initialize(simulation)`、`render(simulation)`、`close()`。没有通用渲染基类、注册器或插件系统。更换 renderer 时先释放旧组件；重复关闭安全。无 `stage_path` 时不初始化 USD 输出。

当前保留共享 `demos.fluid.common.billboard_renderer` 依赖。正式资产生产和 NVDB 工具来自独立 `Sim_Ready`；不得因此擅自修改共享 billboard 或其他 demo 文件。

## 3. Newton 数据契约：Model、State、Config、solver

| 对象 | 当前持有的内容 | 使用原则 |
| --- | --- | --- |
| Model | 粒子初始数据、质量等模型属性，刚体／形状／关节存储，重力 | 是共享场景模型；并非绝对不可变，重力变化通过通知传给 solver |
| 两份 State | 粒子位置速度，以及刚体／关节运动状态和外力等 | Simulation 持有；两子求解器使用相同的一对输入输出 |
| Control | Newton 刚体控制数据 | 由运行层持有并传给刚体 solver；流体当前不消费 |
| PBFConfig | 流体参数、初始化范围等 | 由应用选择，传给运行层和 solver；不需要全部塞进 Newton Model |
| PBF solver | 预测双缓冲、压力工作数组、邻域缓存、HashGrid、派生常量 | 求解器私有工作数据，不应由渲染器管理 |
| BoundarySystem | SDF/VM 资源、共享资产、实例映射、局部变换与 GPU 边界数据 | 场景初始化时创建并交给 solver；不强行挂成 Newton 标准字段 |

Newton 的作用是提供共同的模型、状态与求解接口，使运行层能组织多个 solver、控制和显示。求解器分离不只是把 kernel 搬出主文件，而是把“一步物理计算”与“创建场景、循环帧、显示、计时和生命周期”分开。

当前接口：

```python
PBF(model, *, config, boundaries, verbose=False)
solver.step(state_in, state_out, control, contacts, dt, *, observables=None)
solver.reset(state, world_mask=None, flags=None)
solver.notify_model_changed(flags)
```

`PBF.step()` 只推进一个子步，压力迭代数归 PBF；帧长和每帧子步调度归 Simulation。当前流体不消费 Newton contacts。公开接口对未支持能力明确报错，不静默忽略调用者意图。

### 3.1 双 State 的设计逻辑

双 State 是本实现选择的输入／输出契约，不是“Newton 一律强制”或“为 Graph 才需要”。

一次子步从 A 读取开始状态，把下一时刻结果写入 B，结束后外层交换 A/B 的角色。这样 PBF 和 MuJoCo 可以从同一个时间起点读取，各自填写输出的对应字段；显示和外部调用通过 `Simulation.state/pos/v` 读取当前状态。

公开粒子状态的具体读写：

1. `apply_predict` 从输入位置速度计算预测位置，预测速度只是局部值，不改输入速度。
2. 网格排序后，`buffer_reorder` 直接把排序后的旧位置写进输出 `particle_q`，排序后的预测位置写进预测临时缓冲。
3. 压力迭代修改预测位置。
4. `update_position` 利用输出中暂存的排序旧位置重建速度，原地写最终输出位置及输出速度。
5. 涡量、黏性和最终速度更新继续操作输出速度。

每份 State 的数组绑定稳定，但数组槽位跟随空间排序变化。**槽位不是稳定粒子 ID**；不恢复原编号、不加稳定 ID 映射，也不增加每步完整粒子复制。重置与测试的复制不属于正常子步开销。

“输入只读”在这里指输入运动状态，不能扩大理解为整个输入 State 完全不可写：组合求解器会向输入 `body_f` 累加反力，供 MuJoCo 本子步读取。

### 3.2 配置与初始化

`PBFConfig` 位于 `pbf_helper.py`，检查参数类型、范围及关系。辅助函数负责派生参数、模型存储、工作缓冲与初始粒子生成，主类保持清晰调用。

初始位置由 GPU `init_particles` 生成到模型初始粒子数组，再创建两个 State，避免 CPU 重算改变浮点排列。初始化顺序还要保证 helper 模块首次编译所需的常量已经设置。

模块级常量用于固定配置的高效执行。粒子半径、拓扑和编译期流体系数变化通常需要新进程／重新构建实例。当前明确支持动态通知的是重力；帧长、子步长度等运行参数变化按 Graph 失效规则处理。

## 4. 算法调用层级

### 4.1 流体子步：保持 Prologue → Solver → Epilogue

```text
PBF.step(in, out, ..., dt)
  Prologue
    apply_predict
    grid.build
    pbf_helper.buffer_reorder
    cache_neighbors

  Solver：每次压力迭代
    volume_query             静态与动态边界密度／梯度贡献
    calc_lambda              流体邻域与汇总边界梯度求 lambda
    calc_deltaPos            流体与边界位置修正
    calc_boundaryForce       动态实例压力反力；仅动态场景发射
    update_prePos            应用修正，再做静态／动态 SDF 投影

  Epilogue
    update_position          输出位置、重建输出速度
    calc_curl
    calc_visvor
    update_velocity
```

### 4.2 耦合子步与一帧

```text
Simulation.step()
  → _advance_frame()
      每个子步：
        输入 State.clear_forces()（耦合场景）
        SolverPBFMuJoCo.step(in, out, control, contacts, dt)
          → step_fluid()
              同步输入刚体位姿到边界
              清本子步 boundary_forces
              PBF.step：写输出粒子，跨压力迭代累计反力
              将反力加到输入 body_f
          → SolverMuJoCo.step：写输出刚体／关节
          → 同步输出边界位姿
        交换 State 角色
  → sim_time += frame_dt（整帧只加一次）
```

`_advance_frame` 仅表达普通执行与 Graph 捕获共用的物理循环，`step` 在真实推进之后更新时间；不再加入只转发一步的中间层。Python 时间并不是 GPU Graph 中的运算。

PBF 只更新粒子字段，MuJoCo 只负责刚体／关节字段，不覆盖已写出的粒子。两者共用 Model、State 和 dt，压力迭代期间刚体位姿冻结。

外力责任必须明确：Simulation 当前每子步清输入外力；直接使用组合 solver 时，由调用者先清理并施加本子步外力。组合 solver 只加压力反力，不清除调用者已有外力。不要把帧外写一次 `body_f` 误当成 Simulation 会持续保留的力输入。

### 4.3 坐标、力与力矩

```text
world_from_asset = state.body_q[body_id] × body_from_asset
world_COM        = transform_point(state.body_q[body_id], model.body_com[body_id])

Δx_ib = -lambda_i × grad_ib
F_ib  = -mass_i × Δx_ib / dt²
τ_ib  = (equivalent_boundary_point - world_COM_b) × F_ib
```

- 内部实例索引与 Newton `body_id` 显式映射，不能依赖目录扫描顺序。
- `body_q` 是 body 原点的位姿，不应默认等同于 COM 位姿。
- 反力缓冲按 Newton body 索引排列，`wp.spatial_vector(force, torque)`，对应 `[force, torque]`。
- 每次压力迭代都贡献反力，全部累加，不除以迭代次数；下一子步先清零，避免重复施力。
- 等效施力点由当前边界查询得到。反力 kernel 在 `update_prePos` 之前读取与 delta 相同的未修正预测位置。
- SDF 投影是无反力反馈的穿透纠正；不能用“发生投影”代替“产生压力”的判据。压力作用包括表面外支持范围内的粒子。
- 这是压力显式弱耦合，不宣称经过投影、限速等处理后的全系统严格守恒。

## 5. 当前性能设计与不可随意破坏的条件

| 现用设计 | 解决的问题 | 保持条件 |
| --- | --- | --- |
| 空间排序 | 改善粒子数据局部性 | 保持槽位随排序变化，不恢复原编号 |
| 每子步一次有效邻域缓存 | 多个压力／速度 kernel 复用邻居名单 | 当前布局为 `[邻居槽位, 粒子]`，容量 192，保留容量保护 |
| 独立 `volume_query` | 将复杂体积查询与 lambda 邻域计算分开 | 输出 `volume_Rho/volume_Grad`，保持查询时刻和公式 |
| delta 两层预读 | 重叠邻居编号／位置读取与计算 | 保持当前累加顺序、预取索引保护，不凭直觉简化 |
| 运行时刚体循环 | 避免每个实例展开一份大型查询代码 | 密度、反力、投影循环使用运行时实例数，零刚体保留编译期分支 |
| 资产局部粗筛 | 跳过不可能产生作用的昂贵 SDF/VM 查询 | 扩大影响范围和插值余量，不能漏掉实体内部或临界粒子 |
| 独立 `calc_boundaryForce` | 将反力查询的寄存器负担与流体邻域 delta 分开 | 仍直接原子加法，时序在位置修正前，无额外粒子×刚体缓存 |
| 固定分配、固定模块参数 | 控制步进开销并支持 Graph | 正常步进不烘焙、不重建资产、不扫描粒子到 CPU |
| 自有模块关闭 backward | 减少当前不需要的反向代码生成 | 不全局改 Warp 配置，不修改第三方模块；不等同于整个依赖栈都无反向代码 |

邻域缓存是在子步开始的预测状态上构建的。同一子步后续移入支持半径的粒子不会自动补进名单，这是当前算法行为。邻域遍历成为热点不等于缓存设计无效，也不能仅凭某个平静状态的平均邻居数降低容量。

### 5.1 动态边界粗筛

边界模块统一处理每份资产，不按“球／盒子／复杂模型”分类选算法：

```text
粒子世界坐标 → 资产局部坐标 → 包围范围判断 → SDF → 有效压力条件 → VM
```

初始化从完整资产网格得到局部 AABB，以盒中心为球心、最远顶点距离为球半径；同一资产多个实例共享数据。`BOUNDARY_CULLING` 是边界模块代码常量，当前为 `"box"`，也保留 `"sphere"` 实现。无 CLI、配置项、关闭路径或按资产自动选择。

压力查询外扩支持半径，投影查询外扩粒子半径，二者再加一个 SDF 体素对角线的安全余量。局部盒随资产一起旋转，不需要每帧生成世界轴对齐盒；粒子的局部变换与后续 SDF 采样共用。

循环仍遍历各刚体，粗筛只是让大部分循环项提前跳过重查询。把代码放进 `wp.func` 本身不能保证编译器不展开循环；应检查实际循环边界和生成 CUDA 代码。动态投影必须查询经过前一个刚体纠正后的当前位置。

粗筛包围数据用于粒子—边界查询；MuJoCo 凸碰撞代理用于刚体接触，二者不互相替代。

### 5.2 压力反馈与访存

`volume_Grad` 是各边界贡献的汇总，不能直接用来区分每个刚体的受力。当前独立反力 kernel 逐动态实例重查贡献，向对应 body 的六维合力／力矩直接原子累加。没有粒子×刚体稠密缓存，也没有分层规约。

独立 kernel 的收益来自资源隔离，并非反力运算本身被消除。当前一次编译测量中 delta 使用 42 registers/thread，反力使用 72；这是特定设备／编译结果，不是跨版本保证。性能判断看完整流体子步与耦合帧，不能只看一个 kernel。

## 6. Graph、后端与生命周期

双 State 和预测位置双缓冲有各自的角色交换。CUDA Graph 记录具体 GPU 数组地址，不会在回放时重新执行 Python 赋值。

Simulation 按“输入 State 起点 × 预测缓冲起点”记录对应 Graph，并保存执行后的绑定状态。捕获期间 Python 会执行调度代码，捕获结束必须恢复这些引用及 MuJoCo 的 Python 步数；回放后再更新引用与计数。不能把“Graph 不自动执行 Python 交换”理解为“程序不能使用双缓冲”。

- GPU 后端：捕获完整耦合帧，帧后交换结果由 Simulation 管理。
- CPU 后端：捕获单个流体子步，CPU 刚体桥接与推进在 Graph 外执行，然后同步边界、交换 State，再进行下一子步。
- 重力、dt、子步数和反馈设置参与 Graph 失效判断。前端调用 `graph_needs_recapture()` 等接口，不读取子 solver 私有版本字段。
- 捕获不得推进当前运动状态、模拟时间或 MuJoCo 历史；对比 Graph 时还需恢复 warmstart 等内部历史。
- reset 恢复两份 State、工作缓冲、预测起点、时间、重力、帧长、刚体内部历史和反力，执行 FK 并同步边界。

刚体后端选择位于 `coupling.py` 的 `USE_MUJOCO_CPU`，在创建 solver 前确定；默认 `False`。没有后端 CLI，也不是任意刚体 solver 即插即用的注册机制。

当前 MuJoCo 设置为原生碰撞、每步同步、禁用 sleeping，GPU 容量 `nconmax=2048`、`njmax=8192`。流体粒子不作为 MuJoCo 碰撞对象。

CPU 路径只交换刚体相关状态、控制和力；流体粒子仍留在 GPU。完整帧还包含桥接、同步、临时分配与传输成本，不能用“少量刚体的 CPU 求解更快”直接推出耦合程序更快。当前测量中默认规模 GPU／CPU 加流体 Graph 约为 5.819／6.438 ms，约 49.4 万粒子为 11.907／12.967 ms；这些是同一状态、同步墙钟成对测量，不是通用后端性能承诺。

## 7. 渲染、CLI 与交互归属

CLI 与演示行为在 `app.py`。OpenGL 组件负责原始事件、键码转换、重复按键抑制、释放与失焦；应用注册动作，调用 Simulation 的重力、步长和 reset 接口。

| 操作 | 当前行为 |
| --- | --- |
| Space | 暂停／继续 |
| G | 重力反向 |
| Q / E | 选择水平重力方向，再次选择可恢复竖直方向 |
| X | 慢动作／正常速度 |
| R | 提交重置请求，由主循环安全执行，并暂停 |
| P | 网格显示切换 |
| 窗口／鼠标／相机操作 | renderer 处理底层操作 |

保持“先绘制，再推进”；首帧与暂停重置画面必须是未推进状态。暂停、退出和显示状态不在多个模块重复维护。renderer 每帧读取当前 State，不固定绑定某一份 State，也不增加逐帧粒子复制或 CPU 下载。

保留 billboard shader、速度配色、相机与显示参数。完整 CLI 见 `PBF/README.md` 和两个入口的 `--help`；OGL 与 USD 参数集不同，不能混用。

## 8. 独立离线资产管线

### 8.1 上层编排、下层功能

```text
Sim_Ready.pipelines.preset.pbf
  → 读取代码场景配方与显式参数，形成缓存键
  → 验证并复用构建缓存／发布产物，或完整生产
      container.prepare/build
        地形、兔子放置与对齐 → 布尔容器 → 空腔校验 → 兔子凸块 → SDF
      rigid.prepare/build
        局部实体 → 实体校验 → 按需凸分解 → SDF
  → 生成 OBJ、JSON 与碰撞附件
  → 调用 PBF VM 预计算入口
  → 验证整组产物
  → 暂存、发布到缓存和消费者目录；失败回滚
```

`container.py` 和 `rigid.py` 接收源数据、参数、暂存位置和设备，返回产物；不导入 PBF，不决定发布目录，不各自建立 CLI。PBF preset 才负责场景配方、缓存选择、VM、元数据和整组发布。

底层 `Sim_Ready` 独立维护，不依赖 demo 工具。程序化地形直接调用公开 `newton.Mesh.create_terrain()`，保留项目自身必要的坐标转换、浮点接缝对齐、墙体与碰撞 boxes 提取。不能把有空腔的整个容器转换为一个凸包。

场景描述目前在代码中，不另建场景 JSON。`scene_data.py` 只共享基础尺寸；实例的密度、初始位置和 body 装配在 `PBF/scene.py`。

### 8.2 输入、缓存与发布目录

| 位置 | 内容与规则 |
| --- | --- |
| `Sim_Ready/Assets/bunny_flat_bottom.obj` | 当前容器所需源模型，Git 保留 |
| `Sim_Ready/Assets/spot_triangulated_good.obj` | 保留在 Git 的源模型；当前 PBF 场景不使用它 |
| `.cache/sim-ready/pbf/` | 离线构建缓存及诊断，不是运行时资源 |
| `PBF/Assets/boundary/stairs_scene.*` | OBJ、NVDB、JSON、collision NPZ，共 4 个附件 |
| `PBF/Assets/rigid/stairs_box.*` | OBJ、NVDB、JSON；两个盒子共享 |
| `PBF/Assets/rigid/stairs_sphere.*` | OBJ、NVDB、JSON |
| `PBF/config/*.json` | 流体参数副本／记录，Git 保留，只有显式指定才读取 |

默认场景只需源兔子；盒子、球和地形均由代码生成。兔子文件须是约定的 flat-bottom 模型，SHA-256 为 `5c2fc17c5e851b02dd5503ac50bd6456dbb09959d90492a784c3481cef22d529`。来源和许可证信息见 `Sim_Ready/PROVENANCE.md`。

生成 JSON 不是用户配置：它保存资产格式版本、几何尺寸／种类、坐标约定、摘要、来源、生成参数，以及静态 boxes 等装配数据。由管线自动生成，不应手工维护或单独移配。

除两个明确放行的源 OBJ 外，源资产和全部发布资产忽略；`PBF/Assets/.gitignore` 保留。不要全局忽略 JSON。忽略规则不会取消已跟踪文件，不擅自替用户执行取消跟踪、暂存、提交或删除资产。

缓存键包含源内容、几何、烘焙、凸分解参数、算法／依赖版本与目录布局版本。流体密度、帧长、压力迭代数不应触发几何重建。SDF 身份与可变 VM 分开判断，不用整个 NVDB 文件摘要作为几何有效性的唯一依据。

发布期间停止消费者。所有附件先准备、验证再整组替换；I/O 失败回滚已替换文件。当前机制不是跨文件瞬时原子可见或崩溃恢复协议，不支持并发发布或热更新。

### 8.3 运行时仅加载，VM 失配例外

Simulation 初始化加载本包已发布资源并装配 Newton bodies/shapes。运行阶段不执行几何并集、SDF 烘焙、CoACD 或网格质量积分。

`boundary_model.load_boundary_volumes()` 检查支持半径、积分参数、VM 版本和必要网格信息：

- 命中：直接使用已加载 SDF/VM。
- VM 缺失或参数失配、且 SDF 支持足够：调用 `_precompute_volume_map`，将原 SDF 与新 VM 一起写回同一 NVDB，再加载确认。
- SDF 支持不足：明确报错，要求离线重新生成；运行时不扩大 SDF。

BoundarySystem 持有实际 Volume 生命周期及设备数据；这些资源并未作为 Newton 标准 Model 字段注册。离线 preset 可以复用同一 VM 预计算入口，避免维护两套公式。

默认支持半径 `h=4r=0.05 m`，静态 SDF 体素 0.0075 m，动态体素 0.005 m。离线按 `max(0.05, 4r)` 准备支持范围；在已有范围内改粒子半径通常只需更新 VM。

### 8.4 刚体碰撞、质量与局部坐标

当前动态预设：

| 实例 | 尺寸 | 密度 kg/m³ | 初始世界中心 |
| --- | --- | --- | --- |
| 盒子 A | 边长 0.30 m | 500 | `(2.10, 1.15, 0.55)` |
| 盒子 B | 边长 0.30 m | 1200 | `(2.45, 1.50, 0.55)` |
| 球 | 直径 0.30 m | 750 | `(2.45, 1.15, 1.02)` |

初始旋转为单位旋转，速度为零。基础形状由 Newton 原生 shape 的密度产生质量和惯量，摩擦 0.5、无额外反弹。重复资产共享网格和 Volume，但位姿、body、反力槽位独立。

静态接触由 24 个对齐 boxes 加兔子凸块构成，挂在 world（body 索引 -1），密度为零。兔子凸块只服务于 MuJoCo 的刚体—静态兔子碰撞；流体和渲染仍使用各自完整边界资源。纯流体 `static` 不消费这些凸块，但静态发布仍带完整碰撞附件。

CoACD 当前参数为 1.0.14、seed=0、实际尺度阈值 0.005 m、最多 32 块、每块最多 64 顶点，开启 merge 和 decimate。分解失败报错，不静默降级成整体凸包。

本机生成诊断显示 32 块、约 197.7 秒；当前参数下凸块合并是显著耗时段。CoACD 使用 CPU，合并有串行依赖和无进度输出区间，GPU 空闲不代表卡死。`Max concavity ... exceeds ... due to 32 convex hull limitation` 表示块数限制与目标精度冲突；日志的阈值是归一化量，不可直接当成米。此结果没有 5 mm 精度保证；当前诊断记录代理与原实体的相对体积对称差约 4.47%，它也不是最大表面误差。

普通准备命令复用验证通过的缓存；`--rebuild` 才明确完整重建，不应把它作为每次启动的固定前置步骤。

## 9. 代码结构、风格与验证责任

这些是协作约束，而非要求所有模块套同一模板：

1. **主流程清晰。** 主类显示初始化、算法阶段和调度；辅助计算用明确函数封装。保持主要 kernels 与 helper 的组织，不引入不必要的 kernel 工厂、通用后端框架、注册器或配置转换层。
2. **最小必要改动。** 一个需求先找责任所属模块，不为假设中的扩展重写整套系统。渲染、应用或资产任务不顺带改 solver 公式和优化路径；确需越过用户限定范围时先说明原因、等待决定。
3. **少参数但不隐藏依赖。** 固定参数在实例初始化时常量化，动态边界用少量 `wp.struct` 传递；不能为减少参数而改第三方全局配置或产生隐蔽跨实例状态。
4. **入口负责检查，内部信任契约。** PBFConfig 检查参数关系，JSON 负责解析／未知字段，资产生产负责几何完整性，运行加载负责格式、版本、变换规模和支持范围。避免在多层重复检查同一条件。
5. **保护有算法意义的运行时分支。** 保留邻域容量、预取索引、无效 SDF 梯度、有限值和支持范围保护。输入正数不自动保证积分后的浮点结果有效；不能把所有 `if` 都当成冗余。
6. **减少嵌套。** 对不满足条件的循环项优先使用 `if not valid: continue` 等清晰控制流，但必须保持相同查询、累加和修正顺序。
7. **稳态路径保持轻量。** 不做逐步全粒子有效性扫描、完整 VM tile 下载比较、资产读取或几何校验。完整检查放初始化、离线管线或测试。
8. **功能与参数稳定。** 不顺手改相机、粒子初始化、初始重叠、重力键、billboard、速度配色、压力次数或物理步长。不为新刚体挖空初始流体。
9. **接口与错误清楚。** 报错字符串使用英文。新增 CLI 必须是实际授权需求；粗筛和刚体后端当前都由代码选择，不能自动扩成用户配置系统。
10. **不碰无关文件。** 默认只改被授权的正式包、相关根目录测试和说明；demo、共享工具、资产及 Git 状态操作须遵守当次范围。不要覆盖用户未提交或已暂存的更改。
11. **说明以实现和证据为主。** 明确实测、代码推断和未验证内容；汇报影响、验收与限制，不用“符合 Newton”掩盖额外复杂度。

## 10. 默认参数与新电脑运行

环境：Python 3.11；Newton 1.7.0、Warp 1.18.0、MuJoCo/MuJoCo Warp 3.15.0。离线 CoACD 1.0.14。其余版本以两个 `requirements.txt` 为准；需要可用的 NVIDIA CUDA 环境。

无参数流体配置是 195,525 粒子、`55×79×45` 排列、半径 0.0125 m、帧长 `1/150 s`、3 子步、每子步 3 次压力迭代。初始粒子与地形的重叠完整保留。`container_size` 控制网格／初始化范围，不会重新生成地形。

`PBF/config/default_para.json` 只是显式可选的记录，其值不必等于无参数默认。不要在复现默认场景时擅自加载它。初始速度当前按 GPU 初始化的零速度行为执行，不应仅因 JSON 中存在 `initial_velocity` 字段就宣称实现任意初速度。

以下 PowerShell 命令从仓库根目录执行；`WarpSim` 是当前约定环境名，新机器也可在匹配依赖的其他环境中运行。

```powershell
conda activate WarpSim
python -m pip install -r PBF/requirements.txt -r Sim_Ready/requirements.txt

# 确认约定源模型已拉取；缺少时先补齐。
Test-Path Sim_Ready/Assets/bunny_flat_bottom.obj
Get-FileHash Sim_Ready/Assets/bunny_flat_bottom.obj -Algorithm SHA256

# 通常准备：复用有效资源，无可用资源时完整生成。
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids --particle-radius 0.0125

# 仅当要明确验证从源模型完整生成时使用；发布时关闭仿真。
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids --particle-radius 0.0125 --rebuild

Get-ChildItem PBF/Assets/boundary
Get-ChildItem PBF/Assets/rigid

# 流固耦合场景；一次生成的资源也能运行下面的 static。
python -m PBF.app --scene stairs-rigids --particle-radius 0.0125 --substeps 3 --pressure-iterations 3 --cuda-graph
python -m PBF.app --scene static --particle-radius 0.0125 --substeps 3 --pressure-iterations 3 --cuda-graph

# USD 输出，不需要打开 OGL 窗口。
python -m PBF.PBF --scene stairs-rigids --num-frames 30 --stage-path outputs/pbf-newton/stairs-rigids.usd
```

无需验证完整重建时，不必把两条资产准备命令都执行。只需要纯静态资源也可使用 `--scene static`。

约 50 万粒子配置是 `--particle-radius 0.0092`，当前排列为 494,100 粒子。先用同半径离线准备可避免运行启动时更新 VM。显式 `--config` 与半径等流体参数遵循现有互斥规则；查看 `--help`，不要推断所有入口支持相同参数。

## 11. 验收与性能分析标准

### 11.1 功能与数值

```powershell
python -B -m unittest discover -s tests/pbf -t . -v
python -B -m unittest tests.sim_ready.test_pbf_assets -v
python -B -m tests.pbf.verify_coupled --seconds 10 --output outputs/pbf-newton/acceptance/trajectory.json
python -B -m tests.pbf.verify_viewer --scene static --headless --output outputs/pbf-newton/acceptance/viewer-static
python -B -m tests.pbf.verify_viewer --scene stairs-rigids --headless --output outputs/pbf-newton/acceptance/viewer-coupled
```

`--headless` viewer 测试仍创建真实隐藏 OpenGL 上下文，执行脚本化按键和截图，并检查 USD；不是用假 renderer 替代画面。需要人工窗口检查时移除该参数。

验收重点：

- 静态参考独立加载资源，默认规模连续 30 子步逐位比较运动状态、压力／速度工作数组和邻域计数，不以粒子匹配或宽松容差替代。
- 验证输入运动状态只读、公开数组绑定稳定，无原编号恢复或额外每步完整复制。
- 验证动态实例映射、旋转、局部偏移、偏心 COM、共享资产独立位姿、反力方向／力矩／dt² 比例／迭代累计和外力保留。
- 碰撞检查实际接触对和运动响应，不只检查能否运行；连续运行检查有限状态、明显穿透／逃逸及容量溢出。
- Graph 覆盖奇偶子步、State／预测起点、重力、慢动作、reset 和接触状态，恢复完整 MuJoCo 历史。动态原子累加按预先明确的容差比较，不能承诺所有轨迹逐位相同。
- 资产迁移验证网格、SDF、VM、碰撞、坐标、质量惯量；重新烘焙的 SDF 数据可另外进行采样误差验证，不能把全文件摘要差异直接判成算法回归。
- 缓存命中、VM 失配写回再命中、支持不足、附件缺失、发布失败回滚和输出隔离都要有覆盖。

正式资产管线的主要测试入口是 `tests.sim_ready.test_pbf_assets`；该测试目录还包含其他工具测试，不能把整个目录的所有测试都自动视为本包独立验收。

逐位参考测试 `tests/pbf/reference.py` 还需要 `demos/fluid/PBF` 的独立参考代码和参考 OBJ/NVDB，测试会私有复制资源后读取。这是测试依赖，不是正式运行的资产来源。新电脑只有正式源模型时可以生成并运行正式场景，但不能据此宣称具备全部逐位参考测试条件；缺少参考资源应补齐或明确报告该项未运行，不能让参考和被测 solver 共用同一份 BoundarySystem 来掩盖差异。

仅源模型重建与运行隔离检查可用：

```powershell
python -B -m tests.sim_ready.verify_clean_workspace
python -B -m tests.sim_ready.verify_build --rebuild
```

它们涉及昂贵的离线生成，先读脚本的输出路径和行为，不作为每次小修改的固定前置。

当前验收记录位于 `outputs/pbf-newton/asset-layout/acceptance/`：23 项 PBF 测试和 15 项资产测试通过；195,525 粒子、3 刚体运行 1,500 帧／10 模拟秒，无非有限状态、逃逸或容量溢出；两个场景均通过 OGL 与 USD 验证。此记录是明确配置下的验证，不是任意资产／规模保证。本地 `outputs` 被忽略，新电脑应重跑，而不是假定报告存在。

### 11.2 性能与 profiling

- 完整物理帧默认包含 3 个子步，每子步 3 次压力迭代。`1000 / 帧耗时毫秒` 是纯求解等效 FPS，不是窗口 FPS，也不是 `1 / frame_dt`。
- 区分目标 kernel、流体子步、完整耦合帧、带渲染墙钟四种口径；不要直接混算。
- 性能 A/B 恢复同一粒子状态、两份 State、预测缓冲和完整 MuJoCo 历史。恢复复制在计时区间外；预热充分，交替或随机成对采样，并补连续演化测量。
- 静态完整子步相对独立参考的回退目标不超过 5%；统计波动、设备时钟和轨迹变化都需解释，不能只挑最快一次。
- 当前约 49.4 万粒子的代表性 NSYS 连续窗口约 13 ms／完整耦合帧，主要热点是 delta、邻域缓存构建、lambda 及速度邻域计算；边界查询与反力仍有成本。该值不是当前任何互动状态的保证。
- 激烈运动会改变局部密度、候选邻居数量、缓存命中与线程分歧；仅凭粒子总数不能预测帧耗时。不能把未采样的掉帧直接归因于反力原子竞争或刚体后端。
- 先用 NSYS 看时间线、kernel 占比和同步，再针对当前热点用 NCU。NSYS 本身不能证明寄存器溢出或缓存失效率。
- NCU 采样后确认真正生成 `.ncu-rep`；仿真 JSON 和 `Disconnected` 日志不等于采样报告有效。Windows 优先给可直接运行的命令，不依赖被执行策略拦截的 `.ps1`，不为 profiling 随意修改系统权限。
- 不并行运行 GPU benchmark 与其他仿真／显示测试。冷缓存测试使用独立缓存，不清用户现有缓存，并说明是否清理驱动缓存。

## 12. 接手时的工作顺序

1. 读取用户本次目标与允许范围，检查 `git status`，区分已有修改与本次工作。
2. 从 `app.py → simulation.py → PBF.py/coupling.py` 理解运行；资产任务从 `preset/pbf.py → container.py/rigid.py` 理解生产。
3. 确认依赖、两个源模型、发布目录布局，以及无参数配置与显式 JSON 的区别。
4. 对照本文的数据所有权和流程不变量，确定最少需要修改的模块；不要把领域细节混进应用循环或渲染器。
5. 完成与改动直接相关的验证，只有失败、新变更或明确风险才扩大／重复测试；不要把未跑的验收写成通过。
6. 将报告写到 `outputs/pbf-newton/`，在说明中保留足以复现的命令、规模、版本与结论。交付前核对生产代码、保护文件和用户资产没有被测试意外改写。

本文维护原则：描述当前状态，不叠加补丁说明；实现变化后直接修订对应职责、命令和约束，避免读者需要从多段互相矛盾的文字中推断最终行为。
