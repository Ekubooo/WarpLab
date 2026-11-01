# PBF 独立求解器与 Newton 接入计划

本文记录截至 2026-11-01 的设计讨论，作为后续整理仓库和接入 Newton 的依据。本文描述的是计划，不代表相应重构或耦合功能已经实现。

## 1. 目标与已确定的边界

- 保留当前所有既有 demos 的代码、运行入口与历史实现，作为独立程序和归档，不通过修改旧 demo 完成新架构。
- 在根目录新增 `solver`，从现有 PBF 复制一次实现基线，整理出统一维护的流体核心。
- 新增独立运行程序，调用该核心，保留流体单独运行、渲染和交互的能力。
- Newton 适配代码调用同一份核心，为刚体耦合、机器人和未来父仓库提供接入方式。
- 初期留在当前 Git 仓库，不立即拆分仓库，不要求 editable 安装，不同时迁移整个资产管线。
- 最小目标是跑通一个简单刚体的创建、运动、碰撞、显示和边界位姿同步，再逐步实现流体双向耦合。

未来希望观察 MimicKit 预训练策略控制的机器人在流体环境中的表现。初期只要求策略推理和演示，不要求完整接入其训练框架。

## 2. 当前已经实现到哪里

### 2.1 SimReady 独立刚体资产流程

已有 `demos/sim_Ready/rigid_pipeline.py`，入口函数为 `build_rigid_asset()`。

```text
输入 OBJ
→ 显式缩放／坐标轴变换
→ 封闭部件校验
→ 布尔并集
→ 并集结果和 float32 校验
→ 保存 OBJ
→ 烘焙局部 SDF
→ 保存单网格 NVDB
→ 发布同名 OBJ／NVDB
```

支持一个刚体由多个封闭部件组成；默认保留模型原点、坐标轴与尺寸，不自动居中或排列实例。mesh 与 SDF 使用相同资产局部坐标。

刚体流程中的 `--obj`、`--name` 必填。裸 OBJ 文件名从 `demos/sim_Ready/Assets` 读取；带目录的相对路径相对于当前工作目录，绝对路径直接使用。

当前目录约定：

| 用途 | 当前仓库内路径 |
| --- | --- |
| 原始模型输入 | `demos/sim_Ready/Assets` |
| 刚体本地产物 | `demos/sim_Ready/output/rigid/NAME.obj`、`NAME.nvdb` |
| 刚体发布产物 | `demos/fluid/PBF/Assets/Rigid/NAME.obj`、`NAME.nvdb` |

注意发布目录为首字母大写的 `Rigid`，本地输出目录仍为小写 `rigid`。容器继续使用原来的输出和发布位置。

从仓库根目录生成 Spot 刚体的现有命令：

```powershell
conda run -n WarpSim --no-capture-output python -m demos.sim_Ready.rigid_pipeline --obj spot_triangulated_good.obj --name rigid_spot --device cuda:0
```

默认体素间距为 `0.0075 m`，最大流体支撑半径为 `0.05 m`。需要其他尺寸或上限时显式传入对应参数。

当前发布会保护输入源文件，防止输出覆盖输入；对于已有同名产物则默认备份后覆盖，失败时回滚，不提供同名确认或拒绝覆盖流程。重新发布会用单网格 SDF 替换已有 NVDB，消费者下次加载时重新补齐体积图。本文不改变这些行为。

### 2.2 BoundarySystem 与统一加载

当前 `boundary_model.py` 已提供：

- `BoundaryScene`、`BoundaryLoadOptions`：场景选择与底层加载参数。
- `BoundaryAsset`：OBJ／NVDB 路径，以及 SDF／体积图的强引用。
- `BoundaryInstance`：资产编号、静态／动态类型及内部刚体编号。
- `BoundaryDeviceData`：GPU 资源句柄、实例映射和变换数组。
- `BoundarySystem`：共享资源、实例管理、位姿更新和原位重置。
- 按实例变换采样 SDF／体积图的 GPU 辅助函数。

PBF 当前通过以下入口获得系统：

```python
self.BoundarySys = boundary_model.load_default_boundaries(self.smoothing_length)
```

设备沿用当前 Warp 设备。静态文件名和刚体数量配置位于边界模块；动态资产通过非递归扫描 `Assets/Rigid` 中的 OBJ、配对同名 NVDB 发现，默认每份资产创建一个实例。数量为零时跳过加载，多实例共享 Volume。

体积图仍由 PBF 边界功能计算，并缓存到所读取的同一份 NVDB。缺少缓存或缓存参数不匹配时重算。

当前动态实例尚未参与 PBF 压力、碰撞投影、反作用力、刚体动力学或动态 mesh 渲染。现有流体 kernel 仍查询默认静态容器。内部刚体编号也尚未映射到 Newton `body_id`。

## 3. 新代码的组织方式

以下为拟议结构，名称和拆分粒度可随最小实现调整，避免预先建立空框架：

```text
WarpLab/
├─ demos/
│  ├─ fluid/PBF/                    既有程序，保留原样
│  ├─ fluid/PBF_Origin/             历史实现，保留原样
│  ├─ fluid/PBF_next/               拟新增独立运行程序
│  │  ├─ main.py                    启动与场景选择
│  │  ├─ simulation.py              创建、子步调度、时间与重置
│  │  └─ render_opengl.py           渲染与交互
│  └─ sim_Ready/                    既有资产流程，继续保留
├─ solver/
│  ├─ __init__.py
│  ├─ pbf/                         可独立调用的流体核心
│  │  ├─ solver.py                 PBF 推进与子步
│  │  ├─ kernels.py                流体 GPU 算法
│  │  ├─ boundary_model.py         边界资源、实例与查询
│  │  └─ ...                       按实际需要整理配置、状态和 NVDB 工具
│  └─ adapters/newton/              可选的 Newton 接入代码
│     ├─ rigid.py                  刚体构建、状态与推进
│     ├─ coupling.py               映射、位姿与反馈交换
│     └─ solver_pbf.py             后续正式的 SolverBase 适配
└─ tests/                           新核心与接入功能的测试
```

旧 demo 只作为保留的实现快照；复制基线后，新功能集中维护在新核心，不要求回补旧版本。

### 3.1 依赖方向

```text
新独立程序 ──────────────→ PBF 核心
      └─ 可选 Newton 接入 → Newton 适配层 → PBF 核心

未来父仓库 ──────────────→ PBF 核心／Newton 适配层
```

- 核心不反向导入 demos，不依赖渲染、GUI、MimicKit 或 Newton。
- Newton 适配层依赖核心与 Newton；普通 PBF 导入不触发 Newton 导入。
- 运行程序管理场景、资产路径和外层循环。
- 必要的通用 NVDB 工具应整理为新核心可独立使用的代码，避免新核心通过 `demos.sim_Ready` 反向获取实现；既有管线本身不必同步修改。

仅把调用旧 demo 的包装代码放进 `solver`，还不能让这个目录在未来独立使用。真正的核心实现和必要依赖需要逐步收拢进去。

## 4. PBF 核心需要整理的内容

核心压力、邻域、体积图等算法可以复用，主要调整状态归属与调用边界：

1. 从现有每帧 `step()` 中提取流体子步，明确 `dt` 的归属，避免外层和内部重复划分子步。
2. 将资源句柄、实例相关数据和可变参数归属到实例或显式设备数据，梳理当前模块全局的 SDF／体积图／HashGrid 句柄，避免不同实例互相覆盖。
3. 明确配置、状态、工作缓冲、边界输入、重置与反馈输出的接口，避免大量零散参数和返回值。
4. 将独立运行入口、文件选择和渲染交互留在新程序侧。
5. 梳理位置缓冲交换与 CUDA Graph 的关系；重建资源或改变布局后重新捕获。

这些工作按接入需要逐步进行，不要求先完成通用引擎或一次消除所有历史实现细节。

## 5. 两个层次的 Newton 接入

### 5.1 第一阶段：应用层组合，先跑通

新程序持有 PBF 与 Newton 刚体系统，通过少量适配函数同步数据、分别推进。流体和刚体可以暂时持有各自的状态。

这一阶段可以使用普通类组织，不要求立即继承 Newton `SolverBase`。此时是 Newton 接入 demo，不能据此宣称已经完整兼容 Newton 自定义求解器规范。

第一步可以仅运行 Newton 简单刚体，不修改或调用原有 PBF 程序。

### 5.2 后续阶段：正式自定义求解器

Newton 自定义求解器通常继承 `newton.solvers.SolverBase`，实现：

```python
step(state_in, state_out, control, contacts, dt)
```

适配层将 Newton 状态与 PBF 状态关联，调用 PBF 核心，并正确写回输出状态。根据实际能力实现重置、模型变更通知、自定义属性或可观测量支持。

不能只继承一个类、忽略输入输出状态，再把原有整帧 `step()` 包进去，就认为标准适配完成。需要明确：

- 哪些粒子和刚体由谁推进。
- 哪些数组共享、哪些复制，以及状态交换后引用是否有效。
- 输入的 `dt` 如何传到流体子步。
- 边界位姿与反作用输出如何交换。
- 重置、CUDA Graph 和不同实例的资源生命周期。

PBF 核心继续维护算法；Newton 相关转换留在适配层。

## 6. 外层推进与双向耦合

新独立程序的 `Simulation` 管理帧时间和子步循环，持有流体与刚体系统。PBF 只负责流体推进，Newton 负责刚体动力学、关节和相应碰撞。

初步采用每个物理子步交换一次数据的显式耦合安排：

```text
Simulation.step()
└─ 每个物理子步
   ├─ 读取 Newton 当前状态，更新 BoundarySys 位姿
   ├─ 更新世界包围范围及查询候选
   ├─ 清零流体反作用累计量和本步外力输入
   ├─ PBF 流体子步
   │  ├─ 预测、邻域建立
   │  ├─ 压力迭代：查询边界、修正位置、累计反馈
   │  └─ 更新流体速度
   ├─ 将反馈写入 Newton 外力输入
   ├─ 刚体碰撞与 Newton 推进
   └─ 交换刚体状态，更新边界位姿
└─ 更新时间并供渲染读取
```

这是拟采用的初始时序，后续根据具体耦合算法和稳定性验证调整，不表示已实现强耦合或迭代耦合。

需要明确的关键数据：

- 边界动态实例编号与 Newton `body_id` 的映射。
- 资产局部坐标与 Newton 刚体局部坐标的固定变换；保留资产原点不代表原点就是质心。
- 由 Newton 持有的权威刚体状态，以及供边界查询使用的同步变换。
- 每个刚体的流体反馈与质心力矩。

Newton `State.body_f` 接受世界坐标下的力与质心力矩，顺序为前三维力、后三维力矩。若流体累计的是冲量，应按对应累计时间区间转换为力，不能混用帧 `dt` 与子步 `dt`。反馈应在清零之后写入，并避免重复施加或重复推进刚体。

GPU 流体与 GPU 刚体路径优先在设备端交换数据。将来选择 CPU 刚体路径时，再集中处理必要的反馈与状态传输；不在 PBF kernel 内分散 CPU／GPU 后端判断。

## 7. 资产管线与 Newton 的职责

已有动态刚体烘焙功能继续保留，不因新增框架而删除或重做。

| 部分 | 职责 |
| --- | --- |
| SimReady | 几何处理、封闭性等校验、局部 SDF 烘焙、配对产物发布 |
| 上层场景／Newton 接入 | 选择资产和实例数量，构建质量惯量、关节、碰撞形状和初始状态 |
| PBF 边界模块 | 加载和共享 Volume，计算体积图，按实例位姿查询 |

未来机器人流程可以按“解析机器人模型 → 提取各肢体几何和固定变换 → 调用烘焙功能 → 建立 body 映射”扩展。具有独立运动变换的肢体需要对应边界表示；相同局部几何可以共享资产，不按实例重复烘焙。

Newton 刚体碰撞表示与 PBF 的流体 SDF 分别管理，不能假设现有 NVDB 会自动成为 Newton 碰撞体。当前容器渲染 OBJ 是无墙景观，也不能直接视作完整容器碰撞几何。

新框架初期直接使用已烘焙产物；由新程序指定路径，使用单独的工作副本或运行缓存位置，避免体积图回写改变归档资产。既有资产管线的默认输出暂时不变。

待机器人与整体场景需求明确后，再扩展管线入口、发布目的地和机器人资产组织。烘焙算法仍归资产功能模块，Newton 层负责调用与映射。

## 8. 渲染与 GUI

- 保留当前渲染实现作为可复用基线，在新程序中接入，不修改旧前端。
- 粒子继续读取流体位置和速度数组；新增刚体 mesh 加载与实例变换更新。
- Newton 提供 `ViewerGL` 等 viewer，也可以作为后续可选前端；使用 Newton 求解器不强制切换渲染器。
- Newton Viewer 的 GUI 可以通过 `register_ui_callback()` 扩展，自定义控件放在项目侧。

GUI 只负责展示状态和提交请求，实际操作由 `Simulation` 在适当时机调用对应模块：

```text
GUI 请求 → Simulation 校验与执行 → PBF／边界模块／Newton／策略模块
```

区分可在运行时更新的参数与需要重建模型、缓冲或 CUDA Graph 的操作。初期不实现运行中任意增删资产，不在 UI 绘制回调内直接重建求解器。

## 9. MimicKit 预训练策略演示

最小目标是复用机器人配置和训练好的策略，而不是立即整合整个 MimicKit 项目：

```text
Newton 机器人状态
→ MimicKit 兼容观测与归一化
→ 策略推理
→ 动作还原与关节控制
→ Newton／PBF 耦合推进
```

需要对齐机器人关节与刚体顺序、观测定义、归一化统计、动作缩放、PD 参数、控制频率、坐标约定，以及策略所需的相位或参考动作输入。共用 Newton 并不自动保证旧策略或旧引擎代码兼容当前版本。

先在无流体条件下复现策略，再加入流体，区分接入错误与环境变化。陆地策略在流体中失稳也是可观察的迁移结果；适应流体的训练或微调另行规划。

## 10. SolverCoupled 的定位

Newton 的实验性多求解器框架包括：

- `SolverCoupled`：共享模型视图、对象归属、状态分发与汇总、各求解器子步等公共机制。
- `SolverCoupledProxy`：通过代理对象和交错／滞后反馈组织耦合。
- `SolverCoupledADMM`：基于约束的 ADMM 迭代耦合。

它不会自动识别 PBF 的体积图边界，也不会自动实现反作用力。使用它还需要适配共享 Model／State、对象归属和所选耦合方式需要的接口。

当前将其作为后续方向。先通过应用层组合验证 PBF 与 Newton 的交换关系，再评估迁入该框架的收益。

## 11. 分阶段实施与验收

| 阶段 | 最小工作 | 验收重点 |
| --- | --- | --- |
| 0：保存基线 | 确认旧 demo 的归档边界，复制新核心基线，建立新增入口 | 旧代码不改，新核心不依赖 demos |
| 1：独立流体 | 新程序调用新 PBF 核心，明确子步接口 | 静态场景独立运行、渲染、重置正常 |
| 2：Newton 简单刚体 | 创建方块与地面，推进并显示，再使用 Spot | 刚体运动、碰撞、状态交换正常 |
| 3：位姿同步 | 建立 body 映射和局部固定变换，更新 BoundarySys | mesh、SDF 与刚体位姿一致 |
| 4：单向流体作用 | 包围范围／候选查询、动态边界体积贡献与投影 | 固定或指定运动的刚体能影响流体 |
| 5：双向反馈 | 累计反作用冲量与角冲量，转换并传给 Newton | 反馈方向、时间尺度、质心力矩与刚体响应正确 |
| 6：标准求解器适配 | 实现完整 SolverBase 状态契约和必要生命周期接口 | 外部调用和重置正确，独立 PBF 仍可运行 |
| 7：机器人策略 | 机器人资产映射、观测／控制与预训练策略推理 | 无流体基线正常，再展示流体影响 |

阶段 2 的纯刚体验证可以提前独立开展。每阶段只实现当前需要的功能，避免同时展开全部重构、GUI、资产迁移和训练接入。

验证使用 WarpSim，生成资产、发布回滚和缓存测试使用临时目录。除必要的接口与行为测试外，不为纯文件整理增加重复测试。

## 12. 当前不做与未来分仓库

当前不要求：

- 修改既有 demos 或将其全部改用新核心。
- 立即拆仓库、发布 Python 包或配置 editable 安装。
- 立即完整迁移 SimReady、MimicKit、渲染器或 GUI。
- 首版就支持多环境 RL、运行中任意拓扑变化或所有 CPU／GPU 后端。

未来需要独立使用 `solver` 时，再提供包元数据、依赖声明和稳定入口。父仓库可以依赖其固定版本或 Git commit，验证后更新版本同步。

editable 安装是开发时让 Python 指向本地源码目录，修改后下次运行即可使用；它不是当前同仓库最小 demo 的前置要求，也不会代替 Git 提交或远程同步。

## 13. 官方参考

- [Newton 1.7 SolverBase 接口](https://newton-physics.github.io/newton/1.7.0/api/_generated/newton.solvers.SolverBase.html)
- [Newton State：位姿、速度和质心外力约定](https://newton-physics.github.io/newton/latest/api/_generated/newton.State.html)
- [Newton 1.7 MPM 双向耦合示例](https://github.com/newton-physics/newton/blob/v1.7.0/newton/examples/mpm/example_mpm_twoway_coupling.py)
- [Newton 实验性 Coupled Solvers](https://newton-physics.github.io/newton/latest/concepts/coupling.html)
- [Newton Viewers 与可视化](https://newton-physics.github.io/newton/stable/guide/visualization.html)
- [MimicKit 官方仓库](https://github.com/xbpeng/MimicKit)
- [pip 本地项目与 editable 安装说明](https://pip.pypa.io/en/latest/topics/local-project-installs/)

Newton 参考中带 `latest`／`stable` 的页面可能随版本更新。正式接入时以 WarpSim 实际安装版本的源码和验证结果为准。

## 14. 最新约束：保留现有性能路径，暂不实现固定粒子编号兼容

本节记录后续讨论确定的性能约束；与前文泛化的状态适配设想冲突时，以本节为准。

- 保留当前 PBF 的内部排序、邻域缓存、grid 查询快路径和 GPU 推进顺序。
- 不为 Newton 接口新增粒子编号映射缓冲、逆映射查询、恢复初始排列、全数组复制或额外重排 kernel。
- 保留现有 `query.cell_index - 1` 的内部排序编号获取方式；不能因接口整理而替换成更慢的逐邻居映射。所依赖的 Warp 版本行为通过测试确认。
- 当前阶段不承诺粒子数组槽位对应固定的跨步身份；关联的逐粒子数据必须保持一致。外部按固定粒子 ID 施力、附着、约束或跟踪的通用兼容性暂不实现，作为待验证问题保留。
- Newton 的通用 SolverBase 接口没有自动核验跨步粒子身份的机制。未触发形状、设备等检查，不代表外部索引引用在语义上正确；不能以“不报错”作为完整兼容的验收依据。
- 当前流体到 Newton／MuJoCo 刚体的反馈按 `body_id` 累计合力和质心力矩，不要求流体按初始粒子 ID 排列。该目标不应成为恢复流体顺序的理由。
- 允许通过接口和资源组织方式提取求解器，但 State 缓冲引用、原位／非原位推进及 CUDA Graph 生命周期必须另外核对；不得默认通过额外复制、恢复编号等方式解决这些问题。
- 性能目标为不回退。对原版与提取版使用同配置、同资产、同设备，预热后比较邻域构建、压力迭代和完整推进时间。受测量波动影响，不能在实现和基准前宣称绝对零开销；发现回退时先定位并消除适配成本，不自行放宽性能目标。

当前决定是先保留该兼容性问题，不为其改动已有计算流程，也不将首版标记为已完整适配任意 Newton 粒子约束或 SolverCoupled 场景。

## 15. 本轮实施结果：静态 PBF 已迁移

实际包名按磁盘目录统一为 `Lab_Framework`（不是 `Lab_FrameWork`），求解器包为 `Lab_Solver`。以下记录取代前文尚未定型的目录及适配阶段设想。

```text
Lab_Framework/
├─ pbf/          Newton Model/State 场景构建、帧推进、配置、billboard、交互
└─ sim_ready/    容器与独立刚体两条离线资产流程

Lab_Solver/
├─ pbf/          SolverPBF、编译期特化 kernel/helper、BoundarySystem
└─ nvdb_io.py    资产流程和边界缓存共用的独立 NVDB 工具
```

- 主代理实现求解器、Newton 接口、边界迁移和验收；一个子代理实现框架、渲染及资产流程。旧 demos 和旧资产均未修改。
- `SolverPBF` 继承 Newton `SolverBase`，一次 `step(state, state, None, None, dt)` 推进一个子步。首版只支持原位 State，明确拒绝不同输入／输出 State；不以复制补齐非原位能力。
- State 零拷贝绑定原 PBF 缓冲。原有空间重排交换引用后同步 `state.particle_q`；渲染等消费者每次获取当前引用。没有新增粒子 ID 缓冲、恢复原顺序、逐邻居逆映射或 GPU 全状态复制。
- 原有 kernel 数量、调用顺序、邻居缓存布局、grid sorted-slot 快路径及方程保留。每个实例在创建时独立特化常量和资源句柄，避免不同实例之间的模块全局污染。
- 框架管理帧内子步、时间、重置、交互与图捕获。CUDA Graph 沿用双缓冲奇偶地址捕获／回放，不使用奇数子步后的全状态复制；时间步或重力改变时要求重新捕获。
- 固定粒子 ID、逐粒子外力／约束、原生 Newton Contacts、自动微分和 SolverCoupled 不属于首版支持范围。声明了标准接口及明确能力边界，不宣称所有 Newton 工作流已兼容。
- 新框架默认仅加载静态容器。动态刚体烘焙、资产管理与变换查询已作为既有功能迁移，但没有新增动态刚体动力学、流体查询耦合或反作用。
- 输入只复制原始 OBJ，新地形使用公开 `newton.Mesh.create_terrain()`。默认容器已通过新管线完整重新烘焙到新目录，未复制旧 NVDB；体积图只回写新目录读取的 NVDB。
- 新框架保留原有 flat JSON 字段和默认值。其初始化按旧 GPU FMA 的 float32 算术生成相同点阵；Newton Model 的初速度使用 `initial_velocity` 字段（默认仍为零）。

已验证的规模与结果：

- 默认 `195,525` 粒子：初始化及连续 5 子步的 9 组状态／工作缓冲与原版逐位一致。
- 精确核对原 kernel 启动序列；原位引用、实例隔离、缓存、重置和奇偶子步 CUDA Graph 均有回归验证。
- 已实际执行新入口的无窗口、CUDA Graph、billboard OpenGL 两帧运行。
- 原有 PBF 81 项、原有 SimReady 41 项、新求解器与框架 76 项回归通过，共 198 项；同时实际验证了可选 USD 导出。
- 预热后 60 轮交错 CUDA 事件采样，邻域／压力／完整子步的新旧中位数分别为 `275.10/271.82 μs`、`675.57/677.87 μs`、`1178.77/1187.47 μs`。各阶段配对比值的 95% 区间包含 1，未测出稳定回退；不据此宣称任意配置绝对零开销。
- 基准脚本为 `python -m tests.lab_solver.benchmark_migration`；完整采样记录生成在 `outputs/newton-pbf-migration/performance.json`，代码与测量说明见新增两个包的 README。

运行新程序：

```powershell
conda run -n WarpSim --no-capture-output python -m Lab_Framework.pbf --device cuda:0
conda run -n WarpSim --no-capture-output python -m Lab_Framework.pbf --device cuda:0 --headless --num-frames 2
conda run -n WarpSim --no-capture-output python -m Lab_Framework.pbf --device cuda:0 --headless --num-frames 2 --cuda-graph
```

重新生成默认资产：

```powershell
conda run -n WarpSim --no-capture-output python -m Lab_Framework.sim_ready --obj bunny_flat_bottom.obj --name stairs_scene --device cuda:0
```

下一阶段在该结构上增加动态边界参与查询，再实现按刚体累计反馈的弱耦合；不为刚体引擎的反馈要求流体恢复固定粒子 ID。
