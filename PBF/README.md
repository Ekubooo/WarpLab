# PBF — minimal Newton adaptation and pressure-coupled rigid bodies

在仓库根目录运行，默认已经激活安装了 Newton 的 Conda 环境：

```powershell
python -m PBF.app --particle-radius 0.0125 --substeps 3 --pressure-iterations 3
```

使用 CUDA Graph 时，在同一命令后加 `--cuda-graph`：

```powershell
python -m PBF.app --particle-radius 0.0125 --substeps 3 --pressure-iterations 3 --cuda-graph
```

原阶梯兔子场景加入两个盒子与一个球：

```powershell
python -m PBF.app --scene stairs-rigids --particle-radius 0.0125 --substeps 3 --pressure-iterations 3 --cuda-graph
python -m PBF.PBF --scene stairs-rigids --num-frames 30 --stage-path outputs/pbf-newton/stairs-rigids.usd
```

This independent package starts from `demos/fluid/PBF`, retaining its major
kernels and `Prologue → Solver → Epilogue` flow. It uses Newton 1.7.0 Model/State
and SolverBase. The optional stairs-rigids scene uses MuJoCo Warp 3.15.0 for
native rigid contacts and explicit pressure-feedback coupling.
`tests/pbf/source_manifest.json` (repository root) records the source commit and original file hashes.
The archived `demos/PBF_Rigid` is not a runtime or test dependency.

## Run

From the repository root, in the existing WarpSim environment:

```powershell
python -m PBF.app
python -m PBF.app --cuda-graph
python -m PBF.PBF --num-frames 30 --stage-path outputs/pbf-newton/static.usd
python -m PBF.app --config PBF/config/default_para.json
```

No-argument defaults remain **195,525 particles**, a `55×79×45` lattice,
0.0125 m particle radius, `1/150 s` per frame, three substeps and three pressure
iterations. Initial overlapping particles are retained. The copied JSON preset
has its own original values: passing it is deliberately different from running
without arguments. Initialization and reset preserve the original zero velocity
behavior; the existing `initial_velocity` JSON field is not newly implemented.

The original billboard shader, velocity color mapping, camera and controls are
retained: Space pauses, G reverses gravity, Q/E select horizontal gravity (repeat
to restore vertical gravity), X changes speed, R resets and pauses, P toggles
the grid. Newton Viewer/UI is not used.

Direct `python PBF/app.py` and `python PBF/PBF.py` entrypoints also work.
Configuration export remains local to `PBF/config` via `--export-config NAME`.

## CLI 参数速查

两个入口的参数并不完全相同。完整帮助：

```powershell
python -m PBF.app --help
python -m PBF.PBF --help
```

### 两个入口共有的参数

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `-h`, `--help` | — | 显示帮助并退出 |
| `--device DEVICE` | Warp 自动选择 | 设备，例如 `cuda:0`；当前体积图路径需要 CUDA |
| `--scene static\|stairs-rigids` | `static` | 原静态场景，或同规模流体加三个自由刚体；两个入口均支持 |
| `--num-frames N` | OpenGL：`0`；USD：`480` | OpenGL 的 `0` 表示直到关闭窗口；USD 入口推进 N 帧，`0` 不推进 |
| `--verbose` | 关闭 | 输出更详细的计时信息 |
| `--config PATH` | 不加载 JSON | 从指定 JSON 文件读取流体配置 |
| `--export-config [NAME]` | 不导出；省略 NAME 时为 `default_para.json` | 将选定配置写到 `PBF/config/NAME` 后退出；同名文件会被覆盖 |
| `--clamp-negative-pressure` / `--no-clamp-negative-pressure` | 开启 | 开启或关闭负密度约束钳制 |

### OpenGL 入口的流体参数

以下参数用于 `python -m PBF.app`。默认值指**无参数启动**，不是已保存 JSON 中的值。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--particle-radius R` | `0.0125` m | 粒子半径；间距为 `2R`，支持半径为 `4R` |
| `--substeps N` | `3` | 每帧物理子步数 |
| `--pressure-iterations N` | `3` | 每个子步的压力迭代次数 |
| `--frame-dt DT` | `0.006666666666666667` s | 每帧模拟时间；子步长度为 `DT / substeps` |
| `--rest-density RHO` | `1000.0` kg/m³ | 参考密度，用于粒子质量和流体系数 |
| `--lambda-regularization ALPHA` | `2.0` | lambda 正则化系数，实际 epsilon 为 `ALPHA / h²` |
| `--container-size X Y Z` | `3.0 5.0 2.0` m | 网格范围及初始化范围检查；不会重新生成阶梯几何 |
| `--block-start X Y Z` | `0.1 0.5 0.1` m | 初始粒子块下界 |
| `--block-end X Y Z` | `1.5 2.5 1.25` m | 初始粒子块上界 |

`--config` 与显式流体配置参数互斥，包括压力钳制开关；设备、帧数、配色和 Graph 等运行参数仍可与 JSON 配合使用。黏性、涡量等未提供 CLI 开关的配置通过 JSON 设置。

### OpenGL 绘制参数与 USD 输出参数

| 入口 | 参数 | 默认值 | 用途 |
| --- | --- | --- | --- |
| OpenGL | `--cuda-graph` | 关闭 | 使用整帧 CUDA Graph 推进 |
| OpenGL | `--speed-color-mid SPEED` | `2.0` m/s | 速度配色中间色对应的速度 |
| OpenGL | `--speed-color-max SPEED` | `6.5` m/s | 速度配色最高色对应的速度；必须大于中间值，不改变物理限速 |
| USD | `--stage-path PATH` | `example_pbf.usd` | USD 输出路径；传入 `None` 可关闭输出，仅推进模拟 |

USD 入口 `python -m PBF.PBF` 不支持上面的粒子半径、子步数等 OpenGL 专属参数；需要改动时通过 `--config` 传入。其 `--stage-path` 参数不适用于 OpenGL 入口。

## Responsibilities and data flow

- `PBF.py`: main kernels and `PBF(SolverBase)`. `step(in, out, control, contacts, dt)`
  advances exactly one substep; no frame clock, renderer or asset discovery.
- `pbf_helper.py`: original parameter calculations and math, with separate scene,
  model, particle initialization and solver work-buffer helpers.
- `app.py`: standalone CLI, initial camera, playback loop and demo commands.
  The OpenGL and USD entrypoints retain their separate argument sets.
- `simulation.py`: the Newton application runtime owns scene assembly,
  Model/two States, Control, solvers, substeps, Graph, reset and a renderer.
  `set_renderer()` initializes/replaces it; `render()` delegates display/export;
  `close()` releases it. Rendering never advances physical state.
- `renderers/opengl.py`: existing billboard implementation, mesh registration,
  velocity coloring, window events and native key translation.
  `renderers/usd.py`: existing USD frame output and save-on-close behavior.
  Both implement `initialize(simulation)`, `render(simulation)` and `close()`;
  there is no renderer registry or common abstract base class.
- `app.py` is the OpenGL module/direct-script entrypoint; the old
  `render_opengl.py` forwarding file has been removed. The tail of `PBF.py`
  retains the USD entrypoint, forwarding to `app.main_usd()`. CLI parameters
  and defaults are unchanged.
- `boundary_model.py`: SDF/VM loading and caching, shared asset instances,
  body mapping, local transforms and GPU boundary/COM synchronization.
- `scene.py`: explicit published stairs collision and three rigid instances.
  `coupling.py`: one PBF → pressure reaction → MuJoCo substep.

```text
app.main (OpenGL) / app.main_usd (USD)
  create Simulation → attach renderer → register demo commands (OpenGL)
  loop: pending reset → Simulation.render → step / launch_graph if playing
  finally: Simulation.close

Simulation initialization
  load static boundary → allocate Model → initialize solver/grid constants
  → original GPU particle initialization → create two States

Simulation.step
  repeat substeps:
    PBF.step(in, out, ..., dt)
      predict → hash grid → reorder → neighbor cache
      repeat pressure iterations: boundary query → lambda → correction → SDF
      reconstruct velocity → curl → viscosity/vorticity → final velocity
    swap States
  advance frame time once
```

The application defines what G/Q/E/X/R/P do and registers callbacks through
`Simulation.register_keyboard_callback()`. OpenGL handles key codes, repeat
suppression, release and focus loss. Pause/grid/window state remains in the
renderer; reset requests remain deferred to the application loop. The first
frame is drawn before physics, including a paused reset. Each draw reads the
current State rather than retaining one side of the double buffer.

Help/config export imports no display component. A `Simulation` without
`stage_path` starts with no renderer; `stage_path` still attaches USD output.
Replacing a renderer closes the old one, and repeated close is safe. USD has
no keyboard interface. Future Newton GL/RTX components can use the same runtime;
MimicKit and additional viewers are not introduced by this refactor.

Application/rendering regression artifacts and before/after timings:
`outputs/pbf-newton/app-render-refactor/REPORT.md`.

Warp compiles the helper module as a unit, including reorder. Its grid constant
therefore must exist before launching the particle initialization kernel. Initial
positions are still calculated by the original GPU kernel, not a CPU substitute.

The solver owns prediction buffers, pressure scratch and neighbor cache. Input
motion arrays are read-only. Reorder writes sorted old positions directly into
`out.particle_q`; reconstruction then updates that buffer and `out.particle_qd`.
State array bindings stay fixed; slots follow spatial ordering, with no stable ID
restoration or per-step full-particle copying. `Simulation.state/pos/v` always
refer to the current State. Read pressure scratch through `simulation.fluid_solver`
(the same object as `simulation.solver` for the static scene).

Graph capture handles both State and prediction-buffer parities and does not
advance physical state/time. Replay automatically recaptures after changes to
gravity, frame/substep duration or substep count. Model gravity notifications
refresh the solver's cached launch argument outside the normal substep.

## Supported contract and dependencies

One fixed PBF configuration and solver instance per process, as in the original
module-constant implementation. Model particles must use the uniform radius and
mass implied by this configuration, in one global scene. Changes to topology,
mass, radius or compile-time fluid coefficients require a fresh process; no
automatic rebuilding or multi-instance kernel factory is introduced.

Input/output States must be distinct with non-aliasing arrays. The solver updates
only particle positions/velocities, leaving unrelated fields untouched. Complete
particle reset and gravity notification are supported. Selective reset masks or
flags, observables, differentiation, generic particle forces and Newton collision
contacts are outside the fluid solver's contract. Unsupported optional reset
and observable requests fail explicitly. This is not full Newton feature coverage.

Production dependencies on existing demos are limited to the unchanged public
`demos.fluid.common.billboard_renderer`. NVDB I/O is maintained independently
in the root `Sim_Ready` package. Own simulation modules disable backward generation;
shared and third-party source/configuration is not modified.

Published assets were copied and verified locally; NVDB files remain local ignored
resources, following repository conventions. Runtime volume-map refresh may write
only the new package's NVDB. The independent `Sim_Ready` package produces and
publishes this scene offline; demo and archive resources remain unchanged.

## 离线资产准备

正式离线工具位于根目录 `Sim_Ready`，具体流程与参数见 `Sim_Ready/README.md`。
首次准备或资源缺失时，在停止仿真后运行：

```powershell
python -m pip install -r Sim_Ready/requirements.txt
python -m Sim_Ready.pipelines.preset.pbf --scene static
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids
```

新电脑确认 `Sim_Ready/Assets/bunny_flat_bottom.obj` 已拉取；该源模型与
`spot_triangulated_good.obj` 保留在 Git，全部生成附件忽略。所需源文件摘要、依赖安装和完整还原步骤
见 `Sim_Ready/README.md`。当前场景无需 Spot；盒子、球与地形由代码生成。
场景配方仍在代码中，流体参数 JSON 保留为副本／记录，不默认读取。

发布目录分为 `PBF/Assets/boundary/stairs_scene.*`（静态容器与碰撞附件）和
`PBF/Assets/rigid/stairs_box.*`、`stairs_sphere.*`（动态边界）。
一次 `stairs-rigids` 准备即可同时供静态、耦合场景使用。仅从源模型验证完整生成时：

```powershell
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids --particle-radius 0.0125 --rebuild
python -m PBF.app --scene stairs-rigids --particle-radius 0.0125 --substeps 3 --pressure-iterations 3 --cuda-graph
```

`preset/pbf.py` 统一管理场景、缓存和发布，分别调用 `container.py` 与 `rigid.py`
完成两类边界资产的处理；底层不创建 Simulation，也不自行发布到运行目录。

默认验证复用现有正式资产或独立缓存；缺少可复用产物时从源模型完整生成。
需要更大的 SDF 支持半径时，使用与仿真一致的配置或半径：

```powershell
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids --particle-radius 0.013
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids --config PBF/config/default_para.json
# 完整重建先发布到独立目录，便于和现有基准比较。
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids --rebuild --output-dir outputs/pbf-newton/rebuilt-assets
```

`--config` 与 `--particle-radius` 互斥。现有默认动态资产支持 `h <= 0.05 m`；
较小半径仍由运行时检查 VM、必要时重算写回，不需要重烘焙 SDF。
源几何、SDF、碰撞代理及质量属性的生产留在离线阶段。`scene_data.py`
提供生产与装配共用的基础尺寸，不改变实例、流体参数或运行配置格式。

## Rigid scene and coupling

The original 195,525 particles, overlaps, camera and fluid settings are unchanged.
Two 0.30 m boxes (500/1200 kg/m³) start at `(2.10,1.15,0.55)` and
`(2.45,1.50,0.55)`; a 0.30 m sphere (750 kg/m³) starts at `(2.45,1.15,1.02)`.
Both boxes share one SDF/VM and mesh, but have independent bodies and forces.
Static collision uses 24 aligned boxes plus published bunny convex pieces;
the original boolean mesh remains the rendered and fluid boundary geometry.

Each coupled substep clears external forces in Simulation, freezes input rigid
poses during PBF, adds accumulated pressure forces to input `body_f`, advances
MuJoCo into the same output State, synchronizes boundaries, and swaps States.
Direct users of the composed solver clear external forces before adding their
own force inputs; the solver adds reactions without discarding those inputs.
`Simulation(..., feedback=False)` provides the one-way validation mode.

For each iteration, the reaction is `-mass * boundary_delta_x / dt²`; torque is
about the world COM at the equivalent boundary point. Reactions sum across all
pressure iterations. SDF penetration projection, velocity clamping and vorticity
do not generate compensating forces: this is explicit pressure weak coupling,
not a claim of strict conservation for the entire simulation.

Pressure iteration order is `volume_query → calc_lambda → calc_deltaPos →
calc_boundaryForce → update_prePos`. The separate reaction kernel reads the
same uncorrected prediction as delta, uses direct atomic accumulation, and is
not launched for static scenes. Delta retains its two-level neighbor prefetch.
The frozen pre-split kernel is kept only in tests for numerical/performance
comparison (494,100 particles, private asset copy):

```powershell
python -B -m tests.pbf.benchmark_feedback --warmup 30 --trials 60
python -B -m tests.pbf.benchmark_feedback --warmup 180 --trials 60
```

MuJoCo defaults to GPU with native contacts, update interval 1 and sleeping disabled.
Contact/constraint capacities are 2048/8192. Graph capture preserves the MuJoCo
Python step counter; replay advances it. Full reset also restores backend history
and FK. Particle fields and rigid/joint fields have separate solver ownership.

To select native MuJoCo CPU, set `USE_MUJOCO_CPU = True` in `PBF/coupling.py`
**before creating Simulation**. There is no backend CLI option; `False` retains
the existing GPU path. Model, both States, particles and pressure accumulation
remain on GPU. Newton transfers rigid state/control/wrenches to CPU and returns
the integrated rigid state; it does not transfer the fluid particle arrays.

The existing `--cuda-graph` option keeps its meaning: GPU mode captures a complete
frame; CPU mode captures one fluid substep for each State/prediction binding,
then performs CPU rigid integration outside capture, synchronizes boundaries,
swaps State and repeats. Time advances once per frame. Omitting `--cuda-graph` uses
ordinary stepping with either backend. Changing dt/gravity or feedback rebuilds
the graphs as before; reset clears native CPU history as well as both States.

CPU/GPU tests and full-frame wall-time comparison (test-only backend options):

```powershell
python -B -m tests.pbf.coupled_lifecycle_case --cpu --output outputs/pbf-newton/cpu-backend/lifecycle-cpu.json
python -B -m tests.pbf.verify_viewer --cpu --scene stairs-rigids --headless --output outputs/pbf-newton/cpu-backend/viewer
python -B -m tests.pbf.benchmark_backends --radius .0125 --warmup 180 --trials 60 --output outputs/pbf-newton/cpu-backend/performance-195k.json
python -B -m tests.pbf.benchmark_backends --radius .0092 --warmup 180 --trials 60 --output outputs/pbf-newton/cpu-backend/performance-494k.json
```

This initial CPU path uses Newton's existing bridge, including its temporary
allocations and synchronous transfers. It is not assumed faster for few bodies;
measure the complete coupled frame, not only native `mj_step`.

Assets under `Assets/rigid/stairs_box.*`, `stairs_sphere.*`, and
`Assets/boundary/stairs_scene.*` are independently published by Sim_Ready.
There is no runtime archive import, geometry baking, CoACD invocation or mass
integration. Asset/SDF identity,
convexity and complete tile equality are checked in tests, not per startup.
Dynamic SDFs have 0.005 m voxels and support `h <= 0.05 m` (`r <= 0.0125 m`).
Changing particle radius refreshes VM in this package only; exceeding SDF support
requires offline rebaking and replacing the complete matching published asset.
The dedicated offline PBF pipeline is `Sim_Ready.pipelines.preset.pbf`; it is not a
general asset registry. Geometry processing never runs during simulation startup.

Validation responsibilities: PBFConfig owns parameter types/ranges; JSON owns
parsing and unknown fields; public asset/precompute entries check inputs once.
The old mode-based validation context and full GPU tile downloads were removed.
Runtime neighbor capacity, prefetch bounds, SDF validity and numerical branches
remain. `_substep` was merged into `_advance_frame`; frame timing stays outside
the shared physics loop.

## Verification

```powershell
python -B -m unittest discover -s tests/pbf -t . -v
python -B -m tests.pbf.benchmark
python -B -m tests.pbf.verify_viewer
python -B -m tests.pbf.verify_integrity
python -B -m tests.pbf.coupled_lifecycle_case
python -B -m tests.pbf.verify_coupled --seconds 10
python -B -m tests.pbf.verify_viewer --scene stairs-rigids --output outputs/pbf-newton/viewer-coupled
```

### Current acceptance — 2026-11-06, RTX 5070 / WarpSim

- **17 tests passed**, including isolated pressure, rigid-contact and coupled
  lifecycle processes. Static 30-substep motion/work arrays/neighbors remain
  bitwise equal to the independent original demo; input arrays and bindings
  remain unchanged. The original 1,411 initial overlaps are retained.
- Twelve contact-state Graph combinations (1/2/3 substeps × State parity ×
  prediction parity), gravity/slow motion and full reset passed. Capture leaves
  148 GPU arrays unchanged, including nested MuJoCo contact and warmstart arrays.
  Backend step count and time reset to zero. Dynamic comparison tolerances are
  position 2e-5 m, velocity 2e-4, relative 2e-4; pressure wrench uses absolute
  2e-3, relative 2e-4. Observed particle maxima were 2.41e-7 m and 7.16e-5 m/s.
  Dynamic particle ordering may change near sorting boundaries, so tests use
  bijective spatial matching; static regression never does.
- Controlled tests verify body mapping, translated/rotated assets, local offsets,
  eccentric COM, reaction impulse/torque, iteration accumulation, dt² scaling,
  field ownership and preserved caller forces. With identical one-substep inputs,
  feedback leaves the particle result identical and changes rigid velocity.
- Free fall and box–box, sphere–box, stairs, wall and bunny contacts passed with
  actual contact pairs and motion responses; capacities are checked every rigid
  test substep, not inferred from a successful launch.
- Full-scale coupled run: **1,500 frames / 10 simulated seconds**, finite sampled
  particle/rigid states, no detected escape or capacity overflow. Peak contacts
  **55**, constraints **32**, broadphase pairs **68**. Minimum sampled native
  contact separation was **−0.01143 m**; this is compliant discrete contact, not
  a zero-penetration guarantee. Peak sampled pressure force was **7395.15 N**.
- Static paired CUDA events, 60 samples: original **1178.59 µs**, current
  **1174.98 µs**, ratio **0.9969**, paired bootstrap 95% interval
  **[0.9943, 1.0120]**. No regression beyond the 5% target. Full coupled-frame
  median was **7.516 ms**, excluding the first 60 frames and CPU diagnostics.
- Both real OpenGL scenes completed 150 frames with first-frame, G/Q/E/X/R/P,
  paused reset and USD verification; coupled USD contains terrain and all three
  bodies. Screenshots are in `viewer-static-final/` and `viewer-coupled/`.
- **210 protected demo/archive files unchanged** against this change's snapshot.
  The earlier snapshot had two already-changed demo VM caches before this task;
  normalized SDF identity is unchanged. Formal static OBJ/NVDB were not replaced.

Reports in `outputs/pbf-newton/`: `final-tests.log`, `asset-migration.json`,
`rigid-collisions.json`, `pressure.json`, `coupled-lifecycle.json`,
`coupled-trajectory.json`, `performance-coupled-static.json` and `integrity.json`.
Run GPU timing scripts without other concurrent GPU tests or rendering.

The tests require CUDA and the local published assets. Each numerical test
process creates only one production solver instance. Test-only reference imports
read the original demo; its assets are privately copied before loading. Integrity
verification uses the pre-change demo-tree snapshot at
`outputs/pbf-newton/coupling-demos-before.json` (override with `--baseline`).
Older NVDB whole-file hashes may differ because VM caches are mutable; the
integrity check compares static SDF payload identity separately and verifies
the protected demo tree against the snapshot taken before this change.

Historical static-adapter validation on RTX 5070 / WarpSim, 2026-11-05:

- Six full-scale numerical/lifecycle tests and two CLI tests passed. Motion,
  prediction, pressure/velocity scratch and neighbor counts match the original
  bit for bit across 30 substeps; 1,411 initial SDF overlaps are preserved.
- Input State is unchanged and both States retain array bindings. Eager/Graph
  results match bit for bit for 1/2/3 substeps, both State indices and prediction
  buffer parities, including gravity changes, slow motion and reset.
- Visible real OpenGL ran 150 frames with scripted G/Q/E/X/R/P controls. First
  display and paused reset show the unadvanced initial state. Screenshots and USD
  are in `outputs/pbf-newton/viewer/`. USD retains the original renderer's 60 fps
  integer time-code rounding, so not every 150 Hz simulation frame is a separate
  USD time sample.
- Paired CUDA-event benchmark, 60 samples after warm-up: original **1,175.46 µs**
  vs new **1,179.31 µs** per full substep, ratio **1.0033**; paired bootstrap 95%
  interval **[0.9907, 1.0034]**. Restore copies are outside timing events; no
  observed regression beyond the 5% target.
- Original and archive demo tree: **249 files unchanged**. Copied static assets
  and configuration hashes match the original.
- Generated CUDA audit found six new-package sources and no backward kernel
  entrypoints; shared and third-party generated code is outside this audit.

Detailed reports are under `outputs/pbf-newton/`: `parity.json`, `performance.json`,
`static-tests.log`, `entry-tests.log`, `integrity.json`, `forward-only.json` and
viewer outputs. The module USD CLI also completed 30 frames; the copied JSON
preset completed a separate direct-script smoke run.

### Dynamic boundary query optimization

The three dynamic-body loops use the runtime body count, retaining the compiled
zero-body static path. Dynamic pressure and projection queries first test finite
mesh bounds in asset coordinates; surviving points continue through the original
SDF/volume-map equations. Projection tests the position after each preceding
body correction. No pressure-query cache or force-reduction changes are included;
the two-level delta-position prefetch remains unchanged.

`BOUNDARY_CULLING` near the top of `boundary_model.py` selects `"box"` (default)
or `"sphere"` for **all** dynamic assets. This is a compile-time code choice:
edit it and restart. There is no CLI/config/Simulation option, disabled mode or
asset-type dispatch. Both bounds come from each complete asset mesh, read once
and shared with rendering. Bounds expand by the query reach plus one SDF voxel
diagonal. No particle-by-body buffers or per-step host updates are added.

Reproduce the paired comparisons without changing production choices:

```powershell
python -m tests.pbf.benchmark_boundary --warmup 30 --trials 60
python -m tests.pbf.benchmark_boundary --warmup 180 --trials 60
python -m unittest tests.pbf.test_boundary_culling tests.pbf.test_static tests.pbf.test_coupled_lifecycle tests.pbf.test_pressure
python -m tests.pbf.benchmark --trials 60 --output outputs/pbf-newton/optimization-round1/static-performance.json
```

The test-only reference retains the pre-optimization query kernels. Frozen-state
timings restore both States, fluid scratch and full MuJoCo history outside CUDA
events; all variants start from the reference trajectory. Continuous timings use
three 60-frame runs per variant. Run these measurements without concurrent GPU
tests or rendering. Results and generated test variants are saved under
`outputs/pbf-newton/optimization-round1/`; see its `REPORT.md` for numerical
checks, timing variability and the default-selection evidence.
