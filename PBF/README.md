# PBF — minimal Newton adaptation and pressure-coupled rigid bodies

在仓库根目录运行，默认已经激活安装了 Newton 的 Conda 环境：

```powershell
python -m PBF.render_opengl --particle-radius 0.0125 --substeps 3 --pressure-iterations 3
```

使用 CUDA Graph 时，在同一命令后加 `--cuda-graph`：

```powershell
python -m PBF.render_opengl --particle-radius 0.0125 --substeps 3 --pressure-iterations 3 --cuda-graph
```

原阶梯兔子场景加入两个盒子与一个球：

```powershell
python -m PBF.render_opengl --scene stairs-rigids --particle-radius 0.0125 --substeps 3 --pressure-iterations 3 --cuda-graph
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
python -m PBF.render_opengl
python -m PBF.render_opengl --cuda-graph
python -m PBF.PBF --num-frames 30 --stage-path outputs/pbf-newton/static.usd
python -m PBF.render_opengl --config PBF/config/default_para.json
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

Direct `python PBF/render_opengl.py` and `python PBF/PBF.py` entrypoints also work.
Configuration export remains local to `PBF/config` via `--export-config NAME`.

## CLI 参数速查

两个入口的参数并不完全相同。完整帮助：

```powershell
python -m PBF.render_opengl --help
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

以下参数用于 `python -m PBF.render_opengl`。默认值指**无参数启动**，不是已保存 JSON 中的值。

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
- `simulation.py`: scene assembly, Model/two States, frame loop, Graph, reset and
  USD rendering. `render_opengl.py` retains the existing independent frontend.
- `boundary_model.py`: SDF/VM loading and caching, shared asset instances,
  body mapping, local transforms and GPU boundary/COM synchronization.
- `scene.py`: explicit published stairs collision and three rigid instances.
  `coupling.py`: one PBF → pressure reaction → MuJoCo substep.

```text
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
`demos.fluid.common.billboard_renderer` and `demos.sim_Ready.nvdb_io`. Keep those
packages alongside this one. Own simulation modules disable backward generation;
shared and third-party source/configuration is not modified.

Published assets were copied and verified locally; NVDB files remain local ignored
resources, following repository conventions. Runtime volume-map refresh may write
only the new package's NVDB. No offline producer is added in this stage; future
geometry work must publish/copy explicitly into this package rather than change
the original demo's resources.

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

MuJoCo runs on GPU with native contacts, update interval 1 and sleeping disabled.
Contact/constraint capacities are 2048/8192. Graph capture preserves the MuJoCo
Python step counter; replay advances it. Full reset also restores backend history
and FK. Particle fields and rigid/joint fields have separate solver ownership.

Assets under `Assets/Rigid/stairs_box.*`, `stairs_sphere.*`, and static collision
attachments are independent copies of the previously verified published assets.
There is no runtime archive import, geometry baking, CoACD invocation or mass
integration. The original static OBJ/NVDB were not replaced. Asset/SDF identity,
convexity and complete tile equality are checked in tests, not per startup.
Dynamic SDFs have 0.005 m voxels and support `h <= 0.05 m` (`r <= 0.0125 m`).
Changing particle radius refreshes VM in this package only; exceeding SDF support
requires offline rebaking and replacing the complete matching published asset.
This version consumes published assets and does not add a general publisher.

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
