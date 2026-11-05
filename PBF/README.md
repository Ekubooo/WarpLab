# PBF — minimal static Newton adaptation

在仓库根目录运行，默认已经激活安装了 Newton 的 Conda 环境：

```powershell
python -m PBF.render_opengl --particle-radius 0.0125 --substeps 3 --pressure-iterations 3
```

使用 CUDA Graph 时，在同一命令后加 `--cuda-graph`：

```powershell
python -m PBF.render_opengl --particle-radius 0.0125 --substeps 3 --pressure-iterations 3 --cuda-graph
```

This independent package starts from `demos/fluid/PBF`, retaining its major
kernels and `Prologue → Solver → Epilogue` flow. It uses Newton 1.7.0 Model/State
and SolverBase, with no MuJoCo or dynamic rigid-body coupling in this stage.
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
- `simulation.py`: static assets, Model/two States, frame loop, Graph, reset and
  USD rendering. `render_opengl.py` retains the existing independent frontend.
- `boundary_model.py`: the original boundary/volume-map implementation, pointed
  at this package's own `Assets`. Only the static stairs asset is loaded.

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
refer to the current State. Read pressure scratch through `simulation.solver`.

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
contacts are outside this static adapter's contract. Unsupported optional reset
and observable requests fail explicitly. This is not full Newton feature coverage.

Production dependencies on existing demos are limited to the unchanged public
`demos.fluid.common.billboard_renderer` and `demos.sim_Ready.nvdb_io`. Keep those
packages alongside this one. Own simulation modules disable backward generation;
shared and third-party source/configuration is not modified.

Static OBJ/NVDB assets were copied and verified locally. They remain local ignored
resources, following repository conventions. Runtime volume-map refresh may write
only the new package's NVDB. No offline producer is added in this stage; future
geometry work must publish/copy explicitly into this package rather than change
the original demo's resources.

## Verification

```powershell
python -B -m unittest discover -s tests/pbf -t . -v
python -B -m tests.pbf.benchmark
python -B -m tests.pbf.verify_viewer
python -B -m tests.pbf.verify_integrity
```

The tests require CUDA and the local published assets. Each numerical test
process creates only one production solver instance. Test-only reference imports
read the original demo; its assets are privately copied before loading. Integrity
verification uses the pre-change demo-tree snapshot at
`outputs/pbf-newton/demos-before.json` (override with `--baseline`).

Validation on RTX 5070 / WarpSim, 2026-11-05:

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
