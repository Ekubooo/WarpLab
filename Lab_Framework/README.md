# Independent lab applications

These applications use the Newton-compatible PBF solver from `Lab_Solver`.
Existing programs under `demos` remain independent snapshots.

Run commands from the repository root in the `WarpSim` environment. No editable
installation is needed. `requirements-newton.txt` records the Newton 1.7.0,
Warp 1.18.0 and MuJoCo/MuJoCo Warp 3.15.0 validation environment.

## Prepare assets

Source OBJ inputs stay local and are excluded from Git. Supply the required
models in `Lab_Framework/sim_ready/Assets` after cloning the repository. Build
the default container before starting PBF:

```powershell
conda run -n WarpSim --no-capture-output python -m Lab_Framework.sim_ready --obj bunny_flat_bottom.obj --name stairs_scene --device cuda:0
```

The input directory is `sim_ready/Assets`. Bare OBJ names resolve there; relative
paths containing a directory resolve against the working directory. The pipeline
uses Newton's public `Mesh.create_terrain()` API and retains the existing union,
container validation and CUDA SDF baking.

Container outputs are saved under `sim_ready/output` and published to `pbf/Assets`.
Rigid outputs use local `output/rigid` and published `pbf/Assets/Rigid`:

```powershell
conda run -n WarpSim --no-capture-output python -m Lab_Framework.sim_ready.rigid_pipeline --obj spot_triangulated_good.obj --name rigid_spot --device cuda:0
```

Rigid inputs preserve their local origin and axes unless an explicit transform is
requested. The output is a matching OBJ and single-grid SDF NVDB. Publication
protects the input source and rolls back handled publication failures. Existing
output names are replaced. PBF adds or refreshes the volume map in the same new
NVDB; no original demo cache is read or modified.

## Run PBF

```powershell
conda run -n WarpSim --no-capture-output python -m Lab_Framework.pbf --device cuda:0
conda run -n WarpSim --no-capture-output python -m Lab_Framework.pbf --device cuda:0 --headless --num-frames 2
conda run -n WarpSim --no-capture-output python -m Lab_Framework.pbf --device cuda:0 --headless --num-frames 2 --cuda-graph
```

`--stage-path scene.usd` optionally exports particles during headless execution.
`--config PATH` accepts the original flat PBF JSON fields. `--export-config NAME`
writes a complete preset into the new application's `config` directory and exits.
Help and config export do not initialize CUDA or load assets.

The billboard frontend retains speed coloring, camera controls, particle count,
pause, reset, gravity controls and slow motion from the original application.
Keyboard controls include G (reverse gravity), Q/E (select X gravity), X (slow
motion), R (reset), and P (toggle grid). Particle arrays are read from the current
Newton State each frame. Each State retains its array bindings, while the
simulation exchanges its two States and particles remain spatially sorted.

## Tank and box prototype

```powershell
python -m Lab_Framework.pbf --scene tank-box --device cuda:0
python -m Lab_Framework.pbf --scene tank-box --device cuda:0 --cuda-graph
python -m Lab_Framework.pbf --scene tank-box --headless --num-frames 300 --cuda-graph
python -m Lab_Framework.pbf --scene tank-box --headless --num-frames 30 --stage-path output/tank.usd
```

The default `static` scene remains unchanged. The optional `tank-box` scene uses
one unified Newton Model and two States, with fluid-first pressure-feedback weak
coupling and native MuJoCo Warp rigid collisions. The open one-meter tank has
0.1 m walls; the 0.24 m box starts at (0.5, 0.75, 0.5), with half the fluid rest
density. `config/tank_box.json` is the matching fluid preset. CLI fluid overrides
apply over this preset; an explicit `--config` retains the existing JSON semantics.

Tank/box meshes and SDFs are created on first use in `.cache/pbf/tank-box/` using
the existing solid validation and bake utilities. Geometry, voxel size and
support radius select the cache. Existing stairs/Spot assets are not changed.
First-run volume integration and MuJoCo kernel compilation can take substantially
longer than subsequent starts. Mesh poses update on the GPU in the existing GL
renderer; optional USD exports include the tank and moving box.
The tank preset starts with an elevated view through its open top. Camera
controls and the original static scene's initial viewpoint remain unchanged.

Gravity keys affect both fluid and rigid bodies, and slow motion changes their
common dt. Reset restores both States and the MuJoCo history before rendering the
initial state. Graph capture covers the whole coupled frame; the frontend
recaptures after gravity/time-step changes.

This scene deliberately establishes the physical coupling before imported
assets are enabled. Follow-up work is to connect sim-ready OBJ/NVDB assets to
Newton collision proxies, explicit mass/inertia and body-local transforms;
represent complex static containers with valid rigid collision geometry; then
support multi-link assets and check overlap/double-counting between boundaries.
The existing asset pipeline remains available, but discovery alone does not
create a dynamic body. Fixed particle IDs, multi-world RL and general
SolverCoupled integration are outside this prototype.

Validation and timing commands (CUDA required):

```powershell
python -m unittest discover -s tests/lab_solver -t . -v
python -m unittest discover -s tests/lab_framework -t . -v
python -m tests.lab_solver.benchmark_migration --output outputs/pbf-mujoco-coupling/performance.json
python -m tests.lab_framework.benchmark_coupling
```

The 2026-11-03 validation passed 27 solver tests and 66 framework tests, including
OpenGL camera controls, free fall/contact, floating-box feedback, off-center
pressure rotation, shared-State ownership and Graph/reset lifecycle. Default
tank rendering and USD motion export were also checked. See
[`Lab_Solver/README.md`](../Lab_Solver/README.md#validation) for measured static
regression and coupled timings.
