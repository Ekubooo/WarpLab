# Independent lab applications

These applications use the Newton-compatible PBF solver from `Lab_Solver`.
Existing programs under `demos` remain independent snapshots.

Run commands from the repository root in the `WarpSim` environment. No editable
installation is needed. Newton 1.7.0 and Warp 1.18.0 are the validation baseline.

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
Newton State each frame because spatial sorting can exchange their bindings.

## Current scope

The application initializes one static boundary and advances one homogeneous
fluid in place using `solver.step(state, state, ..., dt)` for each substep.
The spatial sorting and GPU algorithm remain unchanged. Public particle ordering
can change: fixed particle-ID interoperability, out-of-place stepping and
SolverCoupled compatibility are not claimed.

The rigid asset pipeline is available, but this application does not yet simulate
or render rigid dynamics, query dynamic bodies in the pressure solver, or return
fluid reaction forces. Robot policies and GUI extensions remain later work.
