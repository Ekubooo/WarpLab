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

The tank scene remains the small coupling reference. Imported solid geometry
and multiple free bodies are exercised by the stairs scene below. Multi-link
assets, fixed particle IDs, multi-world RL and general SolverCoupled integration
remain outside this prototype.

## Original stairs with four rigid bodies

```powershell
python -m Lab_Framework.pbf
python -m Lab_Framework.pbf --scene stairs-rigids --cuda-graph
python -m Lab_Framework.pbf --scene stairs-rigids --headless --num-frames 1500 --cuda-graph
python -m Lab_Framework.pbf --scene stairs-rigids --headless --num-frames 60 --stage-path output/stairs.usd
```

Both stairs modes use the original no-argument configuration: 195,525 particles
(55 x 79 x 45), radius 0.0125 m, frame duration 1/150 s and three substeps.
Initialization is preserved, including 1,411 particle centers inside the
reference static solid. No particle filtering, preprojection or rigid-body
cutouts are applied. `config/default_para.json` is a different existing preset;
it is used only when explicitly supplied through `--config`.

For exact reproduction when the archived local assets are available, synchronize
them once instead of rebaking the scene (rebaking can change SDF rounding):

```powershell
python -m tests.lab_solver.sync_reference_assets
```

This explicit migration tool backs up the previous framework resources to
`outputs/stairs-rigids/reference-backup`, copies the archived OBJ/SDF, rebuilds
only the destination's volume map at h=0.05 m, and records file hashes. Archived
demo inputs remain untouched; the running application never imports demos.

`stairs-rigids` adds two 0.30 m boxes (densities 500/1200 kg/m3), a 0.30 m
sphere (750 kg/m3), and a Spot solid scaled to a longest edge of 0.35 m
(500 kg/m3). They begin in the empty region to the right of the original fluid
block. The boxes share one boundary asset but have independent states/forces.
Spot's original asset frame, nonzero COM and full inertia are retained.

The static collider is assembled from the same aligned stair boxes and actual
container walls/lid/base as the asset generator, plus convex pieces for the
platform bunny. The cavity is never approximated by a single solid hull.
Spot also uses convex pieces for rigid collision, but one complete mesh/SDF
for rendering and PBF. Imported-solid mass comes from the complete closed mesh,
not the overlapping convex pieces. Collision pieces add no mass.

First use creates cached CoACD 1.0.14 hulls and 0.005 m dynamic SDFs under
`.cache/pbf/stairs-rigids`. Convex decomposition can take several minutes;
generation progress is printed and subsequent launches reuse the cache.
Geometry, scale, sampling parameters, decomposition options and version select
the cache. No asset preparation occurs during a physics step. Decomposition
errors are reported instead of silently falling back to a whole-object hull.
Cached `metrics.json` files record piece counts, volume and sampled distances;
these distances are diagnostics, not a certified Hausdorff error bound.

Full-scale validation is opt-in so ordinary unit tests stay small:

```powershell
$env:PBF_FULL_SCENE_TESTS = '1'
python -m unittest tests.lab_solver.test_legacy_parity tests.lab_framework.test_stairs_lifecycle -v
python -m unittest tests.lab_framework.test_stairs_assets tests.lab_framework.test_stairs_coupling -v
python -m tests.lab_framework.verify_stairs
python -m tests.lab_framework.verify_stairs_viewer
python -m tests.lab_framework.benchmark_coupling --scene stairs-rigids --output outputs/stairs-rigids/coupled-performance.json
```

The trajectory verifier records ten simulated seconds, body poses/velocities,
pressure wrenches, native contact pairs and contact/constraint capacity usage.
Use `--no-feedback --output outputs/stairs-rigids/one-way.json` for a one-way
diagnostic. GL keeps the original stairs view and camera/keyboard behavior;
USD exports contain all four moving bodies.

Validation and timing commands (CUDA required):

```powershell
python -m unittest discover -s tests/lab_solver -t . -v
python -m unittest discover -s tests/lab_framework -t . -v
python -m tests.lab_solver.benchmark_migration --output outputs/pbf-mujoco-coupling/performance.json
python -m tests.lab_framework.benchmark_coupling
```

The 2026-11-03 regression run passed 27 solver tests and 77 framework tests.
Each suite skipped one opt-in full-scale test; both full-scale tests were run
separately and passed. Coverage includes native box/box, sphere/box, Spot/box,
stairs/wall/bunny contacts and motion response, shared-asset force ownership,
off-center pressure rotation, State ownership and Graph/reset lifecycle.

The original-scale coupled trajectory completed 1,500 frames / 10 simulated
seconds with finite motion and four sampled dynamic-body contact combinations
plus each body's world contacts. Only nonpositive contact distances count as
touching pairs; separated native candidates are excluded. Box/box, sphere/box
and Spot/box are also verified in controlled motion tests. Sampled native
contact/constraint peaks were 277/24 against
capacities 512/2048; the deepest sampled native contact was 0.0114 m. No rigid
body escaped. Median GPU time was 8.215 ms per three-substep frame, excluding
rendering. The separate benchmark (60 warm-up + 240 measured frames) recorded
8.417 ms/frame, or 2,805.82 us/substep amortized, in
`outputs/stairs-rigids/coupled-performance.json`.
The cached proxy/source symmetric-difference volume ratios were
4.47% for the bunny and 4.40% for Spot; both use 32 convex pieces. These remain
approximate collision proxies, not exact source surfaces.

`outputs/stairs-rigids/acceptance.json` contains the trajectory and timing
samples. The matching ten-second run with feedback disabled also completed
with finite motion; `one-way.json` retains the comparison trajectory. The real
frontend verifier checks shared flat-shaded box vertices,
initial rendering before physics, G/Q/E/X/R/P, paused reset, and all four USD
mesh animations; screenshots and logs are in the same output directory. See
[`Lab_Solver/README.md`](../Lab_Solver/README.md#validation) for measured static
regression and coupled timings.
