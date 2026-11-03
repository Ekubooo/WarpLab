# Independent lab applications

```
conda run -n WarpSim --no-capture-output python -m Lab_Framework.pbf --scene stairs-rigids --device cuda:0 --cuda-graph --particle-radius 0.0125 --substeps 3 --pressure-iterations 3
```

These applications use the Newton-compatible PBF solver from `Lab_Solver`.
Existing programs under `demos` remain independent snapshots.

Run commands from the repository root in the `WarpSim` environment. No editable
installation is needed. `requirements-newton.txt` records the Newton 1.7.0,
Warp 1.18.0 and MuJoCo/MuJoCo Warp 3.15.0 validation environment.

## Prepare assets

Source OBJ inputs stay local and are excluded from Git. The producer is
`Lab_Framework.sim_ready`; the PBF application consumes published files in
`Lab_Framework/pbf/Assets` and its `Rigid` subdirectory.

For an exact copy of the archived default stairs, run the explicit reference
synchronization once. A fresh installation without the archived resources can
instead build a container using the existing standalone OBJ-to-SDF tool; a
new bake is not promised to reproduce archived SDF rounding bit for bit.

```powershell
python -m tests.lab_solver.sync_reference_assets
# Alternative for freshly generated geometry:
# python -m Lab_Framework.sim_ready --obj bunny_flat_bottom.obj --name stairs_scene

python -m Lab_Framework.sim_ready.pbf_assets --scene static
python -m Lab_Framework.sim_ready.pbf_assets --scene tank-box
python -m Lab_Framework.sim_ready.pbf_assets --scene stairs-rigids
```

The scene publisher validates source geometry and reuses matching existing
baked/CoACD caches. Missing geometry caches are generated **offline**, then all
consumer files are staged and published with rollback on handled failures.
Stop running consumers while publishing. Output copies are kept under
`sim_ready/output/pbf_assets/<scene>`; `.cache/pbf` is only a producer cache.
The runtime never imports the geometry baker or CoACD.

Each asset keeps its original-style filename stem, with these attachments:

| File | Content |
| --- | --- |
| `.obj` | Complete render mesh, preserving the asset frame |
| `.nvdb` | SDF and the current support-radius volume-map cache |
| `.collision.npz` | Convex pieces for imported solids; primitives use native shapes |
| `.json` | Geometry recipe, file/SDF identities, provenance and physical properties |

`stairs_scene` stays at its existing path. Published dynamic stems are
`Rigid/tank_box`, `Rigid/stairs_box`, `Rigid/stairs_sphere`, and `Rigid/stairs_spot`;
the tank boundary is `tank_scene`. The two stair boxes share one asset.
Spot metadata stores volume, local COM and the complete unit-density COM inertia.
Runtime instance density scales these properties; convex pieces add no mass.

Startup only reads and assembles assets, with one deliberate exception inherited
from the demo: a changed support radius, quadrature or volume-map algorithm
recomputes the **volume map**, writes it beside the unchanged SDF in the same
NVDB, and reuses it on the next launch. No geometric SDF bake occurs at startup.
A support radius beyond the sparse SDF's baked band requires offline publication
with a larger particle radius. SDF identity is checked independently of the
mutable volume-map grid and grid-pair header fields.

```powershell
# Same configuration selection rules as the runtime (JSON excludes CLI overrides).
python -m Lab_Framework.sim_ready.pbf_assets --scene tank-box --config my_fluid.json
python -m Lab_Framework.sim_ready.pbf_assets --scene tank-box --particle-radius 0.02 --container-size 1.2 1 1
```

Changing tank dimensions requires republishing; changing density, time step or
pressure iterations does not. Missing, incomplete or geometrically mismatched
assets stop startup with an offline command, without silently repairing them.

The original standalone container and rigid tools remain available:

```powershell
python -m Lab_Framework.sim_ready.rigid_pipeline --obj spot_triangulated_good.obj --name rigid_spot
```

They publish OBJ/SDF pairs. The scene publisher adds the collision/physical
attachments required by the coupled recipes. No USD asset import is involved;
USD remains an optional animation export.

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

Tank/box meshes and SDFs are prepared with the offline command above and loaded
from published Assets. Existing stairs/Spot assets are not changed.
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
python -m Lab_Framework.sim_ready.pbf_assets --scene stairs-rigids
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

Offline publication prepares CoACD 1.0.14 hulls and 0.005 m dynamic SDFs, reusing
`.cache/pbf/stairs-rigids` when available. Convex decomposition can take several
minutes on a cache miss; progress is printed by the producer. Geometry, scale,
sampling parameters, decomposition options and version select its cache.
Startup and physics steps never prepare geometry or recompute imported mass. Decomposition
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

The earlier multi-rigid baseline (before the offline-assets refactor) passed
27 solver tests and 77 framework tests.
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

## Forward-only compilation and startup measurements

Project PBF kernels explicitly disable backward generation, including their
unique module options. Boundary/force helpers, SDF baking and billboard kernels
also disable backward at their own module scope. No global Warp setting or
third-party module is changed. `requires_grad=True` remains unsupported.
MuJoCo's `derivative` module can still appear in load logs: it contains
derivatives used by forward integration and already disables backward code.

`tests.lab_framework.benchmark_startup` measures each scene in fresh processes:
three empty-Warp-cache trials and five trials reusing the first cache. The CUDA
driver cache is not cleared. Asset production is excluded and reported
separately; volume-map-miss startup must also be distinguished from cache hits.
Results include imports/device initialization, asset loading, model assembly,
solver creation, Graph capture and the first hidden OpenGL frame, plus generated
project CUDA backward-entry counts. Use its `--help` for isolated code/cache roots.

```powershell
# Use a new cache directory for each run; existing caches are never cleared.
python -m tests.lab_framework.benchmark_startup --cold 3 --warm 5 --cache-root .cache/startup-check --output outputs/offline-assets/startup-check.json
# Separate volume-map miss/hit trials on private copies, reusing those Warp caches.
python -m tests.lab_framework.benchmark_volume_map_startup --startup-report outputs/offline-assets/startup-check.json --output outputs/offline-assets/volume-map-check
```

The offline-assets refactor passed 92 framework tests (including full-scale
Graph/lifecycle), all 30 solver cases across the suite and parity retry, the
30-substep independent bitwise reference comparison, the real GL/key/USD
verifier, and a 10-second four-body trajectory. One initial archived-reference
test hit a transient Windows file-replacement denial; its isolated rerun and
the subsequent independent benchmark both passed without changing the demo.
Published meshes and complete NVDB pairs are byte-identical to the pre-change
snapshot; convex arrays and Spot physical properties also match. Evidence is
in `outputs/offline-assets/asset-migration.json`, `parity-tests.log`,
`framework-tests.log`, `acceptance.json`, and `static-performance.json`.

Measured on the RTX 5070 / WarpSim environment, with the original scene sizes,
prepared assets and volume-map hits. Each row reports seconds as median
(minimum to maximum); cold has three fresh-process samples and warm has five.
Cold means an empty isolated **Warp** cache; the CUDA driver cache was not cleared.

| Scene | Warp cache | Before | After |
| --- | --- | ---: | ---: |
| static | cold | 10.644 (10.625 to 10.646) | 7.603 (7.561 to 7.670) |
| static | warm | 6.405 (6.377 to 6.564) | 4.968 (4.853 to 5.259) |
| tank-box | cold | 60.947 (59.338 to 61.186) | 51.522 (51.393 to 51.749) |
| tank-box | warm | 20.394 (20.185 to 20.946) | 16.656 (15.769 to 16.767) |
| stairs-rigids | cold | 148.184 (129.832 to 149.682) | 77.289 (77.054 to 78.214) |
| stairs-rigids | warm | 54.729 (54.036 to 69.055) | 23.623 (23.203 to 25.771) |

The 48 canonical trials and per-phase samples are in
`outputs/offline-assets/startup-before.json` and `startup-after.json`;
`startup-comparison.json` separates asset work, project module loading and
third-party module loading. Project cold module-load medians fell from
4.950 to 1.968 s (static), 11.893 to 3.103 s (tank), and 56.729 to 6.284 s
(stairs). These timings include source generation, compilation and loading,
and overlap their enclosing startup phases. The after-cache audit found
122 project CUDA sources with **zero backward kernel entries**, versus 142
entries across 124 baseline sources.

Third-party cold module loading still costs about 1.32 / 44.02 / 65.58 s
respectively, mostly inside rigid-solver initialization and Graph capture.
Third-party code is unchanged; its timing variation is not attributed directly
to disabling project AD. Some early coupled baseline imports put trimesh work
in a different phase (about one second); total first-frame and module-load sums
remain comparable, but asset-phase differences alone are not net savings.

Cached offline migration/publication took 4.037 / 0.209 / 2.135 s for static,
tank and stairs in one process (static includes initial CUDA setup). These are
**cache-reuse** measurements, not fresh SDF/CoACD production times, and are not
counted as compilation savings. See `outputs/offline-assets/publication.json`.

Separate volume-map trials used private copies seeded at h=0.04 m, then the
default h=0.05 m launch followed by a second launch. The existing Warp caches
were reused. These are one diagnostic miss/hit pair per scene, not the repeated
startup samples above; they include volume-map compilation on the first use.

| Scene | Refresh + first frame (s) | Next launch + first frame (s) | Maps rebuilt on first / second launch |
| --- | ---: | ---: | ---: |
| static | 11.786 | 5.036 | 1 / 0 |
| tank-box | 18.932 | 16.937 | 2 / 0 |
| stairs-rigids | 31.334 | 23.646 | 4 / 0 |

The reproducible driver is `tests.lab_framework.benchmark_volume_map_startup`;
results and individual logs are in `outputs/offline-assets/volume-map`.
Its copied SDFs pass published identity checks before and after refresh.
The original published files are untouched by this diagnostic.
