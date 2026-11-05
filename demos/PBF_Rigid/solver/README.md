# Standalone Newton PBF solver

`demos.PBF_Rigid.solver.SolverPBF` derives from Newton 1.7 `SolverBase`. It preserves the
archived PBF's static-boundary equations, launch sequence, spatial sorting and
neighbor-cache layout. It does not import the application or renderer. NVDB
reading/writing reuses `demos.sim_Ready.nvdb_io`.

## Public interface

```python
from demos.PBF_Rigid.solver import SolverPBF

# The caller builds a Newton Model and loads its BoundarySystem explicitly.
state_in, state_out = model.state(), model.state()
solver = SolverPBF(
    model, config=SolverPBF.Config(), boundaries=boundaries,
    grid_bounds=((0.0, 0.0, 0.0), (3.0, 5.0, 2.0)),
)
solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 450.0)
state_in, state_out = state_out, state_in
positions = state_in.particle_q
velocities = state_in.particle_qd
```

One call advances one substep. Frame scheduling and simulation time belong to
the caller. Input and output States must be distinct, with non-aliasing particle
arrays. Input particle positions and velocities are read-only.

The existing spatial reorder gathers old positions directly into the output
State, which becomes the updated position buffer. Each State retains its array
bindings; the caller swaps States. No extra full-particle copy, original-ID
restoration or ID mapping is performed. Slots follow spatial ordering, not
stable cross-step identities. Consumers must select the current State.

## Ownership and configuration

- Newton Model supplies equal positive particle masses/radii and one fluid
  world's gravity. The caller owns particle initialization and its asset paths.
- `SolverPBF.Config` contains density, pressure iterations/regularization,
  viscosity, vorticity, speed limit, artificial pressure and cache capacity.
- PBF owns predicted positions, pressure/velocity scratch, neighbor cache and
  HashGrid. Kernels and fluid functions are specialized with per-instance
  compile-time coefficients/resource handles, without mutable module globals.
- `BoundarySystem` owns the SDF/volume-map resources. `BoundaryScene.asset_dir`
  is explicit; the solver does not choose application filenames or bake meshes.
- `demos.sim_Ready.nvdb_io` reads/writes the same single-segment, one/two-grid NVDB
  format used by the asset pipeline. Runtime volume-map caches replace the
  second grid in the selected file atomically.

The shortest round-trip decimal form of model float32 radius/mass is used when
deriving host coefficients. This preserves existing configured values such as
`0.0125` instead of introducing coefficient changes from float32-to-double noise.

## Lifecycle and current limits

`reset(state, world_mask=None, flags=None)` restores selected particle state from
the model's initial arrays and clears scratch without reallocating particle or
Volume resources. `set_gravity()` updates the model and cached gravity;
`notify_model_changed(ModelFlags.MODEL_PROPERTIES)` refreshes gravity and rejects
changed topology/mass/radius, which require constructing another solver.
Configuration is an immutable dataclass; construct another solver to change its
compile-time physical parameters.

`prepare_graph()`, `get_buffer_bindings()` and
`set_buffer_bindings(bindings)` expose only the two predicted-position scratch
buffers. The framework captures state/scratch parities without whole-state
copies. Gravity, time step or substep-count changes require recapture.

Current support: CUDA NanoVDB, one homogeneous active fluid world, exactly one
static volume boundary plus body-bound dynamic instances. Unsupported:
fixed-ID particle inputs, per-particle external forces/constraints, automatic
differentiation, native Newton Contacts and SolverCoupled integration. `control`
is unused. No automatic scanning or per-step validation of `particle_f` is added
to the hot path; callers must not use it as a supported force channel.

## Weak rigid coupling

`BoundarySystem.bind_bodies(model, body_ids, local_transforms=None)` maps internal
rigid-instance order to Newton bodies. Call once before constructing the solver.
An asset-to-body transform preserves non-COM asset origins; `sync_body_poses`
updates world transforms and COMs on the GPU without reallocating buffers.

`SolverPBFMuJoCo` in `pbf.solver_coupled` composes PBF and MuJoCo Warp on the same
Model and State pair. PBF writes particles; MuJoCo writes bodies/joints. One step
freezes input boundary poses, solves PBF, adds pressure reactions to input
`body_f`, steps MuJoCo by the same dt, and updates boundary poses. The caller
clears external forces once before the next substep. Existing external wrenches
are added to, not erased. `feedback=False` provides a one-way diagnostic mode.

`fluid.boundary_forces` is a persistent body-indexed world-COM wrench buffer in
Newton `[force, torque]` order. Every pressure iteration accumulates
`F = -m * delta_x_boundary / dt**2` and `(equivalent_point - COM) x F`.
The buffer clears once per fluid substep, not per pressure iteration.

This follows SPlisHSPlasH's PBF/Bender2019 pressure-feedback weak coupling:
[pressure reactions](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/master/SPlisHSPlasH/PBF/TimeStepPBF.cpp)
and [outer stepping](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/master/Simulator/SimulatorBase.cpp).
The existing SDF projection remains a correction without reaction feedback.
Speed clipping and vorticity are unchanged; the complete scheme is not claimed
to conserve momentum when these corrections act. Boundary viscosity, strong
coupling, Newton SolverCoupled integration and robot policies remain future work.

## Validation

Validated in WarpSim with Newton 1.7.0, Warp 1.18.0, MuJoCo/MuJoCo Warp 3.15.0
and RTX 5070. See `../requirements.txt`. The verification
tool intentionally imports the old demo; production code does not:

```powershell
conda run -n WarpSim --no-capture-output python -m unittest discover -s demos/PBF_Rigid/tests/solver -t . -v
conda run -n WarpSim --no-capture-output python -m demos.PBF_Rigid.tests.solver.benchmark_migration
```

The default 195,525-particle scene was bitwise equal to the archived solver at
initialization and across thirty substeps for positions, velocities, predicted
positions, corrections, lambda, boundary density/gradient, velocity corrections
and curl. This uses independently loaded archived and framework assets, after
explicit reference synchronization, and retains all 1,411 initially overlapping
particles. Enable `PBF_FULL_SCENE_TESTS=1` to run the full-scene unit test.
The interface test checks the original fluid launch sequence and the
absence of new mapping/copy kernels. Static Graph tests compare even/odd substep
replay with eager execution using exact array equality. Coupled tests use
identical MuJoCo warm-start inputs and floating-point tolerances; spatially sorted
particles are matched by position when atomic roundoff changes their cell order.

GPU benchmark: 12 warm-up pairs, 60 alternating frozen-stage graph trials,
CUDA events, identical configuration/resources. Snapshot restores are outside
the timed events. Times are microseconds per neighborhood/pressure stage or
complete substep:

| Stage | Archived median | New median | New/archived | Paired ratio 95% CI |
| --- | ---: | ---: | ---: | --- |
| Neighborhood | 267.62 | 250.54 | 0.9362 | 0.9261–0.9699 |
| Pressure iterations | 677.41 | 682.61 | 1.0077 | 0.9950–1.0274 |
| Complete substep | 1178.75 | 1175.01 | 0.9968 | 0.9893–1.0014 |

These measurements (2026-11-03) meet the complete-substep 5% regression target
for this configuration; they are not a universal zero-overhead guarantee. Full
samples are in `outputs/offline-assets/static-performance.json` when running the
benchmark with `--output` set to that path.

The default 24,500-particle tank/box scene took 1.691 ms per coupled frame
(three substeps, 563.70 us amortized per substep), measured with CUDA events over
240 advancing frames after 60 warm-up frames, excluding rendering and capture.
At two simulated seconds the box center was at y=0.400 m and all motion arrays
were finite. Reproduce with:

```powershell
python -m demos.PBF_Rigid.tests.framework.benchmark_coupling
python -m unittest demos.PBF_Rigid.tests.framework.test_coupled_simulation -v
```

The coarse integration fixture additionally compares feedback off/on over two
seconds (box center y=0.120/0.474 m), tests free fall and rigid floor collision,
and covers field ownership, capture without advancement, reset and gravity/dt
changes. Controlled kernel tests check pressure impulse balance, translated and
rotated asset offsets, nonzero COM, torque, dt-squared scaling, iteration
accumulation, and force clearing independently of projection and speed limits.


Project-owned PBF/force/boundary kernels are compiled forward-only. This removes
unused generated backward entry points; it does not change the equations or
make the current solver differentiable. Framework asset production is offline;
`load_boundary_volumes` still refreshes only the in-file volume map when its
support radius, quadrature or algorithm version changes.
