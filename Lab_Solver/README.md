# Standalone Newton PBF solver

`Lab_Solver.pbf.SolverPBF` derives from Newton 1.7 `SolverBase`. It preserves the
archived PBF's static-boundary equations, launch sequence, spatial sorting and
neighbor-cache layout. It does not import a framework or anything under `demos`.

## Public interface

```python
from Lab_Solver.pbf import SolverPBF

# The caller builds a Newton Model and loads its BoundarySystem explicitly.
state = model.state()
solver = SolverPBF(
    model, config=SolverPBF.Config(), boundaries=boundaries,
    grid_bounds=((0.0, 0.0, 0.0), (3.0, 5.0, 2.0)),
)
solver.bind_state(state)
solver.step(state, state, control=None, contacts=None, dt=1.0 / 450.0)
positions = state.particle_q
velocities = state.particle_qd
```

One call advances one substep. Frame scheduling and simulation time belong to
the caller. The supported mode is **in-place only**: distinct input and output
State objects raise an error, rather than trigger hidden copies.

The existing spatial reorder swaps array bindings. Read `state.particle_q` after
each step, and do not retain a Torch/NumPy/Warp view of an older position buffer
as if it were the current state. No original-ID restoration or independent ID
mapping buffer is created. Particle slots follow spatial ordering rather than
stable cross-step identities.

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
- `Lab_Solver.nvdb_io` reads/writes the same single-segment, one/two-grid NVDB
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
`set_buffer_bindings(bindings, state)` let a caller capture both existing buffer
parities. The new framework uses these to avoid whole-state copies for odd
substep counts. Changing gravity or time step requires recapture; the framework
checks the solver's revision and its captured time step.

Current support: CUDA NanoVDB, one homogeneous active fluid world, exactly one
static volume boundary, and one bound in-place State per solver. Unsupported:
fixed-ID particle inputs, per-particle external forces/constraints, automatic
differentiation, native Newton Contacts and SolverCoupled integration. `control`
is unused. No automatic scanning or per-step validation of `particle_f` is added
to the hot path; callers must not use it as a supported force channel.

Boundary asset/instance management and the rigid asset baker are retained, but
dynamic boundaries do not yet participate in this fluid solve. Rigid dynamics,
reaction forces and robot policies are later extensions.

## Validation

Validated in WarpSim with Newton 1.7.0, Warp 1.18.0 and RTX 5070. The verification
tool intentionally imports the old demo; production code does not:

```powershell
conda run -n WarpSim --no-capture-output python -m unittest discover -s tests/lab_solver -t . -v
conda run -n WarpSim --no-capture-output python -m tests.lab_solver.benchmark_migration
```

The default 195,525-particle scene was bitwise equal to the archived solver at
initialization and across five substeps for positions, velocities, predicted
positions, corrections, lambda, boundary density/gradient, velocity corrections
and curl. The interface test checks the original fluid launch sequence and the
absence of new mapping/copy kernels. Graph tests compare even/odd substep replay
with eager execution using exact array equality.

GPU benchmark: 12 warm-up pairs, 60 alternating frozen-stage graph trials,
CUDA events, identical configuration/resources. Snapshot restores are outside
the timed events. Times are microseconds per neighborhood/pressure stage or
complete substep:

| Stage | Archived median | New median | New/archived | Paired ratio 95% CI |
| --- | ---: | ---: | ---: | --- |
| Neighborhood | 271.82 | 275.10 | 1.0121 | 0.9813–1.0155 |
| Pressure iterations | 677.87 | 675.57 | 0.9966 | 0.9867–1.0084 |
| Complete substep | 1187.47 | 1178.77 | 0.9927 | 0.9913–1.0024 |

All intervals contain 1: these measurements do not establish a stable slowdown
or speedup. They are evidence for parity on this configuration, not a universal
zero-overhead guarantee. Full samples are in
`outputs/newton-pbf-migration/performance.json` after running the tool.
