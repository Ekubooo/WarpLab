# PBF_Rigid migration record — 2026-11-05

Source commit: `bf3af4586786f40f03289e35ab6bf3e1ec76cf46`.
`archive_manifest.json` records original file SHA-256 hashes and the installed
environment. Hashes describe files **before** import/path adaptations. Large local
assets remain ignored by Git, as before; this migration includes the local copies.

## Scope

- Application: former framework PBF package → this directory.
- Solver: former solver PBF package → `solver/`.
- Asset producers and extensions → `sim_ready/`.
- Regression tests and benchmarks → `tests/solver/` and `tests/framework/`.
- Old implementation and test directories removed; no compatibility entrypoints.

Fourteen core solver/runtime modules match the source commit exactly after
namespace normalization, including kernels, helpers, coupling, simulation,
scenes, configuration, OpenGL frontend and billboard rendering. No equations,
iteration order, capacities, particle layout or scene parameters were changed.

Shared `demos.sim_Ready` functions now provide NVDB I/O, solid validation, SDF
baking, common geometry, path validation and atomic OBJ writing. Archive adapters
pass explicit destinations and do not replace shared configuration. Stair box
extraction, CoACD caching, mass attachments and grouped rollback remain local.
The shared terrain and retained Newton-generated collision boxes pass the exact
vertex/face regression. Other demos' 130 source/config/asset files were checked
against pre-migration hashes and are unchanged.

## Verification

Run from the repository root with the environment in `requirements.txt`:

```powershell
$env:PBF_FULL_SCENE_TESTS = '1'
python -B -m unittest discover -s demos/PBF_Rigid/tests/solver -t . -v
python -B -m unittest discover -s demos/PBF_Rigid/tests/framework -t . -v
python -B -m demos.PBF_Rigid.tests.solver.benchmark_migration --output outputs/pbf-rigid-archive/static-performance.json
python -B -m demos.PBF_Rigid.tests.framework.verify_stairs --output outputs/pbf-rigid-archive/acceptance.json
python -B -m demos.PBF_Rigid.tests.framework.verify_stairs_viewer --scene static --visible
python -B -m demos.PBF_Rigid.tests.framework.verify_stairs_viewer --scene tank-box
python -B -m demos.PBF_Rigid.tests.framework.verify_stairs_viewer --scene stairs-rigids
```

On RTX 5070 / Python 3.11 / Warp 1.18.0 / Newton 1.7.0:

- 30 solver tests passed, including independent-resource, 195,525-particle,
  30-substep bitwise parity for motion, pressure work arrays and neighbor counts.
- 92 migrated framework tests passed; two additional archive sharing/output
  isolation tests passed separately. Coverage includes offline baking, grouped
  publication rollback, volume-map refresh/reuse, collisions, feedback, Graph
  odd/even parity and resets.
- All three scene publication CLIs reused existing geometry/CoACD caches.
  Source meshes, published resources and configuration files (29 files) matched
  original hashes at copy time. Final verification found 27 byte-identical files;
  the two tank NVDBs were republished from cache, with unchanged SDF identity and
  map values identical to the previously validated map reference.
- Default stairs/bunny ran in a visible OpenGL window with original config and
  camera. All three frontend checks saved frames 1, 90 and 151, verified no
  first-frame advancement, G/Q/E/X/R/P behavior and USD output (0/1/4 bodies).
  The actual default headless CLI also exported a three-frame USD animation.
- The four-body scene completed 1,500 frames / 10 simulated seconds with finite
  states and no escaped bodies, retaining 1,411 initial overlapping particles.
  Sampled contacts/constraints peaked at 371/28; median GPU frame time was
  8.449 ms. This is a single trajectory timing, not a paired performance claim.
- Paired static CUDA-event benchmark: original 1,180.86 µs vs archive 1,169.17 µs
  per complete substep, ratio 0.9901; paired ratio bootstrap interval
  [0.9813, 0.9986]. No observed 5% steady-state regression.

**Existing limitation:** both the historical pre-migration acceptance log and
this run emit MuJoCo `broadphase overflow` warnings at capacity 512. Sampled
final contact counts below 512 do not establish that broadphase never overflowed.
The migration preserves this setting and does not claim overflow-free collision
coverage. Capacity adjustment is separate work; warnings were not suppressed.

Publication tests can replace dynamic volume maps from the existing offline
cache. Three such NVDB files had different raw bytes from the pre-migration
published files, while SDF identity, tile layouts and every allocated map cell
were bitwise identical (1,334,784 / 871,936 / 801,792 cells). The pre-migration
byte-identical stair files were restored after these tests. The tank publisher
also replaced its two NVDBs: their map queries match the pre-existing validated
reference at all 6,272,512 / 335,360 allocated cells, but the whole files no longer
match the migration-time hash. Mutable volume-map files
should continue to be checked by geometry identity and numerical contents as
well as recording whole-file hashes at migration time.

Detailed logs, integrity checks, map comparisons, timing samples, screenshots
and USD files are local under `outputs/pbf-rigid-archive/`. Earlier measurements
in README remain historical evidence, not measurements repeated by this move.
