# Warp MMPBF

This directory is a Warp DSL translation of the complete PBF path in
PositionBasedDynamics' `Demos/FluidDemo`. The code layout is Pythonic, while
the numerical model, scene, defaults, and step order follow the official C++
demo at commit `beafc921e21553515b4f406258e5b16054a45268`, except for the
explicitly documented global particle-speed limit.

The previous version mixed SPlisHSPlasH's newer `TimeStepPBF`, a custom box,
custom boundary sampling, Standard viscosity, and Vorticity confinement. That
combination did not correspond to either official program and has been
removed from the active solver.

## Code layout

`MMPBF.py` contains every executable Warp kernel, `MMPBFConfig`, and an
`Example` class with only `__init__`, `step`, and `render`. The complete
kernel/Grid order remains visible in `step`. `mmpbf_functions.py`
contains reusable `@wp.func` mathematics and the small Host result handlers
needed between kernels. `mmpbf_initialization.py` owns scene sampling, storage,
Grid, and static state initialization. This mirrors the neighboring PBF demo
without changing the official numerical path.

Kernel input and output lists intentionally remain explicit in
`step()`. See `LAUNCH_ARGUMENT_PACKING.md` for the rejected alternative
that prebuilds those lists during initialization.

## Official mapping

| Warp implementation | Official source |
| --- | --- |
| `cubic_kernel`, `cubic_kernel_gradient` | `PositionBasedDynamics/SPHKernels.h` |
| density, lambda, position correction | `PositionBasedDynamics/PositionBasedFluids.cpp` |
| five projection iterations and step order | `Demos/FluidDemo/TimeStepFluidModel.cpp` |
| XSPH viscosity | `TimeStepFluidModel::computeXSPHViscosity` |
| fluid block and six duplicate-preserving walls | `Demos/FluidDemo/main.cpp` |
| Akinci pseudo-volume | `Demos/FluidDemo/FluidModel.cpp` |

The default model therefore uses:

- particle radius `0.025`, support radius `0.1`, rest density `1000`;
- an official `15 x 20 x 15` baseline block containing `4500` fluid particles;
- the official `4.0 x 4.0 x 0.8` container and `18630` boundary samples;
- particle mass `0.1`, gravity `(0, -9.81, 0)`;
- exactly five PBF projection iterations with `epsilon = 1e-6`;
- first-order velocity reconstruction (`velocityUpdateMethod = 0`);
- XSPH viscosity `0.02`, with the official boundary-viscosity code disabled;
- a Warp-side global particle-speed limit of `8.0 m/s`, applied after XSPH;
- initial time step `0.0025`, CFL factor `1`, limits `[0.0001, 0.005]`;
- one complete adaptive physics step per `Example.step()` call.

The OpenGL frontend defaults to the `high-resolution` preset. It keeps the
same exact `4.0 x 4.0 x 0.8` container while using radius `2/185`, support
radius `8/185`, and a `36 x 46 x 36` block containing `59,616` visible fluid
particles. Its `97,464` static boundary samples are simulation data and are
not rendered. `MMPBFConfig()` remains the 4,500-particle baseline;
`MMPBFConfig.high_resolution()` selects the denser scene.

There is no artificial pressure, vorticity confinement, Standard SPH
viscosity, random wall offset, explicit restitution, friction, or position
clamp because the official FluidDemo does not execute those operations.
The global speed limit is an intentional stability extension relative to the
official demo: velocities below the limit preserve the official update, while
larger velocity vectors are scaled to the limit without changing direction.

SPlisHSPlasH commit `eccce86155776f6ac52d5080b1f720a52bf29450` remains a
useful second implementation for comparison, but its Poly6/Spiky kernels,
convergence-controlled projection, Standard viscosity, and selectable modern
boundary models are intentionally not mixed into this demo.

## Run

From the repository root:

```bash
.venv/bin/python demos/fluid/MMPBF/render_opengl.py --device cuda:0
```

The frontend advances at `0.5` simulated seconds per wall-clock second by
default. This is wall-clock playback pacing: it does not alter the solver's
adaptive time step, CFL condition, or five projection iterations. Select the
baseline with `--preset official`, change pacing with `--playback-speed`, and
bound catch-up work with `--max-substeps-per-frame`. For example:

- Press `Space` to pause or resume playback.
- Press `R` to recreate the initial fluid state and pause playback while
  restoring default gravity and preserving the current camera and display options.
- Press `G` to reverse the current effective gravity.
- Press `Q` or `E` to rotate effective gravity counterclockwise or clockwise
  by 10 degrees around Z. The boundary particles remain stationary.

```bash
.venv/bin/python demos/fluid/MMPBF/render_opengl.py --device cuda:0 \
    --preset high-resolution --playback-speed 0.5 --max-substeps-per-frame 8
```

For a bounded render run, add `--num-frames N`. For the CPU parity and behavior
suite:

```bash
.venv/bin/python -m unittest demos.fluid.MMPBF.test_mmpbf
```

The OpenGL billboard renderer is only a frontend. It is not part of the
official-behavior comparison.

## Verification

`test_mmpbf.py` checks each official formula independently, the exact default
scene and parameter mapping, zero-gravity equilibrium, CFL ordering, fixed
iteration count, overflow handling, and multi-step containment. Its trajectory
test also compares 200 steps through floor contact against values generated by
compiling the selected official `PositionBasedFluids.cpp` and `SPHKernels.cpp`.
The tolerance only covers Warp float32 arithmetic and hash-neighbor summation
order; it does not hide a different model or parameter set.
