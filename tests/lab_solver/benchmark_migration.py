"""Compare frozen GPU stages and numerical results with the archived PBF.

This verification tool intentionally imports the old demo. Production solver
and framework code never does. Snapshot copies are outside timed CUDA events.
"""

import argparse
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from demos.fluid.PBF import PBF as legacy
from demos.fluid.PBF import pbf_helper as old_fn
from demos.fluid.PBF.simulation import reset_simulation
from tests.fluid.pbf.solver_fixture import bind_solver
from Lab_Framework.pbf.config import PBFConfig
from Lab_Framework.pbf.simulation import Simulation, ASSET_DIR, SDF_FILENAME
from Lab_Solver.pbf import boundary_model as boundary


def neighborhood(solver, kernels):
    solver.grid.build(solver.pre_Pos, solver.smoothing_length)
    wp.launch(kernels.reorder, solver.n, inputs=[solver.pos, solver.pre_Pos],
              outputs=[solver.temp_pos, solver.temp_pPos])
    solver.pos, solver.temp_pos = solver.temp_pos, solver.pos
    solver.pre_Pos, solver.temp_pPos = solver.temp_pPos, solver.pre_Pos
    wp.launch(kernels.cache_neighbors, solver.n, inputs=[solver.pre_Pos], outputs=[solver.n_Cache])


def neighborhood_current(sim):
    solver, kernels = sim.fluid_solver, sim.fluid_solver.kernels
    solver.grid.build(solver.pre_Pos, solver.smoothing_length)
    wp.launch(kernels.reorder, solver.n, inputs=[sim.pos, solver.pre_Pos],
              outputs=[sim.state_out.particle_q, solver.temp_pPos])
    solver.pre_Pos, solver.temp_pPos = solver.temp_pPos, solver.pre_Pos
    sim._state_index = 1 - sim._state_index
    solver.pos, solver.v = sim.pos, sim.v
    wp.launch(kernels.cache_neighbors, solver.n, inputs=[solver.pre_Pos], outputs=[solver.n_Cache])


def pressure(solver, kernels, dt):
    current = hasattr(solver, "boundary_data")
    for _ in range(solver.iterations):
        wp.launch(kernels.volume_query, solver.n,
                  inputs=[solver.pre_Pos] + ([solver.boundary_data] if current else []),
                  outputs=[solver.volume_Rho, solver.volume_Grad])
        wp.launch(kernels.calc_lambda, solver.n,
                  inputs=[solver.pre_Pos, solver.n_Cache, solver.volume_Rho, solver.volume_Grad],
                  outputs=[solver.lambda_Opt])
        wp.launch(kernels.calc_deltaPos, solver.n,
                  inputs=[solver.pre_Pos, solver.lambda_Opt, solver.n_Cache, solver.volume_Grad]
                         + ([dt, solver.boundary_data] if current else []),
                  outputs=[solver.delta_Pos])
        wp.launch(kernels.update_prePos, solver.n,
                  inputs=[solver.delta_Pos] + ([solver.boundary_data] if current else []),
                  outputs=[solver.pre_Pos])


def capture_frozen(solver, state, call):
    bindings = (solver.pos, solver.pre_Pos, solver.temp_pos, solver.temp_pPos)
    snapshots = [(array, wp.clone(array)) for array in (solver.pos, solver.pre_Pos, solver.v)]
    with wp.ScopedCapture() as capture:
        call()
    solver.pos, solver.pre_Pos, solver.temp_pos, solver.temp_pPos = bindings
    if state is not None:
        state.particle_q = solver.pos
    def restore():
        for target, source in snapshots:
            wp.copy(target, source)
    return capture.graph, restore


def capture_current(sim, call):
    solver = sim.fluid_solver
    bindings, index = solver.get_buffer_bindings(), sim._state_index
    arrays = [state.particle_q for state in sim.states] + [state.particle_qd for state in sim.states]
    snapshots = [(array, wp.clone(array)) for array in arrays + list(bindings)]
    with wp.ScopedCapture() as capture:
        call()
    sim._state_index = index
    solver.set_buffer_bindings(bindings)
    solver.pos, solver.v = sim.pos, sim.v
    def restore():
        for target, source in snapshots:
            wp.copy(target, source)
    return capture.graph, restore


def measure_pair(first, second, trials):
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    samples = [[], []]
    for trial in range(trials + 12):
        for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
            graph, restore = (first, second)[index]
            restore()
            wp.record_event(start)
            wp.capture_launch(graph)
            wp.record_event(end)
            elapsed = wp.get_event_elapsed_time(start, end) * 1000.
            if trial >= 12:
                samples[index].append(elapsed)
    old_median, new_median = (float(np.median(values)) for values in samples)
    ratios = np.asarray(samples[1]) / np.asarray(samples[0])
    rng = np.random.default_rng(1)
    resamples = rng.choice(ratios, size=(5000, trials), replace=True)
    interval = np.quantile(np.median(resamples, axis=1), [.025, .975]).tolist()
    return dict(legacy_median_us=old_median, new_median_us=new_median,
                new_over_legacy=new_median/old_median,
                paired_ratio_95pct_bootstrap=interval,
                legacy_samples_us=samples[0], new_samples_us=samples[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/newton-pbf-migration/performance.json"))
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with wp.ScopedDevice(args.device), contextlib.redirect_stdout(io.StringIO()):
        config = old_fn.PBFConfig()
        resources = boundary.load_boundaries(
            boundary.BoundaryScene(ASSET_DIR, static_assets=(SDF_FILENAME,), default_rigid_count=0),
            boundary.BoundaryLoadOptions(4*config.particle_radius, device=args.device))
        with patch.object(legacy.boundary_model, "load_default_boundaries", return_value=resources):
            old = legacy.PBF(stage_path=None, config=config)
        new = Simulation(config=PBFConfig(**asdict(config)), boundaries=resources)
        np.testing.assert_array_equal(old.pos.numpy(), new.pos.numpy())
        old.substep = 1
        bind_solver(config, old.grid, old.static_sdf, old.volume_map)
        for _ in range(5):
            old.step()
            new._substep(old.dt)
            for name in ("pos", "v", "pre_Pos", "delta_Pos", "lambda_Opt", "volume_Rho",
                         "volume_Grad", "delta_Vel", "curl"):
                np.testing.assert_array_equal(getattr(old, name).numpy(),
                                              getattr(new.solver, name).numpy(), err_msg=name)
        old_kernels = type("LegacyKernels", (), dict(
            reorder=old_fn.by_cache_reordering, cache_neighbors=legacy.cache_neighbors,
            volume_query=legacy.volume_query, calc_lambda=legacy.calc_lambda,
            calc_deltaPos=legacy.calc_deltaPos, update_prePos=legacy.update_prePos))
        result = dict(particles=old.n, device=wp.get_device().name, dtype="float32",
                      numerical_parity="bitwise equal: initialization and nine arrays over five substeps",
                      timing="CUDA events around frozen-stage graphs; restores excluded; alternating order",
                      config=asdict(config), trials=args.trials, stages={})
        for label in ("neighborhood", "pressure", "substep"):
            reset_simulation(old)
            new.reset()
            if label != "substep":
                wp.copy(old.pre_Pos, old.pos)
                neighborhood(old, old_kernels)
                wp.copy(new.fluid_solver.pre_Pos, new.pos)
                neighborhood_current(new)
            if label == "neighborhood":
                old_call = lambda: neighborhood(old, old_kernels)
                new_call = lambda: neighborhood_current(new)
            elif label == "pressure":
                old_call = lambda: pressure(old, old_kernels, old.dt)
                new_call = lambda: pressure(new.solver, new.solver.kernels, old.dt)
            else:
                old_call = old.step
                new_call = lambda: new._substep(old.dt)
            # Warm module compilation, radix-sort scratch and launch metadata.
            for _ in range(3):
                old_call()
                new_call()
            pair = (capture_frozen(old, None, old_call), capture_current(new, new_call))
            result["stages"][label] = measure_pair(*pair, args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    for name, stage in result["stages"].items():
        print(f"{name}: old={stage['legacy_median_us']:.2f} us, new={stage['new_median_us']:.2f} us, "
              f"ratio={stage['new_over_legacy']:.4f}, CI={stage['paired_ratio_95pct_bootstrap']}")
    print(result["numerical_parity"])
    print(args.output)


if __name__ == "__main__":
    main()
