"""Bounded steady-state workload for Nsight Systems/Compute; physics is unchanged."""

import argparse
import ctypes
import json
from pathlib import Path
import time

import numpy as np
import warp as wp

try:
    from .simulation import add_simulation_arguments, config_from_args, create_pbf2way_simulation
except ImportError:
    from simulation import add_simulation_arguments, config_from_args, create_pbf2way_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_simulation_arguments(parser)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--render", action="store_true", help="Render once after every step in a hidden window"
    )
    parser.add_argument("--capture", action="store_true", help="Use CUDA profiler start/stop APIs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error("steps must be positive and warmup-seconds finite and nonnegative")

    simulation = create_pbf2way_simulation(config=config_from_args(args), device=args.device)
    # Include at least one complete step to exclude JIT and first-use allocation.
    while simulation.sim_time < args.warmup_seconds or simulation.total_steps == 0:
        simulation.step()
    renderer = None
    if args.render:
        from demos.fluid.PBF2WayCoupling.render_opengl import CouplingRenderer

        renderer = CouplingRenderer(device=simulation.device, hidden=True)
        simulation.renderer = renderer
        simulation.render()  # Allocate GL resources outside the measured interval.
    wp.synchronize_device(simulation.device)
    start_time = simulation.sim_time
    start_step = simulation.total_steps
    contacts_before = int(simulation.contacts.count.numpy()[0])
    contact_candidates_before = int(simulation.contacts.candidate_count.numpy()[0])
    driver = None
    if args.capture:
        if not simulation.device.is_cuda:
            parser.error("CUDA profiling requires a CUDA device")
        driver = ctypes.WinDLL("nvcuda.dll")
        if driver.cuProfilerStart() != 0:
            raise RuntimeError("cuProfilerStart failed")

    samples = []
    started = time.perf_counter()
    try:
        for _ in range(args.steps):
            step_started = time.perf_counter()
            simulation.step()
            if renderer is not None:
                simulation.render()
            samples.append(
                (
                    (time.perf_counter() - step_started) * 1000,
                    simulation.iterations,
                    simulation.substep_dt,
                )
            )
        wp.synchronize_device(simulation.device)
        elapsed = time.perf_counter() - started
    finally:
        if driver is not None and driver.cuProfilerStop() != 0:
            raise RuntimeError("cuProfilerStop failed")
        if renderer is not None:
            renderer.close()

    values = np.asarray(samples)
    result = dict(
        scene=args.scene,
        device=str(simulation.device),
        warp_version=wp.__version__,
        capture=args.capture,
        render=args.render,
        particles=simulation.num_particles,
        boundary_particles=len(simulation.boundary.local),
        start_time=start_time,
        end_time=simulation.sim_time,
        start_step=start_step,
        steps=args.steps,
        substeps=args.steps * simulation.config.substeps,
        frame_dt=simulation.frame_dt,
        elapsed_seconds=elapsed,
        mean_step_ms=elapsed * 1000 / args.steps,
        mean_substep_ms=elapsed * 1000 / (args.steps * simulation.config.substeps),
        step_ms_percentiles=np.percentile(values[:, 0], [0, 50, 95, 100]).tolist(),
        mean_iterations=float(values[:, 1].mean()),
        max_iterations=int(values[:, 1].max()),
        mean_dt=float(values[:, 2].mean()),
        min_dt=float(values[:, 2].min()),
        contacts_before=contacts_before,
        contacts_after=int(simulation.contacts.count.numpy()[0]),
        contact_candidates_before=contact_candidates_before,
        contact_candidates_after=int(simulation.contacts.candidate_count.numpy()[0]),
        simulated_seconds_per_wall_second=(simulation.sim_time - start_time) / elapsed,
        sample_columns=["step_ms", "iterations_per_substep", "substep_dt"],
        samples=samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}), flush=True)


if __name__ == "__main__":
    main()
