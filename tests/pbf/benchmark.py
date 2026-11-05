"""Paired CUDA-event timings of frozen full substeps; restore copies are excluded."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import warp as wp

from .reference import independent_pair, verify_pair, reset_reference, WORK_ARRAYS


def capture_reference(reference):
    arrays = [getattr(reference, n) for n in ("pos", "v", "temp_pos", "temp_pPos") + WORK_ARRAYS]
    snapshots = [(a, wp.clone(a)) for a in arrays]
    bindings = reference.pos, reference.temp_pos, reference.pre_Pos, reference.temp_pPos
    time = reference.sim_time
    with wp.ScopedCapture() as capture:
        reference.step()
    reference.pos, reference.temp_pos, reference.pre_Pos, reference.temp_pPos = bindings
    reference.sim_time = time
    return capture.graph, snapshots


def capture_current(sim):
    solver = sim.solver
    arrays = [a for s in sim.states for a in (s.particle_q, s.particle_qd)]
    arrays += [getattr(solver, n) for n in WORK_ARRAYS + ("temp_pPos",)]
    snapshots = [(a, wp.clone(a)) for a in arrays]
    index, prediction = sim._state_index, (solver.pre_Pos, solver.temp_pPos)
    with wp.ScopedCapture() as capture:
        sim._substep(sim.dt)
    sim._state_index = index
    solver.pre_Pos, solver.temp_pPos = prediction
    return capture.graph, snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("outputs/pbf-newton/performance.json"))
    args = parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()), independent_pair() as (reference, sim):
        parity = verify_pair(reference, sim)
        reset_reference(reference)
        sim.reset()
        for _ in range(3):
            reference.step()
            sim._substep(sim.dt)
        cases = capture_reference(reference), capture_current(sim)
        start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        samples = [[], []]
        for trial in range(args.trials + 12):
            for i in ((0, 1) if trial % 2 == 0 else (1, 0)):
                graph, snapshots = cases[i]
                for target, source in snapshots:
                    wp.copy(target, source)
                wp.record_event(start)
                wp.capture_launch(graph)
                wp.record_event(end)
                elapsed = wp.get_event_elapsed_time(start, end) * 1000.
                if trial >= 12:
                    samples[i].append(elapsed)
    old, new = (float(np.median(s)) for s in samples)
    ratios = np.asarray(samples[1]) / np.asarray(samples[0])
    bootstrap = np.random.default_rng(1).choice(ratios, size=(5000, args.trials), replace=True)
    interval = np.quantile(np.median(bootstrap, axis=1), [.025, .975]).tolist()
    result = dict(parity=parity, legacy_median_us=old, new_median_us=new,
                  new_over_legacy=new / old, paired_ratio_95pct_bootstrap=interval,
                  trials=args.trials, legacy_samples_us=samples[0], new_samples_us=samples[1],
                  timing="CUDA events, frozen complete substeps; restore copies excluded; alternating order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Original {old:.2f} us; new {new:.2f} us; ratio {new / old:.4f}; paired CI {interval}")


if __name__ == "__main__":
    main()
