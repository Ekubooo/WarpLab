"""CUDA-event timings for complete coupled frames, excluding rendering."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import warp as wp

from demos.PBF_Rigid.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", choices=("tank-box", "stairs-rigids"), default="tank-box")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/pbf-mujoco-coupling/coupled-performance.json"))
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("frames must be positive and warmup nonnegative")
    with wp.ScopedDevice(args.device):
        sim = Simulation(scene=args.scene)
        sim.capture_graph()
        for _ in range(args.warmup):
            sim.launch_graph()
        start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        times = []
        for _ in range(args.frames):
            wp.record_event(start)
            sim.launch_graph()
            wp.record_event(end)
            times.append(wp.get_event_elapsed_time(start, end))
        positions, velocities = sim.pos.numpy(), sim.v.numpy()
        bodies, body_velocities = sim.state.body_q.numpy(), sim.state.body_qd.numpy()
        if not all(np.isfinite(values).all() for values in (positions, velocities, bodies, body_velocities)):
            raise RuntimeError("Non-finite simulation state")
        result = dict(scene=args.scene, device=sim.model.device.name, config=asdict(sim.config), particles=sim.n,
            bodies=sim.model.body_count, frames=args.frames, warmup=args.warmup,
            timing="CUDA events around complete coupled frame graphs; rendering and capture excluded",
            median_frame_ms=float(np.median(times)),
            median_amortized_substep_us=float(np.median(times)) * 1000. / sim.substep,
            final_time=sim.sim_time, final_body_q=bodies.tolist(), finite=True,
            frame_samples_ms=times)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Coupled frame: {result['median_frame_ms']:.3f} ms; "
          f"substep: {result['median_amortized_substep_us']:.2f} us (amortized)")
    print(args.output)


if __name__ == "__main__":
    main()
