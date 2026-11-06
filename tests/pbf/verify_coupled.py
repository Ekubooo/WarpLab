"""Full-scale coupled trajectory, contact capacities and CUDA-event timings."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from mujoco_warp._src.types import OverflowType
import numpy as np
import warp as wp

from PBF.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/pbf-newton/coupled-trajectory.json"))
    args = parser.parse_args()
    with wp.ScopedDevice("cuda:0"):
        sim = Simulation(scene="stairs-rigids", feedback=not args.no_feedback)
        assert sim.n == 195525 and sim.model.body_count == 3
        initial = sim.pos.numpy()
        sim.capture_graph()
        np.testing.assert_array_equal(sim.pos.numpy(), initial)
        assert sim.sim_time == 0.
        rigid = sim.solver.rigid
        data = rigid.mjw_data
        mask = int(OverflowType.ALL & ~(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS))
        frames = int(np.ceil(args.seconds / sim.frame_dt))
        start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        times, samples, peaks = [], [], dict(contacts=0, constraints=0, broadphase=0)
        maximum_pressure = 0.
        for frame in range(frames):
            wp.record_event(start)
            sim.launch_graph()
            wp.record_event(end)
            times.append(wp.get_event_elapsed_time(start, end))
            count = int(data.nacon.numpy()[0])
            rows = int(data.nefc.numpy().max())
            pairs = int(data.ncollision.numpy()[0])
            overflow = int(np.bitwise_or.reduce(data.overflow.numpy()))
            assert not overflow & mask, (frame, overflow)
            assert count <= data.naconmax and pairs <= data.naconmax and rows <= data.njmax
            for key, value in zip(peaks, (count, rows, pairs)):
                peaks[key] = max(peaks[key], value)
            poses, velocity = sim.state.body_q.numpy(), sim.state.body_qd.numpy()
            force = sim.solver.boundary_forces.numpy()
            assert all(np.isfinite(a).all() for a in (poses, velocity, force))
            assert np.all(poses[:, :3] > [-.3, -.35, -.3]) and np.all(poses[:, :3] < [3.4, 4.9, 1.9])
            maximum_pressure = max(maximum_pressure, float(np.linalg.norm(force[:, :3], axis=1).max()))
            if frame % 15 == 0 or frame + 1 == frames:
                assert np.isfinite(sim.pos.numpy()).all() and np.isfinite(sim.v.numpy()).all()
                geom = data.contact.geom.numpy()[:count]
                distances = data.contact.dist.numpy()[:count]
                actual = geom[distances <= 0.]
                active_pairs = sorted({tuple(sorted(rigid.mj_model.body(int(rigid.mj_model.geom_bodyid[g])).name
                    or "world" for g in pair)) for pair in actual})
                minimum = float(distances.min()) if count else 0.
                assert minimum > -.05, (frame, "excessive contact penetration", minimum)
                samples.append(dict(time=sim.sim_time, body_q=poses.tolist(), body_qd=velocity.tolist(),
                    boundary_forces=force.tolist(), contacts=count, constraints=rows,
                    broadphase=pairs, minimum_distance=minimum, active_pairs=active_pairs))
            if (frame + 1) % 150 == 0:
                print(f"Verified {sim.sim_time:.2f} simulated seconds", flush=True)
        assert maximum_pressure > 0., "No fluid pressure ever reached the rigid bodies"
        result = dict(scene=sim.scene, feedback=not args.no_feedback, config=asdict(sim.config),
            particles=sim.n, bodies=3, frames=frames, simulated_seconds=sim.sim_time,
            finite=True, escaped=False, capacity_overflow=False, capacities=dict(nconmax=data.naconmax, njmax=data.njmax),
            peaks=peaks, maximum_pressure_force=maximum_pressure, samples=samples,
            median_frame_ms=float(np.median(times[60:])), frame_samples_ms=times,
            timing="CUDA events, full coupled frames, first 60 frames excluded; diagnostics outside events")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"PASS: {sim.sim_time:.3f}s, median frame {result['median_frame_ms']:.3f}ms; peaks {peaks}")


if __name__ == "__main__":
    main()
