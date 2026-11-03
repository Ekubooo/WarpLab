"""Full-scale stairs/rigid acceptance trajectory and CUDA-event timings.

Run separately from unit tests and other GPU work. Downloads and diagnostics
are outside the timing events; samples retain body motion, reactions and the
native MuJoCo contact pairs rather than only reporting a finite end state.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import warp as wp

from Lab_Framework.pbf.simulation import Simulation
from tests.lab_framework.test_rigid_sdf import _sample


def contact_snapshot(rigid):
    data, model = rigid.mjw_data, rigid.mj_model
    count = int(data.nacon.numpy()[0])
    constraints = int(data.nefc.numpy().max())
    if count > data.naconmax or constraints > data.njmax:
        raise AssertionError(f"Contact/constraint overflow: {count}/{constraints}")
    pairs = set()
    geoms = data.contact.geom.numpy()[:count]
    distance = data.contact.dist.numpy()[:count]
    for (a, b), separation in zip(geoms, distance):
        # Native contact storage can include separated candidates. Only report
        # touching/penetrating geometry as evidence of an actual contact pair.
        if a >= 0 and b >= 0 and separation <= 0.:
            names = [model.body(int(model.geom_bodyid[g])).name or "world" for g in (a, b)]
            pairs.add(tuple(sorted(names)))
    return dict(count=count, active_count=int(np.count_nonzero(distance <= 0.)),
                constraints=constraints, pairs=sorted(pairs),
                minimum_distance=float(distance.min()) if count else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/stairs-rigids/acceptance.json"))
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.sample_every < 1:
        parser.error("seconds and sample-every must be positive")
    with wp.ScopedDevice(args.device):
        sim = Simulation(scene="stairs-rigids", feedback=not args.no_feedback)
        if sim.n != 195525 or sim.model.body_count != 4:
            raise AssertionError("Acceptance must use the original particle scale and four bodies")
        initial = sim.pos.numpy()
        distance = wp.empty(sim.n, dtype=float)
        normals = wp.empty(sim.n, dtype=wp.vec3)
        wp.launch(_sample, sim.n, inputs=[sim.fluid_solver.static_sdf.id, sim.pos],
                  outputs=[distance, normals])
        overlap = int(np.count_nonzero(distance.numpy() < 0.))
        if overlap == 0:
            raise AssertionError("The original overlapping particle initialization must be retained")
        sim.capture_graph()
        np.testing.assert_array_equal(sim.pos.numpy(), initial)
        if sim.sim_time != 0.:
            raise AssertionError("Graph capture advanced the physical clock")
        start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        times, samples = [], []
        frames = int(np.ceil(args.seconds / sim.frame_dt))
        for frame in range(frames):
            wp.record_event(start)
            sim.launch_graph()
            wp.record_event(end)
            times.append(wp.get_event_elapsed_time(start, end))
            if frame % args.sample_every == 0 or frame + 1 == frames:
                particles, velocity = sim.pos.numpy(), sim.v.numpy()
                pose, body_velocity = sim.state.body_q.numpy(), sim.state.body_qd.numpy()
                forces = sim.solver.boundary_forces.numpy()
                if not all(np.isfinite(a).all() for a in (particles, velocity, pose, body_velocity, forces)):
                    raise AssertionError(f"Non-finite motion at t={sim.sim_time}")
                # Generous bounds around the closed reference container detect
                # escaped bodies without treating compliant contact as failure.
                if np.any(pose[:, :3] < [-.3, -.35, -.3]) or np.any(pose[:, :3] > [3.4, 4.9, 1.9]):
                    raise AssertionError(f"A rigid body escaped at t={sim.sim_time}: {pose[:, :3]}")
                samples.append(dict(time=sim.sim_time, body_q=pose.tolist(), body_qd=body_velocity.tolist(),
                    boundary_forces=forces.tolist(), contacts=contact_snapshot(sim.solver.rigid),
                    max_fluid_speed=float(np.linalg.norm(velocity, axis=1).max())))
            if (frame + 1) % 150 == 0:
                print(f"Verified {sim.sim_time:.2f} s / {args.seconds:g} s", flush=True)
        contact_pairs = sorted({tuple(pair) for sample in samples for pair in sample["contacts"]["pairs"]})
        result = dict(scene=sim.scene, device=sim.model.device.name, feedback=not args.no_feedback,
            config=asdict(sim.config), particles=sim.n, bodies=sim.model.body_count,
            initial_particles_inside_static=overlap, frames=frames, final_time=sim.sim_time,
            finite=True, captured_without_advancing=True, contact_pairs=contact_pairs,
            maximum_contacts=max(sample["contacts"]["count"] for sample in samples),
            maximum_constraints=max(sample["contacts"]["constraints"] for sample in samples),
            median_frame_ms=float(np.median(times[60:] if len(times) > 60 else times)),
            timing="CUDA events around full coupled frames; first 60 frames excluded from median",
            samples=samples, frame_samples_ms=times)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {frames} frames, {sim.sim_time:.3f} s, {overlap} original overlapping particles")
    print(f"Coupled frame median: {result['median_frame_ms']:.3f} ms; contacts: {contact_pairs}")
    print(args.output)


if __name__ == "__main__":
    main()
