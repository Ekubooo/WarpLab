"""Time old/new contact sweeps on one frozen scene state, including precomputation.

Run with python -m demos.fluid.PBF2WayCoupling.benchmark_contacts.
This explicit diagnostic reads arrays and uses a test-only copy of the old solver.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from .PBF2WayCoupling import (
    CONTACT_MANIFOLD_POINTS,
    Example,
    compact_contact_manifolds,
    initialize_contact_manifolds,
    prepare_contacts,
    prepare_rigid_contacts,
    resolve_contact_manifold_slot,
    score_contact_manifold_slot,
    solve_contacts,
)
from .test_contact_reference import reference_contact_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/pbf2way/contact-benchmark.json")
    )
    args = parser.parse_args()
    simulation = Example(device="cuda:0")
    for _ in range(round(3 / simulation.frame_dt)):
        simulation.step()
    rigid, contacts = simulation.rigid, simulation.contacts
    arrays = [rigid.velocity, rigid.omega, contacts.normal_impulse, contacts.tangent_impulse]
    # Manifolds intentionally do not warm start. Use the same zero-impulse state
    # for the reference sweeps and the compact solver comparison.
    saved = [
        wp.clone(rigid.velocity),
        wp.clone(rigid.omega),
        wp.zeros_like(contacts.normal_impulse),
        wp.zeros_like(contacts.tangent_impulse),
    ]

    def restore():
        for array, original in zip(arrays, saved):
            wp.copy(array, original)

    def original():
        for _ in range(5):
            wp.launch(reference_contact_sweep, 1, [rigid, contacts], device=simulation.device)

    def compression():
        wp.launch(
            initialize_contact_manifolds,
            simulation.max_manifold_contacts,
            [rigid, contacts],
            device=simulation.device,
        )
        for manifold_slot in range(CONTACT_MANIFOLD_POINTS):
            wp.launch(
                score_contact_manifold_slot,
                simulation.config.max_contacts,
                [
                    rigid,
                    contacts,
                    len(simulation.body_models),
                    simulation.config.particle_radius,
                    simulation.config.contact_tolerance,
                    manifold_slot,
                ],
                device=simulation.device,
            )
            wp.launch(
                resolve_contact_manifold_slot,
                simulation.config.max_contacts,
                [rigid, contacts, len(simulation.body_models), manifold_slot],
                device=simulation.device,
            )
        wp.launch(
            compact_contact_manifolds,
            1,
            [rigid, contacts, len(simulation.body_models)],
            device=simulation.device,
        )

    def optimized():
        compression()
        wp.launch(
            prepare_rigid_contacts, len(simulation.body_models), [rigid], device=simulation.device
        )
        wp.launch(
            prepare_contacts,
            simulation.max_manifold_contacts,
            [rigid, contacts],
            device=simulation.device,
        )
        wp.launch(solve_contacts, 1, [rigid, contacts, 5], device=simulation.device)

    restore()
    original()
    reference = [array.numpy() for array in arrays]
    restore()
    optimized()
    actual = [array.numpy() for array in arrays]
    errors = {}
    for name, before, after in zip(
        ["velocity", "omega", "normal_impulse", "tangent_impulse"], reference, actual
    ):
        np.testing.assert_allclose(after, before, atol=2e-6, rtol=2e-5)
        errors[name] = float(np.max(np.abs(after - before)))

    def measure(call):
        wp.synchronize_device(simulation.device)
        with wp.ScopedCapture(device=simulation.device) as capture:
            for _ in range(100):
                restore()
                call()
        start = wp.Event(device=simulation.device, enable_timing=True)
        end = wp.Event(device=simulation.device, enable_timing=True)
        samples = []
        for _ in range(7):
            wp.record_event(start)
            wp.capture_launch(capture.graph)
            wp.record_event(end)
            samples.append(wp.get_event_elapsed_time(start, end) * 1000 / 100)
        return dict(median_us=float(np.median(samples)), samples_us=samples)

    result = dict(
        contact_candidates=int(contacts.candidate_count.numpy()[0]),
        contacts=int(contacts.count.numpy()[0]),
        maximum_absolute_errors=errors,
        timing="CUDA events, 100 frozen-state solves in graph, 7 trials; includes same D2D restores",
        original=measure(original),
        compression=measure(compression),
        optimized=measure(optimized),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
