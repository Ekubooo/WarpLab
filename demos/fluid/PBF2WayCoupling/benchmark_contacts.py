"""Time old/new contact sweeps on one frozen scene state, including precomputation.

Run with python -m demos.fluid.PBF2WayCoupling.benchmark_contacts.
This explicit diagnostic reads arrays and uses a test-only copy of the old solver.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from .PBF2WayCoupling import Example, prepare_contacts, prepare_rigid_contacts, solve_contacts
from .test_contact_reference import reference_contact_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/pbf2way/contact-benchmark.json")
    )
    args = parser.parse_args()
    simulation = Example(device="cuda:0")
    for _ in range(180):
        simulation.step()
    rigid, contacts = simulation.rigid, simulation.contacts
    arrays = [rigid.velocity, rigid.omega, contacts.normal_impulse, contacts.tangent_impulse]
    saved = [wp.clone(array) for array in arrays]

    def restore():
        for array, original in zip(arrays, saved):
            wp.copy(array, original)

    def original():
        for _ in range(5):
            wp.launch(reference_contact_sweep, 1, [rigid, contacts], device=simulation.device)

    def optimized():
        wp.launch(
            prepare_rigid_contacts, len(simulation.body_models), [rigid], device=simulation.device
        )
        wp.launch(
            prepare_contacts,
            simulation.config.max_contacts,
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
        contacts=int(contacts.count.numpy()[0]),
        maximum_absolute_errors=errors,
        timing="CUDA events, 100 frozen-state solves in graph, 7 trials; includes same D2D restores",
        original=measure(original),
        optimized=measure(optimized),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
