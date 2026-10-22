"""Bounded old/new neighbor comparison, GPU timings, and baseline test audit."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
wp.config.kernel_cache_dir = str(ROOT / ".warp_cache")
wp.init()

from demos.fluid.PBF2WayCoupling import PBF2WayCoupling as current


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


before = load("demos.fluid.PBF2WayCoupling._before_native", Path(__file__).parent / "solver/before_native.py")
output = ROOT / "outputs/pbf2way/native-sorted-query-20261022"
output.mkdir(parents=True, exist_ok=True)


def baseline_audit():
    module = load("tests.fluid.pbf2waycoupling._before_native_tests", Path(__file__).parent / "before_test_coupling.py")
    module.solver = before
    suite = unittest.TestSuite(module.CouplingTest(name) for name in (
        "test_cli_lambda_regularization", "test_cpu_cuda_initial_trajectory_and_finite_evolution",
    ))
    with (output / "baseline-existing-tests.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    print("BASELINE AUDIT:", result.testsRun, "tests", len(result.failures), "failures", len(result.errors), "errors", flush=True)
    return dict(tests=result.testsRun, failures=[(test.id(), detail) for test, detail in result.failures],
                errors=[(test.id(), detail) for test, detail in result.errors])


def arguments(module, sim):
    result = [sim.fluid_grid.id, sim.boundary_grid.id, sim.positions]
    if module is before:
        result.append(sim.old_to_sorted)
    return result + [sim.boundary, sim.support_radius, sim.neighbors]


def build_and_cache(module, sim):
    sim.fluid_grid.build(sim.positions, sim.support_radius)
    sim.boundary_grid.build(sim.boundary.position, sim.support_radius)
    sim._reorder_fluid()
    wp.launch(module.cache_neighbors, sim.num_particles, arguments(module, sim), device=sim.device)


def benchmark(radius):
    old = before.Example(before.PBF2WayCouplingConfig(particle_radius=radius), device="cuda:0")
    for _ in range(270):
        old.step()
    new = current.Example(current.PBF2WayCouplingConfig(particle_radius=radius), device="cuda:0")
    for name in ("positions", "velocities", "old_positions", "particle_ids"):
        getattr(new, name).assign(getattr(old, name).numpy())
    new.boundary.position.assign(old.boundary.position.numpy())
    new.boundary.velocity.assign(old.boundary.velocity.numpy())
    for module, sim in ((before, old), (current, new)):
        build_and_cache(module, sim)
    np.testing.assert_array_equal(old.positions.numpy(), new.positions.numpy())
    comparisons = {}
    for field, count_field in (("fluid", "fluid_count"), ("boundary", "boundary_count")):
        count = getattr(old.neighbors, count_field).numpy()
        np.testing.assert_array_equal(count, getattr(new.neighbors, count_field).numpy())
        a, b = getattr(old.neighbors, field).numpy(), getattr(new.neighbors, field).numpy()
        mask = np.arange(a.shape[0])[:, None] < count
        np.testing.assert_array_equal(a[mask], b[mask])
        comparisons[field] = dict(maximum=int(count.max()), entries=int(count.sum()), ordered_entries_identical=True)
    graphs = []
    repetitions = 200
    for module, sim in ((before, old), (current, new)):
        inputs = arguments(module, sim)
        for _ in range(20):
            wp.launch(module.cache_neighbors, sim.num_particles, inputs, device="cuda:0")
        wp.synchronize_device("cuda:0")
        with wp.ScopedCapture(device="cuda:0") as capture:
            for _ in range(repetitions):
                wp.launch(module.cache_neighbors, sim.num_particles, inputs, device="cuda:0")
        graphs.append(capture.graph)
    times = [[], []]
    with wp.ScopedDevice("cuda:0"):
        start, stop = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        for batch in range(9):
            for index in ((0, 1) if batch % 2 == 0 else (1, 0)):
                wp.record_event(start)
                wp.capture_launch(graphs[index])
                wp.record_event(stop)
                times[index].append(wp.get_event_elapsed_time(start, stop) / repetitions)
    old_ms, new_ms = [float(np.median(t)) for t in times]
    result = dict(radius=radius, particles=old.num_particles, frozen_seconds=old.sim_time,
                  cache_comparison=comparisons, before_ms=old_ms, after_ms=new_ms,
                  improvement_percent=(old_ms-new_ms)/old_ms*100, sample_ms=times,
                  note="GPU-event timings of 200 repeated cache kernels in CUDA graphs, 9 alternating batches")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    result = dict(baseline_tests=baseline_audit(), benchmarks=[])
    for radius in (0.025, 0.015625):
        result["benchmarks"].append(benchmark(radius))
        (output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
