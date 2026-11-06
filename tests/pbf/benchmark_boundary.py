"""Paired frozen-state and continuous timings of the first boundary optimization.

Run as python -m tests.pbf.benchmark_boundary --warmup 180 --trials 60.
Production choices are never changed on disk; test modules live under outputs.
All query variants use the current split delta/reaction stage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pbf-newton/optimization-round1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".warp_cache"))
import numpy as np
import warp as wp
from PBF import PBF as kernels, pbf_helper as fn
from PBF.simulation import Simulation
from tests.pbf.coupled_lifecycle_case import checkpoint, restore

parser = argparse.ArgumentParser()
parser.add_argument("--warmup", type=int, default=180)
parser.add_argument("--trials", type=int, default=60)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
from .query_variants import make_variants, install
from types import SimpleNamespace
original = SimpleNamespace(**{name: getattr(kernels, name) for name in
    ("volume_query", "calc_deltaPos", "calc_boundaryForce", "update_prePos")})
source_path = ROOT / "PBF/PBF.py"
digest = hashlib.sha256(source_path.read_bytes()).hexdigest()


def numeric_compare(value, base):
    result = {}
    for key, a in value.items():
        b = base[key]
        difference = a.astype(np.float64) - b.astype(np.float64)
        result[key] = dict(bitwise=bool(np.array_equal(a.view(np.uint8), b.view(np.uint8))),
                           finite=bool(np.isfinite(a).all()),
                           max_abs=float(np.abs(difference).max(initial=0)),
                           relative_l2=float(np.linalg.norm(difference) / max(np.linalg.norm(b.astype(np.float64)), 1e-30)))
    return result


def check_numerical(scope, name, measured):
    for field, error in measured.items():
        assert error["finite"], (scope, name, field, error)
        if field == "forces":
            assert error["max_abs"] <= 2e-3 or error["relative_l2"] < 2e-4, (scope, name, field, error)
        elif scope == "coupled_frame":
            limit = 2e-4 if field.endswith("qd") else 2e-5
            assert error["max_abs"] < limit, (scope, name, field, error)
        else:
            assert error["bitwise"], (scope, name, field, error)


with wp.ScopedDevice("cuda:0"):
    sim = Simulation(scene="stairs-rigids")
    variants = make_variants()
    install(variants["original"])
    sim.capture_graph()
    for _ in range(args.warmup):
        sim.launch_graph()
    wp.synchronize_device()
    solver = sim.fluid_solver
    full_saved = checkpoint(sim)

    def execute(scope):
        if scope == "coupled_frame":
            sim._advance_frame()
            return
        solver.boundary_forces.zero_()
        if scope == "fluid_substep":
            solver.step(sim.state, sim.state_out, None, None, sim.dt)
        elif scope == "volume_query":
            wp.launch(kernels.volume_query, sim.n, inputs=[solver.pre_Pos, solver.boundary_data],
                      outputs=[solver.volume_Rho, solver.volume_Grad])
        elif scope == "calc_deltaPos":
            # Keep the historical stage timing inclusive of pressure feedback.
            wp.launch(kernels.calc_deltaPos, sim.n,
                      inputs=[solver.pre_Pos, solver.lambda_Opt, solver.n_Cache, solver.volume_Grad],
                      outputs=[solver.delta_Pos])
            wp.launch(kernels.calc_boundaryForce, sim.n,
                      inputs=[solver.pre_Pos, solver.lambda_Opt, sim.dt, solver.boundary_data])
        else:
            wp.launch(kernels.update_prePos, sim.n, inputs=[solver.delta_Pos, solver.boundary_data], outputs=[solver.pre_Pos])

    def read_values(scope, end_index, end_prediction):
        if scope == "coupled_frame":
            state = sim.states[end_index]
            fields = {key: getattr(state, key) for key in ("particle_q", "particle_qd", "body_q", "body_qd")}
        elif scope == "fluid_substep":
            fields = dict(particle_q=sim.state_out.particle_q, particle_qd=sim.state_out.particle_qd,
                          lambda_value=solver.lambda_Opt, delta=solver.delta_Pos)
        elif scope == "volume_query":
            fields = dict(rho=solver.volume_Rho, gradient=solver.volume_Grad)
        elif scope == "calc_deltaPos":
            fields = dict(delta=solver.delta_Pos)
        else:
            fields = dict(prediction=end_prediction)
        fields["forces"] = solver.boundary_forces
        return {key: array.numpy().copy() for key, array in fields.items()}

    def benchmark(scope, saved):
        graphs, values = {}, {}
        for name, variant in variants.items():
            install(variant)
            restore(sim, saved)
            execute(scope)
            restore(sim, saved)
            with wp.ScopedCapture() as capture:
                execute(scope)
            end_index, end_prediction = sim._state_index, solver.pre_Pos
            graphs[name] = capture.graph
            restore(sim, saved)
            wp.capture_launch(capture.graph)
            values[name] = read_values(scope, end_index, end_prediction)
        numerical = {name: numeric_compare(value, values["original"]) for name, value in values.items()}
        for name, measured in numerical.items():
            check_numerical(scope, name, measured)
        samples = {key: [] for key in variants}
        begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        rng = np.random.default_rng(241)
        for trial in range(args.trials + 15):
            for name in rng.permutation(list(variants)):
                restore(sim, saved)
                wp.record_event(begin)
                wp.capture_launch(graphs[name])
                wp.record_event(end)
                us = wp.get_event_elapsed_time(begin, end) * 1000
                if trial >= 15:
                    samples[name].append(us)
        result = {}
        for name in variants:
            ratios = np.asarray(samples[name]) / np.asarray(samples["original"])
            bootstrap = np.random.default_rng(1).choice(ratios, size=(5000, args.trials), replace=True)
            result[name] = dict(median_us=float(np.median(samples[name])),
                                paired_ratio=float(np.median(ratios)),
                                ratio_ci95=np.quantile(np.median(bootstrap, axis=1), [.025, .975]).tolist(),
                                numerical=numerical[name],
                                samples_us=samples[name])
        print(scope, {key: (value["median_us"], value["ratio_ci95"]) for key, value in result.items()}, flush=True)
        return result

    results = {"coupled_frame": benchmark("coupled_frame", full_saved),
               "fluid_substep": benchmark("fluid_substep", full_saved)}
    # Compile all binding parities before timing continuous trajectories.
    graph_sets = {}
    for name, variant in variants.items():
        install(variant)
        restore(sim, full_saved)
        sim.capture_graph()
        graph_sets[name] = sim.step_graphs
    continuous = {name: [] for name in variants}
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    rng = np.random.default_rng(92)
    for repeat in range(3):
        for name in rng.permutation(list(variants)):
            install(variants[name])
            restore(sim, full_saved)
            sim.step_graphs = graph_sets[name]
            samples = []
            for frame in range(60):
                wp.record_event(start)
                sim.launch_graph()
                wp.record_event(end)
                samples.append(wp.get_event_elapsed_time(start, end) * 1000.)
            for array in (sim.pos, sim.v, sim.state.body_q, sim.state.body_qd, solver.boundary_forces):
                assert np.isfinite(array.numpy()).all(), name
            continuous[name].append(samples)
    continuous_report = {name: dict(median_us=float(np.median(samples)), batches_us=samples)
                         for name, samples in continuous.items()}
    print("continuous", {k: v["median_us"] for k, v in continuous_report.items()}, flush=True)
    restore(sim, full_saved)
    install(variants["original"])
    # Build true inputs for the first pressure iteration of the next substep.
    wp.launch(kernels.apply_predict, sim.n,
              inputs=[sim.state.particle_q, sim.dt, solver.gravity, sim.state.particle_qd], outputs=[solver.pre_Pos])
    solver.grid.build(solver.pre_Pos, solver.smoothing_length)
    fn.buffer_reorder(solver, sim.state.particle_q, sim.state_out.particle_q)
    wp.launch(kernels.cache_neighbors, sim.n, inputs=[solver.pre_Pos], outputs=[solver.n_Cache])
    execute("volume_query")
    wp.launch(kernels.calc_lambda, sim.n,
              inputs=[solver.pre_Pos, solver.n_Cache, solver.volume_Rho, solver.volume_Grad], outputs=[solver.lambda_Opt])
    execute("calc_deltaPos")
    partial_saved = checkpoint(sim)
    del full_saved
    for scope in ("volume_query", "calc_deltaPos", "update_prePos"):
        results[scope] = benchmark(scope, partial_saved)
    install(original)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == digest
    report = dict(particles=sim.n, bodies=sim.model.body_count, warmup_frames=args.warmup, trials=args.trials,
                  production_sha256=digest,
                  method="Random-order paired CUDA events over captured graphs, identical frozen inputs and complete MuJoCo history restored outside timing; no clock locking.",
                  results=results, continuous=continuous_report)
    path = args.output or OUT / f"boundary-{args.warmup}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RESULT", path, flush=True)
