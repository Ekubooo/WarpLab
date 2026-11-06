"""Paired regression/timing against the frozen pre-split delta/reaction kernel.

Run: python -m tests.pbf.benchmark_feedback --warmup 180 --trials 60
Uses private 494100-particle assets; never writes production assets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pbf-newton/feedback-split"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".warp_cache"))
import numpy as np
import warp as wp
from warp._src.context import get_cuda_kernel_properties
from PBF import PBF as kernels, pbf_helper as fn, scene, boundary_model
from PBF.simulation import Simulation
from tests.pbf.coupled_lifecycle_case import checkpoint, restore

parser = argparse.ArgumentParser()
parser.add_argument("--warmup", type=int, default=180)
parser.add_argument("--trials", type=int, default=60)
parser.add_argument("--seed", type=int, default=241)
parser.add_argument("--continuous-seed", type=int, default=92)
parser.add_argument("--tag", default="")
args = parser.parse_args()

source_paths = [p for p in (ROOT / "PBF").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
private = ROOT / "outputs/pbf-newton/profiling-500k/Assets"
if not private.exists():
    shutil.copytree(ROOT / "PBF/Assets", private)
scene.ASSET_DIR = boundary_model.ASSET_DIR = private


def compare(value, reference):
    result = {}
    for field, a in value.items():
        b = reference[field]
        delta = a.astype(np.float64) - b.astype(np.float64)
        result[field] = dict(bitwise=bool(np.array_equal(a.view(np.uint8), b.view(np.uint8))),
            finite=bool(np.isfinite(a).all()), max_abs=float(np.max(np.abs(delta), initial=0)),
            relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(b.astype(np.float64)), 1e-30)))
    return result


with wp.ScopedDevice("cuda:0"):
    sim = Simulation(scene="stairs-rigids", config=fn.PBFConfig(particle_radius=.0092))
    assert sim.n == 494100
    fluid = sim.fluid_solver
    from . import feedback_reference as reference
    variants = ("fused", "split")
    # The fused and split delta kernels have the same name but different modules.
    properties = {
        "fused_delta": get_cuda_kernel_properties(reference.calc_deltaPos, "cuda:0", 256),
        "split_delta": get_cuda_kernel_properties(kernels.calc_deltaPos, "cuda:0", 256),
        "reaction": get_cuda_kernel_properties(kernels.calc_boundaryForce, "cuda:0", 256),
    }
    print("CUDA properties", properties, flush=True)
    selected = "fused"

    class WarpLaunchOverride:
        def __getattr__(self, name):
            return getattr(wp, name)

        def launch(self, kernel, dim, inputs=(), outputs=(), **kwargs):
            if selected == "fused":
                if kernel is kernels.calc_deltaPos:
                    return wp.launch(reference.calc_deltaPos, dim,
                        inputs=[*inputs, sim.dt, fluid.boundary_data], outputs=outputs, **kwargs)
                if kernel is kernels.calc_boundaryForce:
                    return
            return wp.launch(kernel, dim, inputs=inputs, outputs=outputs, **kwargs)

    launch = WarpLaunchOverride().launch

    original_step = fluid.step
    step_function = original_step.__func__
    # Reuse the exact production step, changing only its local wp.launch lookup.
    scope = dict(step_function.__globals__, wp=WarpLaunchOverride())
    patched_step = types.FunctionType(step_function.__code__, scope, argdefs=step_function.__defaults__)
    patched_step.__kwdefaults__ = step_function.__kwdefaults__
    fluid.step = types.MethodType(patched_step, fluid)
    sim.capture_graph()
    for _ in range(args.warmup):
        sim.launch_graph()
    wp.synchronize_device()
    saved_frame = checkpoint(sim)

    def execute(scope):
        if scope == "coupled_frame":
            sim._advance_frame()
        else:
            fluid.boundary_forces.zero_()
            if scope == "fluid_substep":
                fluid.step(sim.state, sim.state_out, None, None, sim.dt)
            else:
                launch(kernels.calc_deltaPos, sim.n,
                    inputs=[fluid.pre_Pos, fluid.lambda_Opt, fluid.n_Cache, fluid.volume_Grad],
                    outputs=[fluid.delta_Pos])
                launch(kernels.calc_boundaryForce, sim.n,
                    inputs=[fluid.pre_Pos, fluid.lambda_Opt, sim.dt, fluid.boundary_data])
                launch(kernels.update_prePos, sim.n,
                    inputs=[fluid.delta_Pos, fluid.boundary_data], outputs=[fluid.pre_Pos])

    def values(scope):
        if scope == "coupled_frame":
            fields = {name: getattr(sim.state, name) for name in
                      ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")}
        elif scope == "fluid_substep":
            fields = dict(particle_q=sim.state_out.particle_q, particle_qd=sim.state_out.particle_qd,
                          rho=fluid.volume_Rho, gradient=fluid.volume_Grad, lam=fluid.lambda_Opt,
                          prediction=fluid.pre_Pos, delta=fluid.delta_Pos)
        else:
            fields = dict(delta=fluid.delta_Pos, prediction=fluid.pre_Pos)
        fields["forces"] = fluid.boundary_forces
        return {k: v.numpy().copy() for k, v in fields.items()}

    def benchmark(scope, saved):
        global selected
        graphs, numerical = {}, {}
        base = None
        for selected in variants:
            restore(sim, saved)
            execute(scope)  # Compile all dependencies before capture.
            restore(sim, saved)
            with wp.ScopedCapture() as capture:
                execute(scope)
            end_index = sim._state_index
            prediction = fluid.pre_Pos, fluid.temp_pPos
            graphs[selected] = capture.graph
            restore(sim, saved)
            wp.capture_launch(capture.graph)
            sim._state_index = end_index
            fluid.pre_Pos, fluid.temp_pPos = prediction
            current = values(scope)
            if base is None:
                base = current
            measured = compare(current, base)
            for field, error in measured.items():
                assert error["finite"], (scope, selected, field, error)
                if field == "forces":
                    np.testing.assert_allclose(current[field], base[field], atol=2e-3, rtol=2e-4)
                elif scope == "coupled_frame":
                    np.testing.assert_allclose(current[field], base[field],
                        atol=2e-4 if field.endswith("qd") else 2e-5, rtol=2e-4)
                else:
                    assert error["bitwise"], (scope, selected, field, error)
            numerical[selected] = measured
        samples = {name: [] for name in variants}
        begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        rng = np.random.default_rng(args.seed)
        for trial in range(args.trials + 15):
            for name in rng.permutation(list(variants)):
                restore(sim, saved)
                wp.record_event(begin)
                wp.capture_launch(graphs[name])
                wp.record_event(end)
                us = wp.get_event_elapsed_time(begin, end) * 1000.
                if trial >= 15:
                    samples[name].append(us)
        result = {}
        for name in variants:
            ratios = np.array(samples[name]) / np.array(samples["fused"])
            boot = np.random.default_rng(1).choice(ratios, size=(5000, args.trials))
            result[name] = dict(median_us=float(np.median(samples[name])), paired_ratio=float(np.median(ratios)),
                ratio_ci95=np.quantile(np.median(boot, axis=1), [.025, .975]).tolist(),
                samples_us=samples[name], numerical=numerical[name])
        print(scope, {k: (v['median_us'], v['ratio_ci95']) for k,v in result.items()}, flush=True)
        return result

    results = {"coupled_frame": benchmark("coupled_frame", saved_frame),
               "fluid_substep": benchmark("fluid_substep", saved_frame)}
    # Independently capture both State/prediction parities for continuous runs.
    graphs = {}
    for selected in variants:
        restore(sim, saved_frame)
        sim.capture_graph()
        graphs[selected] = sim.step_graphs
    continuous = {name: [] for name in variants}
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    rng = np.random.default_rng(args.continuous_seed)
    for repeat in range(3):
        for selected in rng.permutation(list(variants)):
            restore(sim, saved_frame)
            sim.step_graphs = graphs[selected]
            samples = []
            for _ in range(60):
                wp.record_event(begin)
                sim.launch_graph()
                wp.record_event(end)
                samples.append(wp.get_event_elapsed_time(begin, end) * 1000.)
            assert all(np.isfinite(a.numpy()).all() for a in
                (sim.pos, sim.v, sim.state.body_q, sim.state.body_qd, fluid.boundary_forces))
            continuous[selected].append(samples)
    continuous = {k: dict(median_us=float(np.median(v)), batches_us=v) for k,v in continuous.items()}
    print("continuous", {k: v['median_us'] for k,v in continuous.items()}, flush=True)

    restore(sim, saved_frame)
    selected = "fused"
    # Exact first-iteration inputs, prepared from the same reference checkpoint.
    wp.launch(kernels.apply_predict, sim.n, inputs=[sim.state.particle_q, sim.dt, fluid.gravity,
        sim.state.particle_qd], outputs=[fluid.pre_Pos])
    fluid.grid.build(fluid.pre_Pos, fluid.smoothing_length)
    fn.buffer_reorder(fluid, sim.state.particle_q, sim.state_out.particle_q)
    wp.launch(kernels.cache_neighbors, sim.n, inputs=[fluid.pre_Pos], outputs=[fluid.n_Cache])
    wp.launch(kernels.volume_query, sim.n, inputs=[fluid.pre_Pos, fluid.boundary_data],
        outputs=[fluid.volume_Rho, fluid.volume_Grad])
    wp.launch(kernels.calc_lambda, sim.n, inputs=[fluid.pre_Pos, fluid.n_Cache,
        fluid.volume_Rho, fluid.volume_Grad], outputs=[fluid.lambda_Opt])
    saved_delta = checkpoint(sim)
    del saved_frame
    results["pressure_iteration"] = benchmark("pressure_iteration", saved_delta)
    fluid.step = original_step
    after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    assert before == after, "Production source/assets changed"
    report = dict(particles=sim.n, radius=.0092, warmup_frames=args.warmup, trials=args.trials,
        seed=args.seed, continuous_seed=args.continuous_seed,
        block_size=256, cuda_properties=properties, results=results, continuous=continuous,
        production_unchanged=True, production_sha256=before,
        method="Random-order paired CUDA events, frozen inputs/full MuJoCo history restored outside timing; pressure_iteration includes delta, feedback and position projection for every variant; no clock lock. Continuous: three batches of 60 frames from the same checkpoint.")
    path = OUT / f"feedback-{args.warmup}{args.tag}.json"
    path.write_text(json.dumps(report, indent=2))
    print("RESULT", path, flush=True)
