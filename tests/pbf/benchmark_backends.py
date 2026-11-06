"""Complete CPU/GPU coupled frames: wall time includes transfer and CPU work.

python -m tests.pbf.benchmark_backends --radius .0092 --warmup 180 --trials 60
Only this test exposes options; production backend selection is a code constant.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".warp_cache"))
import mujoco
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo
from PBF import coupling, scene, boundary_model
from PBF.pbf_helper import PBFConfig
from PBF.simulation import Simulation
from .coupled_lifecycle_case import checkpoint, restore, assert_capture_unchanged, state_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=.0125)
    parser.add_argument("--warmup", type=int, default=180)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private = ROOT / "outputs/pbf-newton/cpu-backend" / f"assets-{args.radius}"
    if not private.exists():
        source = ROOT / "outputs/pbf-newton/profiling-500k/Assets" if args.radius == .0092 else ROOT / "PBF/Assets"
        shutil.copytree(source if source.exists() else ROOT / "PBF/Assets", private)
    scene.ASSET_DIR = boundary_model.ASSET_DIR = private
    coupling.USE_MUJOCO_CPU = False
    with wp.ScopedDevice("cuda:0"):
        sim = Simulation(scene="stairs-rigids", config=PBFConfig(particle_radius=args.radius))
        gpu = sim.solver.rigid
        cpu = SolverMuJoCo(sim.model, use_mujoco_cpu=True, use_mujoco_contacts=True,
            update_data_interval=1, enable_sleeping=False, nconmax=2048, njmax=8192)
        sim.capture_graph()
        for _ in range(args.warmup):
            sim.launch_graph()
        saved_gpu = checkpoint(sim)
        # Common physical state and generalized acceleration warmstart. Each
        # backend retains its own native contact/scratch layout for replay.
        cpu._update_mjc_data(cpu.mj_data, sim.model, sim.state)
        cpu.mj_data.time = float(gpu.mjw_data.time.numpy()[0])
        mujoco.mj_forward(cpu.mj_model, cpu.mj_data)
        cpu.mj_data.qacc_warmstart[:] = gpu.mjw_data.qacc_warmstart.numpy()[0]
        cpu._step = gpu._step
        sim.solver.rigid, sim.use_mujoco_cpu = cpu, True
        saved_cpu = checkpoint(sim)
        backends = {"gpu_graph": (gpu, saved_gpu), "cpu_graph": (cpu, saved_cpu),
                    "cpu_eager": (cpu, saved_cpu)}
        graphs = {}
        for name, (rigid, saved) in backends.items():
            sim.solver.rigid, sim.use_mujoco_cpu = rigid, rigid.use_mujoco_cpu
            restore(sim, saved)
            sim.capture_graph()
            assert_capture_unchanged(sim, saved)
            graphs[name] = sim.step_graphs

        def select(name):
            rigid, saved = backends[name]
            sim.solver.rigid, sim.use_mujoco_cpu = rigid, rigid.use_mujoco_cpu
            restore(sim, saved)
            sim.step_graphs = graphs[name]

        # A single frozen-body fluid substep must be identical across backends.
        values = {}
        for name in ("gpu_graph", "cpu_graph"):
            select(name)
            sim.substep = 1
            sim.frame_dt = sim.config.frame_dt / sim.config.substeps
            sim.step()
            values[name] = state_values(sim)
        for name in ("particle_q", "particle_qd"):
            np.testing.assert_array_equal(values["cpu_graph"][name], values["gpu_graph"][name])
        np.testing.assert_allclose(values["cpu_graph"]["boundary_forces"],
                                   values["gpu_graph"]["boundary_forces"], atol=2e-3, rtol=2e-4)
        sim.substep, sim.frame_dt = sim.config.substeps, sim.config.frame_dt
        sim._graph_settings = sim._graph_parameters()
        difference = {name: float(np.max(np.abs(values["cpu_graph"][name] - values["gpu_graph"][name])))
                      for name in ("body_q", "body_qd", "boundary_forces")}
        samples = {name: [] for name in backends}
        rng = np.random.default_rng(715)
        for trial in range(args.trials + 15):
            for name in rng.permutation(list(backends)):
                select(name)
                wp.synchronize_device()
                start = time.perf_counter()
                if name == "cpu_eager":
                    sim._advance_frame()
                    sim.sim_time += sim.frame_dt
                else:
                    sim.launch_graph()
                wp.synchronize_device()
                elapsed = (time.perf_counter() - start) * 1000
                if trial >= 15:
                    samples[name].append(elapsed)
                assert all(np.isfinite(a).all() for a in state_values(sim).values())
        result = {}
        for name, timings in samples.items():
            ratio = np.array(timings) / samples["gpu_graph"]
            boot = np.random.default_rng(1).choice(ratio, size=(5000, args.trials))
            result[name] = dict(median_ms=float(np.median(timings)),
                p10_p90_ms=np.quantile(timings, [.1, .9]).tolist(),
                paired_ratio=float(np.median(ratio)),
                ratio_ci95=np.quantile(np.median(boot, axis=1), [.025, .975]).tolist(), samples_ms=timings)
        continuous = {}
        for name in ("gpu_graph", "cpu_graph"):
            select(name)
            wp.synchronize_device()
            start = time.perf_counter()
            for _ in range(180):
                sim.launch_graph()
            wp.synchronize_device()
            continuous[name] = (time.perf_counter() - start) * 1000 / 180
            assert all(np.isfinite(a).all() for a in state_values(sim).values())
        report = dict(particles=sim.n, warmup=args.warmup, trials=args.trials,
            method="Wall time with device synchronization, restore excluded, randomized pairs; common particle/rigid state and qacc warmstart; separate native backend histories.",
            results=result, continuous_ms=continuous, frozen_fluid_bitwise=True,
            single_substep_backend_differences=difference)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print({name: (r["median_ms"], r["ratio_ci95"]) for name, r in result.items()}, flush=True)
        print("Continuous", continuous, "Backend differences", difference, flush=True)


if __name__ == "__main__":
    main()
