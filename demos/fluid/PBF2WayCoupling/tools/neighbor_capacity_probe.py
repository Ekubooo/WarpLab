"""Measure uncapped fluid neighborhoods before every pressure solve on CUDA.

The production solver is not edited. A launch interceptor counts neighbors before
cache_neighbors and grows the TEST cache before writing it, so 256 is not a
censored observation. Requested time settings bypass only the demo's fixed-time
configuration restriction in this harness. This is a correctness/stress probe,
not a performance benchmark (it synchronizes every substep).

Run: .venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.tools.neighbor_capacity_probe
"""

import argparse
import csv
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import warp as wp

from .. import PBF2WayCoupling as solver
from .. import coupling_functions as fn
from .. import coupling_initialization as init


@wp.kernel
def exact_counts(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    h: float,
    volume: float,
    counts: wp.array(dtype=int),
    density: wp.array(dtype=float),
    speeds: wp.array(dtype=float),
):
    i = wp.tid()
    count = int(0)
    rho = volume * fn.poly6(0.0, h)
    if wp.isfinite(x[i][0]) and wp.isfinite(x[i][1]) and wp.isfinite(x[i][2]):
        query = wp.hash_grid_query(grid, x[i], h)
        input_index = int(0)
        while wp.hash_grid_query_next(query, input_index):
            j = solver.hash_grid_query_sorted_index(query)
            distance_sq = wp.length_sq(x[i] - x[j])
            if j != i and distance_sq < h * h:
                count += 1
                rho += volume * fn.poly6(wp.sqrt(distance_sq), h)
    else:
        count = -1
    counts[i] = count
    density[i] = rho
    speeds[i] = wp.length(v[i])


@wp.kernel
def runtime_impulse(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    center: wp.vec3,
    extent: float,
    speed: float,
    mode: int,
):
    i = wp.tid()
    if mode == 0:
        # Affine radial inflow: the outermost sample receives the requested speed.
        v[i] = -(speed / extent) * (x[i] - center)
    else:
        direction = float(-1.0)
        if x[i][0] < center[0]:
            direction = 1.0
        v[i] = wp.vec3(direction * speed, 0.0, 0.0)


def next_capacity(count):
    return max(32, 1 << (int(count) - 1).bit_length())


def write_stress_scenes(output):
    wall = dict(
        geometryFile=str((init.ROOT / "assets/UnitBox.obj").resolve()),
        scale=[1.6, 1.6, 1.6], translation=[0, 0.8, 0], isWall=True,
    )
    blocks = {
        "radial": [dict(start=[-0.5, 0.3, -0.5], end=[0.5, 1.3, 0.5])],
        "head_on": [
            dict(start=[-0.6, 0.3, -0.5], end=[-0.1, 1.3, 0.5]),
            dict(start=[0.1, 0.3, -0.5], end=[0.6, 1.3, 0.5]),
        ],
    }
    paths = {}
    for name, fluid_blocks in blocks.items():
        path = output / (name + "_scene.json")
        path.write_text(json.dumps(dict(RigidBodies=[wall], FluidBlocks=fluid_blocks), indent=2))
        paths[name] = str(path.resolve())
    return paths


def case_id(case):
    return "_".join(str(case[k]) for k in ("profile", "scene_name", "fps", "substeps", "iterations", "cap", "kick"))


class Probe:
    def __init__(self, simulation, case, output):
        self.sim = simulation
        self.case = case
        self.output = output
        self.raw_launch = wp.launch
        n = simulation.num_particles
        self.counts = wp.zeros(n, dtype=int, device=simulation.device)
        self.density = wp.zeros(n, dtype=float, device=simulation.device)
        self.speeds = wp.zeros(n, dtype=float, device=simulation.device)
        self.histogram = np.zeros(n, dtype=np.int64)
        self.substep = 0
        self.peak = -1
        self.peak_row = None
        self.first_over_256 = None
        self.first_over_128 = None
        self.max_speed = 0.0
        self.max_density = 0.0
        self.kicked = False
        self.growth = []
        self.validation = []
        self.history = (output / (case_id(case) + ".jsonl")).open("w", encoding="utf-8")

    def validate(self, index, expected, capture=False):
        sim = self.sim
        positions = sim.positions.numpy()
        difference = positions - positions[index]
        distances_sq = np.sum(difference * difference, axis=1, dtype=np.float32)
        h2 = np.float32(sim.support_radius) ** np.float32(2)
        mask = distances_sq < h2
        mask[index] = False
        actual = int(mask.sum())
        near_cutoff = int(np.count_nonzero(np.abs(distances_sq - h2) < 2e-7 * h2))
        if abs(actual - expected) > near_cutoff:
            raise AssertionError(f"independent brute force: {actual} != GPU {expected}")
        result = dict(substep=self.substep, gpu=int(expected), brute_force=actual, cutoff_ties=near_cutoff)
        self.validation.append(result)
        if capture:
            np.savez_compressed(
                self.output / (case_id(self.case) + "_peak.npz"),
                positions=positions, velocities=sim.velocities.numpy(),
                counts=self.counts.numpy(), index=index, h=sim.support_radius,
            )

    def observe(self, inputs):
        sim = self.sim
        self.raw_launch(
            exact_counts, sim.num_particles,
            [inputs[0], inputs[2], sim.velocities, sim.support_radius,
             sim.fluid_volume, self.counts, self.density, self.speeds], device=sim.device,
        )
        counts = self.counts.numpy()
        if np.any(counts < 0):
            raise FloatingPointError("Non-finite predicted positions before neighbor caching")
        self.histogram += np.bincount(counts, minlength=sim.num_particles)
        index = int(np.argmax(counts))
        maximum = int(counts[index])
        densities = self.density.numpy()
        speeds = self.speeds.numpy()
        self.max_speed = max(self.max_speed, float(speeds.max()))
        self.max_density = max(self.max_density, float(densities.max()))
        row = dict(
            substep=self.substep, time=(self.substep + 1) * sim.substep_dt,
            maximum=maximum, mean=float(counts.mean()),
            p99=float(np.percentile(counts, 99)),
            over_128=int(np.count_nonzero(counts > 128)),
            over_256=int(np.count_nonzero(counts > 256)),
            max_speed=float(speeds.max()), max_fluid_only_density=float(densities.max()),
            density_at_max_count=float(densities[index]),
        )
        if maximum > self.peak:
            self.peak = maximum
            self.peak_row = row.copy()
            if maximum > 128 or self.substep == 0:
                self.validate(index, maximum, capture=maximum > 256)
        if maximum > 128 and self.first_over_128 is None:
            self.first_over_128 = row.copy()
        if maximum > 256 and self.first_over_256 is None:
            self.first_over_256 = row.copy()
        capacity = int(sim.neighbors.fluid.shape[0])
        if maximum > capacity:
            new_capacity = next_capacity(maximum)
            sim.neighbors.fluid = wp.zeros((new_capacity, sim.num_particles), dtype=int, device=sim.device)
            self.growth.append(dict(substep=self.substep, old=capacity, new=new_capacity))
        self.history.write(json.dumps(row) + "\n")
        self.substep += 1

    def launch(self, kernel, dim=None, inputs=None, *args, **kwargs):
        if kernel is solver.predict and self.case["profile"] in ("radial", "head_on"):
            if not self.kicked and self.substep * self.sim.substep_dt >= 0.5 - 1e-8:
                positions = self.sim.positions.numpy()
                center = positions.mean(axis=0)
                extent = float(np.linalg.norm(positions - center, axis=1).max())
                self.raw_launch(
                    runtime_impulse, self.sim.num_particles,
                    [self.sim.positions, self.sim.velocities, wp.vec3(*center), extent,
                     float(self.case["kick"]), int(self.case["profile"] == "head_on")],
                    device=self.sim.device,
                )
                self.kicked = True
        if kernel is solver.cache_neighbors:
            self.observe(inputs)
        result = self.raw_launch(kernel, dim, inputs, *args, **kwargs)
        if kernel is solver.cache_neighbors and (self.substep == 1 or self.peak > 256):
            np.testing.assert_array_equal(self.counts.numpy(), self.sim.neighbors.fluid_count.numpy())
        return result

    def finish(self, status, wall_seconds, error=None):
        self.history.close()
        cumulative = np.cumsum(self.histogram)
        total = int(cumulative[-1])
        percentiles = {
            str(p): int(np.searchsorted(cumulative, total * p / 100, side="left"))
            for p in (50, 95, 99, 99.9, 99.99)
        } if total else {}
        return dict(
            **self.case, status=status, error=error, particles=self.sim.num_particles,
            substeps_observed=self.substep, simulated_seconds=self.substep * self.sim.substep_dt,
            peak=self.peak, peak_event=self.peak_row, first_over_128=self.first_over_128,
            first_over_256=self.first_over_256, percentiles=percentiles,
            max_predicted_speed=self.max_speed, max_fluid_only_density=self.max_density,
            test_cache_growth=self.growth, brute_force_checks=self.validation,
            count_histogram={str(i): int(n) for i, n in enumerate(self.histogram) if n},
            wall_seconds=wall_seconds, effective_config=vars(self.sim.config),
            note="Exact every-substep count; test cache grows before solver use. Density excludes boundary samples.",
        )


def run_case(case, output, device, seconds):
    config = solver.PBF2WayCouplingConfig(
        scene=case["scene"], max_fluid_neighbors=256, max_boundary_neighbors=2048,
        pressure_iterations=case["iterations"], max_speed=6.0 if case["cap"] == "6" else 1e30,
        gravity=(0.0, 0.0, 0.0) if case["profile"] in ("radial", "head_on") else (0.0, -9.81, 0.0),
    )
    sim = solver.Example(config=config, device=device)
    # The demo rejects non-default time settings in __post_init__. Change only
    # the test instance after valid initialization; all original kernels remain.
    sim.config = SimpleNamespace(**asdict(config))
    sim.config.frame_dt = 1.0 / case["fps"]
    sim.config.substeps = case["substeps"]
    sim.frame_dt = sim.config.frame_dt
    sim.substep_dt = sim.frame_dt / sim.config.substeps
    sim.current_dt = sim.last_dt = sim.frame_dt
    probe = Probe(sim, case, output)
    started = time.perf_counter()
    status, error = "complete", None
    original_launch = wp.launch
    wp.launch = probe.launch
    try:
        for step in range(math.ceil(seconds * case["fps"])):
            if case["profile"] == "gravity_switch":
                phase = min(int(step * sim.frame_dt), 5)
                sim.config.gravity = [
                    (0, -9.81, 0), (0, 9.81, 0), (9.81, 0, 0),
                    (0, -9.81, 0), (-9.81, 0, 0), (0, -9.81, 0),
                ][phase]
            sim.step()
            fault = int(sim.rigid.fault.numpy()[0])
            if fault:
                status, error = "device_fault", f"fault bits={fault}"
                break
    except Exception as exc:
        status, error = "error", repr(exc)
    finally:
        wp.launch = original_launch
    wp.synchronize_device(sim.device)
    result = probe.finish(status, time.perf_counter() - started, error)
    (output / (case_id(case) + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "profile", "scene_name", "fps", "substeps", "iterations", "cap", "kick",
        "status", "peak", "simulated_seconds", "wall_seconds",
    )}), flush=True)
    del probe, sim
    gc.collect()
    return result


def build_cases(scenes, suite):
    cases = []
    combinations = [(2, 2), (3, 5), (5, 3)]
    def add(profile, scene_name, scene, fps, substeps, iterations, cap, kick=0):
        cases.append(dict(profile=profile, scene_name=scene_name, scene=scene, fps=fps,
                          substeps=substeps, iterations=iterations, cap=cap, kick=kick))
    if suite in ("all", "natural", "pilot"):
        for scene in ("dam-break-objects", "floating-equilibrium"):
            for fps in (60, 90, 120, 180):
                for substeps, iterations in combinations:
                    for cap in ("6", "none"):
                        add("natural", scene, scene, fps, substeps, iterations, cap)
        if suite == "pilot":
            return cases[:2]
    if suite in ("all", "gravity"):
        for scene in ("dam-break-objects", "floating-equilibrium"):
            for cap in ("6", "none"):
                add("gravity_switch", scene, scene, 60, 2, 2, cap)
    if suite in ("all", "stress"):
        for profile in ("radial", "head_on"):
            for fps in (60, 180):
                for substeps, iterations in combinations:
                    for cap, kick in [("6", 6)] + [("none", v) for v in (6, 12, 24, 48, 96)]:
                        add(profile, profile, scenes[profile], fps, substeps, iterations, cap, kick)
    return cases


def write_report(output):
    results = []
    for path in output.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(row, dict) and "peak_event" in row:
            results.append(row)
    results.sort(key=lambda r: (r["profile"], r["scene_name"], r["fps"], r["substeps"], r["cap"], r["kick"]))
    columns = ["profile", "scene_name", "fps", "substeps", "iterations", "cap", "kick", "particles",
               "status", "peak", "substeps_observed", "simulated_seconds", "max_predicted_speed", "wall_seconds"]
    with (output / "cases.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    lines = [
        "# PBF fluid neighbor capacity measurements", "",
        "Exact uncapped counts on NVIDIA RTX 5070 / Warp 1.15.0. Radius 0.025 m; search radius 0.1 m.", "",
        "Every physical substep is sampled after prediction and before pressure correction. "
        "The test grows the rectangular cache on the host before writing; no GPU-thread allocation "
        "and no change to production solver files. All original pressure/viscosity/vorticity kernels are used.", "",
        "The demo's fixed frame/substep validation is bypassed ONLY on test instances to exercise requested settings.", "",
        "Natural cases: two bundled scenes, 60/90/120/180 Hz, 2x2/3x5/5x3 substeps x iterations, "
        "6 m/s end-of-substep cap or effectively uncapped (1e30). Each case runs ten simulated seconds.", "",
        "Stress cases: normally spaced fluid, zero gravity, 0.5 s of normal evolution, "
        "then a single velocity-only kick toward the center or opposing block. No runtime position edits. "
        "Uncapped kick speeds 6/12/24/48/96 m/s; capped kick 6 m/s. Each runs two seconds including warmup.", "",
        "Gravity-switch cases: reverse/rotate gravity every simulated second at 60 Hz and 2x2; ten seconds.", "",
        "The cap is applied by the original solver after correction; it does not bound intermediate position "
        "corrections. Uncapped speeds may exceed the kick because subsequent reconstructed velocities are unrestricted.", "",
        "These are observed maxima over the stated experiments, not mathematical worst-case bounds.", "",
        "| Profile | Speed cap | Cases | Completed | Observed maximum | 25% margin, rounded to 32 | 50% margin, rounded to 32 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    groups = {}
    for profile in ("natural", "gravity_switch", "radial", "head_on"):
        for cap in ("6", "none"):
            group = [r for r in results if r["profile"] == profile and r["cap"] == cap]
            if not group:
                continue
            peak = max(r["peak"] for r in group)
            groups[f"{profile}_{cap}"] = dict(cases=len(group), peak=peak,
                complete=sum(r["status"] == "complete" for r in group))
            margin25 = math.ceil(1.25 * peak / 32) * 32
            margin50 = math.ceil(1.5 * peak / 32) * 32
            lines.append(f"| {profile} | {cap} | {len(group)} | {groups[f'{profile}_{cap}']['complete']} | {peak} | {margin25} | {margin50} |")
    lines += ["", "## Natural-run maxima by requested time settings", "",
              "| Hz | Substeps x iterations | Cap 6 m/s | Uncapped |", "|---:|---|---:|---:|"]
    for fps in (60, 90, 120, 180):
        for substeps, iterations in ((2, 2), (3, 5), (5, 3)):
            values = []
            for cap in ("6", "none"):
                matches = [r["peak"] for r in results if r["profile"] == "natural" and r["fps"] == fps
                           and r["substeps"] == substeps and r["iterations"] == iterations and r["cap"] == cap]
                values.append(str(max(matches)) if matches else "-")
            lines.append(f"| {fps} | {substeps} x {iterations} | {values[0]} | {values[1]} |")
    lines += ["", "## Largest recorded events", "",
              "| Profile | Hz | Substeps x iterations | Cap | Kick m/s | Neighbors | Time s | Max speed at event m/s | Fluid-only normalized density at peak particle | Status |",
              "|---|---:|---|---|---:|---:|---:|---:|---:|---|"]
    for row in sorted(results, key=lambda r: r["peak"], reverse=True)[:20]:
        event = row["peak_event"] or {}
        lines.append(f"| {row['profile']} | {row['fps']} | {row['substeps']} x {row['iterations']} | {row['cap']} | "
                     f"{row['kick']} | {row['peak']} | {event.get('time', 0):.4f} | {event.get('max_speed', 0):.2f} | "
                     f"{event.get('density_at_max_count', 0):.3f} | {row['status']} |")
    lines += ["", "## Uncapped impulse sweep (maximum across both stress profiles and all settings)", "",
              "| Applied kick m/s | Maximum neighbors | Power-of-two capacity with at least 25% margin |",
              "|---:|---:|---:|"]
    sweep = []
    for kick in (6, 12, 24, 48, 96):
        group = [r for r in results if r["profile"] in ("radial", "head_on") and r["cap"] == "none" and r["kick"] == kick]
        if group:
            peak = max(r["peak"] for r in group)
            capacity = next_capacity(math.ceil(1.25 * peak))
            sweep.append((kick, peak, capacity))
            lines.append(f"| {kick} | {peak} | {capacity} |")
    failures = [r for r in results if r["status"] != "complete"]
    total_checks = sum(len(r["brute_force_checks"]) for r in results)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    unchanged = hashlib.sha256(Path(manifest["source"]).read_bytes()).hexdigest() == manifest["source_sha256"]
    full_check_path = output / "independent_full_snapshot_checks.json"
    full_checks = json.loads(full_check_path.read_text()) if full_check_path.exists() else []
    lines += ["", "## Validation and limitations", "",
              f"- Cases: {len(results)}; non-complete: {len(failures)}.",
              f"- Substeps observed: {sum(r['substeps_observed'] for r in results):,}.",
              f"- Independent CPU brute-force point checks: {total_checks}; differences at the float32 cutoff are tracked.",
              f"- Full CPU all-pairs snapshot checks: {len(full_checks)}; mismatching particles: "
              f"{sum(r['mismatching_particles'] for r in full_checks)}.",
              "- GPU counts match production cache counts for every particle on the first substep, "
              "and on every substep after a case exceeds 256 (after test-only growth).",
              f"- Production solver SHA-256 unchanged: {unchanged}.",
              "- Fluid-only density excludes surface samples; use it as compression context, not total density near walls.",
              "- Timings include measurement synchronization and are not performance benchmarks.",
              "- Test duration, scenes and applied forcing are finite; more severe forcing/resolution changes need new coverage.",
              "- A failed case is only measured until its failure; it is not a completed stability claim.", ""]
    for row in failures:
        lines.append(f"- {case_id(row)}: {row['error']}")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "aggregate.json").write_text(json.dumps(dict(groups=groups, cases=len(results),
        incomplete=len(failures), source_unchanged=unchanged, brute_force_checks=total_checks), indent=2), encoding="utf-8")
    zh = [
        "# 流体邻域缓存容量实测", "",
        f"RTX 5070 / Warp 1.15.0；粒子半径 0.025 m，搜索半径 0.1 m。共 {len(results)} 组，"
        f"累计观察 {sum(r['substeps_observed'] for r in results):,} 个子步，未完成组数 {len(failures)}。", "",
        "## 如何统计", "",
        "每个子步在预测后、压力修正前，遍历 HashGrid 并按距离严格小于 h 统计所有流体邻居，排除自身。"
        "统计不受原有 256 容量限制；测试程序在写入前由 CPU 统一扩容矩形缓存，避免截断改变轨迹。"
        "这不是在 GPU 线程内申请不等长数组，也不是对生产缓存结构的修改。", "",
        f"进行了 {total_checks} 次独立 CPU 距离枚举检查；每组首个子步及突破 256 后的全部子步，"
        "还核对了每个粒子的原缓存计数。原求解器 SHA-256 保持不变。", "",
        "## 覆盖范围", "",
        "- 常规：两个现有场景；step 为 1/60、1/90、1/120、1/180 秒；子步×迭代为 2×2、3×5、5×3；"
        "6 m/s 限速及不限速；48 组，每组 10 秒。",
        "- 重力扰动：每秒反转或旋转重力；1/60 秒、2×2；4 组，每组 10 秒。",
        "- 冲击：正常间距、无初始重叠；零重力下正常演化 0.5 秒后，施加一次径向汇聚或两块对撞速度。"
        "全程没有运行时位置重设。覆盖 1/60 和 1/180 秒、三种子步/迭代组合；72 组，每组 2 秒。",
        "- 常规场景分别为 24,389、31,747 个流体粒子；径向汇聚为 6,859 个，对撞为 6,498 个。", "",
        "## 测得的最大值", "",
        "| 测试 | 限速 6 m/s | 不限速 |", "|---|---:|---:|",
    ]
    for profile, label in (("natural", "常规运行"), ("gravity_switch", "重力反转/旋转"),
                           ("radial", "径向汇聚冲击"), ("head_on", "两块流体对撞")):
        values = [str(groups[f"{profile}_{cap}"]["peak"]) if f"{profile}_{cap}" in groups else "-" for cap in ("6", "none")]
        zh.append(f"| {label} | {values[0]} | {values[1]} |")
    zh += ["", "不限速冲击中，6–96 m/s 是施加速度，不是之后的全程速度上限；压力修正重建速度后可能更高。", "",
           "| 施加冲击速度 | 实测最大邻居数 | 加 25% 余量后取 2 的幂 |", "|---:|---:|---:|"]
    for kick, peak, capacity in sweep:
        zh.append(f"| {kick} m/s | {peak} | {capacity} |")
    zh += ["", "## 容量取舍", ""]
    capped = [r for r in results if r["cap"] == "6"]
    if capped:
        peak = max(r["peak"] for r in capped)
        capacity = next_capacity(math.ceil(1.25 * peak))
        zh.append(f"- 只覆盖本轮限速 6 m/s 的所有场景：峰值 {peak}，加 25% 余量取 2 的幂建议 {capacity}。")
    for kick, peak, capacity in sweep:
        if kick in (12, 24, 48, 96):
            zh.append(f"- 不限速、覆盖本轮 {kick} m/s 冲击：峰值 {peak}，同样余量规则建议 {capacity}。")
    if results:
        worst = max(results, key=lambda r: r["peak"])
        event = worst["peak_event"] or {}
        zh += ["", f"本轮最大值 {worst['peak']} 出现在 {worst['profile']}、1/{worst['fps']} 秒、"
               f"{worst['substeps']} 子步×{worst['iterations']} 迭代、施加速度 {worst['kick']} m/s、限速设置 {worst['cap']}。"
               f"峰值粒子处仅流体贡献的归一化密度约 {event.get('density_at_max_count', 0):.2f}。"]
    if full_checks:
        zh += ["", f"对超过 256 的 {len(full_checks)} 个峰值快照完成了完整 CPU 两两距离枚举，"
               f"GPU/CPU 计数不一致的粒子总数为 {sum(r['mismatching_particles'] for r in full_checks)}。"]
    zh += ["",
           "所有组完成且保持有限数值，不代表所有瞬时状态的不可压缩性都合格。"
           "这些是有限时长、有限场景的观测峰值；改变粒径、外力、运行时长或压力算法后需重新覆盖。", "",
           "完整逐组数据见 cases.csv，逐子步记录见各组 .jsonl，超过 256 的峰值位置/速度/计数快照见 *_peak.npz。", "",
           "## 复现", "",
           "```powershell",
           ".venv/Scripts/python.exe -B -m demos.fluid.PBF2WayCoupling.tools.neighbor_capacity_probe --suite all --resume",
           ".venv/Scripts/python.exe -B -m demos.fluid.PBF2WayCoupling.tools.neighbor_capacity_probe --report-only",
           "```", ""]
    (output / "SUMMARY_zh.md").write_text("\n".join(zh), encoding="utf-8")
    print(json.dumps(groups, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--suite", choices=("all", "natural", "gravity", "stress", "pilot"), default="all")
    parser.add_argument("--output", default="outputs/pbf2way/neighbor-capacity-20261020")
    parser.add_argument("--natural-seconds", type=float, default=10.0)
    parser.add_argument("--stress-seconds", type=float, default=2.0)
    parser.add_argument("--case-filter", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        write_report(output)
        return
    wp.config.kernel_cache_dir = str(Path(".warp_cache").resolve())
    wp.init()
    source = Path(solver.__file__)
    manifest = dict(
        device=str(wp.get_device(args.device)), device_name=wp.get_device(args.device).name,
        warp_version=wp.__version__, source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        measure="All fluid neighbors r<h excluding self, after prediction before every pressure solve",
        radius=0.025, support_radius=0.1, natural_seconds=args.natural_seconds,
        stress_seconds=args.stress_seconds,
        stress="Regular nonoverlapping samples; evolve .5s, then velocity-only inward impulse, no position edits",
        cache="Test-only growth prevents truncation; production solver/config files unchanged",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    scenes = write_stress_scenes(output)
    results = []
    for case in build_cases(scenes, args.suite):
        name = case_id(case)
        if args.case_filter and args.case_filter not in name:
            continue
        destination = output / (name + ".json")
        if args.resume and destination.exists():
            results.append(json.loads(destination.read_text(encoding="utf-8")))
            continue
        seconds = args.stress_seconds if case["profile"] in ("radial", "head_on") else args.natural_seconds
        results.append(run_case(case, output, args.device, seconds))
        (output / ("summary_" + args.suite + ".json")).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Finished {len(results)} cases; peak={max((r['peak'] for r in results), default=-1)}", flush=True)


if __name__ == "__main__":
    main()
