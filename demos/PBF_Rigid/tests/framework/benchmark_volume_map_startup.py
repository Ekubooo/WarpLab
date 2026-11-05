"""Measure volume-map refresh and reuse on private published-asset copies.

Run benchmark_startup first. Reuse its warm Warp caches; leave the actual
published assets and the user's normal compiler cache untouched.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
SCENE_ASSETS = {
    "static": ("stairs_scene",),
    "tank-box": ("tank_scene", "Rigid/tank_box"),
    "stairs-rigids": ("stairs_scene", "Rigid/stairs_box", "Rigid/stairs_sphere", "Rigid/stairs_spot"),
}


def run(args):
    if args.output.exists():
        raise ValueError(f"Use a new output directory to retain previous results: {args.output}")
    after = json.loads(args.startup_report.read_text(encoding="utf-8"))
    results = {}
    for scene, stems in SCENE_ASSETS.items():
        private = args.output / scene / "Assets"
        files = []
        for stem in stems:
            for extension in (".obj", ".nvdb", ".json", ".collision.npz"):
                source = ROOT / "demos/PBF_Rigid/Assets" / (stem + extension)
                if not source.exists():
                    continue
                destination = private / (stem + extension)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if extension == ".nvdb":
                    files.append(str(destination))
        cache = next(t["cache"] for t in after["trials"]
                     if t["scene"] == scene and t["mode"] == "cold" and t["trial"] == 0)
        env = dict(os.environ, WARP_CACHE_PATH=cache, PYTHONDONTWRITEBYTECODE="1")
        with (private.parent / "seed.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-B", "-m", "demos.PBF_Rigid.tests.framework.benchmark_volume_map_startup", "--seed", *files,
                            "--device", args.device], cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        results[scene] = {}
        for mode in ("miss", "hit"):
            target = private.parent / (mode + ".json")
            log = target.with_suffix(".log")
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run([sys.executable, "-B", "-m", "demos.PBF_Rigid.tests.framework.benchmark_startup",
                                "--worker", "--scene", scene, "--asset-root", str(private),
                                "--code-root", str(ROOT), "--device", args.device, "--output", str(target)],
                               cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            entry = json.loads(target.read_text(encoding="utf-8"))
            text = log.read_text(encoding="utf-8")
            entry["volume_map_rebuilds"] = text.count("Boundary cache rebuild:")
            entry["volume_map_hits"] = text.count("Boundary cache hit:")
            assert entry["volume_map_rebuilds"] == (len(stems) if mode == "miss" else 0)
            results[scene][mode] = entry
            print(scene, mode, entry["phases"]["first_frame_total_seconds"], flush=True)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-report", type=Path, default=Path("outputs/offline-assets/startup-after.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/offline-assets/volume-map"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seed:
        from demos.PBF_Rigid.solver.boundary_model import load_boundary_volumes
        for filename in args.seed:
            load_boundary_volumes(filename, .04, device=args.device)
    else:
        run(args)


if __name__ == "__main__":
    main()
