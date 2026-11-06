"""Rebuild and run PBF from code and a manually supplied bunny, without old assets."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from Sim_Ready.asset_io import file_digest


ROOT = Path(__file__).resolve().parents[2]


def prepare_workspace(workspace):
    for package in ("PBF", "Sim_Ready"):
        for source in (ROOT / package).rglob("*.py"):
            destination = workspace / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    # Production rendering still uses this shared source file, never demo assets.
    shared = Path("demos/fluid/common/billboard_renderer.py")
    destination = workspace / shared
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / shared, destination)
    for package in ("demos", "demos/fluid", "demos/fluid/common", "tests", "tests/pbf"):
        directory = workspace / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
    for name in ("verify_viewer.py", "verify_coupled.py", "coupled_lifecycle_case.py"):
        shutil.copy2(ROOT / "tests/pbf" / name, workspace / "tests/pbf" / name)
    source = ROOT / "Sim_Ready/Assets/bunny_flat_bottom.obj"
    destination = workspace / "Sim_Ready/Assets" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    assert not (workspace / "PBF/Assets").exists()
    assert not (workspace / ".cache").exists()
    assert not (workspace / "demos/PBF_Rigid").exists()
    assets = [p for p in workspace.rglob("*") if p.suffix in (".obj", ".nvdb", ".npz", ".json")]
    assert assets == [destination], assets
    return file_digest(source)


def run(workspace, output, name, args, env):
    command = [sys.executable, "-B", *args]
    print(f"Running {name}: {' '.join(command)}", flush=True)
    with (output / f"{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} failed ({result.returncode}); see {output / (name + '.log')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pbf-newton/pipeline-split")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="clean-workspace-", dir=output))
    source_digest = prepare_workspace(workspace)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONUTF8"] = "1"
    # This is a compiler cache, not a geometry cache; keep it writable in sandboxes.
    env["WARP_CACHE_PATH"] = str(ROOT / ".warp_cache")
    temporary = workspace / "tmp"
    temporary.mkdir()
    env["TEMP"] = env["TMP"] = str(temporary)
    # No --rebuild: demonstrate that a clean checkout's ordinary command can bootstrap.
    run(workspace, output, "clean-build", ["-m", "Sim_Ready.pipelines.preset.pbf", "--scene", "stairs-rigids"], env)
    origin_check = """
from pathlib import Path
import json, sys
from Sim_Ready.pipelines.preset import pbf
from PBF import scene
root = Path.cwd().resolve()
assert pbf.PROJECT == root and pbf.OUTPUT == root / 'PBF/Assets'
assert pbf.CACHE == root / '.cache/sim-ready/pbf'
assert scene.ASSET_DIR == root / 'PBF/Assets'
assert all(Path(module.__file__).resolve().is_relative_to(root)
    for name, module in sys.modules.items()
    if name.split('.')[0] in ('Sim_Ready', 'PBF', 'demos', 'tests') and getattr(module, '__file__', None))
assert not any(name.startswith(('demos.sim_Ready', 'demos.PBF_Rigid')) for name in sys.modules)
print('Isolated import and asset paths verified')
"""
    run(workspace, output, "clean-origins", ["-c", origin_check], env)
    run(workspace, output, "clean-coupled", ["-m", "tests.pbf.verify_coupled", "--seconds", "10",
        "--output", str(output / "clean-coupled.json")], env)
    run(workspace, output, "clean-lifecycle", ["-m", "tests.pbf.coupled_lifecycle_case",
        "--output", str(output / "clean-lifecycle.json")], env)
    run(workspace, output, "clean-viewer", ["-m", "tests.pbf.verify_viewer", "--scene", "stairs-rigids",
        "--output", str(output / "clean-viewer")], env)
    from pxr import Usd
    stage = Usd.Stage.Open(str(output / "clean-viewer/static.usd"))
    meshes = sorted(str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "Mesh")
    assert meshes == [f"/root/boundary_{i}" for i in range(4)], meshes
    artifacts = sorted(str(p.relative_to(workspace)) for p in (workspace / "PBF/Assets").rglob("*") if p.is_file())
    report = dict(workspace=str(workspace), source_sha256=source_digest,
                  initial_asset_files=["Sim_Ready/Assets/bunny_flat_bottom.obj"],
                  no_prior_assets_or_geometry_cache=True, isolated_imports=True,
                  generated=artifacts, coupled_seconds=10, lifecycle=True, ogl_and_usd=True,
                  usd_boundary_meshes=meshes)
    (output / "clean-workspace.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
