"""Check the source snapshot, independent assets and protected demo tree."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sdf_digest(path):
    # The VM and NanoVDB multi-grid header are mutable cache data.
    import warp as wp
    with wp.ScopedDevice("cuda:0"), path.open("rb") as stream:
        sdf = wp.Volume.load_from_nvdb(stream)
        raw = wp.array(ptr=sdf.array().ptr, dtype=wp.uint8,
            shape=sdf.get_grid_info().size_in_bytes, device=sdf.device).numpy()
    result = hashlib.sha256()
    for chunk in (raw[:8], bytes(8), raw[16:20], bytes(12), raw[32:]):
        result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/pbf-newton/coupling-demos-before.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pbf-newton/integrity.json")
    args = parser.parse_args()
    protected = json.loads(args.baseline.read_text())
    changed = [name for name, value in protected.items()
               if not (ROOT / name).is_file() or digest(ROOT / name) != value]
    assert not changed, changed
    manifest = json.loads(Path(__file__).with_name("source_manifest.json").read_text())
    original = ROOT / "demos/fluid/PBF"
    for name, value in manifest["source_files"].items():
        if name.endswith(".nvdb"):
            assert sdf_digest(original / name) == sdf_digest(ROOT / "PBF" / name), name
            continue
        assert digest(original / name) == value, name
        if name.startswith(("config/", "Assets/")):
            assert digest(ROOT / "PBF" / name) == value, name
    for path in (ROOT / "PBF").glob("*.py"):
        assert "PBF_Rigid" not in path.read_text(), path
    result = dict(protected_demo_files=len(protected), unchanged=True,
                  independent_static_sdf_identical=True, archive_runtime_dependency=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
