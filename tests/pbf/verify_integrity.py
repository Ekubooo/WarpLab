"""Check the source snapshot, independent assets and protected demo tree."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/pbf-newton/demos-before.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pbf-newton/integrity.json")
    args = parser.parse_args()
    protected = json.loads(args.baseline.read_text())
    changed = [name for name, value in protected.items()
               if not (ROOT / name).is_file() or digest(ROOT / name) != value]
    assert not changed, changed
    manifest = json.loads(Path(__file__).with_name("source_manifest.json").read_text())
    original = ROOT / "demos/fluid/PBF"
    for name, value in manifest["source_files"].items():
        assert digest(original / name) == value, name
        if name.startswith(("config/", "Assets/")):
            assert digest(ROOT / "PBF" / name) == value, name
    for path in (ROOT / "PBF").glob("*.py"):
        assert "PBF_Rigid" not in path.read_text(), path
    result = dict(protected_demo_files=len(protected), unchanged=True,
                  independent_static_assets_identical=True, archive_runtime_dependency=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
