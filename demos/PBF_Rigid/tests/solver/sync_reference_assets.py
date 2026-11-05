"""Explicitly synchronize the archived default scene into the independent app.

This migration/verification tool may read demos; production code never does.
Only the destination volume-map cache is rebuilt. Archived inputs stay intact.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "demos" / "fluid" / "PBF" / "Assets"
DESTINATION = ROOT / "demos" / "PBF_Rigid" / "Assets"
REPORT_DIR = ROOT / "outputs" / "stairs-rigids"
FILES = ("stairs_scene.obj", "stairs_scene.nvdb")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sync_reference_assets(*, device="cuda:0"):
    from demos.PBF_Rigid.solver.boundary_model import load_boundary_volumes

    backup = REPORT_DIR / "reference-backup"
    backup.mkdir(parents=True, exist_ok=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: digest(SOURCE / name) for name in FILES}
    previous = {}
    for name in FILES:
        target = DESTINATION / name
        if target.is_file():
            previous[name] = digest(target)
            if not (backup / name).exists():
                shutil.copy2(target, backup / name)
        shutil.copy2(SOURCE / name, target)
    sdf, volume_map = load_boundary_volumes(DESTINATION / FILES[1], .05, device=device)
    final_source = {name: digest(SOURCE / name) for name in FILES}
    if final_source != source_hashes:
        raise RuntimeError("Archived scene changed during synchronization")
    result = dict(source=str(SOURCE), destination=str(DESTINATION),
                  source_sha256=source_hashes, previous_destination_sha256=previous,
                  destination_sha256={name: digest(DESTINATION / name) for name in FILES},
                  sdf_grid=sdf.get_grid_info().name, volume_grid=volume_map.get_grid_info().name,
                  support_radius=.05, archived_inputs_unchanged=True)
    path = REPORT_DIR / "reference-assets.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(sync_reference_assets(device=args.device), indent=2))


if __name__ == "__main__":
    main()
