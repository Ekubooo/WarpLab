"""Verify fixed-step kernel counts and zero DtoH transfers in an nsys SQLite export."""

import argparse
import json
from pathlib import Path
import sqlite3


def verify(path, steps, rendered=False):
    with sqlite3.connect(path) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        copies = []
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            copies = list(
                db.execute(
                    "SELECT copyKind, COUNT(*), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
                )
            )
        kernels = list(
            db.execute(
                "SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6 "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName "
                "GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC"
            )
        )
    dtoh = sum(count for kind, count, _ in copies if kind == 2)
    expected = {
        "predict": steps * 3,
        "pressure_correction": steps * 15,
        "solve_contacts": steps * 3,
    }
    if rendered:
        expected["write_rigid_matrices"] = steps
    counts = {
        name: sum(row[1] for row in kernels if row[0].startswith(name + "_")) for name in expected
    }
    assert dtoh == 0, f"Found {dtoh} device-to-host copies"
    assert counts == expected, (counts, expected)
    result = dict(
        dtoh_copies=dtoh,
        counts=counts,
        copies=copies,
        kernel_columns=["name", "calls", "total_ms"],
        kernels=kernels,
    )
    Path(path).with_suffix(".verified.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    result = verify(args.sqlite, args.steps, args.render)
    print(json.dumps({k: v for k, v in result.items() if k != "kernels"}, indent=2))
