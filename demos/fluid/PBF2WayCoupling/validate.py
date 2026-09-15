"""Run the full-resolution 10-second behavioral acceptance scenarios.

Run from the repository root:
    .venv/Scripts/python.exe -m demos.fluid.PBF2WayCoupling.validate
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .PBF2WayCoupling import PBF2WayCouplingConfig
from .simulation import run_headless


def check_stability(simulation, history):
    final = history[-1]
    assert final["time"] >= 10, "Must advance at least ten simulated seconds"
    assert len(simulation.positions) == history[0]["particles"]
    assert final["max_fluid_violation_ever"] < 0.5 * simulation.config.particle_radius
    assert final["max_contact_penetration_ever"] < simulation.config.particle_radius
    assert final["max_quaternion_error_ever"] < 2e-6
    assert simulation.total_steps == 600
    assert simulation.total_substeps == 1800
    assert all(np.isfinite(row["density_error_percent"]) for row in history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="outputs/pbf2way/acceptance")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    config = PBF2WayCouplingConfig()
    simulation, history = run_headless(
        config, args.device, 10, output / "dam-break", log_interval=1
    )
    check_stability(simulation, history)
    initial = np.array(history[0]["rigid_positions"])
    final = np.array(history[-1]["rigid_positions"])
    assert np.all(np.linalg.norm(final[1:] - initial[1:], axis=1) > 0.2)
    assert np.all(final[1:, 1] > 0.35), "Dynamic bodies should finish supported by the water"
    assert any(
        np.linalg.norm(np.asarray(row["rigid_omega"])[1:], axis=1).max() > 0.5 for row in history
    )
    result["dam_break"] = history[-1]
    del simulation
    floating = replace(config, scene="floating-equilibrium")
    simulation, history = run_headless(
        floating, args.device, 10, output / "floating", log_interval=1
    )
    check_stability(simulation, history)
    heights = simulation.rigid.position.numpy()[:, 1]
    assert heights[1] > 0.7, "Light sphere should rise from y=.5"
    assert heights[2] < 0.25, "Heavy sphere should sink to the floor"
    result["floating"] = history[-1]
    coupled_height = float(heights[1])
    del simulation
    simulation, history = run_headless(
        replace(floating, two_way=False), args.device, 10, output / "one-way", log_interval=1
    )
    check_stability(simulation, history)
    height = float(simulation.rigid.position.numpy()[1, 1])
    assert height < 0.25, "Without feedback, the light body falls to the floor"
    assert coupled_height - height > 0.4
    result["one_way"] = history[-1]
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "PASS: dam break, floating/sinking, one-way comparison, and per-step stability audits",
        flush=True,
    )


if __name__ == "__main__":
    main()
