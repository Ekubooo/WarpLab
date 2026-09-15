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
from .simulation import create_pbf2way_simulation, diagnostics, run_headless


def check_stability(simulation, history):
    final = history[-1]
    assert final["time"] >= 10, "Must advance at least ten simulated seconds"
    assert len(simulation.positions) == history[0]["particles"]
    assert final["max_fluid_violation_ever"] == 0
    assert all(row["fluid_boundary_violation"] == 0 for row in history[1:])
    assert final["max_contact_penetration_ever"] < simulation.config.particle_radius
    assert final["max_quaternion_error_ever"] < 2e-6
    expected_steps = round(10 / simulation.frame_dt)
    assert simulation.total_steps == expected_steps
    assert simulation.total_substeps == expected_steps * simulation.config.substeps
    assert all(np.isfinite(row["density_error_percent"]) for row in history)


def check_gravity_interactions(config, device, output):
    """Exercise the same gravity operations as G/Q/E, without opening a window."""
    from .render_opengl import reverse_gravity, rotate_gravity

    simulation = create_pbf2way_simulation(config=config, device=device)
    history = [diagnostics(simulation)]
    events = []
    steps_per_second = round(1 / simulation.frame_dt)
    total_steps = 10 * steps_per_second
    try:
        for step in range(total_steps):
            if step in (2 * steps_per_second, 6 * steps_per_second):
                reverse_gravity(simulation)
            elif step == 4 * steps_per_second:
                rotate_gravity(simulation, 90)
            elif step == 8 * steps_per_second:
                rotate_gravity(simulation, 45)
            if step % steps_per_second == 0 and step // steps_per_second in (2, 4, 6, 8):
                events.append(dict(time=simulation.sim_time, gravity=simulation.config.gravity))
            simulation.step()
            if simulation.total_steps % steps_per_second == 0:
                row = diagnostics(simulation)
                history.append(row)
                assert row["particles"] == history[0]["particles"]
                assert row["fluid_boundary_violation"] == 0
                assert row["max_fluid_violation_ever"] == 0
        assert simulation.sim_time == 10
        assert simulation.total_substeps == total_steps * config.substeps
    finally:
        output.write_text(
            json.dumps(dict(events=events, history=history), indent=2), encoding="utf-8"
        )
    return history[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="outputs/pbf2way/acceptance")
    parser.add_argument(
        "--interactions",
        action="store_true",
        help="Also run ten-second gravity reversal/rotation tests in both scenes",
    )
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
    if args.interactions:
        del simulation
        for scene_config in (config, floating):
            result[scene_config.scene + "_gravity"] = check_gravity_interactions(
                scene_config, args.device, output / (scene_config.scene + "-gravity.json")
            )
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "PASS: dam break, floating/sinking, one-way comparison, and per-step stability audits",
        flush=True,
    )


if __name__ == "__main__":
    main()
