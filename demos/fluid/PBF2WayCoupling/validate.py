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
    assert final["max_fluid_violation_ever"] < .5*simulation.config.particle_radius
    assert final["max_contact_penetration_ever"] < simulation.config.particle_radius
    assert final["max_quaternion_error_ever"] < 2e-6
    assert final["pressure_limit_streak"] == 0
    assert all(row["density_error_percent"] <= .011 for row in history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="outputs/pbf2way/acceptance")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    config = PBF2WayCouplingConfig()
    sim, history = run_headless(config, args.device, 10, output/"dam-break")
    check_stability(sim, history)
    initial = np.array(history[0]["rigid_positions"])
    final = np.array(history[-1]["rigid_positions"])
    assert np.all(np.linalg.norm(final[1:]-initial[1:],axis=1)>.2)
    assert np.all(final[1:,1] > .35), "Dynamic bodies should finish supported by the water"
    assert any(np.linalg.norm(np.asarray(row["rigid_omega"])[1:],axis=1).max()>.5 for row in history)
    result["dam_break"] = history[-1]
    del sim
    floating = replace(config, scene="floating-equilibrium")
    sim, history = run_headless(floating, args.device, 10, output/"floating")
    check_stability(sim, history)
    heights = sim.rigid.position.numpy()[:,1]
    assert heights[1] > .7, "Light sphere should rise from y=.5"
    assert heights[2] < .25, "Heavy sphere should sink to the floor"
    result["floating"] = history[-1]
    coupled_height = float(heights[1])
    del sim
    sim, history = run_headless(replace(floating,two_way=False),args.device,10,output/"one-way")
    check_stability(sim, history)
    height = float(sim.rigid.position.numpy()[1,1])
    assert height < .25, "Without feedback, the light body falls to the floor"
    assert coupled_height-height > .4
    result["one_way"] = history[-1]
    (output/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("PASS: dam break, floating/sinking, one-way comparison, and per-step stability audits",flush=True)


if __name__ == "__main__":
    main()
