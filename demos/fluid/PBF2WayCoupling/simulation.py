"""Factory and bounded, window-free simulation entry point."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

try:
    from .PBF2WayCoupling import Example, PBF2WayCouplingConfig
except ImportError:
    from PBF2WayCoupling import Example, PBF2WayCouplingConfig


def create_pbf2way_simulation(verbose=False, config=None, device=None):
    return Example(config=config, device=device, verbose=verbose)


def diagnostics(sim):
    positions, velocities = sim.positions.numpy(), sim.velocities.numpy()
    rigid_positions, rigid_rotations = sim.rigid.position.numpy(), sim.rigid.rotation.numpy()
    omega = sim.rigid.omega.numpy()
    boundary = sim.boundary.position.numpy()
    contact_count = min(int(sim.contacts.count.numpy()[0]), sim.config.max_contacts)
    gaps = sim.contacts.gap.numpy()[:contact_count]
    maxima = sim.audit_maxima.numpy()
    if not all(
        np.isfinite(array).all()
        for array in (
            positions,
            velocities,
            rigid_positions,
            rigid_rotations,
            omega,
            boundary,
            sim.densities.numpy(),
        )
    ):
        raise FloatingPointError("Non-finite fluid or rigid state")
    fluid_violation = float(
        max(0, (sim.container_min - positions).max(), (positions - sim.container_max).max())
    )
    return dict(
        time=sim.sim_time,
        steps=sim.total_steps,
        dt=sim.last_dt,
        next_dt=sim.current_dt,
        particles=sim.num_particles,
        boundary_particles=len(boundary),
        iterations=sim.iterations,
        density_error_percent=sim.density_error_percent,
        max_speed=float(np.linalg.norm(velocities, axis=1).max()),
        fluid_container_violation=fluid_violation,
        contacts=contact_count,
        max_contact_penetration=float(max(0, -gaps.min())) if contact_count else 0.0,
        max_fluid_violation_ever=float(maxima[0]),
        max_contact_penetration_ever=float(maxima[1]),
        max_quaternion_error_ever=float(maxima[2]),
        pressure_limit_streak=sim.iteration_limit_streak,
        rigid_positions=rigid_positions.tolist(),
        rigid_rotations=rigid_rotations.tolist(),
        rigid_omega=omega.tolist(),
    )


def positive_float(value):
    parsed_value = float(value)
    if not np.isfinite(parsed_value) or parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed_value


def add_simulation_arguments(parser):
    parser.add_argument(
        "--device", default=None, help="Defaults to CUDA when available, otherwise CPU"
    )
    parser.add_argument(
        "--scene", default="dam-break-objects", help="Bundled scene name or JSON path"
    )
    parser.add_argument("--particle-radius", type=positive_float, default=0.025)
    parser.add_argument(
        "--one-way", action="store_true", help="Disable reaction forces for comparison"
    )
    parser.add_argument("--verbose", action="store_true")


def config_from_args(args):
    return PBF2WayCouplingConfig(
        scene=args.scene, particle_radius=args.particle_radius, two_way=not args.one_way
    )


def run_headless(config, device=None, seconds=10.0, output=None, log_interval=1.0, verbose=False):
    simulation = create_pbf2way_simulation(config=config, device=device, verbose=verbose)
    started = time.perf_counter()
    history = [diagnostics(simulation)]
    print(json.dumps(history[-1]), flush=True)
    next_log = log_interval
    while simulation.sim_time < seconds:
        simulation.step()
        if simulation.sim_time >= next_log or simulation.sim_time >= seconds:
            row = diagnostics(simulation)
            row["wall_seconds"] = time.perf_counter() - started
            row["mean_step_ms"] = 1000 * row["wall_seconds"] / simulation.total_steps
            history.append(row)
            print(json.dumps(row), flush=True)
            next_log += log_interval
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        np.savez_compressed(
            path.with_suffix(".npz"),
            positions=simulation.positions.numpy(),
            velocities=simulation.velocities.numpy(),
            rigid_positions=simulation.rigid.position.numpy(),
            rigid_rotations=simulation.rigid.rotation.numpy(),
        )
    return simulation, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_simulation_arguments(parser)
    parser.add_argument("--seconds", type=positive_float, default=10.0)
    parser.add_argument("--output", help="Output prefix for JSON diagnostics and NPZ final state")
    args = parser.parse_args()
    run_headless(
        config_from_args(args), args.device, args.seconds, args.output, verbose=args.verbose
    )
