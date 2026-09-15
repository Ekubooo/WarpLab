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
    x, v = sim.positions.numpy(), sim.velocities.numpy()
    p, q = sim.rigid.position.numpy(), sim.rigid.rotation.numpy()
    omega = sim.rigid.omega.numpy()
    boundary = sim.boundary.position.numpy()
    contacts = min(int(sim.contacts.count.numpy()[0]), sim.config.max_contacts)
    gaps = sim.contacts.gap.numpy()[:contacts]
    maxima = sim.audit_maxima.numpy()
    if not all(np.isfinite(a).all() for a in (x, v, p, q, omega, boundary, sim.densities.numpy())):
        raise FloatingPointError("Non-finite fluid or rigid state")
    fluid_violation = float(max(0, (sim.container_min-x).max(), (x-sim.container_max).max()))
    return dict(time=sim.sim_time, steps=sim.total_steps, dt=sim.last_dt, next_dt=sim.current_dt,
                particles=sim.num_particles, boundary_particles=len(boundary), iterations=sim.iterations,
                density_error_percent=sim.density_error_percent, max_speed=float(np.linalg.norm(v, axis=1).max()),
                fluid_container_violation=fluid_violation, contacts=contacts,
                max_contact_penetration=float(max(0, -gaps.min())) if contacts else 0.0,
                max_fluid_violation_ever=float(maxima[0]), max_contact_penetration_ever=float(maxima[1]),
                max_quaternion_error_ever=float(maxima[2]), pressure_limit_streak=sim.iteration_limit_streak,
                rigid_positions=p.tolist(), rigid_rotations=q.tolist(), rigid_omega=omega.tolist())


def positive_float(value):
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return x


def add_simulation_arguments(parser):
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU")
    parser.add_argument("--scene", default="dam-break-objects", help="Bundled scene name or JSON path")
    parser.add_argument("--particle-radius", type=positive_float, default=.025)
    parser.add_argument("--one-way", action="store_true", help="Disable reaction forces for comparison")
    parser.add_argument("--verbose", action="store_true")


def config_from_args(args):
    return PBF2WayCouplingConfig(scene=args.scene, particle_radius=args.particle_radius, two_way=not args.one_way)


def run_headless(config, device=None, seconds=10.0, output=None, log_interval=1.0, verbose=False):
    sim = create_pbf2way_simulation(config=config, device=device, verbose=verbose)
    started = time.perf_counter()
    history = [diagnostics(sim)]
    print(json.dumps(history[-1]), flush=True)
    next_log = log_interval
    while sim.sim_time < seconds:
        sim.step()
        if sim.sim_time >= next_log or sim.sim_time >= seconds:
            row = diagnostics(sim)
            row["wall_seconds"] = time.perf_counter() - started
            row["mean_step_ms"] = 1000 * row["wall_seconds"] / sim.total_steps
            history.append(row)
            print(json.dumps(row), flush=True)
            next_log += log_interval
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        np.savez_compressed(path.with_suffix(".npz"), positions=sim.positions.numpy(), velocities=sim.velocities.numpy(),
                            rigid_positions=sim.rigid.position.numpy(), rigid_rotations=sim.rigid.rotation.numpy())
    return sim, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_simulation_arguments(parser)
    parser.add_argument("--seconds", type=positive_float, default=10.0)
    parser.add_argument("--output", help="Output prefix for JSON diagnostics and NPZ final state")
    args = parser.parse_args()
    run_headless(config_from_args(args), args.device, args.seconds, args.output, verbose=args.verbose)
