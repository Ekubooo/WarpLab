"""Isolated full-scale coupled Graph, feedback and reset acceptance.

Run with ``python -m tests.pbf.coupled_lifecycle_case``. One fixed PBF instance
is used throughout; every eager/Graph pair starts from identical GPU history.
"""

from dataclasses import fields, is_dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import warp as wp

from PBF.simulation import Simulation


STATE_FIELDS = ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")


def nested_arrays(value, name):
    if isinstance(value, wp.array):
        if value.size:
            yield name, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from nested_arrays(getattr(value, field.name), f"{name}.{field.name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from nested_arrays(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            yield from nested_arrays(item, f"{name}[{i}]")


def checkpoint(sim):
    arrays = []
    for i, state in enumerate(sim.states):
        arrays.extend(nested_arrays(vars(state), f"state[{i}]"))
    arrays.extend(nested_arrays(vars(sim.fluid_solver), "fluid"))
    cache = sim.fluid_solver.n_Cache
    arrays.extend((("neighbors.count", cache.NeighborCount), ("neighbors.cache", cache.CacheBuffer)))
    data = sim.BoundarySys.device_data
    arrays.extend((("boundary.transforms", data.transforms), ("boundary.com", data.world_com)))
    arrays.extend(nested_arrays(sim.solver.rigid.mjw_data, "mujoco"))
    seen = set()
    unique = []
    for name, array in arrays:
        key = array.ptr, array.shape, array.dtype
        if key not in seen:
            seen.add(key)
            unique.append((name, array, wp.clone(array)))
    fluid = sim.fluid_solver
    return dict(arrays=unique, index=sim._state_index,
                prediction=(fluid.pre_Pos, fluid.temp_pPos),
                time=sim.sim_time, counter=sim.solver.rigid._step)


def restore(sim, saved):
    for _, target, source in saved["arrays"]:
        wp.copy(target, source)
    sim._state_index = saved["index"]
    sim.fluid_solver.pre_Pos, sim.fluid_solver.temp_pPos = saved["prediction"]
    sim.sim_time = saved["time"]
    sim.solver.rigid._step = saved["counter"]


def assert_capture_unchanged(sim, saved):
    assert sim._state_index == saved["index"]
    assert sim.sim_time == saved["time"]
    assert sim.solver.rigid._step == saved["counter"]
    assert (sim.fluid_solver.pre_Pos.ptr, sim.fluid_solver.temp_pPos.ptr) == tuple(
        array.ptr for array in saved["prediction"])
    for name, array, previous in saved["arrays"]:
        # Compare raw bytes: uninitialized backend scratch may contain NaNs.
        np.testing.assert_array_equal(array.numpy().view(np.uint8), previous.numpy().view(np.uint8), err_msg=name)


def state_values(sim):
    result = {name: getattr(sim.state, name).numpy().copy() for name in STATE_FIELDS}
    result["boundary_forces"] = sim.solver.boundary_forces.numpy().copy()
    return result


def compare_result(sim, expected):
    positions = sim.pos.numpy()
    # Atomics may cross a hash-cell edge, changing only spatial slot order.
    distance, match = cKDTree(expected["particle_q"]).query(positions)
    assert len(np.unique(match)) == sim.n, "Particle matching must be bijective"
    position_error = float(distance.max())
    assert position_error < 2e-5, position_error
    velocity_error = float(np.max(np.abs(sim.v.numpy() - expected["particle_qd"][match])))
    np.testing.assert_allclose(sim.v.numpy(), expected["particle_qd"][match], atol=2e-4, rtol=2e-4)
    for name in STATE_FIELDS[2:]:
        np.testing.assert_allclose(getattr(sim.state, name).numpy(), expected[name],
                                   atol=2e-4 if name.endswith("qd") else 2e-5,
                                   rtol=2e-4, err_msg=name)
    np.testing.assert_allclose(sim.solver.boundary_forces.numpy(), expected["boundary_forces"],
                               atol=2e-3, rtol=2e-4, err_msg="boundary_forces")
    return dict(particle_position_max_error=position_error,
                particle_velocity_max_error=velocity_error)


def main():
    with wp.ScopedDevice("cuda:0"):
        sim = Simulation(scene="stairs-rigids")
        assert sim.n == 195525 and sim.model.body_count == 3
        initial = state_values(sim)
        pointers = [(s.particle_q.ptr, s.particle_qd.ptr, s.body_q.ptr) for s in sim.states]
        report = dict(particles=sim.n, bodies=sim.model.body_count, cases=[],
                      tolerances=dict(position_atol=2e-5, velocity_atol=2e-4, rtol=2e-4))
        initial_saved = checkpoint(sim)
        sim.capture_graph()
        assert_capture_unchanged(sim, initial_saved)
        del initial_saved
        report["initial_capture_unchanged"] = True
        for frame in range(300):
            sim.launch_graph()
            contacts = int(sim.solver.rigid.mjw_data.nacon.numpy()[0])
            pressure = float(np.linalg.norm(sim.solver.boundary_forces.numpy()))
            if frame >= 60 and contacts > 0 and pressure > 1e-5:
                break
        assert contacts > 0 and pressure > 1e-5, (contacts, pressure)
        report["contact_start"] = dict(frame=frame + 1, contacts=contacts, pressure_wrench_norm=pressure)
        contact = checkpoint(sim)
        report["saved_arrays"] = len(contact["arrays"])
        assert any(name.startswith("mujoco.contact.") for name, _, _ in contact["arrays"])

        for substeps in (1, 2, 3):
            for start_index in (0, 1):
                for reverse_prediction in (False, True):
                    restore(sim, contact)
                    sim.substep = substeps
                    # Use the same real contact state in either public slot.
                    sim.states[start_index].assign(sim.state)
                    sim._state_index = start_index
                    if reverse_prediction:
                        fluid = sim.fluid_solver
                        fluid.pre_Pos, fluid.temp_pPos = fluid.temp_pPos, fluid.pre_Pos
                    saved = checkpoint(sim)
                    sim.capture_graph()
                    assert_capture_unchanged(sim, saved)
                    sim.step()
                    expected = state_values(sim)
                    end_index = sim._state_index
                    end_prediction = (sim.fluid_solver.pre_Pos.ptr, sim.fluid_solver.temp_pPos.ptr)
                    restore(sim, saved)
                    sim.launch_graph()
                    errors = compare_result(sim, expected)
                    assert sim._state_index == end_index
                    assert (sim.fluid_solver.pre_Pos.ptr, sim.fluid_solver.temp_pPos.ptr) == end_prediction
                    assert sim.solver.rigid._step == saved["counter"] + substeps
                    assert abs(sim.sim_time - saved["time"] - sim.frame_dt) < 1e-14
                    report["cases"].append(dict(substeps=substeps, state=start_index,
                        reversed_prediction=reverse_prediction, **errors))
                    print("Graph lifecycle:", report["cases"][-1], flush=True)
                    del saved

        # A single-substep frame sees identical frozen body poses with feedback
        # enabled/disabled. Only the rigid result should differ at this point.
        restore(sim, contact)
        sim.substep = 1
        sim.frame_dt = sim.config.frame_dt / sim.config.substeps
        sim.solver.feedback = False
        sim.step()
        one_way = state_values(sim)
        restore(sim, contact)
        sim.solver.feedback = True
        sim.step()
        two_way = state_values(sim)
        for name in STATE_FIELDS[:2]:
            np.testing.assert_array_equal(two_way[name], one_way[name], err_msg=name)
        response = float(np.max(np.abs(two_way["body_qd"] - one_way["body_qd"])))
        assert response > 1e-7, response
        report["feedback"] = dict(particles_bitwise_equal=True, body_velocity_max_difference=response)

        restore(sim, contact)
        sim.substep = sim.config.substeps
        sim.frame_dt = sim.config.frame_dt
        sim.gravity = (9.81, 0., 0.)
        assert sim.graph_needs_recapture()
        np.testing.assert_allclose(sim.solver.rigid.mjw_model.opt.gravity.numpy(), [[9.81, 0., 0.]], atol=1e-6)
        sim.frame_dt *= .5
        saved = checkpoint(sim)
        sim.capture_graph()
        assert_capture_unchanged(sim, saved)
        sim.step()
        expected = state_values(sim)
        restore(sim, saved)
        sim.launch_graph()
        report["gravity_slow_motion"] = compare_result(sim, expected)
        del saved, contact

        sim.reset()
        assert sim.sim_time == 0. and sim.frame_dt == sim.config.frame_dt
        assert sim.solver.rigid._step == 0
        np.testing.assert_array_equal(sim.solver.rigid.mjw_data.time.numpy(), 0.)
        assert sim.graph_needs_recapture()
        np.testing.assert_allclose(tuple(sim.gravity), sim.config.gravity)
        for state in sim.states:
            for name in STATE_FIELDS:
                np.testing.assert_array_equal(getattr(state, name).numpy(), initial[name], err_msg=f"reset {name}")
            np.testing.assert_array_equal(state.body_f.numpy(), 0.)
            np.testing.assert_array_equal(state.particle_f.numpy(), 0.)
        np.testing.assert_array_equal(sim.solver.boundary_forces.numpy(), 0.)
        np.testing.assert_array_equal(sim.solver.rigid.mjw_data.qacc_warmstart.numpy(), 0.)
        for name in ("pre_Pos", "temp_pPos", "delta_Pos", "lambda_Opt", "volume_Rho",
                     "volume_Grad", "delta_Vel", "curl"):
            np.testing.assert_array_equal(getattr(sim.fluid_solver, name).numpy(), 0., err_msg=name)
        data = sim.BoundarySys.device_data
        np.testing.assert_allclose(data.transforms.numpy()[data.rigid_instance_ids.numpy()],
                                   sim.state.body_q.numpy()[data.body_ids.numpy()], atol=1e-7)
        assert [(s.particle_q.ptr, s.particle_qd.ptr, s.body_q.ptr) for s in sim.states] == pointers
        report["reset"] = dict(both_states=True, scratch=True, warmstart=True, backend_time=True, boundary_pose=True,
                               stable_bindings=True, paused_display_ready=True)
        # Continue from the reset state through automatic recapture.
        sim.launch_graph()
        assert sim.sim_time == sim.frame_dt
        assert np.isfinite(sim.pos.numpy()).all() and np.isfinite(sim.state.body_q.numpy()).all()
        report["passed"] = True
        path = Path(__file__).resolve().parents[2] / "outputs/pbf-newton/coupled-lifecycle.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
