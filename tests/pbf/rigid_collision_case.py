"""Controlled GPU rigid contacts on the published stairs, without fluid kernels."""

import argparse
import json
import math
from pathlib import Path

import newton
from newton.solvers import SolverMuJoCo
from mujoco_warp._src.types import OverflowType
import numpy as np
import warp as wp

from PBF import scene


ROOT = Path(__file__).resolve().parents[2]


class RigidCollisionCases:
    def __init__(self, use_cpu=False):
        self.dt = 1. / 450.
        self.device = wp.get_device("cuda:0")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.gravity = (0., 0., 0.)
        SolverMuJoCo.register_custom_attributes(builder)
        self.bodies = scene.add_rigid_scene(builder, scene.load_rigid_assets())
        # A sentinel confirms that the rigid backend leaves particle output alone.
        builder.add_particles(pos=[(1., 2., .8)], vel=[(.1, .2, .3)], mass=[.0125], radius=[.0125])
        self.model = builder.finalize(device=self.device)
        self.states = (self.model.state(), self.model.state())
        self.control = self.model.control()
        self.rigid = SolverMuJoCo(self.model, use_mujoco_cpu=use_cpu,
            use_mujoco_contacts=True, update_data_interval=1, enable_sleeping=False,
            nconmax=2048, njmax=8192)
        self.com = self.model.body_com.numpy()
        self.shape_body = self.model.shape_body.numpy()
        self.geom_shape = self.rigid.mjc_geom_to_newton_shape.numpy()[0]
        self.body_joint = {int(body): i for i, body in enumerate(self.model.joint_child.numpy())}
        self.q_start = self.model.joint_q_start.numpy()
        self.qd_start = self.model.joint_qd_start.numpy()
        self.graphs = None
        self.index = 0

    @property
    def state(self):
        return self.states[self.index]

    def place(self, centers, velocities=None, gravity=(0., 0., 0.)):
        self.index = 0
        self.model.set_gravity(gravity)
        self.rigid.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        q = self.model.joint_q.numpy().copy()
        qd = np.zeros_like(self.model.joint_qd.numpy())
        velocities = {} if velocities is None else velocities
        for local, body in enumerate(self.bodies):
            center = centers.get(local, (.5 + .65 * local, 3.2, .35))
            joint = self.body_joint[body]
            start, dof = self.q_start[joint], self.qd_start[joint]
            q[start:start + 3] = np.asarray(center) - self.com[body]
            q[start + 3:start + 7] = (0., 0., 0., 1.)
            qd[dof:dof + 3] = velocities.get(local, (0., 0., 0.))
        for state in self.states:
            self.rigid.reset(state)
            state.clear_forces()
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        self.stats = dict(peak_contacts=0, peak_constraints=0, peak_broadphase_pairs=0,
                          overflow_bits=0, minimum_contact_distance=0.)
        if self.rigid.use_mujoco_cpu:
            self.stats["peak_broadphase_pairs"] = None

    def advance(self):
        self.state.clear_forces()
        self.rigid.step(self.state, self.states[1 - self.index], self.control, None, self.dt)
        self.index = 1 - self.index

    def prepare_graphs(self):
        self.place({0: (1., 2., .8)})
        self.advance()  # Load the rigid backend's kernels outside capture.
        if self.rigid.use_mujoco_cpu:
            return
        index, counter = self.index, self.rigid._step
        self.graphs = {}
        for start in (0, 1):
            self.index = start
            with wp.ScopedCapture() as capture:
                self.advance()
            self.graphs[start] = capture.graph
        self.index, self.rigid._step = index, counter

    def contacts(self):
        if self.rigid.use_mujoco_cpu:
            data = self.rigid.mj_data
            assert not np.any(data.warning.number), data.warning.number
            self.stats["peak_contacts"] = max(self.stats["peak_contacts"], data.ncon)
            self.stats["peak_constraints"] = max(self.stats["peak_constraints"], data.nefc)
            return self.contact_pairs(data.contact.geom[:data.ncon], data.contact.dist[:data.ncon])
        data = self.rigid.mjw_data
        count = int(data.nacon.numpy()[0])
        constraints = int(data.nefc.numpy().max())
        broadphase = int(data.ncollision.numpy()[0])
        overflow = int(np.bitwise_or.reduce(data.overflow.numpy()))
        self.stats["peak_contacts"] = max(self.stats["peak_contacts"], count)
        self.stats["peak_constraints"] = max(self.stats["peak_constraints"], constraints)
        self.stats["peak_broadphase_pairs"] = max(self.stats["peak_broadphase_pairs"], broadphase)
        self.stats["overflow_bits"] |= overflow
        capacity_mask = int(OverflowType.ALL & ~(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS))
        assert not overflow & capacity_mask, f"MuJoCo capacity overflow: {overflow}"
        assert count < data.naconmax and broadphase < data.naconmax and constraints < data.njmax
        geoms = data.contact.geom.numpy()[:count]
        distances = data.contact.dist.numpy()[:count]
        return self.contact_pairs(geoms, distances)

    def contact_pairs(self, geoms, distances):
        if not len(distances):
            return []
        self.stats["minimum_contact_distance"] = min(self.stats["minimum_contact_distance"], float(distances.min()))
        assert distances.min() > -.02, "Excessive controlled-contact penetration"
        pairs = []
        for geoms, distance in zip(geoms, distances):
            if distance <= .001:
                assert np.all(geoms >= 0)
                shapes = self.geom_shape[geoms]
                assert np.all(shapes >= 0)
                pairs.append(tuple(int(shape) for shape in shapes))
        return pairs

    def run_motion(self, seconds, observe=None):
        particles = [(s.particle_q.numpy().copy(), s.particle_qd.numpy().copy()) for s in self.states]
        pairs = set()
        steps = math.ceil(seconds / self.dt)
        for step in range(steps):
            if self.rigid.use_mujoco_cpu:
                self.advance()
            else:
                wp.capture_launch(self.graphs[self.index])
                self.index = 1 - self.index
                self.rigid._step += 1
            current = self.contacts()  # Check all capacities after every substep.
            pairs.update(current)
            if observe is not None:
                observe(current, (step + 1) * self.dt)
        for state, (q, qd) in zip(self.states, particles):
            np.testing.assert_array_equal(state.particle_q.numpy(), q)
            np.testing.assert_array_equal(state.particle_qd.numpy(), qd)
        assert np.isfinite(self.state.body_q.numpy()).all()
        assert np.isfinite(self.state.body_qd.numpy()).all()
        return pairs

    def result(self, name, pairs):
        return dict(name=name, **self.stats,
                    pairs=sorted([self.model.shape_label[a], self.model.shape_label[b]] for a, b in pairs),
                    final_body_q=self.state.body_q.numpy().tolist(),
                    final_body_qd=self.state.body_qd.numpy().tolist())

    def run(self):
        self.prepare_graphs()
        results = []
        self.place({0: (1., 2., .8)}, gravity=(0., -9.81, 0.))
        seconds = 20 * self.dt
        pairs = self.run_motion(seconds)
        assert not pairs
        np.testing.assert_allclose(self.state.body_qd.numpy()[0, 1], -9.81 * seconds, atol=1e-5)
        results.append(self.result("free-fall", pairs))

        for name, moving, target in (("box-box", 0, 1), ("sphere-box", 2, 0)):
            self.place({moving: (1.15, 2., .8), target: (1.65, 2., .8)}, {moving: (.6, 0., 0.)})
            pairs = self.run_motion(.8)
            expected = {self.bodies[moving], self.bodies[target]}
            assert any(set(self.shape_body[list(pair)]) == expected for pair in pairs), name
            velocity = self.state.body_qd.numpy()
            assert velocity[target, 0] > .01 and velocity[moving, 0] < .59, name
            assert self.state.body_q.numpy()[target, 0] > 1.655, name
            results.append(self.result(name, pairs))

        for name, center, velocity, gravity in (
            ("stairs", (2.1, .8, .525), (0., 0., 0.), (0., -9.81, 0.)),
            ("wall", (2.55, 2., .8), (1., 0., 0.), (0., 0., 0.)),
        ):
            self.place({0: center}, {0: velocity}, gravity)
            pairs = self.run_motion(.7)
            touched = [self.model.shape_label[s] for pair in pairs if 0 in self.shape_body[list(pair)]
                       for s in pair if self.shape_body[s] == -1]
            assert any(label.startswith("stairs_solid_") for label in touched), name
            q, qd = self.state.body_q.numpy()[0], self.state.body_qd.numpy()[0]
            if name == "stairs":
                assert abs(q[1] - .55) < .02 and abs(qd[1]) < .1, (q, qd)
            else:
                assert 2.80 < q[0] <= 2.885 and abs(qd[0]) < .15, (q, qd)
            results.append(self.result(name, pairs))

        self.place({0: (1.55, 1.6, .8)}, gravity=(0., -9.81, 0.))
        response = []
        def observe(pairs, time):
            for pair in pairs:
                labels = [self.model.shape_label[s] for s in pair]
                if 0 in self.shape_body[list(pair)] and any(s.startswith("static_bunny_") for s in labels):
                    response.append(float(self.state.body_qd.numpy()[0, 1]) + 9.81 * time)
                    break
        pairs = self.run_motion(.7, observe)
        assert response and max(response) > .5, "Bunny collision did not interrupt free fall"
        result = self.result("platform-bunny", pairs)
        result["max_vertical_response_vs_free_fall"] = max(response)
        results.append(result)
        return dict(passed=True, backend="MuJoCo CPU" if self.rigid.use_mujoco_cpu else "MuJoCo Warp GPU", dt=self.dt,
                    nconmax=None if self.rigid.use_mujoco_cpu else self.rigid.mjw_data.naconmax,
                    njmax=None if self.rigid.use_mujoco_cpu else self.rigid.mjw_data.njmax,
                    checked_every_substep=True, particle_output_unchanged=True, cases=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pbf-newton/rigid-collisions.json")
    args = parser.parse_args()
    with wp.ScopedDevice("cuda:0"):
        report = RigidCollisionCases(use_cpu=args.cpu).run()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": [case["name"] for case in report["cases"]]}))


if __name__ == "__main__":
    main()
