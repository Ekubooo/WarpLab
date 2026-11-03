"""Pressure-feedback weak coupling on one Newton model and two states."""

import newton
import warp as wp
from newton.solvers import SolverBase, SolverMuJoCo

from .solver_pbf import SolverPBF


@wp.kernel
def add_boundary_forces(boundary_forces: wp.array(dtype=wp.spatial_vector),
                        body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    body_f[i] += boundary_forces[i]


@wp.kernel
def clear_body_forces(body_world: wp.array(dtype=int), mask: wp.array(dtype=wp.bool),
                      body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    if not mask or mask[body_world[i]]:
        body_f[i] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))


class SolverPBFMuJoCo(SolverBase):
    """Freeze boundaries during PBF, then advance rigid bodies with its reaction.

    The caller clears external forces before each substep. Only pressure
    corrections produce reactions; SDF projection remains a fallback.
    """

    def __init__(self, model, *, boundaries, config=None, grid_bounds=None,
                 feedback=True):
        super().__init__(model)
        self.boundaries = boundaries
        self.feedback = feedback
        self.fluid = SolverPBF(model, boundaries=boundaries, config=config,
                               grid_bounds=grid_bounds)
        self.rigid = SolverMuJoCo(model, use_mujoco_cpu=False,
            use_mujoco_contacts=True, update_data_interval=1, enable_sleeping=False)

    @property
    def _graph_revision(self):
        return self.fluid._graph_revision

    @property
    def boundary_forces(self):
        return self.fluid.boundary_forces

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        if observables is not None:
            self.validate_observables(observables, contacts)
        self.fluid.step(state_in, state_out, None, None, dt)
        if self.feedback:
            wp.launch(add_boundary_forces, self.model.body_count,
                inputs=[self.boundary_forces], outputs=[state_in.body_f], device=self.device)
        self.rigid.step(state_in, state_out, control, contacts, dt)
        self.boundaries.sync_body_poses(state_out)

    def reset(self, state, world_mask=None, flags=None):
        self.fluid.reset(state, world_mask, flags)
        self.rigid.reset(state, world_mask, flags)
        selected = newton.StateFlags.ALL if flags is None else flags
        if selected & newton.StateFlags.BODY_F:
            mask = self._normalize_reset_world_mask(world_mask)
            wp.launch(clear_body_forces, self.model.body_count,
                inputs=[self.model.body_world, mask], outputs=[state.body_f], device=self.device)
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        self.boundaries.sync_body_poses(state)

    def notify_model_changed(self, flags):
        self.fluid.notify_model_changed(flags)
        self.rigid.notify_model_changed(flags)

    def prepare_graph(self):
        self.fluid.prepare_graph()
        wp.load_module(module=__name__, device=self.device)

