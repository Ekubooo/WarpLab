"""One pressure-coupled fluid/rigid substep on a shared Newton model."""

import newton
import warp as wp
from newton.solvers import SolverBase, SolverMuJoCo

from .PBF import PBF

wp.set_module_options({"enable_backward": False})

# Select once when constructing the solver; particles always remain on GPU.
USE_MUJOCO_CPU = False


@wp.kernel
def add_boundary_forces(forces: wp.array(dtype=wp.spatial_vector),
                        body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    body_f[i] += forces[i]


class SolverPBFMuJoCo(SolverBase):
    def __init__(self, model, *, config, boundaries, verbose=False, feedback=True):
        super().__init__(model)
        self.boundaries = boundaries
        self.feedback = feedback
        self.fluid = PBF(model, config=config, boundaries=boundaries, verbose=verbose)
        self.rigid = SolverMuJoCo(model, use_mujoco_cpu=USE_MUJOCO_CPU,
            use_mujoco_contacts=True, update_data_interval=1, enable_sleeping=False,
            nconmax=2048, njmax=8192)

    @property
    def boundary_forces(self):
        return self.fluid.boundary_forces

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        # The caller clears external forces before adding this substep's inputs.
        if observables is not None:
            raise NotImplementedError("PBF coupling observables are not supported")
        self.step_fluid(state_in, state_out, dt)
        self.rigid.step(state_in, state_out, control, contacts, dt)
        self.boundaries.sync_body_poses(state_out)

    def step_fluid(self, state_in, state_out, dt):
        """GPU pressure exchange, also captured separately for CPU rigid steps."""
        self.boundaries.sync_body_poses(state_in)
        self.boundary_forces.zero_()
        self.fluid.step(state_in, state_out, None, None, dt)
        if self.feedback:
            wp.launch(add_boundary_forces, self.model.body_count,
                inputs=[self.boundary_forces], outputs=[state_in.body_f], device=self.device)

    def reset(self, state, world_mask=None, flags=None):
        self.fluid.reset(state, world_mask, flags)
        self.rigid.reset(state)
        self.rigid._step = 0
        if self.rigid.use_mujoco_cpu:
            import mujoco
            mujoco.mj_resetData(self.rigid.mj_model, self.rigid.mj_data)
        else:
            self.rigid.mjw_data.time.zero_()
        state.clear_forces()
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        self.boundaries.sync_body_poses(state)

    def notify_model_changed(self, flags):
        self.fluid.notify_model_changed(flags)
        self.rigid.notify_model_changed(flags)
