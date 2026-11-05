"""A Newton particle solver preserving the original spatially sorted PBF path."""

import math

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverBase

from . import pbf_helper as fn
from .boundary_model import BoundarySystem, BoundaryCouplingData
from .kernels import create_kernels


# =============================================================================
# Model and state ownership
# Only homogeneous fluid particles are integrated. Particle slots follow the
# existing spatial permutation; fixed-ID inputs and native contacts are unused.
# =============================================================================

class SolverPBF(SolverBase):
    """Advance one fluid substep into a distinct State, without ID restoration.

    This first version supports one CUDA fluid world with equal positive mass
    and radius, volume boundaries, no autodiff, and no native Contacts.
    Particle forces/constraints and fixed cross-step particle identities are not
    supported. State array bindings remain fixed, but particle slots follow
    the existing spatial permutation rather than stable cross-step identities.
    """

    Config = fn.PBFConfig

    def __init__(self, model: newton.Model, *, config: fn.PBFConfig | None = None,
                 boundaries: BoundarySystem, grid_bounds=None):
        super().__init__(model)
        self.config = self.Config() if config is None else config
        self.boundaries = boundaries
        self.n = model.particle_count
        self._validate_model()
        radii = model.particle_radius.numpy()
        masses = model.particle_mass.numpy()
        # Use the shortest decimal representation of the model's float32
        # scalars, preserving the original configured coefficients (e.g. .0125).
        self.particle_radius = float(str(radii[0]))
        self.fluid_mass = float(str(masses[0]))
        self.smoothing_length = self.support_radius = 4.0 * self.particle_radius
        self.iterations = self.config.pressure_iterations
        self.gravity = wp.vec3(*model.gravity.numpy()[self._particle_world])
        if boundaries.device != model.device or boundaries.static_count != 1:
            raise ValueError("SolverPBF requires one static boundary on the model device")
        if boundaries.rigid_count and boundaries.bound_model is not model:
            raise ValueError("Bind dynamic boundaries to this Newton model before constructing SolverPBF")
        primary = next(item for item in boundaries.instances if not item.is_dynamic)
        asset = boundaries.assets[primary.asset_id]
        self.static_sdf, self.volume_map = asset.sdf, asset.volume_map
        if grid_bounds is None:
            positions = model.particle_q.numpy()
            grid_bounds = (positions.min(0) - self.support_radius,
                           positions.max(0) + self.support_radius)
        self.hash_grid_dims = fn.compute_hash_grid_dims(*grid_bounds, self.support_radius)
        with wp.ScopedDevice(model.device):
            self.grid = wp.HashGrid(*self.hash_grid_dims)
            self.pre_Pos = wp.zeros(self.n, dtype=wp.vec3)
            self.delta_Pos = wp.zeros(self.n, dtype=wp.vec3)
            self.lambda_Opt = wp.zeros(self.n, dtype=float)
            self.volume_Rho = wp.zeros(self.n, dtype=float)
            self.volume_Grad = wp.zeros(self.n, dtype=wp.vec3)
            self.delta_Vel = wp.zeros(self.n, dtype=wp.vec3)
            self.curl = wp.zeros(self.n, dtype=wp.vec4)
            self.temp_pPos = wp.zeros(self.n, dtype=wp.vec3)
            self.boundary_forces = wp.zeros(model.body_count, dtype=wp.spatial_vector)
            self.n_Cache = fn.NeighborCache().allocateNCache(self.n, self.config.neighbor_capacity)
        parameters = fn.KernelParameters(self.config, self.particle_radius, self.fluid_mass)
        self.boundary_data = BoundaryCouplingData()
        self.boundary_data.instances = boundaries.device_data
        self.boundary_data.forces = self.boundary_forces
        self.kernels = create_kernels(parameters, self.grid.id, self.static_sdf.id,
                                      self.volume_map.id, boundaries.rigid_count)
        # Diagnostic aliases only; State owns the arrays and their bindings.
        self.pos = self.v = None
        self._model_signature = (self.n, radii.tobytes(), masses.tobytes())
        self._graph_revision = 0

    def _validate_model(self):
        if not self.model.device.is_cuda:
            raise ValueError("SolverPBF static NanoVDB boundaries require CUDA")
        if self.model.requires_grad:
            raise ValueError("SolverPBF does not support automatic differentiation")
        if self.n < 1 or self.model.world_count > 1:
            raise ValueError("SolverPBF requires a nonempty single fluid world")
        worlds = np.unique(self.model.particle_world.numpy())
        if len(worlds) != 1:
            raise ValueError("SolverPBF requires all fluid particles to belong to one world")
        self._particle_world = int(worlds[0])
        for label in ("particle_radius", "particle_mass"):
            values = getattr(self.model, label).numpy()
            if not np.isfinite(values).all() or np.any(values <= 0) or np.any(values != values[0]):
                raise ValueError(f"SolverPBF requires equal, finite, positive {label}")
        flags = self.model.particle_flags.numpy()
        if np.any((flags & int(newton.ParticleFlags.ACTIVE)) == 0):
            raise ValueError("SolverPBF requires all fluid particles to be active")

    def _validate_state(self, state: newton.State) -> None:
        for name in ("particle_q", "particle_qd"):
            values = getattr(state, name, None)
            if (not isinstance(values, wp.array) or values.dtype != wp.vec3
                    or values.shape != (self.n,) or values.device != self.model.device):
                raise ValueError(f"Expected {name} with shape ({self.n},) on the model device")
            if values.requires_grad:
                raise ValueError("SolverPBF does not support automatic differentiation")

    # =========================================================================
    # One original GPU substep
    # Preserve launch order and cache layout; reorder directly into output.
    # =========================================================================

    def step(self, state_in: newton.State, state_out: newton.State,
             control: newton.Control | None, contacts: newton.Contacts | None, dt: float,
             *, observables: SolverBase.Observables | None = None) -> None:
        """Advance a single substep; control and particle force channels are unused."""
        if state_in is state_out:
            raise ValueError("SolverPBF requires distinct input and output States")
        if contacts is not None:
            raise NotImplementedError("SolverPBF uses SDF boundaries, not Newton Contacts")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if observables is not None:
            self.validate_observables(observables, contacts)
        self._validate_state(state_in)
        self._validate_state(state_out)
        input_pointers = {state_in.particle_q.ptr, state_in.particle_qd.ptr}
        output_pointers = {state_out.particle_q.ptr, state_out.particle_qd.ptr}
        if input_pointers & output_pointers or len(output_pointers) != 2:
            raise ValueError("Input and output particle arrays must not alias")
        self.pos, self.v = state_out.particle_q, state_out.particle_qd
        if self.boundaries.rigid_count:
            self.boundaries.sync_body_poses(state_in)
            self.boundary_forces.zero_()
        k = self.kernels

        wp.launch(k.apply_predict, self.n,
            inputs=[state_in.particle_q, dt, self.gravity, self.pre_Pos, state_in.particle_qd],
            device=self.device)
        self.grid.build(self.pre_Pos, self.smoothing_length)
        wp.launch(k.reorder, self.n, inputs=[state_in.particle_q, self.pre_Pos],
            outputs=[self.pos, self.temp_pPos], device=self.device)
        self.pre_Pos, self.temp_pPos = self.temp_pPos, self.pre_Pos
        wp.launch(k.cache_neighbors, self.n, inputs=[self.pre_Pos], outputs=[self.n_Cache], device=self.device)

        for _ in range(self.iterations):
            wp.launch(k.volume_query, self.n, inputs=[self.pre_Pos, self.boundary_data],
                outputs=[self.volume_Rho, self.volume_Grad], device=self.device)
            wp.launch(k.calc_lambda, self.n,
                inputs=[self.pre_Pos, self.n_Cache, self.volume_Rho, self.volume_Grad],
                outputs=[self.lambda_Opt], device=self.device)
            wp.launch(k.calc_deltaPos, self.n,
                inputs=[self.pre_Pos, self.lambda_Opt, self.n_Cache, self.volume_Grad,
                        dt, self.boundary_data],
                outputs=[self.delta_Pos], device=self.device)
            wp.launch(k.update_prePos, self.n, inputs=[self.delta_Pos, self.boundary_data],
                outputs=[self.pre_Pos], device=self.device)

        wp.launch(k.update_position, self.n, inputs=[dt, self.pre_Pos],
            outputs=[self.pos, self.v], device=self.device)
        wp.launch(k.calc_curl, self.n, inputs=[self.pre_Pos, self.v, self.n_Cache],
            outputs=[self.curl], device=self.device)
        wp.launch(k.calc_visvor, self.n, inputs=[dt, self.pre_Pos, self.v, self.curl, self.n_Cache],
            outputs=[self.delta_Vel], device=self.device)
        wp.launch(k.update_velocity, self.n, inputs=[self.delta_Vel], outputs=[self.v], device=self.device)

    # =========================================================================
    # Reset, model updates and graph bindings
    # These operations run outside steady-state fluid advancement.
    # =========================================================================

    def reset(self, state: newton.State, world_mask: wp.array | None = None,
              flags: newton.StateFlags | int | None = None) -> None:
        self._validate_state(state)
        mask = self._normalize_reset_world_mask(world_mask)
        if mask is not None:
            selected = mask.numpy()
            worlds = self.model.particle_world.numpy()
            if not np.any(selected[worlds]):
                return
        selected_flags = newton.StateFlags.ALL if flags is None else flags
        with wp.ScopedDevice(self.device):
            if selected_flags & newton.StateFlags.PARTICLE_Q:
                wp.copy(state.particle_q, self.model.particle_q)
            if selected_flags & newton.StateFlags.PARTICLE_QD:
                wp.copy(state.particle_qd, self.model.particle_qd)
            if selected_flags & newton.StateFlags.PARTICLE_F and state.particle_f is not None:
                state.particle_f.zero_()
            for buffer in (self.pre_Pos, self.delta_Pos, self.lambda_Opt, self.volume_Rho,
                           self.volume_Grad, self.delta_Vel, self.curl, self.temp_pPos):
                buffer.zero_()
            self.n_Cache.NeighborCount.zero_()
            self.boundary_forces.zero_()
            self.grid.build(state.particle_q, self.smoothing_length)
        self.pos, self.v = state.particle_q, state.particle_qd

    def set_gravity(self, gravity):
        values = np.asarray(gravity, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("gravity must have three finite components")
        self.model.set_gravity(values)
        self.gravity = wp.vec3(*values)
        self._graph_revision += 1

    def notify_model_changed(self, flags: newton.ModelFlags | int) -> None:
        if flags & newton.ModelFlags.MODEL_PROPERTIES:
            signature = (self.model.particle_count, self.model.particle_radius.numpy().tobytes(),
                         self.model.particle_mass.numpy().tobytes())
            if signature != self._model_signature:
                raise ValueError("Particle topology, radius or mass changed; recreate SolverPBF")
            self.gravity = wp.vec3(*self.model.gravity.numpy()[self._particle_world])
            self._graph_revision += 1

    def get_buffer_bindings(self):
        return self.pre_Pos, self.temp_pPos

    def set_buffer_bindings(self, bindings):
        self.pre_Pos, self.temp_pPos = bindings

    def prepare_graph(self):
        with wp.ScopedDevice(self.device):
            for kernel in vars(self.kernels).values():
                wp.load_module(kernel.module, device=self.device)
            self.grid.reserve(self.n)
