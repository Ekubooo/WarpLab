"""Scene construction and frame scheduling around the Newton PBF solver."""

from pathlib import Path

import numpy as np
import warp as wp

from .config import PBFConfig


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
SDF_FILENAME = "stairs_scene.nvdb"


class Simulation:
    """One model, two complete states, and a common fluid/rigid substep clock.

    Each State keeps its array bindings. Particle slots retain spatial ordering;
    consumers read the current state through pos/v after each frame.
    """

    def __init__(self, *, config=None, verbose=False, boundaries=None, scene="static",
                 feedback=True):
        import newton
        from Lab_Solver.pbf import SolverPBF, boundary_model

        if scene not in ("static", "tank-box", "stairs-rigids"):
            raise ValueError(f"Unknown scene: {scene}")
        self.scene = scene
        if scene == "tank-box":
            from .scenes import tank_box_config, add_tank_box, load_tank_box_boundaries
        if scene != "static":
            from newton.solvers import SolverMuJoCo
        self.config = config if config is not None else (
            tank_box_config() if scene == "tank-box" else PBFConfig())
        self.verbose = verbose
        self.sim_time = 0.0
        self.frame_dt = self.config.frame_dt
        self.substep = self.config.substeps
        self.particle_radius = self.config.particle_radius
        self.width, self.height, self.length = self.config.container_size
        spacing = 2.0 * self.particle_radius
        self.smoothing_length = 4.0 * self.particle_radius
        counts = np.floor((np.asarray(self.config.block_end) - self.config.block_start)
                          / spacing + .5).astype(int) - 1
        if np.any(counts < 1):
            raise ValueError("Fluid block must fit at least one particle on every axis")
        # Preserve the original lattice numbering: Z varies fastest, then Y, X.
        lattice = np.indices(tuple(counts)).reshape(3, -1).T
        # Match the old GPU's fused multiply-add: float32 inputs, one rounding.
        positions = (np.asarray(self.config.block_start, dtype=np.float32).astype(np.float64)
                     + float(np.float32(spacing)) * (lattice + 1)).astype(np.float32)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.gravity = tuple(self.config.gravity)
        body_ids = []
        if scene != "static":
            SolverMuJoCo.register_custom_attributes(builder)
        if scene == "tank-box":
            body_ids = [add_tank_box(builder, self.config)]
        elif scene == "stairs-rigids":
            from .stairs_scene import load_stairs_assets, add_stairs_rigids, load_stairs_boundaries
            scene_assets = load_stairs_assets(self.config)
            body_ids = add_stairs_rigids(builder, scene_assets)
        self.is_coupled = bool(body_ids)
        builder.add_particles(
            pos=positions.tolist(),
            vel=[self.config.initial_velocity] * len(positions),
            mass=[.8 * spacing**3 * self.config.rest_density] * len(positions),
            radius=[self.particle_radius] * len(positions),
        )
        self.model = builder.finalize(device=wp.get_device())
        self.states = (self.model.state(), self.model.state())
        self._state_index = 0
        self.control = self.model.control()
        if boundaries is None:
            if scene == "tank-box":
                boundaries = load_tank_box_boundaries(self.config, self.model.device)
            elif scene == "stairs-rigids":
                boundaries = load_stairs_boundaries(scene_assets, self.config, self.model.device)
            else:
                from .asset_io import load_asset, load_asset_volumes, build_command
                from .scene_data import asset_geometry
                asset = load_asset(ASSET_DIR / "stairs_scene", asset_geometry("stairs_scene", self.config),
                                   command=build_command("static", self.config))
                sdf, volume_map = load_asset_volumes(asset, self.smoothing_length, self.model.device)
                boundaries = boundary_model.BoundarySystem(
                    [boundary_model.BoundaryAsset(asset.sdf_path, asset.mesh_path, sdf, volume_map)],
                    [boundary_model.BoundaryInstance(0)], self.model.device)
        if self.is_coupled:
            boundaries.bind_bodies(self.model, body_ids)
            for state in self.states:
                newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
            boundaries.sync_body_poses(self.state)
        self.BoundarySys = boundaries
        solver_config = SolverPBF.Config(
            rest_density=self.config.rest_density,
            pressure_iterations=self.config.pressure_iterations,
            lambda_regularization=self.config.lambda_regularization,
            viscosity=self.config.viscosity,
            vorticity_confinement=self.config.vorticity_confinement,
            max_speed=self.config.max_speed,
            artificial_pressure_strength=self.config.artificial_pressure_strength,
            artificial_pressure_q=self.config.artificial_pressure_q,
            clamp_negative_pressure=self.config.clamp_negative_pressure,
        )
        solver_type = SolverPBF
        solver_options = {}
        if self.is_coupled:
            from Lab_Solver.pbf.solver_coupled import SolverPBFMuJoCo
            solver_type = SolverPBFMuJoCo
            solver_options["feedback"] = feedback
            if scene == "stairs-rigids":
                solver_options.update(nconmax=512, njmax=2048)
        self.solver = solver_type(
            self.model, config=solver_config, boundaries=boundaries,
            grid_bounds=((0., 0., 0.), self.config.container_size),
            **solver_options,
        )
        self.fluid_solver = self.solver.fluid if self.is_coupled else self.solver
        self._gravity = wp.vec3(*self.config.gravity)
        self.n = self.model.particle_count
        self.step_graphs = None
        self._initial_bindings = self.fluid_solver.get_buffer_bindings()

    @property
    def state(self):
        return self.states[self._state_index]

    @property
    def state_out(self):
        return self.states[1 - self._state_index]

    @property
    def pos(self):
        return self.state.particle_q

    @property
    def v(self):
        return self.state.particle_qd

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, value):
        import newton
        values = np.asarray(value, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("gravity must have three finite components")
        self._gravity = wp.vec3(*value)
        self.model.set_gravity(values)
        self.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

    @property
    def dt(self):
        return self.frame_dt / self.substep

    def _advance_frame(self):
        for _ in range(self.substep):
            self._substep(self.dt)

    def _substep(self, dt):
        self.state.clear_forces()
        self.solver.step(self.state, self.state_out, self.control, None, dt)
        self._state_index = 1 - self._state_index

    def step(self):
        with wp.ScopedDevice(self.model.device), wp.ScopedTimer(
            "PBF frame", active=self.verbose, synchronize=False,
        ):
            self._advance_frame()
        self.sim_time += self.frame_dt

    def reset(self):
        with wp.ScopedDevice(self.model.device):
            for state in self.states:
                self.solver.reset(state)
                state.clear_forces()
            self._state_index = 0
            self.fluid_solver.set_buffer_bindings(self._initial_bindings)
            self.fluid_solver.pos, self.fluid_solver.v = self.pos, self.v
            if self.is_coupled:
                self.BoundarySys.sync_body_poses(self.state)
            else:
                self.BoundarySys.reset()
            self.gravity = self.config.gravity
        self.frame_dt = self.config.frame_dt
        self.sim_time = 0.0

    def capture_graph(self):
        """Capture both state and scratch parities; no physics is advanced."""
        if not self.model.device.is_cuda:
            raise RuntimeError("PBF graph capture requires a CUDA device")
        original = self.fluid_solver.get_buffer_bindings()
        state_index = self._state_index
        rigid_step = self.solver.rigid._step if self.is_coupled else None
        graphs = {}
        with wp.ScopedDevice(self.model.device):
            self.solver.prepare_graph()
            try:
                # Four variants also support capturing after externally inspected
                # substeps/reset, without assuming state and scratch parity match.
                for index in (0, 1):
                    for start in (original, original[::-1]):
                        self._state_index = index
                        self.fluid_solver.set_buffer_bindings(start)
                        with wp.ScopedCapture() as capture:
                            self._advance_frame()
                        key = (index, start[0].ptr)
                        graphs[key] = (capture.graph, self._state_index,
                                       self.fluid_solver.get_buffer_bindings())
            finally:
                self._state_index = state_index
                self.fluid_solver.set_buffer_bindings(original)
                if rigid_step is not None:
                    self.solver.rigid._step = rigid_step
                self.fluid_solver.pos, self.fluid_solver.v = self.pos, self.v
        self.step_graphs = graphs
        self._graph_dt = self.dt
        self._graph_substeps = self.substep
        self._graph_revision = self.solver._graph_revision

    def graph_needs_recapture(self):
        return (self.step_graphs is None or self.dt != self._graph_dt
                or self.substep != self._graph_substeps
                or self._graph_revision != self.solver._graph_revision)

    def launch_graph(self):
        if self.step_graphs is None:
            raise RuntimeError("Call capture_graph() before launch_graph()")
        if self.dt != self._graph_dt or self.substep != self._graph_substeps:
            raise RuntimeError("Time step changed; call capture_graph() again")
        if self._graph_revision != self.solver._graph_revision:
            raise RuntimeError("Solver parameters changed; call capture_graph() again")
        bindings = self.fluid_solver.get_buffer_bindings()
        key = (self._state_index, bindings[0].ptr)
        if key not in self.step_graphs:
            raise RuntimeError("Particle buffers changed; call capture_graph() again")
        graph, end_index, end = self.step_graphs[key]
        with wp.ScopedDevice(self.model.device), wp.ScopedTimer(
            "PBF frame", active=self.verbose, synchronize=False,
        ):
            wp.capture_launch(graph)
            self._state_index = end_index
            self.fluid_solver.set_buffer_bindings(end)
            self.fluid_solver.pos, self.fluid_solver.v = self.pos, self.v
            if self.is_coupled:
                self.solver.rigid._step += self.substep
        self.sim_time += self.frame_dt


def create_pbf_simulation(verbose=False, config=None, scene="static"):
    return Simulation(verbose=verbose, config=config, scene=scene)


def reset_simulation(simulation):
    simulation.reset()


def reverse_gravity(simulation):
    simulation.gravity = -simulation.gravity
