"""Scene initialization and frame scheduling around the single-substep solver."""

import warp as wp
from . import pbf_helper as fn
from . import boundary_model


class Simulation:
    def __init__(self, stage_path=None, verbose=False, config=None, scene="static", feedback=True):
        from .PBF import PBF

        self.config = config if config is not None else fn.PBFConfig()
        self.verbose = verbose
        self.sim_time = 0.0
        self.scene = scene
        fn.init_scene_parameters(self)
        if scene == "static":
            self.BoundarySys = boundary_model.load_boundaries(
                boundary_model.BoundaryScene(static_assets=(boundary_model.SDF_FILENAME,), default_rigid_count=0),
                boundary_model.BoundaryLoadOptions(self.smoothing_length, device=wp.get_device()),
            )
            self.model = fn.init_model(self)
        elif scene == "stairs-rigids":
            import newton
            from .scene import load_rigid_assets, add_rigid_scene, load_rigid_boundaries
            assets = load_rigid_assets()
            builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
            bodies = add_rigid_scene(builder, assets)
            self.model = fn.init_model(self, builder=builder)
            self.BoundarySys = load_rigid_boundaries(assets, self.smoothing_length, self.model.device)
            self.BoundarySys.bind_bodies(self.model, bodies)
        else:
            raise ValueError(f"Unknown PBF scene: {scene}")
        self.is_coupled = self.BoundarySys.rigid_count > 0
        if self.is_coupled:
            from .coupling import SolverPBFMuJoCo
            self.solver = SolverPBFMuJoCo(self.model, config=self.config,
                boundaries=self.BoundarySys, verbose=verbose, feedback=feedback)
            self.fluid_solver = self.solver.fluid
        else:
            self.solver = self.fluid_solver = PBF(self.model, config=self.config,
                boundaries=self.BoundarySys, verbose=verbose)
        self.use_mujoco_cpu = self.is_coupled and self.solver.rigid.use_mujoco_cpu
        fn.init_particle_state(self)
        self.states = (self.model.state(), self.model.state())
        self.control = self.model.control() if self.is_coupled else None
        self._state_index = 0
        self._initial_predict = (self.fluid_solver.pre_Pos, self.fluid_solver.temp_pPos)
        self.step_graphs = None
        if self.is_coupled:
            self.BoundarySys.sync_body_poses(self.state)

        self.renderer = None
        if stage_path:
            from .renderers.usd import PBFUsdRenderer
            self.set_renderer(PBFUsdRenderer(stage_path))

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
    def dt(self):
        return self.frame_dt / self.substep

    @property
    def gravity(self):
        return self.fluid_solver.gravity

    @gravity.setter
    def gravity(self, value):
        import newton
        self.model.set_gravity(value)
        self.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

    def _advance_frame(self):
        for _ in range(self.substep):
            if self.is_coupled:
                self.state.clear_forces()
            self.solver.step(self.state, self.state_out, self.control, None, self.dt)
            self._state_index = 1 - self._state_index

    def step(self):
        with wp.ScopedDevice(self.model.device), wp.ScopedTimer("sub-step", synchronize=False):
            self._advance_frame()
        self.sim_time += self.frame_dt

    def reset(self):
        for state in self.states:
            self.solver.reset(state)
            state.clear_forces()
        self._state_index = 0
        self.fluid_solver.pre_Pos, self.fluid_solver.temp_pPos = self._initial_predict
        self.gravity = wp.vec3(*self.config.gravity)
        self.frame_dt = self.config.frame_dt
        self.sim_time = 0.0
        if self.is_coupled:
            self.BoundarySys.sync_body_poses(self.state)
        else:
            self.BoundarySys.reset()

    def _graph_parameters(self):
        return self.dt, self.substep, tuple(self.gravity), self.solver.feedback if self.is_coupled else False

    def graph_needs_recapture(self):
        return self.step_graphs is None or self._graph_settings != self._graph_parameters()

    def capture_graph(self):
        """Capture State and prediction-buffer parities without advancing physics."""
        from . import PBF as kernels

        solver = self.fluid_solver
        index = self._state_index
        rigid_step = self.solver.rigid._step if self.is_coupled else None
        prediction = (solver.pre_Pos, solver.temp_pPos)
        graphs = {}
        with wp.ScopedDevice(self.model.device):
            wp.load_module(module=fn.__name__, device=self.model.device)
            wp.load_module(module=kernels.__name__, device=self.model.device)
            wp.load_module(module=boundary_model.__name__, device=self.model.device)
            if self.is_coupled:
                from . import coupling
                wp.load_module(module=coupling.__name__, device=self.model.device)
            solver.grid.reserve(self.n)
            try:
                for start_index in (0, 1):
                    for start in (prediction, prediction[::-1]):
                        self._state_index = start_index
                        solver.pre_Pos, solver.temp_pPos = start
                        with wp.ScopedCapture() as capture:
                            if self.use_mujoco_cpu:
                                # CPU integration cannot be captured. Record one GPU substep.
                                self.state.clear_forces()
                                self.solver.step_fluid(self.state, self.state_out, self.dt)
                                self._state_index = 1 - self._state_index
                            else:
                                self._advance_frame()
                        graphs[(start_index, start[0].ptr)] = (
                            capture.graph, self._state_index, solver.pre_Pos, solver.temp_pPos,
                        )
            finally:
                self._state_index = index
                solver.pre_Pos, solver.temp_pPos = prediction
                if self.is_coupled:
                    self.solver.rigid._step = rigid_step
        self.step_graphs = graphs
        self._graph_settings = self._graph_parameters()

    def launch_graph(self):
        if self.graph_needs_recapture():
            self.capture_graph()
        if self.use_mujoco_cpu:
            with wp.ScopedDevice(self.model.device):
                for _ in range(self.substep):
                    graph, index, prediction, temporary = self.step_graphs[
                        (self._state_index, self.fluid_solver.pre_Pos.ptr)]
                    wp.capture_launch(graph)
                    self.fluid_solver.pre_Pos, self.fluid_solver.temp_pPos = prediction, temporary
                    self.solver.rigid.step(self.state, self.state_out, self.control, None, self.dt)
                    self.BoundarySys.sync_body_poses(self.state_out)
                    self._state_index = index
            self.sim_time += self.frame_dt
            return
        graph, index, prediction, temporary = self.step_graphs[(self._state_index, self.fluid_solver.pre_Pos.ptr)]
        with wp.ScopedDevice(self.model.device):
            wp.capture_launch(graph)
        self._state_index = index
        self.fluid_solver.pre_Pos, self.fluid_solver.temp_pPos = prediction, temporary
        if self.is_coupled:
            self.solver.rigid._step += self.substep
        self.sim_time += self.frame_dt

    def set_renderer(self, renderer):
        """Attach a display/export component without changing physical state."""
        if renderer is self.renderer:
            return
        self.close()
        self.renderer = renderer
        if renderer is not None:
            try:
                renderer.initialize(self)
            except Exception:
                self.close()
                raise

    def register_keyboard_callback(self, key, callback, *, repeat=True):
        self.renderer.register_keyboard_callback(key, callback, repeat=repeat)

    def render(self):
        if self.renderer is not None:
            self.renderer.render(self)

    def close(self):
        renderer, self.renderer = self.renderer, None
        if renderer is not None:
            renderer.close()


def create_pbf_simulation(verbose=False, config=None, scene="static"):
    return Simulation(verbose=verbose, config=config, scene=scene)


def reverse_gravity(sim):
    sim.gravity = -sim.gravity


def reset_simulation(sim):
    sim.reset()
