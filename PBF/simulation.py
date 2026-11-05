"""Scene initialization and frame scheduling around the single-substep solver."""

import warp as wp
from . import pbf_helper as fn
from . import boundary_model


class Simulation:
    def __init__(self, stage_path=None, verbose=False, config=None):
        from .PBF import PBF

        self.config = config if config is not None else fn.PBFConfig()
        self.verbose = verbose
        self.sim_time = 0.0
        fn.init_scene_parameters(self)
        self.BoundarySys = boundary_model.load_boundaries(
            boundary_model.BoundaryScene(static_assets=(boundary_model.SDF_FILENAME,), default_rigid_count=0),
            boundary_model.BoundaryLoadOptions(self.smoothing_length, device=wp.get_device()),
        )
        self.model = fn.init_model(self)
        self.solver = PBF(self.model, config=self.config, boundaries=self.BoundarySys, verbose=verbose)
        fn.init_particle_state(self)
        self.states = (self.model.state(), self.model.state())
        self._state_index = 0
        self._initial_predict = (self.solver.pre_Pos, self.solver.temp_pPos)
        self.step_graphs = None

        from warp.render import UsdRenderer
        self.renderer = UsdRenderer(stage_path) if stage_path else None

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
        return self.solver.gravity

    @gravity.setter
    def gravity(self, value):
        import newton
        self.model.set_gravity(value)
        self.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

    def _substep(self, dt):
        self.solver.step(self.state, self.state_out, None, None, dt)
        self._state_index = 1 - self._state_index

    def _advance_frame(self):
        for _ in range(self.substep):
            self._substep(self.dt)

    def step(self):
        with wp.ScopedDevice(self.model.device), wp.ScopedTimer("sub-step", synchronize=False):
            self._advance_frame()
        self.sim_time += self.frame_dt

    def reset(self):
        for state in self.states:
            self.solver.reset(state)
            state.clear_forces()
        self._state_index = 0
        self.solver.pre_Pos, self.solver.temp_pPos = self._initial_predict
        self.gravity = wp.vec3(*self.config.gravity)
        self.frame_dt = self.config.frame_dt
        self.sim_time = 0.0
        self.BoundarySys.reset()

    def _graph_parameters(self):
        return self.dt, self.substep, tuple(self.gravity)

    def graph_needs_recapture(self):
        return self.step_graphs is None or self._graph_settings != self._graph_parameters()

    def capture_graph(self):
        """Capture State and prediction-buffer parities without advancing physics."""
        from . import PBF as kernels

        solver = self.solver
        index = self._state_index
        prediction = (solver.pre_Pos, solver.temp_pPos)
        graphs = {}
        with wp.ScopedDevice(self.model.device):
            wp.load_module(module=fn.__name__, device=self.model.device)
            wp.load_module(module=kernels.__name__, device=self.model.device)
            solver.grid.reserve(self.n)
            try:
                for start_index in (0, 1):
                    for start in (prediction, prediction[::-1]):
                        self._state_index = start_index
                        solver.pre_Pos, solver.temp_pPos = start
                        with wp.ScopedCapture() as capture:
                            self._advance_frame()
                        graphs[(start_index, start[0].ptr)] = (
                            capture.graph, self._state_index, solver.pre_Pos, solver.temp_pPos,
                        )
            finally:
                self._state_index = index
                solver.pre_Pos, solver.temp_pPos = prediction
        self.step_graphs = graphs
        self._graph_settings = self._graph_parameters()

    def launch_graph(self):
        if self.graph_needs_recapture():
            self.capture_graph()
        graph, index, prediction, temporary = self.step_graphs[(self._state_index, self.solver.pre_Pos.ptr)]
        with wp.ScopedDevice(self.model.device):
            wp.capture_launch(graph)
        self._state_index = index
        self.solver.pre_Pos, self.solver.temp_pPos = prediction, temporary
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return
        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.pos.numpy(), radius=self.particle_radius, name="points", colors=(0.8, 0.3, 0.2)
            )
            self.renderer.end_frame()


def create_pbf_simulation(verbose=False, config=None):
    return Simulation(verbose=verbose, config=config)


def reverse_gravity(sim):
    sim.gravity = -sim.gravity


def reset_simulation(sim):
    sim.reset()
