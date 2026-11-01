"""Scene construction and frame scheduling around the Newton PBF solver."""

from pathlib import Path

import numpy as np
import warp as wp

from .config import PBFConfig


ASSET_DIR = Path(__file__).resolve().parent / "Assets"
SDF_FILENAME = "stairs_scene.nvdb"


class Simulation:
    """Own one Newton state and advance its fluid particles in place.

    Particle arrays may change bindings during spatial sorting. Consumers must
    read pos/v (or the current Newton State) each frame rather than cache them.
    """

    def __init__(self, *, config=None, verbose=False, boundaries=None):
        import newton
        from Lab_Solver.pbf import SolverPBF, boundary_model

        self.config = config if config is not None else PBFConfig()
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
        builder.add_particles(
            pos=positions.tolist(),
            vel=[self.config.initial_velocity] * len(positions),
            mass=[.8 * spacing**3 * self.config.rest_density] * len(positions),
            radius=[self.particle_radius] * len(positions),
        )
        self.model = builder.finalize(device=wp.get_device())
        self.state = self.model.state()
        if boundaries is None:
            scene = boundary_model.BoundaryScene(
                asset_dir=ASSET_DIR, static_assets=(SDF_FILENAME,), default_rigid_count=0,
            )
            boundaries = boundary_model.load_boundaries(
                scene, boundary_model.BoundaryLoadOptions(self.smoothing_length, device=self.model.device),
            )
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
        self.solver = SolverPBF(
            self.model, config=solver_config, boundaries=boundaries,
            grid_bounds=((0., 0., 0.), self.config.container_size),
        )
        self.solver.bind_state(self.state)
        self._gravity = wp.vec3(*self.config.gravity)
        self.n = self.model.particle_count
        self.step_graphs = None

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
        self._gravity = wp.vec3(*value)
        self.solver.set_gravity(self._gravity)

    @property
    def dt(self):
        return self.frame_dt / self.substep

    def _advance_frame(self):
        for _ in range(self.substep):
            self.solver.step(self.state, self.state, control=None, contacts=None, dt=self.dt)

    def step(self):
        with wp.ScopedDevice(self.model.device), wp.ScopedTimer(
            "PBF frame", active=self.verbose, synchronize=False,
        ):
            self._advance_frame()
        self.sim_time += self.frame_dt

    def reset(self):
        with wp.ScopedDevice(self.model.device):
            self.solver.reset(self.state)
            self.BoundarySys.reset()
            self.gravity = self.config.gravity
        self.frame_dt = self.config.frame_dt
        self.sim_time = 0.0

    def capture_graph(self):
        """Capture both buffer parities without restoring particle order or copies."""
        if not self.model.device.is_cuda:
            raise RuntimeError("PBF graph capture requires a CUDA device")
        original = self.solver.get_buffer_bindings()
        graphs = {}
        with wp.ScopedDevice(self.model.device):
            self.solver.prepare_graph()
            try:
                for start in (original, (original[2], original[3], original[0], original[1])):
                    self.solver.set_buffer_bindings(start, self.state)
                    with wp.ScopedCapture() as capture:
                        self._advance_frame()
                    key = (start[0].ptr, start[1].ptr)
                    graphs[key] = (capture.graph, self.solver.get_buffer_bindings())
            finally:
                self.solver.set_buffer_bindings(original, self.state)
        self.step_graphs = graphs
        self._graph_dt = self.dt
        self._graph_revision = self.solver._graph_revision

    def launch_graph(self):
        if self.step_graphs is None:
            raise RuntimeError("Call capture_graph() before launch_graph()")
        if self.dt != self._graph_dt:
            raise RuntimeError("Time step changed; call capture_graph() again")
        if self._graph_revision != self.solver._graph_revision:
            raise RuntimeError("Solver parameters changed; call capture_graph() again")
        bindings = self.solver.get_buffer_bindings()
        key = (bindings[0].ptr, bindings[1].ptr)
        if key not in self.step_graphs:
            raise RuntimeError("Particle buffers changed; call capture_graph() again")
        graph, end = self.step_graphs[key]
        with wp.ScopedDevice(self.model.device), wp.ScopedTimer(
            "PBF frame", active=self.verbose, synchronize=False,
        ):
            wp.capture_launch(graph)
            self.solver.set_buffer_bindings(end, self.state)
        self.sim_time += self.frame_dt


def create_pbf_simulation(verbose=False, config=None):
    return Simulation(verbose=verbose, config=config)


def reset_simulation(simulation):
    simulation.reset()


def reverse_gravity(simulation):
    simulation.gravity = -simulation.gravity
