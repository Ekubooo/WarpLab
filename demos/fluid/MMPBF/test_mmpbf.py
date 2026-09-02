"""CPU parity tests for the Warp port of PBD's official FluidDemo."""

# ruff: noqa: E402 - configure Warp's kernel cache before importing solver modules.

import inspect
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

wp.config.kernel_cache_dir = "/tmp/warp-mmpbf-tests"

from . import MMPBF as solver
from . import mmpbf_functions as functions
from . import mmpbf_initialization as initialization
from .MMPBF import Example, MMPBFConfig
from .render_opengl import (
    MMPBF_CAMERA_FRONT,
    MMPBF_CAMERA_POS,
    MMPBF_GRAVITY_ROTATION_STEP,
    MMPBF_SCENE_SCALING,
    PlaybackScheduler,
    create_rendered_simulation,
    register_keyboard_controls,
    reverse_gravity,
    rotate_gravity,
    parse_args,
)


def cubic_value(distance: float, support_radius: float) -> float:
    q = distance / support_radius
    if q > 1.0:
        return 0.0
    coefficient = 8.0 / (math.pi * support_radius**3)
    if q <= 0.5:
        return coefficient * (6.0 * q**3 - 6.0 * q**2 + 1.0)
    return coefficient * 2.0 * (1.0 - q) ** 3


def cubic_gradient(displacement: np.ndarray, support_radius: float) -> np.ndarray:
    distance = float(np.linalg.norm(displacement))
    q = distance / support_radius
    if distance <= 1.0e-6 or q > 1.0:
        return np.zeros(3, dtype=np.float64)
    coefficient = 48.0 / (math.pi * support_radius**3)
    grad_q = displacement / (distance * support_radius)
    if q <= 0.5:
        return coefficient * q * (3.0 * q - 2.0) * grad_q
    return -coefficient * (1.0 - q) ** 2 * grad_q


@wp.kernel
def sample_cubic_kernel(
    displacement: wp.vec3,
    support_radius: float,
    value: wp.array(dtype=float),
    gradient: wp.array(dtype=wp.vec3),
):
    value[0] = functions.cubic_kernel(wp.length(displacement), support_radius)
    gradient[0] = functions.cubic_kernel_gradient(displacement, support_radius)


class MMPBFTest(unittest.TestCase):
    device = "cpu"

    @staticmethod
    def small_config(**overrides):
        values = dict(
            particle_radius=0.1,
            fluid_width=4,
            fluid_height=4,
            fluid_depth=3,
            container_height=1.2,
        )
        values.update(overrides)
        return MMPBFConfig(**values)

    def test_default_parameters_and_scene_match_fluid_demo(self):
        config = MMPBFConfig()
        simulation = Example(config, device=self.device)

        self.assertEqual(config.particle_radius, 0.025)
        self.assertEqual(config.initial_time_step, 0.0025)
        self.assertEqual(config.pressure_iterations, 5)
        self.assertEqual(config.velocity_update_method, 0)
        self.assertEqual(config.xsph_viscosity, 0.02)
        self.assertEqual(config.cfl_factor, 1.0)
        self.assertEqual(simulation.support_radius, 0.1)
        self.assertEqual(simulation.container_size, (4.0, 4.0, 0.8))
        self.assertEqual(simulation.n, 15 * 20 * 15)
        self.assertEqual(simulation.boundary_n, 18630)
        self.assertAlmostEqual(simulation.fluid_volume, 0.0001)
        self.assertAlmostEqual(simulation.particle_mass, 0.1)

        positions = simulation.pos.numpy()
        np.testing.assert_allclose(positions[0], (-1.95, 0.05, -0.35), atol=1.0e-7)
        np.testing.assert_allclose(positions[-1], (-1.25, 1.0, 0.35), atol=1.0e-7)

        boundaries = simulation.boundary_positions.numpy()
        np.testing.assert_allclose(boundaries.min(axis=0), (-2.0, 0.0, -0.4), atol=1.0e-7)
        np.testing.assert_allclose(boundaries.max(axis=0), (2.0, 4.0, 0.4), atol=1.0e-7)
        # The official addWall calls intentionally duplicate edge and corner samples.
        self.assertGreater(len(boundaries), len(np.unique(boundaries, axis=0)))

    def test_example_has_three_methods_and_initialization_has_no_reverse_import(self):
        methods = {
            name
            for name, value in Example.__dict__.items()
            if inspect.isfunction(value)
        }
        self.assertEqual(methods, {"__init__", "step", "render"})

        initialization_source = inspect.getsource(initialization)
        self.assertNotIn("import MMPBF", initialization_source)
        self.assertNotIn("from .MMPBF", initialization_source)

    def test_default_render_method_uses_attached_billboard_renderer(self):
        class Renderer:
            def __init__(self):
                self.begin_time = None
                self.billboards = None
                self.ended = False

            def begin_frame(self, sim_time):
                self.begin_time = sim_time

            def render_billboards(self, **kwargs):
                self.billboards = kwargs

            def end_frame(self):
                self.ended = True

        simulation = Example(self.small_config(), device=self.device)
        renderer = Renderer()
        simulation.renderer = renderer
        simulation.render()

        self.assertEqual(renderer.begin_time, simulation.sim_time)
        self.assertIs(renderer.billboards["points"], simulation.pos)
        self.assertIs(renderer.billboards["velocities"], simulation.v)
        self.assertEqual(renderer.billboards["radius"], simulation.particle_radius)
        self.assertEqual(renderer.billboards["speed_color_mid"], 2.0)
        self.assertEqual(renderer.billboards["speed_color_max"], 6.0)
        self.assertTrue(renderer.ended)

    def test_high_resolution_preset_preserves_scene_dimensions_and_counts(self):
        config = MMPBFConfig.high_resolution()
        simulation = Example(config, device=self.device)

        self.assertAlmostEqual(config.particle_radius, 2.0 / 185.0)
        self.assertEqual(
            (config.fluid_width, config.fluid_height, config.fluid_depth),
            (36, 46, 36),
        )
        self.assertEqual(simulation.n, 59616)
        self.assertEqual(simulation.boundary_n, 97464)
        self.assertEqual(simulation.n + simulation.boundary_n, 157080)
        np.testing.assert_allclose(simulation.container_size, (4.0, 4.0, 0.8))
        self.assertAlmostEqual(simulation.support_radius, 8.0 / 185.0)

    def test_high_resolution_substep_is_finite_inside_and_within_capacity(self):
        simulation = Example(MMPBFConfig.high_resolution(), device=self.device)
        simulation.step()

        positions = simulation.pos.numpy()
        velocities = simulation.v.numpy()
        fluid_neighbor_counts = simulation.fluid_neighbor_counts.numpy()
        boundary_neighbor_counts = simulation.boundary_neighbor_counts.numpy()
        boundary_min = np.asarray(simulation.boundary_min, dtype=np.float32)
        boundary_max = np.asarray(simulation.boundary_max, dtype=np.float32)

        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(np.isfinite(velocities)))
        self.assertLessEqual(
            int(fluid_neighbor_counts.max()),
            simulation.config.max_fluid_neighbors,
        )
        self.assertLessEqual(
            int(boundary_neighbor_counts.max()),
            simulation.config.max_boundary_neighbors,
        )
        self.assertTrue(np.all(positions >= boundary_min))
        self.assertTrue(np.all(positions <= boundary_max))
        self.assertEqual(simulation.last_boundary_violation, 0.0)

    def test_official_scene_is_visibly_framed_by_mmpbf_camera(self):
        self.assertEqual(MMPBF_SCENE_SCALING, 1.0)
        camera = np.asarray(MMPBF_CAMERA_POS, dtype=np.float64)
        front = np.asarray(MMPBF_CAMERA_FRONT, dtype=np.float64)
        front /= np.linalg.norm(front)
        initial_fluid_center = np.array((-1.6, 0.525, 0.0), dtype=np.float64)
        direction = initial_fluid_center - camera
        distance = np.linalg.norm(direction)
        direction /= distance
        angle_degrees = math.degrees(math.acos(float(np.dot(front, direction))))
        self.assertLess(angle_degrees, 15.0)
        # At 720 px and a 45-degree vertical FOV, the official diameter is
        # several pixels wide rather than the sub-pixel old 0.05-scaled size.
        projected_diameter_pixels = (
            2.0 * 0.025 / distance * 720.0 / math.radians(45.0)
        )
        self.assertGreater(projected_diameter_pixels, 4.0)

    def test_runtime_gravity_controls_are_reversible(self):
        simulation = SimpleNamespace(config=MMPBFConfig())
        initial_gravity = np.asarray(simulation.config.gravity, dtype=np.float64)

        reverse_gravity(simulation)
        np.testing.assert_allclose(simulation.config.gravity, -initial_gravity)
        reverse_gravity(simulation)
        np.testing.assert_allclose(simulation.config.gravity, initial_gravity)

        rotate_gravity(simulation, MMPBF_GRAVITY_ROTATION_STEP)
        rotated_gravity = np.asarray(simulation.config.gravity, dtype=np.float64)
        self.assertAlmostEqual(np.linalg.norm(rotated_gravity), np.linalg.norm(initial_gravity))
        self.assertEqual(rotated_gravity[2], initial_gravity[2])
        rotate_gravity(simulation, -MMPBF_GRAVITY_ROTATION_STEP)
        np.testing.assert_allclose(simulation.config.gravity, initial_gravity, atol=1.0e-12)

    def test_keyboard_controls_dispatch_and_leave_unknown_keys_unhandled(self):
        key = SimpleNamespace(SPACE=1, R=2, G=3, Q=4, E=5)
        handled = object()
        fake_pyglet = SimpleNamespace(
            window=SimpleNamespace(key=key),
            event=SimpleNamespace(EVENT_HANDLED=handled),
        )

        class Renderer:
            paused = False

            def register_key_press_callback(self, callback):
                self.callback = callback

        renderer = Renderer()
        events = []
        with patch.dict(sys.modules, {"pyglet": fake_pyglet}):
            callback = register_keyboard_controls(
                renderer,
                on_reset=lambda: events.append("reset"),
                on_reverse_gravity=lambda: events.append("gravity"),
                on_rotate_gravity=lambda angle: events.append(angle),
                extra_actions={6: lambda: events.append("extra")},
            )

            self.assertIs(callback(key.SPACE, 0), handled)
            self.assertTrue(renderer.paused)
            self.assertIs(callback(key.SPACE, 0), handled)
            self.assertFalse(renderer.paused)

            renderer.paused = True
            self.assertIs(callback(key.R, 0), handled)
            self.assertFalse(renderer.paused)
            self.assertIs(callback(key.G, 0), handled)
            self.assertIs(callback(key.Q, 0), handled)
            self.assertIs(callback(key.E, 0), handled)
            self.assertIs(callback(6, 0), handled)
            self.assertIsNone(callback(99, 0))

        self.assertEqual(
            events,
            ["reset", "gravity", MMPBF_GRAVITY_ROTATION_STEP, -MMPBF_GRAVITY_ROTATION_STEP, "extra"],
        )

    def test_rendered_simulation_attachment_preserves_renderer_state(self):
        renderer = SimpleNamespace(camera_pos=object())
        original_camera = renderer.camera_pos
        simulation = SimpleNamespace()
        config = MMPBFConfig()

        with patch(
            "demos.fluid.MMPBF.render_opengl.create_mmpbf_simulation",
            return_value=simulation,
        ) as create_simulation:
            result = create_rendered_simulation(
                config=config,
                verbose=True,
                device="cpu",
                renderer=renderer,
                speed_color_mid=2.0,
                speed_color_max=6.0,
            )

        self.assertIs(result, simulation)
        self.assertIs(simulation.renderer, renderer)
        self.assertIs(renderer.camera_pos, original_camera)
        self.assertEqual(simulation.speed_color_mid, 2.0)
        self.assertEqual(simulation.speed_color_max, 6.0)
        create_simulation.assert_called_once_with(verbose=True, config=config, device="cpu")

    def test_cubic_kernel_and_gradient_match_official_scalar_formulas(self):
        for displacement in (
            np.array((0.025, 0.0, 0.0), dtype=np.float32),
            np.array((0.075, 0.0, 0.0), dtype=np.float32),
            np.array((0.12, 0.0, 0.0), dtype=np.float32),
        ):
            value = wp.zeros(1, dtype=float, device=self.device)
            gradient = wp.zeros(1, dtype=wp.vec3, device=self.device)
            wp.launch(
                sample_cubic_kernel,
                dim=1,
                inputs=[wp.vec3(*displacement), 0.1],
                outputs=[value, gradient],
                device=self.device,
            )
            expected_value = cubic_value(float(np.linalg.norm(displacement)), 0.1)
            expected_gradient = cubic_gradient(displacement.astype(np.float64), 0.1)
            self.assertAlmostEqual(float(value.numpy()[0]), expected_value, places=3)
            np.testing.assert_allclose(
                gradient.numpy()[0], expected_gradient, rtol=3.0e-5, atol=3.0e-3
            )

    def test_akinci_volume_matches_official_direct_sum_with_duplicates(self):
        simulation = Example(self.small_config(), device=self.device)
        positions = simulation.boundary_positions.numpy().astype(np.float64)
        volumes = simulation.boundary_volumes.numpy()
        for sample_index in (0, len(positions) // 2, len(positions) - 1):
            distances = np.linalg.norm(positions - positions[sample_index], axis=1)
            expected_sum = sum(
                cubic_value(float(distance), simulation.support_radius)
                for distance in distances
                if distance <= simulation.support_radius
            )
            self.assertAlmostEqual(
                float(volumes[sample_index]), 1.0 / expected_sum, places=6
            )

    def test_density_lambda_and_pair_correction_match_official_formulas(self):
        positions_np = np.array(((-0.1, 0.0, 0.0), (0.1, 0.0, 0.0)), dtype=np.float32)
        positions = wp.array(positions_np, dtype=wp.vec3, device=self.device)
        boundary_positions = wp.array(((10.0, 10.0, 10.0),), dtype=wp.vec3, device=self.device)
        boundary_volumes = wp.ones(1, dtype=float, device=self.device)
        fluid_counts = wp.array((1, 1), dtype=int, device=self.device)
        fluid_indices = wp.array((1, 0), dtype=int, device=self.device)
        boundary_counts = wp.zeros(2, dtype=int, device=self.device)
        boundary_indices = wp.zeros(2, dtype=int, device=self.device)
        lambdas = wp.zeros(2, dtype=float, device=self.device)
        densities = wp.zeros(2, dtype=float, device=self.device)
        density_error = wp.zeros(1, dtype=float, device=self.device)
        corrections = wp.zeros(2, dtype=wp.vec3, device=self.device)
        volume = 0.4
        support = 1.0

        wp.launch(
            solver.compute_lambdas,
            dim=2,
            inputs=[
                positions,
                boundary_positions,
                boundary_volumes,
                volume,
                support,
                1,
                1,
                fluid_counts,
                fluid_indices,
                boundary_counts,
                boundary_indices,
            ],
            outputs=[lambdas, densities, density_error],
            device=self.device,
        )
        wp.launch(
            solver.compute_position_corrections,
            dim=2,
            inputs=[
                positions,
                boundary_positions,
                boundary_volumes,
                lambdas,
                volume,
                support,
                1,
                1,
                fluid_counts,
                fluid_indices,
                boundary_counts,
                boundary_indices,
            ],
            outputs=[corrections],
            device=self.device,
        )

        displacement = positions_np[0].astype(np.float64) - positions_np[1]
        expected_density = volume * (
            cubic_value(0.0, support) + cubic_value(float(np.linalg.norm(displacement)), support)
        )
        grad_j = -volume * cubic_gradient(displacement, support)
        expected_lambda = -(expected_density - 1.0) / (
            2.0 * float(np.dot(grad_j, grad_j)) + 1.0e-6
        )
        expected_correction = -2.0 * expected_lambda * grad_j

        np.testing.assert_allclose(densities.numpy(), expected_density, rtol=2.0e-6)
        np.testing.assert_allclose(lambdas.numpy(), expected_lambda, rtol=3.0e-6)
        np.testing.assert_allclose(corrections.numpy()[0], expected_correction, rtol=3.0e-6)
        np.testing.assert_allclose(corrections.numpy()[1], -expected_correction, rtol=3.0e-6)

    def test_xsph_matches_official_velocity_update(self):
        positions_np = np.array(((0.0, 0.0, 0.0), (0.25, 0.0, 0.0)), dtype=np.float32)
        velocities_np = np.array(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), dtype=np.float32)
        positions = wp.array(positions_np, dtype=wp.vec3, device=self.device)
        velocities = wp.array(velocities_np, dtype=wp.vec3, device=self.device)
        normalized_densities = wp.array((1.0, 1.0), dtype=float, device=self.device)
        counts = wp.array((1, 1), dtype=int, device=self.device)
        indices = wp.array((1, 0), dtype=int, device=self.device)
        output = wp.zeros(2, dtype=wp.vec3, device=self.device)
        volume = 0.4
        viscosity = 0.02

        wp.launch(
            solver.compute_xsph_viscosity,
            dim=2,
            inputs=[
                positions,
                velocities,
                normalized_densities,
                volume,
                viscosity,
                1.0,
                1,
                counts,
                indices,
            ],
            outputs=[output],
            device=self.device,
        )
        weight = cubic_value(0.25, 1.0)
        expected = velocities_np.copy()
        expected[0] -= viscosity * volume * (velocities_np[0] - velocities_np[1]) * weight
        expected[1] -= viscosity * volume * (velocities_np[1] - velocities_np[0]) * weight
        np.testing.assert_allclose(output.numpy(), expected, rtol=2.0e-6, atol=2.0e-6)

    def test_velocity_update_enum_matches_official(self):
        projected = wp.array(((3.0, 0.0, 0.0),), dtype=wp.vec3, device=self.device)
        old = wp.array(((1.0, 0.0, 0.0),), dtype=wp.vec3, device=self.device)
        last = wp.array(((0.0, 0.0, 0.0),), dtype=wp.vec3, device=self.device)
        positions = wp.zeros(1, dtype=wp.vec3, device=self.device)
        velocities = wp.zeros(1, dtype=wp.vec3, device=self.device)

        wp.launch(
            solver.reconstruct_velocities,
            dim=1,
            inputs=[projected, old, last, 0.5, 0],
            outputs=[positions, velocities],
            device=self.device,
        )
        np.testing.assert_allclose(velocities.numpy()[0], (4.0, 0.0, 0.0))
        wp.launch(
            solver.reconstruct_velocities,
            dim=1,
            inputs=[projected, old, last, 0.5, 1],
            outputs=[positions, velocities],
            device=self.device,
        )
        np.testing.assert_allclose(velocities.numpy()[0], (5.0, 0.0, 0.0))

    def test_cfl_order_and_fixed_iteration_count_match_official(self):
        simulation = Example(self.small_config(), device=self.device)
        simulation.step()
        self.assertEqual(simulation.last_dt, simulation.config.initial_time_step)
        self.assertEqual(simulation.current_dt, simulation.config.cfl_max_time_step)
        self.assertEqual(simulation.sim_time, simulation.config.initial_time_step)
        self.assertEqual(simulation.last_pressure_iterations, 5)

    def test_zero_gravity_official_scene_remains_exactly_at_rest(self):
        simulation = Example(
            MMPBFConfig(gravity=(0.0, 0.0, 0.0), xsph_viscosity=0.0),
            device=self.device,
        )
        initial_positions = simulation.pos.numpy().copy()
        simulation.step()
        np.testing.assert_allclose(simulation.pos.numpy(), initial_positions, atol=1.0e-7)
        np.testing.assert_allclose(simulation.v.numpy(), 0.0, atol=1.0e-7)
        self.assertEqual(simulation.last_density_error_percent, 0.0)

    def test_multistep_dam_break_is_finite_and_inside_container(self):
        simulation = Example(self.small_config(), device=self.device)
        elapsed = 0.0
        for _ in range(100):
            used_dt = simulation.current_dt
            simulation.step()
            elapsed += used_dt
        self.assertAlmostEqual(simulation.sim_time, elapsed, places=10)
        self.assertTrue(np.all(np.isfinite(simulation.pos.numpy())))
        self.assertTrue(np.all(np.isfinite(simulation.v.numpy())))
        self.assertEqual(simulation.last_pressure_iterations, 5)
        self.assertEqual(simulation.last_boundary_violation, 0.0)

    def test_trajectory_matches_compiled_official_cpp_reference(self):
        """Compare through wall impact, not only free fall or local formulas.

        The reference values were produced by compiling the selected official
        ``PositionBasedFluids.cpp`` and ``SPHKernels.cpp`` commit in double
        precision, with the same reduced FluidDemo scene and XSPH disabled.
        Warp uses float32 and hash-order accumulation, hence the small tolerance.
        """
        simulation = Example(
            self.small_config(xsph_viscosity=0.0),
            device=self.device,
        )
        references = {
            25: (-95.75178626103879, 21.262132813238065, 9.430104046242516, -31.54955979569716, 19.003401627975617, 1.2017249999998647),
            50: (-92.63792682586559, 17.79612564259741, 34.312107942695526, -26.995748598449453, 45.48105981436031, 2.9220391531215566),
            100: (-79.61344978228091, 12.527487784022897, 65.32331736644477, -15.550272067810173, 88.23789218021194, 4.637825127571071),
            200: (-40.22349440127277, 7.637408191441459, 87.86812096253408, -5.430758754250471, 129.8799131277995, 4.853673510320003),
        }
        for step in range(1, 201):
            simulation.step()
            if step not in references:
                continue
            positions = simulation.pos.numpy()
            velocities = simulation.v.numpy()
            speed = np.linalg.norm(velocities, axis=1)
            actual = np.array(
                (
                    np.sum(positions[:, 0], dtype=np.float64),
                    np.sum(positions[:, 1], dtype=np.float64),
                    np.sum(velocities[:, 0], dtype=np.float64),
                    np.sum(velocities[:, 1], dtype=np.float64),
                    0.5 * np.sum(velocities * velocities, dtype=np.float64),
                    np.max(speed),
                )
            )
            np.testing.assert_allclose(
                actual,
                references[step],
                rtol=2.0e-4,
                atol=3.0e-3,
            )

    def test_neighbor_capacity_overflow_is_not_silent(self):
        simulation = Example(
            self.small_config(max_boundary_neighbors=1),
            device=self.device,
        )
        with self.assertRaisesRegex(RuntimeError, "boundary neighbor capacity exceeded"):
            simulation.step()

    def test_wall_clock_scheduler_is_frame_rate_independent(self):
        class FixedStepSimulation:
            def __init__(self):
                self.current_dt = 0.005
                self.last_dt = 0.0
                self.sim_time = 0.0

            def step(self):
                self.last_dt = self.current_dt
                self.sim_time += self.last_dt

        for frame_rate in (60, 120):
            simulation = FixedStepSimulation()
            scheduler = PlaybackScheduler(
                playback_speed=0.5,
                max_substeps_per_frame=8,
            )
            for _ in range(frame_rate):
                scheduler.advance(simulation, 1.0 / frame_rate)
            self.assertLessEqual(abs(simulation.sim_time - 0.5), simulation.current_dt)

    def test_wall_clock_scheduler_clamps_long_frames_and_drops_backlog(self):
        class FixedStepSimulation:
            current_dt = 0.005
            last_dt = 0.0

            def __init__(self):
                self.steps = 0

            def step(self):
                self.last_dt = self.current_dt
                self.steps += 1

        simulation = FixedStepSimulation()
        scheduler = PlaybackScheduler(
            playback_speed=0.5,
            max_substeps_per_frame=8,
            max_wall_delta=0.1,
        )
        executed = scheduler.advance(simulation, 1.0)

        self.assertEqual(executed, 8)
        self.assertEqual(simulation.steps, 8)
        self.assertGreaterEqual(scheduler.accumulator, 0.0)
        self.assertLess(scheduler.accumulator, simulation.current_dt)

    def test_wall_clock_scheduler_consumes_the_dt_actually_used(self):
        class AdaptiveStepSimulation:
            def __init__(self):
                self.current_dt = 0.0025
                self.last_dt = 0.0
                self.sim_time = 0.0

            def step(self):
                self.last_dt = self.current_dt
                self.sim_time += self.last_dt
                self.current_dt = 0.005

        simulation = AdaptiveStepSimulation()
        scheduler = PlaybackScheduler(playback_speed=0.5)
        executed = scheduler.advance(simulation, 0.02)

        self.assertEqual(executed, 2)
        self.assertAlmostEqual(simulation.sim_time, 0.0075)
        self.assertAlmostEqual(scheduler.accumulator, 0.0025)

    def test_opengl_cli_defaults_and_overrides(self):
        defaults = parse_args([])
        self.assertEqual(defaults.preset, "high-resolution")
        self.assertEqual(defaults.playback_speed, 0.5)
        self.assertEqual(defaults.max_substeps_per_frame, 8)

        overrides = parse_args(
            [
                "--preset",
                "official",
                "--playback-speed",
                "0.25",
                "--max-substeps-per-frame",
                "3",
            ]
        )
        self.assertEqual(overrides.preset, "official")
        self.assertEqual(overrides.playback_speed, 0.25)
        self.assertEqual(overrides.max_substeps_per_frame, 3)


if __name__ == "__main__":
    unittest.main()
