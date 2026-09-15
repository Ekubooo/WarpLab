"""CPU/CUDA checks for the container fallback, through the actual update kernels."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from . import PBF2WayCoupling as solver


class WallClampTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(Path(__file__).resolve().parents[3] / ".warp_cache")
        wp.init()
        cls.devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        cls.lower, cls.upper = wp.vec3(-1, -2, -3), wp.vec3(1, 2, 3)

    def array(self, data, device):
        return wp.array(np.asarray(data, dtype=np.float32), dtype=wp.vec3, device=device)

    def reflected(self, position, velocity, damping):
        result = velocity.copy()
        outward = ((position <= np.asarray(self.lower)) & (velocity < 0)) | (
            (position >= np.asarray(self.upper)) & (velocity > 0)
        )
        result[outward] *= -damping
        return result

    def test_position_projection_each_iteration(self):
        initial = np.array(
            [
                [-10000, 0.2, 0.3],
                [10000, 0.2, 0.3],
                [0.2, -10000, 0.3],
                [0.2, 10000, 0.3],
                [0.2, 0.3, -10000],
                [0.2, 0.3, 10000],
                [-10, -20, 0.3],
                [10, 20, 30],
                [-10, -20, -30],
                [0.1, 0.2, 0.3],
            ],
            dtype=np.float32,
        )
        delta = np.full_like(initial, 0.01)
        for device in self.devices:
            with self.subTest(device=device):
                x, correction = self.array(initial, device), self.array(delta, device)
                fault = wp.zeros(2, dtype=int, device=device)
                expected = initial.copy()
                for _ in range(5):
                    with patch.object(wp.array, "numpy", side_effect=AssertionError("Readback")):
                        wp.launch(
                            solver.apply_correction,
                            len(x),
                            [x, correction, self.lower, self.upper, fault],
                            device=device,
                        )
                    expected = np.clip(expected + delta, self.lower, self.upper).astype(np.float32)
                    np.testing.assert_array_equal(x.numpy(), expected)

    def test_reflection_viscosity_and_no_double_damping(self):
        positions = np.array(
            [
                [-1, 0.2, 0.3],
                [1, 0.2, 0.3],
                [0.2, -2, 0.3],
                [0.2, 2, 0.3],
                [0.2, 0.3, -3],
                [0.2, 0.3, 3],
                [-1, -2, 0.3],
                [1, 2, 3],
                [-1, 0.2, 0.3],
                [0, 0, 0],
                [-1, 0.2, 0.3],
            ],
            dtype=np.float32,
        )
        velocity = np.array(
            [
                [-2, 3, 4],
                [2, 3, 4],
                [2, -3, 4],
                [2, 3, 4],
                [2, 3, -4],
                [2, 3, 4],
                [-2, -3, 4],
                [2, 3, 4],
                [2, 3, 4],
                [500, 600, -700],
                [0, 3, 4],
            ],
            dtype=np.float32,
        )
        dt = 0.25
        old = positions - dt * velocity
        for device in self.devices:
            for damping in (0.0, 0.8, 1.0):
                with self.subTest(device=device, damping=damping):
                    x, old_x = self.array(positions, device), self.array(old, device)
                    v = wp.zeros(len(x), dtype=wp.vec3, device=device)
                    fault = wp.zeros(2, dtype=int, device=device)
                    wp.launch(
                        solver.reconstruct_velocity,
                        len(x),
                        [x, old_x, dt, v, self.lower, self.upper, damping, fault],
                        device=device,
                    )
                    expected = self.reflected(positions, (positions - old) / dt, damping)
                    np.testing.assert_allclose(v.numpy(), expected, atol=2e-6)
                    acceleration = wp.zeros(len(x), dtype=wp.vec3, device=device)
                    wp.launch(
                        solver.apply_viscosity,
                        len(x),
                        [v, acceleration, dt, x, self.lower, self.upper, damping, fault],
                        device=device,
                    )
                    np.testing.assert_allclose(v.numpy(), expected, atol=2e-6)
                    # Viscosity can turn an already reflected velocity outward again.
                    acceleration_host = velocity * 20
                    acceleration.assign(acceleration_host)
                    wp.launch(
                        solver.apply_viscosity,
                        len(x),
                        [v, acceleration, dt, x, self.lower, self.upper, damping, fault],
                        device=device,
                    )
                    expected = self.reflected(positions, expected + dt * acceleration_host, damping)
                    np.testing.assert_allclose(v.numpy(), expected, atol=2e-6)

    def test_nonfinite_state_is_not_hidden(self):
        invalid = np.array([[np.nan, 0, 0], [np.inf, 0, 0], [-np.inf, 0, 0]], dtype=np.float32)
        for device in self.devices:
            with self.subTest(device=device):
                x, v = self.array(invalid, device), self.array(invalid, device)
                zeros = wp.zeros(3, dtype=wp.vec3, device=device)
                fault = wp.zeros(2, dtype=int, device=device)
                wp.launch(
                    solver.apply_correction,
                    3,
                    [x, zeros, self.lower, self.upper, fault],
                    device=device,
                )
                np.testing.assert_array_equal(x.numpy(), invalid)
                wp.launch(
                    solver.apply_viscosity,
                    3,
                    [v, zeros, 0.25, x, self.lower, self.upper, 0.8, fault],
                    device=device,
                )
                np.testing.assert_array_equal(v.numpy(), invalid)
                maxima, invalid_state = wp.zeros(3, device=device), wp.zeros(
                    1, dtype=int, device=device
                )
                wp.launch(
                    solver.audit_fluid,
                    3,
                    [x, v, self.lower, self.upper, maxima, invalid_state],
                    device=device,
                )
                self.assertEqual(invalid_state.numpy()[0], 1)

    def test_fault_prevents_clamp_writes(self):
        initial = np.array([[-10, 20, 30]], dtype=np.float32)
        for device in self.devices:
            with self.subTest(device=device):
                x, v, old = [self.array(initial, device) for _ in range(3)]
                delta = self.array([[1, 2, 3]], device)
                fault = wp.array([1, 1], dtype=int, device=device)
                wp.launch(
                    solver.apply_correction,
                    1,
                    [x, delta, self.lower, self.upper, fault],
                    device=device,
                )
                wp.launch(
                    solver.reconstruct_velocity,
                    1,
                    [x, old, 0.25, v, self.lower, self.upper, 0.8, fault],
                    device=device,
                )
                wp.launch(
                    solver.apply_viscosity,
                    1,
                    [v, delta, 0.25, x, self.lower, self.upper, 0.8, fault],
                    device=device,
                )
                np.testing.assert_array_equal(x.numpy(), initial)
                np.testing.assert_array_equal(v.numpy(), initial)

    def test_damping_configuration(self):
        self.assertEqual(solver.PBF2WayCouplingConfig().wall_damping, 0.8)
        for value in (-0.1, 1.1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "wall_damping"):
                solver.PBF2WayCouplingConfig(wall_damping=value)


if __name__ == "__main__":
    unittest.main()
