"""Reset the current PBF state without loading scene assets or constructing a solver."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from demos.fluid.PBF import pbf_helper as fn
from demos.fluid.PBF.simulation import reset_simulation, reverse_gravity


class SimulationResetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("PBF hash-grid construction requires CUDA")

    def test_reset_after_buffer_swaps_preserves_resources_and_invalidates_neighbors(self):
        device = wp.get_device("cuda:0")
        config = fn.PBFConfig()
        sim = SimpleNamespace(
            n=8, config=config, particle_spacing=0.1,
            block_counts=wp.vec3i(2, 2, 2), smoothing_length=0.2,
            sim_time=12.0, gravity=wp.vec3(*config.gravity),
            static_sdf=object(), volume_map=object(), renderer=object(),
        )
        with wp.ScopedDevice(device):
            for name in ("pos", "v", "pre_Pos", "delta_Pos", "delta_Vel",
                         "volume_Grad", "temp_pos", "temp_pPos"):
                setattr(sim, name, wp.full(sim.n, wp.vec3(9.0), dtype=wp.vec3))
            sim.lambda_Opt = wp.full(sim.n, 9.0, dtype=float)
            sim.curl = wp.full(sim.n, wp.vec4(9.0), dtype=wp.vec4)
            sim.n_Cache = fn.NeighborCache()
            sim.n_Cache.NeighborCount = wp.full(sim.n, 3, dtype=int)
            sim.n_Cache.CacheBuffer = wp.full((16, sim.n), 7, dtype=int)
            sim.grid = wp.HashGrid(8, 8, 8)

        expected = np.array([
            np.array(config.block_start) + sim.particle_spacing * np.array([x, y, z])
            for x in (1, 2) for y in (1, 2) for z in (1, 2)
        ], dtype=np.float32)
        zero_names = ("v", "pre_Pos", "delta_Pos", "lambda_Opt", "delta_Vel", "curl", "volume_Grad")
        resources = (sim.config, sim.grid, sim.static_sdf, sim.volume_map, sim.renderer)
        cache_pointers = (sim.n_Cache.NeighborCount.ptr, sim.n_Cache.CacheBuffer.ptr)

        # The helper module also contains the reorder kernel, whose grid id must resolve.
        with patch.object(fn, "GRID_id", wp.constant(wp.uint64(sim.grid.id)), create=True):
            for _ in range(2):
                # Reordering swaps these references; reset must use the current buffers.
                sim.pos, sim.temp_pos = sim.temp_pos, sim.pos
                sim.pre_Pos, sim.temp_pPos = sim.temp_pPos, sim.pre_Pos
                for name in ("pos",) + zero_names:
                    getattr(sim, name).fill_(9)
                sim.n_Cache.NeighborCount.fill_(3)
                reverse_gravity(sim)
                sim.sim_time = 12.0
                names = ("pos",) + zero_names + ("temp_pos", "temp_pPos")
                pointers = tuple(getattr(sim, name).ptr for name in names)

                # Reset chooses the simulation device and restores the caller's scope.
                with wp.ScopedDevice("cpu"):
                    reset_simulation(sim)
                    self.assertEqual(wp.get_device(), wp.get_device("cpu"))

                np.testing.assert_allclose(sim.pos.numpy(), expected, rtol=0, atol=1e-7)
                for name in zero_names:
                    np.testing.assert_array_equal(getattr(sim, name).numpy(), 0)
                np.testing.assert_array_equal(sim.n_Cache.NeighborCount.numpy(), 0)
                np.testing.assert_array_equal(sim.n_Cache.CacheBuffer.numpy(), 7)
                self.assertEqual(sim.sim_time, 0)
                self.assertEqual(tuple(sim.gravity), tuple(wp.vec3(*config.gravity)))
                self.assertEqual(tuple(getattr(sim, name).ptr for name in names), pointers)
                self.assertEqual((sim.n_Cache.NeighborCount.ptr, sim.n_Cache.CacheBuffer.ptr), cache_pointers)
                for current, original in zip(
                    (sim.config, sim.grid, sim.static_sdf, sim.volume_map, sim.renderer), resources
                ):
                    self.assertIs(current, original)


if __name__ == "__main__":
    unittest.main()
