"""Numerical equivalence against the untouched PBF demo on identical resources."""

import contextlib
from dataclasses import asdict, replace
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from demos.fluid.PBF import PBF as old
from demos.fluid.PBF import pbf_helper as old_helper
from tests.fluid.pbf.solver_fixture import bind_solver
from Lab_Framework.pbf.config import PBFConfig
from Lab_Framework.pbf.simulation import Simulation
from .fixture import make_boundaries


class LegacyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA required")

    def test_initialization_and_each_substep_match_legacy(self):
        config = replace(old_helper.PBFConfig(), container_size=(1., 1., 1.),
                         block_start=(.1, .025, .1), block_end=(.35, .275, .35))
        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cuda:0"), \
                contextlib.redirect_stdout(io.StringIO()):
            boundaries = make_boundaries(directory)
            with patch.object(old.boundary_model, "load_default_boundaries", return_value=boundaries):
                reference = old.PBF(stage_path=None, config=config)
            current = Simulation(config=PBFConfig(**asdict(config)), boundaries=boundaries)
            np.testing.assert_array_equal(current.pos.numpy(), reference.pos.numpy())
            np.testing.assert_array_equal(current.v.numpy(), reference.v.numpy())
            # Advance exactly one old substep and compare every persistent and
            # pressure work array in the same spatial order, without tolerances.
            reference.substep = 1
            bind_solver(config, reference.grid, reference.static_sdf, reference.volume_map)
            for _ in range(5):
                reference.step()
                current._substep(reference.dt)
                for name in ("pos", "v", "pre_Pos", "delta_Pos", "lambda_Opt", "volume_Rho",
                             "volume_Grad", "delta_Vel", "curl"):
                    actual = getattr(current.solver, name).numpy()
                    expected = getattr(reference, name).numpy()
                    np.testing.assert_array_equal(actual, expected, err_msg=name)
                np.testing.assert_array_equal(current.solver.n_Cache.NeighborCount.numpy(),
                                              reference.n_Cache.NeighborCount.numpy())


if __name__ == "__main__":
    unittest.main()
