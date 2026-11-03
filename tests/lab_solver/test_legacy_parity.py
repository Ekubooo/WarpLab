"""Numerical equivalence with the archived PBF, including independent assets.

Set PBF_FULL_SCENE_TESTS=1 to include the 195,525-particle, 30-substep check.
The small fixture remains suitable for the ordinary unit-test loop.
"""

import contextlib
from dataclasses import asdict, replace
import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from demos.fluid.PBF import PBF as old
from demos.fluid.PBF import pbf_helper as old_helper
from tests.fluid.pbf.solver_fixture import bind_solver
from Lab_Framework.pbf.config import PBFConfig
from Lab_Framework.pbf.simulation import Simulation, SDF_FILENAME
from .fixture import make_boundaries
from .sync_reference_assets import SOURCE, digest


WORK_ARRAYS = ("pos", "v", "pre_Pos", "delta_Pos", "lambda_Opt", "volume_Rho",
               "volume_Grad", "delta_Vel", "curl")


@wp.kernel
def sample_initial_sdf(sdf: wp.uint64, positions: wp.array(dtype=wp.vec3),
                       distances: wp.array(dtype=float)):
    i = wp.tid()
    index = wp.volume_world_to_index(sdf, positions[i])
    distances[i] = wp.volume_sample_f(sdf, index, wp.Volume.LINEAR)


@contextlib.contextmanager
def independent_default_pair(device="cuda:0"):
    """Load each implementation's own SDF/map; never rewrite archived inputs."""
    before = digest(SOURCE / SDF_FILENAME)
    with tempfile.TemporaryDirectory(prefix="pbf-reference-") as directory, wp.ScopedDevice(device):
        copied = Path(directory)
        for name in (SDF_FILENAME, "stairs_scene.obj"):
            shutil.copy2(SOURCE / name, copied / name)
        config = old_helper.PBFConfig()
        np.testing.assert_equal(asdict(config), asdict(PBFConfig()))
        # The old loader uses a module-level directory. Scope the override to
        # the copy so its normal support-radius cache refresh cannot touch demos.
        with patch.object(old.boundary_model, "ASSET_DIR", copied):
            resources = old.boundary_model.load_boundaries(
                old.boundary_model.BoundaryScene(static_assets=(SDF_FILENAME,), default_rigid_count=0),
                old.boundary_model.BoundaryLoadOptions(4 * config.particle_radius, device=device))
        with patch.object(old.boundary_model, "load_default_boundaries", return_value=resources):
            reference = old.PBF(stage_path=None, config=config)
        current = Simulation()
        if (reference.static_sdf.id == current.fluid_solver.static_sdf.id
                or reference.volume_map.id == current.fluid_solver.volume_map.id):
            raise AssertionError("Reference and migrated solver must own independent boundary volumes")
        reference.substep = 1
        bind_solver(config, reference.grid, reference.static_sdf, reference.volume_map)
        try:
            yield reference, current
        finally:
            if digest(SOURCE / SDF_FILENAME) != before:
                raise AssertionError("Archived NVDB was modified by verification")


def verify_default_pair(reference, current, substeps=30):
    """Check the unfiltered lattice, pressure arrays and State ownership."""
    if reference.n != 195525 or current.n != 195525:
        raise AssertionError("Default scene must retain the full 55 x 79 x 45 lattice")
    if current.sim_time != 0.0 or reference.sim_time != 0.0:
        raise AssertionError("Initialization must not advance physics")
    np.testing.assert_array_equal(reference.block_counts, (55, 79, 45))
    np.testing.assert_array_equal(current.pos.numpy(), reference.pos.numpy())
    np.testing.assert_array_equal(current.v.numpy(), reference.v.numpy())
    distances = wp.empty(current.n, dtype=float, device=current.model.device)
    wp.launch(sample_initial_sdf, current.n,
              inputs=[reference.static_sdf.id, reference.pos], outputs=[distances], device=current.model.device)
    old_distances = distances.numpy()
    wp.launch(sample_initial_sdf, current.n,
              inputs=[current.fluid_solver.static_sdf.id, current.pos], outputs=[distances],
              device=current.model.device)
    np.testing.assert_array_equal(distances.numpy(), old_distances)
    overlap_count = int(np.count_nonzero(old_distances < 0.))
    if overlap_count == 0:
        raise AssertionError("The reference's initially overlapping particles must be retained")
    pointers = [(state.particle_q.ptr, state.particle_qd.ptr) for state in current.states]
    for step in range(substeps):
        source = current.state
        source_q, source_qd = source.particle_q.numpy(), source.particle_qd.numpy()
        reference.step()
        current._substep(reference.dt)
        for name in WORK_ARRAYS:
            np.testing.assert_array_equal(getattr(current.fluid_solver, name).numpy(),
                                          getattr(reference, name).numpy(), err_msg=f"step {step + 1}: {name}")
        np.testing.assert_array_equal(current.fluid_solver.n_Cache.NeighborCount.numpy(),
                                      reference.n_Cache.NeighborCount.numpy())
        np.testing.assert_array_equal(source.particle_q.numpy(), source_q)
        np.testing.assert_array_equal(source.particle_qd.numpy(), source_qd)
        if [(state.particle_q.ptr, state.particle_qd.ptr) for state in current.states] != pointers:
            raise AssertionError("Public State array bindings changed")
    return dict(particles=current.n, lattice=[55, 79, 45], substeps=substeps,
                initially_overlapping_particles=overlap_count, independent_boundary_volumes=True,
                motion_input_readonly=True, state_array_bindings_stable=True,
                numerical_parity="bitwise equal: initialization, nine work arrays and neighbor counts")


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

    @unittest.skipUnless(os.environ.get("PBF_FULL_SCENE_TESTS") == "1",
                         "Set PBF_FULL_SCENE_TESTS=1 for independent full-scene regression")
    def test_full_default_scene_with_independent_assets(self):
        with contextlib.redirect_stdout(io.StringIO()), independent_default_pair() as pair:
            result = verify_default_pair(*pair)
        self.assertGreater(result["initially_overlapping_particles"], 0)


if __name__ == "__main__":
    unittest.main()
