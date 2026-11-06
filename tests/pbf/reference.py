"""Independent original-demo resources for numerical regression, never production."""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import warp as wp

from demos.fluid.PBF import PBF as old
from demos.fluid.PBF import pbf_helper as old_fn
from demos.fluid.PBF.simulation import reset_simulation as reset_reference
from PBF.simulation import Simulation


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "demos/fluid/PBF/Assets"
WORK_ARRAYS = ("pre_Pos", "delta_Pos", "lambda_Opt", "volume_Rho", "volume_Grad", "delta_Vel", "curl")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def independent_pair():
    """One original and one new instance, each with its own module constants."""
    before = digest(SOURCE / "stairs_scene.nvdb")
    with tempfile.TemporaryDirectory(prefix="pbf-reference-") as directory, wp.ScopedDevice("cuda:0"):
        private = Path(directory)
        for name in ("stairs_scene.obj", "stairs_scene.nvdb"):
            shutil.copy2(SOURCE / name, private / name)
        with patch.object(old.boundary_model, "ASSET_DIR", private):
            reference = old.PBF(stage_path=None)
        current = Simulation()
        reference.substep = 1
        assert asdict(reference.config) == asdict(current.config)
        assert reference.static_sdf.id != current.solver.static_sdf.id
        assert reference.volume_map.id != current.solver.volume_map.id
        try:
            yield reference, current
        finally:
            assert digest(SOURCE / "stairs_scene.nvdb") == before


def compare_step(reference, current):
    np.testing.assert_array_equal(current.pos.numpy(), reference.pos.numpy(), err_msg="position")
    np.testing.assert_array_equal(current.v.numpy(), reference.v.numpy(), err_msg="velocity")
    for name in WORK_ARRAYS:
        np.testing.assert_array_equal(getattr(current.solver, name).numpy(),
                                      getattr(reference, name).numpy(), err_msg=name)
    np.testing.assert_array_equal(current.solver.n_Cache.NeighborCount.numpy(),
                                  reference.n_Cache.NeighborCount.numpy())


def verify_pair(reference, current, steps=30):
    reset_reference(reference)
    current.reset()
    assert current.n == reference.n == 195525
    np.testing.assert_array_equal(current.block_counts, (55, 79, 45))
    assert current.sim_time == reference.sim_time == 0.
    np.testing.assert_array_equal(current.pos.numpy(), reference.pos.numpy())
    np.testing.assert_array_equal(current.v.numpy(), reference.v.numpy())
    pointers = [(s.particle_q.ptr, s.particle_qd.ptr) for s in current.states]
    for _ in range(steps):
        state_in = current.state
        positions, velocities = state_in.particle_q.numpy(), state_in.particle_qd.numpy()
        reference.step()
        current.solver.step(current.state, current.state_out, None, None, reference.dt)
        current._state_index = 1 - current._state_index
        compare_step(reference, current)
        np.testing.assert_array_equal(state_in.particle_q.numpy(), positions)
        np.testing.assert_array_equal(state_in.particle_qd.numpy(), velocities)
        assert [(s.particle_q.ptr, s.particle_qd.ptr) for s in current.states] == pointers
    return dict(particles=current.n, lattice=[55, 79, 45], substeps=steps,
                bitwise_equal=True, independent_assets=True, input_readonly=True, stable_state_bindings=True)
