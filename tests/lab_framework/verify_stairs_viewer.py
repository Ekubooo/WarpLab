"""Exercise the real frontend with a hidden OpenGL context and scripted keys.

Run with python -m tests.lab_framework.verify_stairs_viewer. Requires an NVIDIA
OpenGL context; writes screenshots and a four-body USD animation under outputs.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from PIL import Image
from pyglet.window import key

from Lab_Framework.pbf import render_opengl as app


def main():
    root = Path("outputs/stairs-rigids")
    root.mkdir(parents=True, exist_ok=True)
    simulation, initial = None, None
    create = app.create_pbf_simulation

    def create_simulation(*args, **kwargs):
        nonlocal simulation, initial
        simulation = create(*args, **kwargs)
        initial = (simulation.pos.numpy(), simulation.state.body_q.numpy())
        return simulation

    class SmokeRenderer(app.PBFRenderer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, headless=True, **kwargs)
            self.smoke_frame = 0

        def end_frame(self):
            super().end_frame()
            self.smoke_frame += 1
            frame = self.smoke_frame
            assert self.gl.glGetError() == self.gl.GL_NO_ERROR
            if frame == 1:
                # Two box instances must share intact flat-shaded GPU vertices.
                shape = self._instances["rigid_1"][2]
                assert self._instances["rigid_2"][2] == shape
                expected = self._shapes[shape][0]
                resource = self._shape_gl_buffers[shape][-1]
                vertices = resource.map(dtype=wp.float32, shape=expected.shape)
                try:
                    np.testing.assert_array_equal(vertices.numpy()[:, :3], expected[:, :3])
                finally:
                    resource.unmap()
                assert simulation.sim_time == 0.
                np.testing.assert_array_equal(simulation.pos.numpy(), initial[0])
            if frame in (1, 90, 151):
                pixels = wp.empty((self.screen_height, self.screen_width, 3), dtype=wp.uint8)
                assert self.get_pixels(pixels, split_up_tiles=False, use_uint8=True)
                Image.fromarray(pixels.numpy()).save(root / f"viewer-frame-{frame}.png")
                print("Rendered", frame, "at", simulation.sim_time, flush=True)
            keys = {90: key.G, 105: key.Q, 120: key.E, 135: key.X, 150: key.R, 154: key.P}
            if frame in keys:
                self._key_press_callback(keys[frame], 0)
            if frame in (90, 105, 120):
                expected = {90: (0., 9.81, 0.), 105: (-9.81, 0., 0.), 120: (9.81, 0., 0.)}[frame]
                np.testing.assert_allclose(tuple(simulation.gravity), expected, atol=1e-6)
                np.testing.assert_allclose(simulation.solver.rigid.mjw_model.opt.gravity.numpy(), [expected], atol=1e-6)
            if frame == 135:
                assert simulation.frame_dt == simulation.config.frame_dt / 2
            if frame == 151:
                assert self.paused and simulation.sim_time == 0.
                np.testing.assert_array_equal(simulation.pos.numpy(), initial[0])
                np.testing.assert_array_equal(simulation.state.body_q.numpy(), initial[1])
            if frame == 152:
                self.paused = False

    arguments = [sys.argv[0], "--scene", "stairs-rigids", "--num-frames", "180", "--cuda-graph"]
    with patch.object(app, "PBFRenderer", SmokeRenderer), \
            patch.object(app, "create_pbf_simulation", create_simulation), patch.object(sys, "argv", arguments):
        app.main()
    app.run_headless(simulation, SimpleNamespace(stage_path=str(root / "stairs.usd"),
                                                num_frames=30, cuda_graph=True))
    from pxr import Usd
    stage = Usd.Stage.Open(str(root / "stairs.usd"))
    for i in range(1, 5):
        prim = stage.GetPrimAtPath(f"/root/rigid_{i}")
        assert prim and prim.GetTypeName() == "Mesh"
        assert len(prim.GetAttribute("xformOp:translate").GetTimeSamples()) > 1
    print("PASS: shared mesh, GL motion, G/Q/E, slow motion, paused reset, grid toggle, USD four bodies", flush=True)


if __name__ == "__main__":
    main()
