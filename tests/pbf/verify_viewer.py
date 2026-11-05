"""Run the actual billboard frontend, scripted controls and USD output."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import warp as wp
from pyglet.window import key

from PBF import render_opengl as app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Use a hidden real OpenGL context")
    parser.add_argument("--output", type=Path, default=Path("outputs/pbf-newton/viewer"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    simulation, initial = None, None
    create = app.create_pbf_simulation

    def create_simulation(*a, **kw):
        nonlocal simulation, initial
        simulation = create(*a, **kw)
        initial = simulation.pos.numpy(), simulation.v.numpy()
        return simulation

    class VerifyRenderer(app.PBFRenderer):
        def __init__(self, *a, **kw):
            super().__init__(*a, headless=args.headless, **kw)
            self.frame = 0

        def end_frame(self):
            super().end_frame()
            self.frame += 1
            frame = self.frame
            assert self.gl.glGetError() == self.gl.GL_NO_ERROR
            if frame == 1:
                assert simulation.n == 195525 and simulation.sim_time == 0.
                np.testing.assert_array_equal(simulation.pos.numpy(), initial[0])
                np.testing.assert_array_equal(simulation.v.numpy(), initial[1])
            if frame in (1, 60, 121):
                pixels = wp.empty((self.screen_height, self.screen_width, 3), dtype=wp.uint8)
                assert self.get_pixels(pixels, split_up_tiles=False, use_uint8=True)
                Image.fromarray(pixels.numpy()).save(args.output / f"frame-{frame}.png")
                print(f"Rendered frame {frame}, time={simulation.sim_time}", flush=True)
            keys = {60: key.G, 75: key.Q, 90: key.E, 105: key.X, 120: key.R, 124: key.P}
            if frame in keys:
                self._key_press_callback(keys[frame], 0)
            if frame in (60, 75, 90):
                expected = {60: (0., 9.81, 0.), 75: (-9.81, 0., 0.), 90: (9.81, 0., 0.)}[frame]
                np.testing.assert_allclose(tuple(simulation.gravity), expected, atol=1e-6)
                np.testing.assert_allclose(simulation.model.gravity.numpy()[-1], expected, atol=1e-6)
            if frame == 105:
                assert simulation.frame_dt == simulation.config.frame_dt / 2
            if frame == 121:
                assert self.paused and simulation.sim_time == 0.
                np.testing.assert_array_equal(simulation.pos.numpy(), initial[0])
                np.testing.assert_array_equal(simulation.v.numpy(), initial[1])
            if frame == 122:
                self.paused = False

    argv = [sys.argv[0], "--num-frames", "150", "--cuda-graph"]
    with patch.object(app, "PBFRenderer", VerifyRenderer), \
            patch.object(app, "create_pbf_simulation", create_simulation), patch.object(sys, "argv", argv):
        app.main()
    from warp.render import UsdRenderer
    simulation.reset()
    simulation.renderer = UsdRenderer(str(args.output / "static.usd"))
    times = []
    for _ in range(10):
        times.append(round(simulation.sim_time * simulation.renderer.fps))
        simulation.render()
        simulation.step()
    simulation.renderer.save()
    from pxr import Usd
    stage = Usd.Stage.Open(str(args.output / "static.usd"))
    particles = [p for p in stage.Traverse() if p.GetTypeName() == "PointInstancer"]
    assert len(particles) == 1
    samples = particles[0].GetAttribute("positions").GetTimeSamples()
    assert samples == sorted(set(times))
    assert len(particles[0].GetAttribute("positions").Get(samples[0])) == simulation.n
    result = dict(particles=simulation.n, frames=150, first_frame_unadvanced=True,
                  paused_reset=True, controls="G/Q/E/X/R/P", gl_errors=0,
                  usd_render_calls=10, usd_time_samples=samples)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: billboard, first frame, controls, paused reset and USD", flush=True)


if __name__ == "__main__":
    main()
