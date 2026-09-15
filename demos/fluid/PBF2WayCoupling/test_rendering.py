"""Optional hidden-window GL checks; requires an NVIDIA CUDA/OpenGL device.

Run explicitly with: python -m unittest demos.fluid.PBF2WayCoupling.test_rendering -v
"""

from pathlib import Path
import unittest

import numpy as np
import warp as wp

from .simulation import create_pbf2way_simulation
from .render_opengl import (
    CouplingRenderer,
    register_keyboard_controls,
    reverse_gravity,
    rotate_gravity,
)


class RenderingTest(unittest.TestCase):
    def test_shadow_map_billboards_depth_and_controls(self):
        simulation = create_pbf2way_simulation(device="cuda:0")
        renderer = CouplingRenderer(device=simulation.device, hidden=True)
        simulation.renderer = renderer
        output = Path(__file__).resolve().parents[3] / "outputs/pbf2way/render-check"
        output.mkdir(parents=True, exist_ok=True)
        try:
            renderer.update_rigid_scene(simulation)
            renderer.begin_frame(0)
            renderer.end_frame()
            renderer.screenshot(output / "shadow-on.png")
            renderer.shadows = False
            renderer.begin_frame(0)
            renderer.end_frame()
            renderer.screenshot(output / "shadow-off.png")
            import pyglet

            def pixels(path):
                data = pyglet.image.load(str(path)).get_image_data().get_data("RGB")
                return np.frombuffer(data, dtype=np.uint8).astype(int)

            delta = pixels(output / "shadow-off.png") - pixels(output / "shadow-on.png")
            self.assertGreater(np.count_nonzero(delta > 10), 1000)
            renderer.shadows = True
            simulation.render()
            renderer.screenshot(output / "initial-scene.png")
            fluid_delta = np.abs(
                pixels(output / "initial-scene.png") - pixels(output / "shadow-on.png")
            )
            self.assertGreater(np.count_nonzero(fluid_delta > 10), 10000)
            self.assertEqual(renderer.gl.glGetError(), renderer.gl.GL_NO_ERROR)
            events = []
            callback = register_keyboard_controls(
                renderer,
                on_reset=lambda: events.append("reset"),
                on_reverse_gravity=lambda: reverse_gravity(simulation),
                on_rotate_gravity=lambda angle: rotate_gravity(simulation, angle),
            )
            keys = pyglet.window.key
            callback(keys.SPACE, 0)
            self.assertTrue(renderer.paused)
            callback(keys.G, 0)
            self.assertAlmostEqual(simulation.config.gravity[1], 9.81)
            callback(keys.Q, 0)
            callback(keys.E, 0)
            np.testing.assert_allclose(simulation.config.gravity, [0, 9.81, 0], atol=1e-12)
            callback(keys.R, 0)
            self.assertEqual(events, ["reset"])
            self.assertFalse(renderer.paused)  # Releases the base renderer's pause loop.
            print(
                f"GL verified: {np.count_nonzero(delta>10)} shadow channel samples; screenshots: {output}"
            )
        finally:
            renderer.paused = False
            renderer.close()
            renderer.close()  # Window callback + finally must be harmless.


if __name__ == "__main__":
    unittest.main()
