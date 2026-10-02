# NanoVDB OpenGL demo

`example_nvdb.py` is an unchanged copy of Warp 1.15's official example. It
uses the `rocks.nvdb` asset bundled with Warp. `render_opengl.py` displays
the same simulation with Warp's built-in OpenGL renderer, without USD output.
The rock mesh is extracted once from the SDF using Warp Marching Cubes.

Run from the repository root in an environment containing Warp and pyglet:

```powershell
python -m demos.nvbd_Demo.render_opengl
```

The simulation starts paused. Press Space to pause/resume simulation steps,
and S to toggle normal real-time playback (1x) and slow motion (0.1x).
The title bar shows the current state and speed. Camera controls remain active
while paused: drag with the left mouse button to look around, use W/A/D or
the arrow keys to move, and scroll to change the field of view. Close the window
or press Escape to exit. `--num-frames 180` limits the run; the default is
unlimited. `--headless --num-frames 3` runs a rendering smoke check without
showing a window. A CUDA GPU is required.
