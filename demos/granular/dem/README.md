# DEM demo

This demo contains two source files:

- `simulation.py` creates the official `warp.examples.core.example_dem.Example`
  with USD output disabled.
- `render_opengl.py` is the executable real-time OpenGL upright-cube viewer.

Run from the repository root:

```bash
.venv/bin/python demos/granular/dem/render_opengl.py --device cuda:0
```

The official simulation keeps its original 65,536 particles, HashGrid contact
search, 64 substeps per frame, and CUDA Graph execution. The cubes are only a
visual representation; collision detection and contact response still use the
official demo's spherical particles. Each cube has edge length
`2 * simulation.point_radius` and remains aligned to the world axes.
