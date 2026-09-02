# Simulation demos

Demos are grouped first by simulation family and then by algorithm:

```text
demos/
├── rigid_body/<algorithm>/
├── soft_body/<algorithm>/
├── cloth/<algorithm>/
├── fluid/<algorithm>/
└── granular/<algorithm>/
```

Each algorithm directory should keep its simulation source separate from its
executable rendering frontend. The current SPH demo can be run with:

```bash
python demos/fluid/sph/render_opengl.py
```

Generated recordings and simulation outputs belong in an ignored `output/` or
`outputs/` directory, not beside the source files.
