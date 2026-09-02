# Warp launch argument packing

`MMPBF.py` intentionally writes every `wp.launch()` input and output directly
inside `Example.step()`. This makes the PBF data flow visible next to
the algorithm order: CFL, prediction, neighbor caching, five constraint
iterations, velocity reconstruction, XSPH, and diagnostics.

An alternative is to assemble the mostly static launch argument lists once
during initialization. For example, a solver could store:

```python
self.lambda_inputs = [
    self.predicted_positions,
    self.boundary_positions,
    self.boundary_volumes,
    self.fluid_volume,
    self.support_radius,
    self.config.max_fluid_neighbors,
    self.config.max_boundary_neighbors,
    self.fluid_neighbor_counts,
    self.fluid_neighbor_indices,
    self.boundary_neighbor_counts,
    self.boundary_neighbor_indices,
]
self.lambda_outputs = [
    self.lambdas,
    self.normalized_densities,
    self.density_error_sum,
]
```

The step would then use shorter launches:

```python
wp.launch(
    compute_lambdas,
    dim=self.n,
    inputs=self.lambda_inputs,
    outputs=self.lambda_outputs,
    device=self.device,
)
```

This reduces repeated Python list construction and shortens the source, but it
also moves the kernel's data dependencies away from the algorithm schedule.
Reviewing a launch then requires jumping back to initialization, and changing
an array without rebuilding the stored list can leave a stale reference.

Values that change by value every substep, especially `dt`, cannot be treated
as fully static arguments. They must still be supplied at launch time or the
corresponding packed list must be updated before each launch. The same caution
applies to any future state that swaps buffers rather than mutating an existing
Warp array in place.

For this demo, the small Host overhead is not worth obscuring the PBF pipeline.
Therefore launch argument packing is documented only as an optional design and
is not enabled in the current implementation.
