"""Isolated check of a native accessor; does not change Warp or the solver."""
from pathlib import Path

import numpy as np
import warp as wp

wp.config.kernel_cache_dir = str(Path(__file__).resolve().parents[1] / ".warp_cache")
wp.set_module_options({"enable_backward": False})


@wp.func_native("return query.cell_index - 1;")
def sorted_column(query: wp.HashGridQuery) -> int:
    ...


@wp.kernel
def inspect_query(grid: wp.uint64, result: wp.array2d(dtype=int)):
    query = wp.hash_grid_query(grid, wp.vec3(0.0, 0.0, 0.0), 10.0)
    pid = int(0)
    k = int(0)
    while wp.hash_grid_query_next(query, pid):
        column = sorted_column(query)
        result[k, 0] = column
        result[k, 1] = pid
        result[k, 2] = wp.hash_grid_point_id(grid, column)
        k += 1


if __name__ == "__main__":
    wp.init()
    for device in ("cpu", "cuda:0"):
        positions = wp.array([(2.1, 0.0, 0.0), (0.1, 0.0, 0.0), (1.1, 0.0, 0.0)],
                             dtype=wp.vec3, device=device)
        grid = wp.HashGrid(8, 8, 8, device=device)
        grid.build(positions, 1.0)
        result = wp.full((3, 3), -1, dtype=int, device=device)
        wp.launch(inspect_query, 1, [grid.id, result], device=device)
        actual = result.numpy()
        np.testing.assert_array_equal(actual[:, 1], actual[:, 2])
        np.testing.assert_array_equal(np.sort(actual[:, 0]), np.arange(3))
        print(device, "[sorted column, input index, verification]:", actual.tolist())
