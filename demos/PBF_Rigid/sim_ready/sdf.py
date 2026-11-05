"""Use the shared baker with forward-only compilation for this application."""

import warp as wp
from demos.sim_Ready import sdf as _shared

# Set options before the shared kernels are first loaded, without global changes.
wp.set_module_options({"enable_backward": False}, module=_shared)

from demos.sim_Ready.sdf import bake, bake_rigid, check_options, save_volume
