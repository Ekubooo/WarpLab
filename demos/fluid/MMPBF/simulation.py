"""Simulation adapter for the Warp MMPBF example."""

try:
    from .MMPBF import Example as MMPBFExample, MMPBFConfig
except ImportError:
    from MMPBF import Example as MMPBFExample, MMPBFConfig


def create_mmpbf_simulation(
    verbose: bool = False,
    config: MMPBFConfig | None = None,
    device=None,
) -> MMPBFExample:
    """Create the Warp PBF simulation used by the OpenGL frontend."""
    return MMPBFExample(config=config, verbose=verbose, device=device)
