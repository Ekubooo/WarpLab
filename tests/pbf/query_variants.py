"""Isolated test modules; production has no culling-disable or loop-mode switch."""

import importlib.util
from pathlib import Path
import sys

import warp as wp

from PBF import PBF as kernels, boundary_model


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pbf-newton/optimization-round1"
NAMES = ("volume_query", "calc_deltaPos", "update_prePos")


def load_module(name, source):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (name + ".py")
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def boundary_variant(mode):
    source = Path(boundary_model.__file__).read_text()
    start = source.index("@wp.func\ndef sample_boundary_sdf")
    end = source.index("# =============================================================================\n# 4.", start)
    return load_module("pbf_bounds_" + mode,
        "from PBF.boundary_model import *\n"
        f"BOUNDARY_CULLING = {mode!r}\n"
        "wp.set_module_options({'enable_backward': False})\n" + source[start:end])


def make_variants():
    """Call after Simulation initializes the fixed PBF constants."""
    from . import rigid_query_reference as reference
    for name in ("SDF_id", "VMAP_id", "Rigid_Count"):
        setattr(reference, name, getattr(kernels, name))
    source = Path(reference.__file__).read_text()
    source = source.replace("SDF_id = wp.constant(wp.uint64(0))\nVMAP_id = wp.constant(wp.uint64(0))\nRigid_Count = wp.constant(0)",
                            "from PBF.PBF import SDF_id, VMAP_id, Rigid_Count")
    lines = source.splitlines()
    for first in reversed([i for i, line in enumerate(lines) if line == "    for rigid in range(Rigid_Count):"]):
        last = first + 1
        while last < len(lines) and (not lines[last].strip() or lines[last].startswith("        ")):
            last += 1
        lines[first] = "    for rigid in range(boundaries.instances.rigid_count):"
        lines[first:last] = ["    if wp.static(Rigid_Count > 0):"] + ["    " + line for line in lines[first:last]]
    runtime = load_module("pbf_runtime_reference", "\n".join(lines))
    for name in ("SDF_id", "VMAP_id", "Rigid_Count"):
        setattr(runtime, name, getattr(kernels, name))
    variants = dict(original=reference, runtime=runtime)
    source = Path(kernels.__file__).read_text()
    chunks = []
    for name in NAMES:
        start = source.index("@wp.kernel\ndef " + name + "(")
        end = source.index("@wp.kernel\n", start + 1)
        chunks.append(source[start:end])
    for mode in ("sphere", "box"):
        boundary = boundary_variant(mode)
        header = ("import warp as wp\nfrom PBF import pbf_helper as fn\n"
                  "from PBF.PBF import SDF_id, VMAP_id, Rigid_Count\n"
                  f"import {boundary.__name__} as boundary\n"
                  "wp.set_module_options({'enable_backward': False})\n")
        variants[mode] = load_module("pbf_query_" + mode, header + "\n".join(chunks))
    for module in variants.values():
        wp.load_module(module=module, device="cuda:0")
    return variants


def install(module):
    for name in NAMES:
        setattr(kernels, name, getattr(module, name))
