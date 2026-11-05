"""Fresh-process PBF startup timings, including the first hidden OpenGL frame.

The runner owns isolated Warp caches. A cold trial uses a new empty cache;
warm trials reuse cold-0, without deleting the user's cache or the CUDA driver
cache. Assets must already exist: this benchmark never prepares them itself.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def audit_generated_kernels(cache_root):
    """Inspect generated project CUDA sources without counting third-party AD."""
    sources = []
    for path in cache_root.rglob('*.cu'):
        if path.name.startswith('wp_demos_PBF_Rigid_tests_'):
            continue
        if not path.name.startswith(('wp_create_kernels__locals__', 'wp_demos_PBF_Rigid', 'wp_demos_sim_Ready_sdf')):
            continue
        code = path.read_text(encoding='utf-8')
        sources.append(dict(path=str(path), backward_entries=code.count('_cuda_kernel_backward(')))
    return dict(source_count=len(sources), backward_entries=sum(s['backward_entries'] for s in sources),
                sources=sources)


def measure_worker(args):
    started = time.perf_counter()
    sys.path.insert(0, str(args.code_root.resolve()))
    from unittest.mock import patch
    import warp as wp
    from warp._src.context import Module
    from demos.PBF_Rigid import render_opengl as app
    from demos.PBF_Rigid.solver.solver_pbf import SolverPBF
    import newton
    if args.scene != 'static':
        from newton.solvers import SolverMuJoCo

    wp.init()
    wp.get_device(args.device)
    wp.synchronize_device(args.device)
    phases = dict(import_and_device_seconds=time.perf_counter() - started)
    setup_started = time.perf_counter()
    modules = []
    original_load = Module.load

    def load(module, *positional, **kwargs):
        begin = time.perf_counter()
        result = original_load(module, *positional, **kwargs)
        elapsed = time.perf_counter() - begin
        if elapsed >= .001:
            modules.append(dict(name=module.name, seconds=elapsed))
        return result

    def timed(function, name):
        def call(*positional, **kwargs):
            begin = time.perf_counter()
            result = function(*positional, **kwargs)
            wp.synchronize_device(args.device)
            phases[name] = phases.get(name, 0.) + time.perf_counter() - begin
            return result
        return call

    render_started = None
    first_frame_started = None
    simulation = None
    create = app.create_pbf_simulation
    renderer_class = app.PBFRenderer

    def create_simulation(*positional, **kwargs):
        nonlocal simulation
        begin = time.perf_counter()
        simulation = create(*positional, **kwargs)
        wp.synchronize_device(args.device)
        phases['simulation_init_seconds'] = time.perf_counter() - begin
        return simulation

    class FirstFrameRenderer(renderer_class):
        def __init__(self, *positional, **kwargs):
            nonlocal render_started
            render_started = time.perf_counter()
            super().__init__(*positional, headless=True, **kwargs)

        def begin_frame(self, t=None):
            nonlocal first_frame_started
            first_frame_started = time.perf_counter()
            return super().begin_frame(t)

        def end_frame(self):
            super().end_frame()
            self.gl.glFinish()
            wp.synchronize_device(args.device)
            now = time.perf_counter()
            phases['first_draw_seconds'] = now - first_frame_started
            phases['renderer_setup_and_first_draw_seconds'] = now - render_started
            phases['first_frame_total_seconds'] = now - started
            if simulation.sim_time != 0.:
                raise AssertionError('Startup advanced physical time before first display')
            if self.gl.glGetError() != self.gl.GL_NO_ERROR:
                raise AssertionError('OpenGL error during first frame')
            self.paused = True

    patches = [patch.object(Module, 'load', load),
        patch.object(app, 'create_pbf_simulation', create_simulation),
        patch.object(app, 'PBFRenderer', FirstFrameRenderer),
        patch.object(app.Simulation, 'capture_graph', timed(app.Simulation.capture_graph, 'graph_seconds')),
        patch.object(newton.ModelBuilder, 'finalize', timed(newton.ModelBuilder.finalize, 'model_finalize_seconds')),
        patch.object(SolverPBF, '__init__', timed(SolverPBF.__init__, 'fluid_solver_seconds'))]
    scene_modules = []
    if args.scene != 'static':
        patches.append(patch.object(SolverMuJoCo, '__init__', timed(SolverMuJoCo.__init__, 'rigid_solver_seconds')))
    if args.scene == 'tank-box':
        from demos.PBF_Rigid import scenes
        scene_modules.append((scenes, ('load_tank_box_boundaries',)))
    elif args.scene == 'stairs-rigids':
        from demos.PBF_Rigid import stairs_scene
        scene_modules.append((stairs_scene, ('prepare_stairs_assets', 'load_stairs_assets', 'load_stairs_boundaries')))
    for module, names in scene_modules:
        for name in names:
            if hasattr(module, name):
                patches.append(patch.object(module, name, timed(getattr(module, name), 'asset_seconds')))
    if args.asset_root is not None:
        import importlib
        import importlib.util
        for name in ('demos.PBF_Rigid.simulation', 'demos.PBF_Rigid.asset_io',
                     'demos.PBF_Rigid.scenes', 'demos.PBF_Rigid.stairs_scene'):
            if importlib.util.find_spec(name) is not None:
                module = importlib.import_module(name)
                if hasattr(module, 'ASSET_DIR'):
                    patches.append(patch.object(module, 'ASSET_DIR', args.asset_root.resolve()))
    # Static boundary loading bypasses scene helpers. Measure it separately to
    # avoid double-counting the same load inside coupled scene helper timings.
    if args.scene == 'static':
        import importlib.util
        if importlib.util.find_spec('demos.PBF_Rigid.asset_io') is not None:
            from demos.PBF_Rigid import asset_io
            for name in ('load_asset', 'load_asset_volumes'):
                patches.append(patch.object(asset_io, name, timed(getattr(asset_io, name), 'asset_seconds')))
        else:
            from demos.PBF_Rigid.solver import boundary_model
            patches.append(patch.object(boundary_model, 'load_boundaries',
                timed(boundary_model.load_boundaries, 'asset_seconds')))
    argv = [sys.argv[0], '--scene', args.scene, '--device', args.device, '--num-frames', '1', '--cuda-graph']
    from contextlib import ExitStack
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        stack.enter_context(patch.object(sys, 'argv', argv))
        phases['import_and_device_seconds'] += time.perf_counter() - setup_started
        app.main()
    phases['other_simulation_assembly_seconds'] = phases['simulation_init_seconds'] - sum(
        phases.get(name, 0.) for name in ('asset_seconds', 'fluid_solver_seconds', 'rigid_solver_seconds'))
    # Loading the visible static OBJ precedes renderer construction. Include
    # that work (and the small CLI/frontend setup) in the complete render phase.
    phases['renderer_setup_and_first_draw_seconds'] = phases['first_frame_total_seconds'] - sum(
        phases[name] for name in ('import_and_device_seconds', 'simulation_init_seconds', 'graph_seconds'))
    result = dict(scene=args.scene, code_root=str(args.code_root.resolve()), device=args.device,
        warp_version=wp.__version__, cache=os.environ.get('WARP_CACHE_PATH'),
        asset_root=str(args.asset_root.resolve()) if args.asset_root else None,
        particles=simulation.n, bodies=simulation.model.body_count, sim_time=simulation.sim_time,
        phases=phases, module_loads=modules)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('STARTUP', args.scene, json.dumps(phases), flush=True)


def run_trials(args):
    directory = args.output.parent.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_root.resolve()
    if cache_root.exists():
        raise ValueError(f'Use a new cache-root to preserve cold-trial isolation: {cache_root}')
    script = Path(__file__).resolve()
    trials = []
    for scene in args.scenes:
        for mode, count in (('cold', args.cold), ('warm', args.warm)):
            for index in range(count):
                cache = cache_root / scene / (f'cold-{index}' if mode == 'cold' else 'cold-0')
                cache.mkdir(parents=True, exist_ok=True)
                output = directory / f'{args.output.stem}-{scene}-{mode}-{index}.json'
                log = output.with_suffix('.log')
                env = dict(os.environ, WARP_CACHE_PATH=str(cache), PYTHONDONTWRITEBYTECODE='1')
                command = [sys.executable, '-B', '-u', str(script), '--worker', '--scene', scene,
                    '--code-root', str(args.code_root.resolve()), '--device', args.device, '--output', str(output)]
                if args.asset_root:
                    command += ['--asset-root', str(args.asset_root.resolve())]
                begin = time.perf_counter()
                with log.open('w', encoding='utf-8') as stream:
                    subprocess.run(command, cwd=args.code_root, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
                result = json.loads(output.read_text(encoding='utf-8'))
                log_text = log.read_text(encoding='utf-8')
                result.update(mode=mode, trial=index, process_seconds=time.perf_counter() - begin,
                              log=str(log), volume_map_rebuilds=log_text.count('Boundary cache rebuild:'),
                              volume_map_hits=log_text.count('Boundary cache hit:'))
                trials.append(result)
                print(scene, mode, index, f"{result['phases']['first_frame_total_seconds']:.3f} s", flush=True)
                # Persist progress so a failed trial never discards completed measurements.
                args.output.write_text(json.dumps(dict(trials=trials), indent=2) + '\n', encoding='utf-8')
    summary = {}
    for scene in args.scenes:
        summary[scene] = {}
        for mode in ('cold', 'warm'):
            selected = [t for t in trials if t['scene'] == scene and t['mode'] == mode]
            names = sorted(set().union(*(t['phases'] for t in selected)))
            summary[scene][mode] = {name: dict(median=statistics.median(t['phases'].get(name, 0.) for t in selected),
                minimum=min(t['phases'].get(name, 0.) for t in selected),
                maximum=max(t['phases'].get(name, 0.) for t in selected)) for name in names}
    generated = audit_generated_kernels(cache_root)
    if not generated['source_count']:
        raise RuntimeError('No project CUDA sources found; cannot audit backward generation')
    args.output.write_text(json.dumps(dict(code_root=str(args.code_root.resolve()),
        timing='Fresh processes; hidden real OpenGL first frame; three-substep graph capture; no physics advancement',
        phase_notes='model_finalize is a subset of other_simulation_assembly; first_draw is a subset of renderer_setup_and_first_draw; module_loads overlap their enclosing phases',
        cache_policy='Fresh isolated Warp caches for cold trials; warm trials reuse cold-0; CUDA driver cache NOT cleared',
        assets='Pre-existing assets; offline construction excluded; volume-map misses must be reported from logs',
        summary=summary, generated_kernels=generated,
        trials=trials), indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, default=Path.cwd())
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--asset-root', type=Path,
                        help='Use a private published-asset copy, e.g. for volume-map hit/miss trials')
    parser.add_argument('--scenes', nargs='+', choices=('static', 'tank-box', 'stairs-rigids'),
                        default=['static', 'tank-box', 'stairs-rigids'])
    parser.add_argument('--cold', type=int, default=3)
    parser.add_argument('--warm', type=int, default=5)
    parser.add_argument('--cache-root', type=Path, default=Path('.cache/startup-benchmark'))
    parser.add_argument('--output', type=Path, default=Path('outputs/offline-assets/startup.json'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--scene', choices=('static', 'tank-box', 'stairs-rigids'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold < 1 or args.warm < 1:
        parser.error('cold and warm counts must be positive')
    if args.worker:
        measure_worker(args)
    else:
        run_trials(args)


if __name__ == '__main__':
    main()
