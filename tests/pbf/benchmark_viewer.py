"""Measure complete real-OpenGL frames, with the original or separated frontend."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import warp as wp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=('static', 'stairs-rigids'), default='stairs-rigids')
    parser.add_argument('--warmup', type=int, default=180)
    parser.add_argument('--frames', type=int, default=360)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Saved pre-refactor render_opengl.py')
    args = parser.parse_args()
    if args.reference is None:
        from PBF import app
        from PBF.renderers import opengl
    else:
        spec = importlib.util.spec_from_file_location('PBF.viewer_reference', args.reference)
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        opengl = app

    samples = []
    class TimedRenderer(opengl.PBFRenderer):
        def __init__(self, *a, **kw):
            super().__init__(*a, headless=True, **kw)
            self.measured_frame = 0
            self.previous_start = None

        def begin_frame(self, t=None):
            wp.synchronize()
            now = time.perf_counter()
            if self.previous_start is not None and self.measured_frame > args.warmup:
                samples.append(1000. * (now - self.previous_start))
            self.previous_start = now
            self.measured_frame += 1
            super().begin_frame(t)

        def end_frame(self):
            super().end_frame()
            if self.measured_frame == 1:
                from PIL import Image
                pixels = wp.empty((self.screen_height, self.screen_width, 3), dtype=wp.uint8)
                assert self.get_pixels(pixels, split_up_tiles=False, use_uint8=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels.numpy()).save(args.output.with_suffix('.png'))

    argv = ['PBF', '--scene', args.scene, '--cuda-graph', '--num-frames',
            str(args.warmup + args.frames + 1)]
    with patch.object(opengl, 'PBFRenderer', TimedRenderer), patch.object(sys, 'argv', argv):
        app.main()
    values = np.asarray(samples)
    report = dict(scene=args.scene, reference=str(args.reference), frames=len(samples), warmup=args.warmup,
                  median_ms=float(np.median(values)), mean_ms=float(np.mean(values)),
                  p10_ms=float(np.percentile(values, 10)), p90_ms=float(np.percentile(values, 90)),
                  batches_mean_ms=[float(np.mean(v)) for v in np.array_split(values, 3)],
                  samples_ms=samples,
                  method='Real hidden OpenGL, existing vsync/settings, Graph, synchronized consecutive frame starts; includes drawing, events and physics; startup excluded')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print({k:v for k,v in report.items() if k != 'samples_ms'})


if __name__ == '__main__':
    main()
