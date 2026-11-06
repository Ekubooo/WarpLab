"""USD recording of the current PBF state; no window or input handling."""

import warp as wp


class PBFUsdRenderer:
    def __init__(self, stage_path):
        self.stage_path = stage_path
        self.renderer = None
        self._meshes = None

    def initialize(self, simulation):
        from warp.render import UsdRenderer
        self.renderer = UsdRenderer(self.stage_path)

    @property
    def fps(self):
        return self.renderer.fps

    def render(self, simulation):
        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(simulation.sim_time)
            self.renderer.render_points(points=simulation.pos.numpy(),
                radius=simulation.particle_radius, name="points", colors=(0.8, 0.3, 0.2))
            if simulation.is_coupled:
                if self._meshes is None:
                    from ..scene import load_render_meshes
                    self._meshes = load_render_meshes(simulation.BoundarySys)
                poses = simulation.BoundarySys.device_data.transforms.numpy()
                for index, (points, faces) in enumerate(self._meshes):
                    pose = poses[index]
                    self.renderer.render_mesh(f"boundary_{index}", points, faces,
                        pos=tuple(pose[:3]), rot=tuple(pose[3:]),
                        colors=(.55, .55, .55) if index == 0 else (.85, .45, .15))
            self.renderer.end_frame()

    def close(self):
        if self.renderer is not None:
            self.renderer.save()
            self.renderer = None
