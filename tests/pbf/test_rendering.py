"""Renderer ownership must not advance physics or retain a stale State."""

import unittest
from unittest.mock import Mock

from PBF.simulation import Simulation


class RenderingTests(unittest.TestCase):
    def test_attach_replace_render_and_close(self):
        sim = Simulation.__new__(Simulation)
        sim.renderer = None
        sim.states = (object(), object())
        sim._state_index = 0
        sim.sim_time = 0.
        first, second = Mock(), Mock()
        observed = []
        first.render.side_effect = lambda s: observed.append(s.state)
        sim.set_renderer(first)
        first.initialize.assert_called_once_with(sim)
        sim.set_renderer(first)
        first.initialize.assert_called_once_with(sim)
        sim.render()
        sim._state_index = 1
        sim.render()
        self.assertEqual(observed, list(sim.states))
        sim.register_keyboard_callback('G', lambda: None, repeat=False)
        self.assertFalse(first.register_keyboard_callback.call_args.kwargs['repeat'])
        events = []
        first.close.side_effect = lambda: events.append('close')
        second.initialize.side_effect = lambda s: events.append('initialize')
        sim.set_renderer(second)
        self.assertEqual(events, ['close', 'initialize'])
        sim.close()
        sim.close()
        second.close.assert_called_once()
        self.assertIsNone(sim.renderer)
        sim.render()
        self.assertEqual(sim.sim_time, 0.)

    def test_failed_initialization_releases_renderer(self):
        sim = Simulation.__new__(Simulation)
        sim.renderer = None
        renderer = Mock()
        renderer.initialize.side_effect = RuntimeError('display unavailable')
        with self.assertRaisesRegex(RuntimeError, 'display unavailable'):
            sim.set_renderer(renderer)
        renderer.close.assert_called_once()
        self.assertIsNone(sim.renderer)


if __name__ == '__main__':
    unittest.main()
