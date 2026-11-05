"""The production PBF path must not compile unused backward kernels."""

import importlib
import unittest

import warp as wp

from demos.PBF_Rigid.solver import pbf_helper
from demos.PBF_Rigid.solver.kernels import create_kernels


class ForwardOnlyTests(unittest.TestCase):
    def test_unique_kernels_disable_both_module_and_kernel_backward(self):
        parameters = pbf_helper.KernelParameters(pbf_helper.PBFConfig(), .0125, .0125)
        kernels = create_kernels(parameters, 1, 2, 3, rigid_count=4)
        self.assertEqual(len(vars(kernels)), 11)
        for name, kernel in vars(kernels).items():
            with self.subTest(kernel=name):
                self.assertIs(kernel.options['enable_backward'], False)
                self.assertIs(kernel.module.options['enable_backward'], False)

    def test_auxiliary_modules_disable_backward_without_global_override(self):
        from demos.PBF_Rigid.sim_ready import sdf  # configure the shared module before loading
        before = wp.config.enable_backward
        for name in ('demos.PBF_Rigid.solver.boundary_model', 'demos.PBF_Rigid.solver.solver_coupled',
                     'demos.PBF_Rigid.billboard_renderer', 'demos.sim_Ready.sdf'):
            importlib.import_module(name)
            with self.subTest(module=name):
                self.assertIs(wp.get_module_options(module=name)['enable_backward'], False)
        self.assertEqual(wp.config.enable_backward, before)


if __name__ == '__main__':
    unittest.main()
