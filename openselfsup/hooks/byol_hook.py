from math import cos, pi
from mmcv.runner import Hook
from mmcv.parallel import is_module_wrapper
import torch
from .registry import HOOKS


@HOOKS.register_module
class BYOLHook(Hook):
    """Hook for BYOL.

    This hook includes momentum adjustment in BYOL following:
        m = 1 - ( 1- m_0) * (cos(pi * k / K) + 1) / 2,
        k: current step, K: total steps.

    Args:
        end_momentum (float): The final momentum coefficient
            for the target network. Default: 1.
    """

    def __init__(self, end_momentum=1., update_interval=1,**kwargs):
        self.end_momentum = end_momentum
        self.update_interval = update_interval

    def before_train_iter(self, runner):
        assert hasattr(runner.model.module, 'momentum'), \
            "The runner must have attribute \"momentum\" in BYOLHook."
        assert hasattr(runner.model.module, 'base_momentum'), \
            "The runner must have attribute \"base_momentum\" in BYOLHook."
        assert hasattr(runner.model.module, 'lamda'), \
            "The runner must have attribute \"lamda\" in BYOLHook."
        assert hasattr(runner.model.module, 'base_lamda'), \
            "The runner must have attribute \"base_lamda\" in BYOLHook."
        assert hasattr(runner.model.module, 'end_lamda'), \
            "The runner must have attribute \"end_lamda\" in BYOLHook."
        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            max_iter = runner.max_iters
            base_m = runner.model.module.base_momentum
            m = self.end_momentum - (self.end_momentum - base_m) * (
                cos(pi * cur_iter / float(max_iter)) + 1) / 2
            runner.model.module.momentum = m
            
            base_lamda = runner.model.module.base_lamda
            if base_lamda is not None:
                end_lamda = runner.model.module.end_lamda
                if end_lamda < 1.0:
                    lamda = end_lamda - (end_lamda - base_lamda) * (
                        cos(pi * cur_iter / float(max_iter)) + 1) / 2
                else:
                    if cur_iter < int(max_iter/2):
                        lamda = 0.0
                    else:
                        lamda = 1.0
                runner.model.module.lamda = lamda

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                runner.model.module.momentum_update()
            else:
                runner.model.momentum_update()
