"""
BURN
Copyright (c) 2022-present NAVER Corp.
CC BY-NC 4.0
"""

from math import cos, pi
from mmcv.runner import Hook
from mmcv.parallel import is_module_wrapper
import torch
from .registry import HOOKS


@HOOKS.register_module
class BURNHook(Hook):

    def __init__(self, update_interval=1,**kwargs):
        self.update_interval = update_interval

    def before_train_iter(self, runner):

        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            max_iter = runner.max_iters
            
            base_lamda = runner.model.module.base_lamda
            if base_lamda is not None:
                end_lamda = runner.model.module.end_lamda
                lamda = end_lamda - (end_lamda - base_lamda) * (
                        cos(pi * cur_iter / float(max_iter)) + 1) / 2
                runner.model.module.lamda = lamda

