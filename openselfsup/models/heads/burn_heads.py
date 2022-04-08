"""
BURN
Copyright (c) 2022-present NAVER Corp.
CC BY-NC 4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..registry import HEADS
from .. import builder


    
@HEADS.register_module
class KLDivMSEMovingLambdaHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, size_average=True, T=0.2):
        super(KLDivMSEMovingLambdaHead, self).__init__()
        self.size_average = size_average
        self.kldiv = nn.KLDivLoss(reduction='sum')
        self.T = T
        self.mse = nn.MSELoss(reduction='none')
    def init_weights(self, init_linear='normal'):
        pass

    def forward(self, input, target, lamda):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        target_norm = nn.functional.normalize(target, dim=1)
        input_norm = nn.functional.normalize(input, dim=1)
        target_prob = F.softmax(target/self.T, dim=1)
        input_prob = F.log_softmax(input/self.T, dim=1)
        kldiv = self.kldiv(input_prob, target_prob)
        mse = self.mse(input_norm, target_norm).sum()
        loss = (1-lamda)*mse + lamda*kldiv
        if self.size_average:
            loss /= input.size(0)
        return dict(loss=loss)

@HEADS.register_module
class KLDivL2MovingLambdaHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, size_average=True, T=0.2):
        super(KLDivL2MovingLambdaHead, self).__init__()
        self.size_average = size_average
        self.kldiv = nn.KLDivLoss(reduction='sum')
        self.T = T
        self.mse = nn.MSELoss(reduction='none')
    def init_weights(self, init_linear='normal'):
        pass

    def forward(self, input, target,lamda):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        target_prob = F.softmax(target/self.T, dim=1)
        input_prob = F.log_softmax(input/self.T, dim=1)
        kldiv = self.kldiv(input_prob, target_prob)
        mse = self.mse(input, target).sum()
        loss = (1-lamda)*mse + lamda*kldiv
        if self.size_average:
            loss /= input.size(0)
        return dict(loss=loss)

@HEADS.register_module
class KLDivL1MovingLambdaHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, size_average=True, T=0.2):
        super(KLDivL1MovingLambdaHead, self).__init__()
        self.size_average = size_average
        self.kldiv = nn.KLDivLoss(reduction='sum')
        self.T = T
        self.mae = nn.L1Loss(reduction='none')
    def init_weights(self, init_linear='normal'):
        pass

    def forward(self, input, target,lamda):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        target_prob = F.softmax(target/self.T, dim=1)
        input_prob = F.log_softmax(input/self.T, dim=1)
        kldiv = self.kldiv(input_prob, target_prob)
        mae = self.mae(input, target).sum()
        loss = (1-lamda)*mae + lamda*kldiv
        if self.size_average:
            loss /= input.size(0)
        return dict(loss=loss)
