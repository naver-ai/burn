"""
BURN
Copyright (c) 2022-present NAVER Corp.
CC BY-NC 4.0
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from .binaryfunction import *

import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from openselfsup.utils import get_root_logger
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer

stage_out_channel = [32] + [64] + [128] * \
    2 + [256] * 2 + [512] * 6 + [1024] * 2


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def binaryconv3x3(in_planes, out_planes, stride=1):
    return BinaryConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def binaryconv1x1(in_planes, out_planes, stride=1):
    return BinaryConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


def binactconv3x3(in_planes, out_planes, stride=1):
    return BinaryActConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def binactconv1x1(in_planes, out_planes, stride=1):
    return BinaryActConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return out


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + \
                (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + \
                                 (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * \
                                 (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(
            1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand(
            (self.number_of_weights, 1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(
            real_weights), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)
        # print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + \
                                                       cliped_weights
        # print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride,
                     padding=self.padding)

        return y


class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BinaryConv2d, self).__init__(in_channels, out_channels,
              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        if self.training:
            input = SignSTE.apply(input)
            self.weight_bin_tensor = SignWeight.apply(self.weight)
        else:
            # We clone the input here because it causes unexpected behaviors
            # to edit the data of `input` tensor.
            input = input.clone()
            input.data = input.sign()
            # Even though there is a UserWarning here, we have to use `new_tensor`
            # rather than the "recommended" way
            self.weight_bin_tensor = self.weight.new_tensor(self.weight.sign())
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(self.weight), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        self.weight_bin_tensor = self.weight_bin_tensor * scaling_factor
            # 1. The input of binary convolution shoule be only +1 or -1,
            #    so instead of padding 0 automatically, we need pad -1 by ourselves
            # 2. `padding` of nn.Conv2d is always a tuple of (padH, padW),
            #    while the parameter of F.pad should be (padLeft, padRight, padTop, padBottom)
        out = F.conv2d(input, self.weight_bin_tensor, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out

class BinaryActConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(BinaryActConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           groups, bias)

    def forward(self, input):
        if self.training:
            input = SignSTE.apply(input)
        else:
            # We clone the input here because it causes unexpected behaviors 
            # to edit the data of `input` tensor.
            input = input.clone()
            input.data = input.sign()
            # Even though there is a UserWarning here, we have to use `new_tensor`
            # rather than the "recommended" way
        # 1. The input of binary convolution shoule be only +1 or -1, 
        #    so instead of padding 0 automatically, we need pad -1 by ourselves
        # 2. `padding` of nn.Conv2d is always a tuple of (padH, padW), 
        #    while the parameter of F.pad should be (padLeft, padRight, padTop, padBottom)

        out = F.conv2d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3 = binaryconv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = binaryconv1x1(inplanes, planes)
            self.bn2 = norm_layer(planes)

        else:
            self.binary_pw_down1 = binaryconv1x1(inplanes, inplanes)
            self.binary_pw_down2 = binaryconv1x1(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)


        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes


        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):

        out1 = self.move11(x)

        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)


        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)


        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2


class BasicBlock_FP(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_FP, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3 = conv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = conv1x1(inplanes, planes)
            self.bn2 = norm_layer(planes)

        else:
            self.binary_pw_down1 = conv1x1(inplanes, inplanes)
            self.binary_pw_down2 = conv1x1(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)


        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes


        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):

        out1 = self.move11(x)

        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)


        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)


        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2

class BasicBlock_Binact(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_Binact, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3 = binactconv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = binactconv1x1(inplanes, planes)
            self.bn2 = norm_layer(planes)

        else:
            self.binary_pw_down1 = binactconv1x1(inplanes, inplanes)
            self.binary_pw_down2 = binactconv1x1(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)


        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes


        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):

        out1 = self.move11(x)

        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)


        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)


        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2

@BACKBONES.register_module
class ReActnet(nn.Module):
    def __init__(self, num_classes=1000, out_indices=(13), frozen_stages=-1,norm_eval=False, use_mlp=False):
        super(ReActnet, self).__init__()
        self.feature = nn.ModuleList()
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.use_mlp = use_mlp
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1))
            # self.pool1 = nn.AdaptiveAvgPool2d(1)


            # codes for swav

        if use_mlp:
            self.pool1 = nn.AdaptiveAvgPool2d(1)
            dim_in = 1024
            self.fc = nn.Sequential(
                 nn.Linear(dim_in, dim_in),
                 nn.ReLU(inplace=True)
            )
        

    def forward(self, x):
        outs = []
        for i, block in enumerate(self.feature):
            x = block(x)
            if i in self.out_indices:
                if self.use_mlp:
                    x = self.pool1(x)
                    x = torch.flatten(x, 1)
                    x = self.fc(x)
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = self.feature[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(ReActnet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
            
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)


@BACKBONES.register_module
class ReActnet_Binact(nn.Module):
    def __init__(self, num_classes=1000, out_indices=(13), frozen_stages=-1,norm_eval=False, use_mlp=False):
        super(ReActnet_Binact, self).__init__()
        self.feature = nn.ModuleList()
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock_Binact(stage_out_channel[i-1], stage_out_channel[i], 2))
            else:
                self.feature.append(BasicBlock_Binact(stage_out_channel[i-1], stage_out_channel[i], 1))
        self.use_mlp = use_mlp
        if use_mlp:
            self.pool1 = nn.AdaptiveAvgPool2d(1)
            dim_in = 1024
            self.fc = nn.Sequential(
                 nn.Linear(dim_in, dim_in),
                 nn.ReLU(inplace=True)
            )
        

    def forward(self, x):
        outs = []
        for i, block in enumerate(self.feature):
            x = block(x)
            if i in self.out_indices:
                if self.use_mlp:
                    x = self.pool1(x)
                    x = torch.flatten(x, 1)
                    x = self.fc(x)
                outs.append(x)
        # x = self.pool1(x)
        # x = torch.flatten(x, 1)

        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = self.feature[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(ReActnet_Binact, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
            
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
