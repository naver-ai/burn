"""
BURN
Copyright (c) 2022-present NAVER Corp.
CC BY-NC 4.0
"""

import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class BURN(nn.Module):
    
    def __init__(self,
                 backbone,
                 diff_branch,
                 pretrained_teacher,
                 neck=None,
                 head=None,
                 diff_neck = None,
                 pretrained=None,
                 base_lamda = None,
                 end_lamda = None,
                 multi=False,
                 **kwargs):
        super(BURN, self).__init__()
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.online_net[0]
        if diff_neck is not None:
            self.diff_branch_net = nn.Sequential(
                builder.build_backbone(diff_branch), builder.build_neck(diff_neck))
        else:
            self.diff_branch_net = nn.Sequential(
                builder.build_backbone(diff_branch))

        for param in self.diff_branch_net.parameters():
            param.requires_grad = False


        self.diff_branch_head = builder.build_head(head)
        self.init_weights(pretrained=pretrained, pretrained_teacher = pretrained_teacher, multi=multi)

        self.base_lamda = base_lamda
        self.lamda = base_lamda
        self.end_lamda = end_lamda


    def init_weights(self, pretrained=None, pretrained_teacher = None, multi=False):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
            if multi:
                checkpoint = torch.load(pretrained, map_location="cpu")
                self.online_net[0].load_state_dict(checkpoint['backbone']['state_dict']) # backbone
                self.online_net[1].load_state_dict(checkpoint['classifier']['state_dict']) # projection
                self.diff_branch_net[1].load_state_dict(checkpoint['teacher_classifier']['state_dict']) # teacher classifier
        if pretrained_teacher is not None:
            print_log('load teacher model from: {}'.format(pretrained_teacher), logger='root')
            checkpoint = torch.load(pretrained_teacher, map_location="cpu")
            self.diff_branch_net[0].load_state_dict(checkpoint['state_dict'])
        if not multi:
            if len(self.diff_branch_net) > 1:
                self.diff_branch_net[1].init_weights(init_linear='kaiming')   
            self.online_net[0].init_weights(pretrained=pretrained) # backbone
            self.online_net[1].init_weights(init_linear='kaiming') # projection

        # init the predictor in the head
        self.diff_branch_head.init_weights()


    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        # print(img_v1)
        # compute query features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        with torch.no_grad():
            proj_diff_branch_v1 = self.diff_branch_net(img_v1)[0].clone().detach()
            proj_diff_branch_v2 = self.diff_branch_net(img_v2)[0].clone().detach()
            loss = self.diff_branch_head(proj_online_v1, proj_diff_branch_v2, lamda=self.lamda)['loss'] + self.diff_branch_head(proj_online_v2, proj_diff_branch_v1,lamda=self.lamda)['loss']

        return dict(loss=loss)

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
