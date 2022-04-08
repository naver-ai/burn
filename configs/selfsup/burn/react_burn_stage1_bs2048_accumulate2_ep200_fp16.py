"""
BURN
Copyright (c) 2022-present NAVER Corp.
CC BY-NC 4.0
"""

import copy
_base_ = '../../base.py'
# model settings
model = dict(
    type='BURN',
    pretrained=None,
    base_lamda = 0.1,
    end_lamda = 0.3,
    pre_conv=True,
    pretrained_teacher = "fp_teacher/moco/moco_v2_800ep.pth",
    multi=False,
    backbone=dict(
        type='ReActnet_Binact',
        out_indices=[13],
        use_mlp=True
    ),
    diff_branch=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckSimCLR',
        in_channels=1024,
        hid_channels=2000,
        out_channels=2000,
        num_layers=2,
        sync_bn=True,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=False),
    diff_neck=dict(
        type='NonLinearNeckSimCLR',
        in_channels=2048,
        hid_channels=2000,
        out_channels=2000,
        num_layers=2,
        sync_bn=True,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(type='KLDivMSEMovingLambdaHead',
              size_average=True, T=0.2))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/Imagenet/meta/train.txt'
data_train_root = 'data/Imagenet/train'
dataset_type = 'BYOLDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.6),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=0.2),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')], p=0.),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
train_pipeline1 = copy.deepcopy(train_pipeline)
train_pipeline2 = copy.deepcopy(train_pipeline)
train_pipeline2[4]['p'] = 0.2 # gaussian blur
train_pipeline2[5]['p'] = 0.0 # solarization

data = dict(
    imgs_per_gpu=256,  # total 256*8(gpu)*2(interval)=4096
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline1=train_pipeline1,
        pipeline2=train_pipeline2,
        prefetch=prefetch,
    ))
# additional hooks
update_interval = 2  # interval for accumulate gradient
custom_hooks = [
    dict(type='BURNHook', update_interval=update_interval)
]
# optimizer
optimizer = dict(type='Adam', lr=3e-4, weight_decay=0.0)
# apex
use_fp16 = True
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
