"""
BURN
Copyright (c) 2022-present NAVER Corp.
CC BY-NC 4.0
"""

import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output', type=str, help='destination file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(backbone =dict(state_dict=dict()), classifier=dict(state_dict=dict()), teacher_classifier=dict(state_dict=dict()),author="OpenSelfSup")
    has_backbone = False
    for key, value in ck['state_dict'].items():
        if key.startswith("module"):
            key = key.replace("module.","")
        if key.startswith('backbone'):
            output_dict['backbone']['state_dict'][key[9:]] = value
            has_backbone = True
        if key.startswith('online_net.1.'):
            key = key.replace("online_net.1.","")
            output_dict['classifier']['state_dict'][key] = value
        if key.startswith('diff_branch_net.1.'):
            key = key.replace("diff_branch_net.1.","")
            output_dict['teacher_classifier']['state_dict'][key] = value
    if not has_backbone:
        raise Exception("Cannot find a backbone module in the checkpoint.")
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()
