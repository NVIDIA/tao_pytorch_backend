# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model Conversion Tool for timm (GCViT & FAN) to MMCls Weights."""

import argparse
import os
from collections import OrderedDict

import torch


def convert_timm(model_path, output_file):
    """ Convert timm (GCViT & FAN) Model """
    tmp = torch.load(model_path, map_location='cpu')
    if 'state_dict' in tmp:
        model = tmp['state_dict']
    else:
        model = tmp
    state_dict = OrderedDict()

    for k, v in model.items():
        if not k.startswith('head'):
            state_dict['backbone.' + k] = v
        else:
            state_dict['head.fc.' + k[5:]] = v

    torch.save({"state_dict": state_dict}, output_file)


def build_command_line_parser(parser=None):
    """Build command line parser for model_convert."""
    if parser is None:
        parser = argparse.ArgumentParser(
            prog='model_converter',
            description='Convert timm to mmclassification.'
        )
    parser.add_argument(
        '-m',
        '--model_path',
        required=True,
        help='Path to timm pth file.')
    parser.add_argument(
        "-o",
        "--out_file",
        type=str,
        required=True,
        help="Path to the result mmcls pretrained weights."
    )
    return parser


def parse_command_line_args(cl_args=None):
    """Parse sys.argv arguments from commandline.

    Args:
        cl_args: List of command line arguments.

    Returns:
        args: list of parsed arguments.
    """
    parser = build_command_line_parser()
    args = parser.parse_args(cl_args)
    return args


def main(args=None):
    """
    Convert a torchvision model to MMCls weights.

    Args:
        args(list): list of arguments to be parsed if called from another module.
    """
    args = parse_command_line_args(cl_args=args)

    # Defining the results directory.
    results_dir = os.path.abspath(os.path.join(args.out_file, os.pardir))
    if results_dir:

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    try:
        convert_timm(args.model_path, args.out_file)
        print("Successfully Converted !")
    except Exception as e:
        raise e


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise e
