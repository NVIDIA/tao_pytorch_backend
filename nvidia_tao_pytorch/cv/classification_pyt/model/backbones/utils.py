# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Python utility functions."""
from typing import Union
import torch


def cast_tensor_to_int(x: Union[torch.Tensor, int]):
    """Cast a scalar tensor to int."""
    if isinstance(x, torch.Tensor):
        return int(x.detach().cpu().numpy())
    return int(x)


def cast_tensor_to_float(x: Union[torch.Tensor, float]):
    """Cast a scalar tensor to float."""
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().numpy())
    return float(x)
