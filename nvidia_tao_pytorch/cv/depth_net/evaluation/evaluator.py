# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

""" Depth Net Evaluator in distributed mode. """

from typing import Tuple, List, Dict

import torch
from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


def align_depth_least_square(
    gt: Tensor,
    pred: Tensor,
):
    """Align depth using least square method.
    Args:
        gt (torch.Tensor): Ground truth disparity/depth tensor
        pred (torch.Tensor): Predicted disparity/depth tensor
    Returns:
        aligned_pred (torch.Tensor): Aligned disparity/depth tensor
    """
    ori_shape = pred.shape  # input shape
    gt = gt.squeeze()  # [H, W]
    pred = pred.squeeze()
    assert (
        gt.shape == pred.shape
    ), f"GT shape: {gt.shape}, Pred shape: {pred.shape} are not matched."

    gt_masked = gt.reshape((-1, 1))
    pred_masked = pred.reshape((-1, 1))

    # numpy solver
    _ones = torch.ones_like(pred_masked)
    A = torch.cat([pred_masked, _ones], dim=-1)
    X = torch.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)
    return aligned_pred


def _delta_log_update(preds: Tensor, target: Tensor, num_outputs: int) -> Tuple[Tensor, Tensor, Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor
        num_outputs (int): Number of outputs in multioutput setting

    Returns:
        d1 (torch.Tensor): Delta 1
        d2 (torch.Tensor): Delta 2
        d3 (torch.Tensor): Delta 3
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if num_outputs == 1:
        preds = preds.view(-1)
        target = target.view(-1)

    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo

    thresh = torch.max((target / preds), (preds / target))

    d1 = torch.sum(thresh < 1.25, dim=0)
    d2 = torch.sum(thresh < 1.25 ** 2, dim=0)
    d3 = torch.sum(thresh < 1.25 ** 3, dim=0)

    return d1, d2, d3, target.shape[0]


def _rmse_update(preds: Tensor, target: Tensor, num_outputs: int) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor
        num_outputs (int): Number of outputs in multioutput setting

    Returns:
        sum_abs_error (torch.Tensor): Sum of root mean squared error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if num_outputs == 1:
        preds = preds.view(-1)
        target = target.view(-1)
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    sum_abs_error = torch.sum(torch.pow((preds - target), 2) / target, dim=0)
    return sum_abs_error, target.shape[0]


def _rmse_log_update(preds: Tensor, target: Tensor, num_outputs: int) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor
        num_outputs (int): Number of outputs in multioutput setting

    Returns:
        sum_log_error (torch.Tensor): Sum of log error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if num_outputs == 1:
        preds = preds.view(-1)
        target = target.view(-1)
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    sum_log_error = torch.sum(torch.pow(torch.log(preds) - torch.log(target), 2), dim=0)
    return sum_log_error, target.shape[0]


def _abs_rel_update(preds: Tensor, target: Tensor, num_outputs: int) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor
        num_outputs (int): Number of outputs in multioutput setting

    Returns:
        sum_abs_error (torch.Tensor): Sum of absolute relative error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if num_outputs == 1:
        preds = preds.view(-1)
        target = target.view(-1)
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo

    sum_abs_error = torch.sum(torch.abs(preds - target) / target, dim=0)
    return sum_abs_error, target.shape[0]


def _sq_rel_update(preds: Tensor, target: Tensor, num_outputs: int) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor
        num_outputs (int): Number of outputs in multioutput setting

    Returns:
        sum_sq_error (torch.Tensor): Sum of squared relative error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if num_outputs == 1:
        preds = preds.view(-1)
        target = target.view(-1)
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    sum_sq_error = torch.sum(torch.pow((preds - target), 2) / target, dim=0)
    return sum_sq_error, target.shape[0]


class DepthMetric(Metric):
    """Depth Evaluation Metric Class."""

    def __init__(self, align_gt: bool = True, num_outputs: int = 1, min_depth: float = 0.001, max_depth: float = 10, **kwargs):
        """Initialize for Depth Metric Class.
        Args:
            align_gt (bool): Whether to align the ground truth disparity/depth tensor.
            num_outputs (int): Number of outputs in multioutput setting.
            min_depth (float): Minimum depth value.
            max_depth (float): Maximum depth value.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(f"Expected num_outputs to be a positive integer but got {num_outputs}")
        self.align_gt = align_gt
        self.num_outputs = num_outputs
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.add_state("sum_abs_rel", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_sq_rel", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_rmse", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_rmse_log", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d1", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d2", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d3", default=torch.zeros(num_outputs), dist_reduce_fx="sum")

        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, post_processed_results: List[Dict]) -> None:
        """Function to accumulate the metric results to reduce in the final step.
        Args:
            post_processed_results (List[Dict]): List of post-processed results.
        """
        pred_list = []
        target_list = []
        for result in post_processed_results:
            pred_i = result['depth_pred']
            gt_i = result['disp_gt']
            valid_mask_i = result['valid_mask']
            if self.align_gt:
                pred_aligned = align_depth_least_square(
                    gt=gt_i[valid_mask_i],
                    pred=pred_i[valid_mask_i],
                )
                pred_aligned = torch.clip(pred_aligned, min=1e-8, max=None)  # avoid 0 disparity
                target = torch.clip(gt_i[valid_mask_i], min=1e-8, max=None)  # avoid 0 disparity
            else:
                pred_aligned = torch.clip(pred_i[valid_mask_i], min=self.min_depth, max=self.max_depth)  # avoid 0 disparity
                target = torch.clip(gt_i[valid_mask_i], min=self.min_depth, max=self.max_depth)
            pred_list.append(pred_aligned)
            target_list.append(target)
        pred_aligned = torch.concat(pred_list, dim=0)
        target = torch.concat(target_list, dim=0)
        sum_abs_rel, num_obs = _abs_rel_update(pred_aligned, target, num_outputs=self.num_outputs)

        sum_sq_rel, _ = _sq_rel_update(pred_aligned, target, num_outputs=self.num_outputs)
        sum_rmse, _ = _rmse_update(pred_aligned, target, num_outputs=self.num_outputs)
        sum_rmse_log, _ = _rmse_log_update(pred_aligned, target, num_outputs=self.num_outputs)
        sum_d1, sum_d2, sum_d3, _ = _delta_log_update(pred_aligned, target, num_outputs=self.num_outputs)

        self.sum_abs_rel += sum_abs_rel
        self.sum_sq_rel += sum_sq_rel
        self.sum_rmse += sum_rmse
        self.sum_rmse_log += sum_rmse_log
        self.sum_d1 += sum_d1
        self.sum_d2 += sum_d2
        self.sum_d3 += sum_d3

        self.total += num_obs

    def compute(self):
        """Reduce and compute the final depth evaluation metric.
        Returns:
            dict: Dictionary containing the depth evaluation metrics.
        """
        abs_rel = self.sum_abs_rel / self.total
        sq_rel = self.sum_sq_rel / self.total
        rmse = torch.sqrt(self.sum_rmse / self.total)
        rmse_log = torch.sqrt(self.sum_rmse_log / self.total)
        d1 = self.sum_d1 / self.total
        d2 = self.sum_d2 / self.total
        d3 = self.sum_d3 / self.total

        return {"d1": d1.item(), "d2": d2.item(), "d3": d3.item(), "abs_rel": abs_rel.item(), "sq_rel": sq_rel.item(),
                "rmse": rmse.item(), 'rmse_log': rmse_log.item()}
