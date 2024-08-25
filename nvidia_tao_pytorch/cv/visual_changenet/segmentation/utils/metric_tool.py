# Copyright (c) 2023 Chaminda Bandara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Original source taken from https://github.com/wgcban/ChangeFormer
#
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

"""Visual ChangeNet Segmentation metric utils."""
import numpy as np


# TODO: Add other metrics from CFPL for eval and visualise?
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        """Initialize AverageMeter"""
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        """
        Initialize counter variables for metric calculation.

        Args:
            val (float): Initial value.
            weight (int): Initial weight.
        """
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        """
        Update AverageMeter with new value and weight.

        Args:
            val (float): New value to update with.
            weight (int, optional): Weight for the new value (default is 1).
        """
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        """
        Update AverageMeter with new value and weight.

        Args:
            val (float): New value to update with.
            weight (int, optional): Weight for the new value (default is 1).
        """
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        """Get the current value."""
        return self.val

    def average(self):
        """Get the average value."""
        return self.avg

    def get_scores(self):
        """Get scores and mean score using cm2score function."""
        scores_dict, mean_score_dict = cm2score(self.sum)
        return scores_dict, mean_score_dict

    def clear(self):
        """Clear the initialized status."""
        self.initialized = False


class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class):
        """init"""
        super().__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """Get the current confusion matrix, calculate the current F1 score, and update the confusion matrix"""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        """get scores from confusion matrix"""
        scores_dict, mean_score_dict = cm2score(self.sum)
        return scores_dict, mean_score_dict


def harmonic_mean(xs):
    """Compute Harmonic mean"""
    harmonic_mean = len(xs) / sum((x + 1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    """Compute F1 Score"""
    hist = confusion_matrix
    # n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    # acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # Acc = total acc, recall, precision, F1 are per class, mean_f1 is average F1 over all classes
    return mean_F1


def cm2score(confusion_matrix):
    """Compute Scores from Confusion Matrix"""
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    # freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    cls_iou = dict(zip(['iou_' + str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_' + str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_' + str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_' + str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1': mean_F1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)

    # Add mean metrics
    mean_recall = np.nanmean(recall)
    mean_precision = np.nanmean(precision)
    mean_score_dict = {'mprecision': mean_precision, 'mrecall': mean_recall}  # 'macc_': acc, 'miou_':mean_iu, 'mf1_':mean_F1,

    return score_dict, mean_score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """Compute the confusion matrix for a set of predictions"""
    def __fast_hist(label_gt, label_pred):
        """Collect values for Confusion Matrix

        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)

        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    """Get mIoU"""
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']  # pylint: disable=E1126
