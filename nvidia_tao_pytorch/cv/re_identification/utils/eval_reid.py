# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Original source taken from https://github.com/michuanhaohao/reid-strong-baseline
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

"""Eval for Re-Identification Module."""

import numpy as np
from nvidia_tao_pytorch.cv.re_identification.utils.common_utils import plot_evaluation_results


def eval_func(cfg, distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths, prepare_for_training):
    """Evaluates person re-identification (ReID) performance using Market1501 metric.

    For each query identity, it discards gallery images from the same camera view.
    After that, it calculates the cumulative matching characteristics (CMC) curve and mean Average Precision (mAP).
    If the program is not in training mode and if plotting is enabled, it also plots the evaluation results.

    Args:
        cfg (DictConfig): Configuration file.
        distmat (numpy.ndarray): Pairwise distance matrix between query and gallery features.
        q_pids (numpy.ndarray): Array containing query person IDs.
        g_pids (numpy.ndarray): Array containing gallery person IDs.
        q_camids (numpy.ndarray): Array containing query camera IDs.
        g_camids (numpy.ndarray): Array containing gallery camera IDs.
        q_img_paths (list of str): List containing query image paths.
        g_img_paths (list of str): List containing gallery image paths.
        prepare_for_training (bool): Flag indicating whether the system is in training mode.

    Returns:
        list: The Cumulative Matching Characteristics (CMC) rank list.
        float: The mean Average Precision (mAP) score.
    """
    num_q, num_g = distmat.shape
    max_rank = cfg["re_ranking"]["max_rank"]
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    query_maps = []

    for q_idx in range(num_q):
        query_map = []

        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_img_path = q_img_paths[q_idx]

        # build the first column of the sampled matches image output
        query_map.append([q_img_path, False])

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        res_list = list(map(g_img_paths.__getitem__, order))

        # build the rest of the columns of the sampled matches image output
        for g_img_path, value in zip(res_list[:max_rank], matches[q_idx][:max_rank]):
            query_map.append([g_img_path, value])
        query_maps.append(query_map)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if not prepare_for_training and cfg["evaluate"]["output_sampled_matches_plot"]:
        plot_evaluation_results(cfg["re_ranking"]["num_query"], query_maps, max_rank, cfg["evaluate"]["output_sampled_matches_plot"])

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery."

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
