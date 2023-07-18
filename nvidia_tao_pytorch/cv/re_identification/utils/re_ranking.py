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

"""Re-Ranking Module for getting metrics."""
import numpy as np
from typing import List

EPSILON = 1e-10


def calc_euclidean_dist(qf: np.array, gf: np.array) -> np.array:
    """Calculate the Euclidean distance between query features and gallery features.

    Args:
        qf (np.array): Query features of shape (m x n).
        gf (np.array): Gallery features of shape (p x q).

    Returns:
        np.array: Distance matrix of shape (m x p).

    """
    dist_mat = 2 - (2 * np.dot(qf, gf.T))
    dist_mat = np.sqrt(np.clip(dist_mat, 0, 4)) / 2
    return dist_mat


def calc_batch_euclidean_dist(qf: np.array, gf: np.array, N: int = 6000) -> np.array:
    """Calculate the Euclidean distance between query features and gallery features in batches.

    Args:
        qf (np.array): Query features of shape (m x n).
        gf (np.array): Gallery features of shape (p x q).
        N (int, optional): Batch size. Defaults to 6000.

    Returns:
        np.array: Distance matrix of shape (m x p).

    """
    m = qf.shape[0]
    n = gf.shape[0]

    dist_mat: List[np.array] = list()
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd: List[np.array] = list()
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = calc_euclidean_dist(temp_qf, temp_gf)
            temp_qd.append(temp_d)
        temp_qd = np.concatenate(temp_qd, axis=0)
        temp_qd = temp_qd / (np.max(temp_qd, axis=0) + EPSILON)
        dist_mat.append(temp_qd.T)
    dist_mat = np.concatenate(dist_mat, axis=0)
    return dist_mat


def compute_batch_topk(qf: np.array, gf: np.array, k1: int, N: int = 6000) -> np.array:
    """Compute the top-k nearest neighbors and return (k+1) results.

    Args:
        qf (np.array): Query features of shape (m x n).
        gf (np.array): Gallery features of shape (p x q).
        k1 (int): k value for computing k-reciprocal feature.
        N (int, optional): Batch size. Defaults to 6000.

    Returns:
        np.array: Initial rank matrix of shape (m x k1).

    """
    m = qf.shape[0]
    n = gf.shape[0]

    initial_rank: List[np.array] = list()
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd: List[np.array] = list()
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = calc_euclidean_dist(temp_qf, temp_gf)
            temp_qd.append(temp_d)
        temp_qd = np.concatenate(temp_qd, axis=0)
        temp_qd = temp_qd / (np.max(temp_qd, axis=0) + EPSILON)
        temp_qd = temp_qd.T
        initial_rank.append(np.argsort(temp_qd, axis=1)[:, :k1])

    initial_rank = np.concatenate(initial_rank, axis=0)
    return initial_rank


def compute_batch_v(feat: np.array, R: List[np.array], all_num: int) -> np.array:
    """Compute the vectors of k-reciprocal nearest neighbors.

    Args:
        feat (np.array): Feature embeddings.
        R (List[np.array]): k-reciprocal expansion indices.
        all_num (int): Length of all the features.

    Returns:
        np.array: k-reciprocal nearest neighbors matrix of shape (all_num x all_num).

    """
    V = np.zeros((all_num, all_num), dtype=np.float32)
    m = feat.shape[0]
    for i in range(m):
        temp_gf = feat[i].reshape(1, -1)
        temp_qd = calc_euclidean_dist(temp_gf, feat)
        temp_qd = temp_qd / (np.max(temp_qd) + EPSILON)
        temp_qd = temp_qd.squeeze()
        temp_qd = temp_qd[R[i]]
        weight = np.exp(-temp_qd)
        weight_sum = np.sum(weight)
        if weight_sum > 0:
            weight = weight / weight_sum
        V[i, R[i]] = weight.astype(np.float32)
    return V


def get_k_reciprocal_index(initial_rank: np.array, i: int, k1: int) -> np.array:
    """Get the k-reciprocal nearest neighbor index.

    Args:
        initial_rank (np.array): Initial rank matrix.
        i (int): Index in the k-reciprocal neighbor set.
        k1 (int): k value for computing k-reciprocal feature.

    Returns:
        np.array: k-reciprocal nearest neighbor index.

    """
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def re_rank(prob_feat: np.array, gal_feat: np.array, k1: int, k2: int, lambda_value: float) -> np.array:
    """Apply re-ranking for distance computation.

    Args:
        prob_feat (np.array): Probe features.
        gal_feat (np.array): Gallery features.
        k1 (int): k value for computing k-reciprocal feature.
        k2 (int): k value for local value expansion.
        lambda_value (float): Lambda for original distance when combining with Jaccard distance.

    Returns:
        np.array: Final distance matrix.

    """
    query_num = prob_feat.shape[0]
    all_num = query_num + gal_feat.shape[0]
    feat = np.append(prob_feat, gal_feat, axis=0)
    initial_rank = compute_batch_topk(feat, feat, k1 + 1, N=6000)
    del prob_feat
    del gal_feat

    R: List[np.array] = list()
    for i in range(all_num):
        k_reciprocal_index = get_k_reciprocal_index(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = get_k_reciprocal_index(initial_rank, candidate, int(np.around(k1 / 2.)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        R.append(k_reciprocal_expansion_index)

    V = compute_batch_v(feat, R, all_num)
    del R
    initial_rank = initial_rank[:, :k2]

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    inv_index: List[int] = list()

    for i in range(all_num):
        inv_index.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        ind_non_zero = np.where(V[i, :] != 0)[0]
        ind_images = [inv_index[ind] for ind in ind_non_zero]
        for j in range(len(ind_non_zero)):
            temp_min[0, ind_images[j]] = temp_min[0, ind_images[j]] + np.minimum(V[i, ind_non_zero[j]],
                                                                                 V[ind_images[j], ind_non_zero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)
    del V
    original_dist = calc_batch_euclidean_dist(feat, feat[:query_num, :])
    del feat
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del jaccard_dist

    final_dist = final_dist[:query_num, query_num:]
    final_dist = np.clip(final_dist, 0, 1)
    return final_dist
