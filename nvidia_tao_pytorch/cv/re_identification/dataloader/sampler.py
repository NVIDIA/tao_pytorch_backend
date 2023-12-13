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

"""Sampler Module for Re-Identification."""

from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities, then for each identity, randomly samples K instances, therefore batch size is N*K.

    RandomIdentitySampler is a subclass of torch.utils.data.sampler. It overrides the __iter__ and __len__ methods based
    on the batch_size, num_samples and num_pids_per_batch. It ensures that for each identity, K consecutive identities
    are obtained.

    Args:
        data_source (list): A list of tuples, where each tuple contains (img_path, pid, camid).
        num_instances (int): Number of instances per identity in a batch.
        batch_size (int): Number of examples in a batch.

    Attributes:
        data_source (list): The list of data provided as input.
        batch_size (int): The number of examples per batch.
        num_instances (int): The number of instances per identity.
        num_pids_per_batch (int): The number of unique identities per batch.
        index_dic (defaultdict): A dictionary where the keys are unique identities (pid) and the values are
                                 lists of indices corresponding to the identities in the data_source.
        pids (list): A list of unique identities (pid) in the data.
        length (int): The estimated number of examples in an epoch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        """Initialize the sampler with the data, batch size, and number of instances.

        Args:
            data_source (list): The list of data.
            batch_size (int): The size of each batch of data.
            num_instances (int): The number of instances per identity.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        """Create an iterator for the sampler.

        Returns:
            final_idxs (iterator): An iterator over the indices of the images to be included in the batch.
        """
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        """Return the length of the sampler.

        Returns:
            length (int): The total number of images to be included in the epoch.
        """
        return self.length
