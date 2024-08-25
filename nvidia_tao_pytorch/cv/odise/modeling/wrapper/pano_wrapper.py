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

from collections import OrderedDict
import torch
import torch.nn as nn


class OpenPanopticInference(nn.Module):
    def __init__(
        self,
        model,
        labels,
        metadata=None,
        semantic_on=True,
        instance_on=True,
        panoptic_on=True,
        test_topk_per_image=100,
    ):
        super().__init__()
        self.model = model
        self.metadata = metadata

        # Calculate the chunksizes and start index to support a more efficient ensemble_logits_with_labels implementation.
        num_templates = []
        for l in labels:
            num_templates.append(len(l))
        chunk_sizes_pyt = torch.tensor(num_templates, dtype=torch.int32, device='cuda')
        chunk_start_idx = [0,]
        for i in range(len(num_templates) - 1):
            chunk_start_idx.append(chunk_start_idx[i] + num_templates[i])
        chunk_start_idx_pyt = torch.tensor(chunk_start_idx, dtype=torch.int32, device='cuda')

        self.model.semantic_on = semantic_on
        self.model.instance_on = instance_on
        self.model.panoptic_on = panoptic_on
        self.model.test_topk_per_image = test_topk_per_image
        self.model.metadata = self.metadata

        # TODO(@yuw): potentially remove self.x
        if hasattr(self.model, "category_head"):
            self.labels = self.model.category_head.test_labels
            self.model.category_head.test_labels = labels

            category_overlapping_list = []
            train_labels = {l for label in self.model.train_labels for l in label}
            for test_label in self.model.category_head.test_labels:
                category_overlapping_list.append(not set(train_labels).isdisjoint(set(test_label)))
            self.model.category_overlapping_mask = torch.tensor(
                category_overlapping_list, dtype=torch.long, device=self.model.device
            )

        elif hasattr(self.model, "word_head"):
            self.labels = self.model.word_head.test_labels
            self.model.word_head.test_labels = labels

            category_overlapping_list = []
            train_labels = {l for label in self.model.train_labels for l in label}
            for test_label in self.model.word_head.test_labels:
                category_overlapping_list.append(not set(train_labels).isdisjoint(set(test_label)))
            self.model.category_overlapping_mask = torch.tensor(
                category_overlapping_list, dtype=torch.long, device=self.model.device
            )

        self.chunk_sizes_pyt = self.model.chunk_sizes_pyt
        self.chunk_start_idx_pyt = self.model.chunk_start_idx_pyt
        self.model.chunk_sizes_pyt = chunk_sizes_pyt
        self.model.chunk_start_idx_pyt = chunk_start_idx_pyt

        self.num_classes = self.model.sem_seg_head.num_classes
        self.model.sem_seg_head.num_classes = len(labels)

    def forward(self, batched_inputs):
        assert not self.training
        results = self.model(batched_inputs)
        return results
