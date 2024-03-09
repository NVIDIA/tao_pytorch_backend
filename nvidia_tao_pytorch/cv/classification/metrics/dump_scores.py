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

""" Module for dumping model predictions """

from typing import Any, List, Optional, Sequence

from mmengine.registry import METRICS
from mmengine.evaluator.metric import BaseMetric, _to_cpu

import pandas as pd


@METRICS.register_module()
class DumpResultsScores(BaseMetric):
    """Dump model predictions to a pickle file for offline evaluation.

    Args:
        out_file_path (str): Path of the dumped file.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        collect_dir: (str, optional): Synchronize directory for collecting data
            from different ranks. This argument should only be configured when
            ``collect_device`` is 'cpu'. Defaults to None.
            `New in version 0.7.3.`
    """

    def __init__(self,
                 out_file_path: str,
                 collect_device: str = 'cpu',
                 classes: List[str] = None,
                 collect_dir: Optional[str] = None) -> None:
        """Initialize DumpResultsScores."""
        super().__init__(
            collect_device=collect_device, collect_dir=collect_dir)
        self.out_file_path = out_file_path
        self.classes = classes

    def process(self, data_batch: Any, predictions: Sequence[dict]) -> None:
        """transfer tensors in predictions to CPU."""
        self.results.extend(_to_cpu(predictions))

    def compute_metrics(self, results: list) -> dict:
        """dump the prediction results to a pickle file."""
        results_header = ["img_name"]
        results_header.extend(self.classes)
        results_header.extend(["pred_label", "pred_score", "gt_label"])
        results_new = [results_header]

        for result in results:
            tmp_result = [result["img_path"]]
            pred_label_num = result["pred_label"].numpy().tolist()[0]
            pred_label = self.classes[pred_label_num]
            pred_scores = result["pred_score"].numpy().tolist()
            gt_label_idx = result["gt_label"].numpy().tolist()
            gt_label = self.classes[gt_label_idx[0]]
            tmp_result.extend(pred_scores)
            tmp_result.extend([pred_label, pred_scores[pred_label_num], gt_label])
            results_new.append(tmp_result)

        with open(self.out_file_path, 'w', encoding='utf-8') as csv_f:
            # Write predictions to file
            df = pd.DataFrame(results_new)
            df.to_csv(csv_f, header=False, index=False)

        return {}
