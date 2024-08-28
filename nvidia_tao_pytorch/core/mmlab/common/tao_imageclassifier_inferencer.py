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

""" TAO Image Classification Inferencer Class."""

from pathlib import Path
from typing import List, Optional
import os
import pandas as pd

import torch

from mmpretrain.visualization import UniversalVisualizer
from mmpretrain.apis.base import InputType
from mmpretrain.apis import ImageClassificationInferencer
from mmpretrain.structures import DataSample
from mmcv.image import imread

from mmengine.runner import Runner


class TAOImageClassificationInferencer(ImageClassificationInferencer):
    """Inferencer class for TAO Image Classification."""

    def __init__(self, results_dir, *args, **kwargs):  # noqa pylint: disable=W0235
        """Initialize TAOImageClassificationInferencer."""
        super(TAOImageClassificationInferencer, self).__init__(*args, **kwargs)
        self.results_dir = results_dir

    def visualize(self,
                  ori_inputs: List[InputType],
                  preds: List[DataSample],
                  show: bool = False,
                  wait_time: int = 0,
                  resize: Optional[int] = None,
                  rescale_factor: Optional[float] = None,
                  draw_score=True,
                  show_dir=None):
        """Visualize the classification results.

        Args:
            ori_inputs (List[InputType]): List of original input images.
            preds (List[DataSample]): List of prediction results.
            show (bool, optional): Whether to display the visualizations. Defaults to False.
            wait_time (int, optional): Time to wait before closing the visualization window. Defaults to 0.
            resize (Optional[int], optional): Resize the visualizations to a specific size. Defaults to None.
            rescale_factor (Optional[float], optional): Rescale factor for visualizations. Defaults to None.
            draw_score (bool, optional): Whether to draw prediction scores. Defaults to True.
            show_dir (Optional[str], optional): Directory path to save visualizations. Defaults to None.

        """
        if not show and show_dir is None:
            return None

        if self.visualizer is None:
            self.visualizer = UniversalVisualizer()

        visualization = []
        predictions = []
        for i, (input_, data_sample) in enumerate(zip(ori_inputs, preds)):
            image = imread(input_)
            if isinstance(input_, str):
                # The image loaded from path is BGR format.
                image = image[..., ::-1]
                name = input_.split('/')[-1]
            else:
                name = str(i) + '.png'

            if show_dir is not None:
                show_dir = Path(show_dir)
                show_dir.mkdir(exist_ok=True)
                out_file = str((show_dir / name))
            else:
                out_file = None
            pred_scores = data_sample.pred_score
            predictions.append((input_, self.classes[torch.argmax(pred_scores).item()], float(torch.max(pred_scores).item())))
            self.visualizer.visualize_cls(
                image,
                data_sample,
                classes=self.classes,
                resize=resize,
                show=show,
                wait_time=wait_time,
                rescale_factor=rescale_factor,
                draw_gt=False,
                draw_pred=True,
                draw_score=draw_score,
                name=name,
                out_file=out_file)
            visualization.append(self.visualizer.get_image())

        result_csv_path = os.path.join(self.results_dir, 'result.csv')
        with open(result_csv_path, 'w', encoding='utf-8') as csv_f:
            # Write predictions to file
            df = pd.DataFrame(predictions)
            df.to_csv(csv_f, header=False, index=False)
        if show:
            self.visualizer.close()
        return visualization


def get_classes_list(experiment_config, head, results_dir, checkpoint):
    """Get a list of classes for the given experiment configuration.

    Args:
        experiment_config (dict): Experiment configuration.
        head (str): Type of head.
        results_dir (str): Directory to store results.
        checkpoint (str): Path to the checkpoint.
    """
    experiment_config["train_dataloader"] = None
    experiment_config["work_dir"] = results_dir
    experiment_config["load_from"] = checkpoint
    runner = Runner.from_cfg(experiment_config)
    classes = sorted(list(runner.val_dataloader.dataset.CLASSES))

    return classes
