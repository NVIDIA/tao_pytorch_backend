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

""" MMClassification Inference Module """

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.mmlab.mmclassification.classification_default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.classification.heads import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.models import *  # noqa pylint: disable=W0401, W0614
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.mmlab.common.tao_imageclassifier_inferencer import TAOImageClassificationInferencer, get_classes_list
from nvidia_tao_pytorch.core.mmlab.mmclassification.utils import MMPretrainConfig

from mmengine import Config

from glob import glob
import os

VALID_IMAGE_EXT = [".jpg", ".png", ".jpeg"]


def run_experiment(experiment_config):
    """Start Inference."""
    results_dir, model_path, _ = initialize_inference_experiment(experiment_config)
    mmpretrain_config = MMPretrainConfig(experiment_config, phase="inference")
    eval_cfg = mmpretrain_config.updated_config
    mmpretrain_model_config = Config(eval_cfg)
    classes = get_classes_list(experiment_config=eval_cfg,
                               head=experiment_config["model"]["head"]["type"],
                               results_dir=results_dir,
                               checkpoint=experiment_config.inference.checkpoint)

    inferencer = TAOImageClassificationInferencer(
        model=mmpretrain_model_config,
        pretrained=model_path,
        device='cuda',
        results_dir=results_dir,
        classes=classes)

    val_dir = eval_cfg["test_dataloader"]["dataset"]["data_prefix"]
    SUFFIXES = [".jpg", ".jpeg", ".JPEG", ".JPG", ".png", ".PNG"]
    image_list = []
    for suffix in SUFFIXES:
        image_list += glob(val_dir + "/**/*" + suffix, recursive=True)

    vis_dir = os.path.join(results_dir, "visualize")
    status_logger = status_logging.get_status_logger()
    status_logger.write(message="********************** Start logging for Inference **********************.")
    inferencer(image_list, show_dir=vis_dir)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="test_cats_and_dogs", schema=ExperimentConfig
)
@monitor_status(name="Classification", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
