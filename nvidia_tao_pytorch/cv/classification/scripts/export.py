# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Original source taken from https://github.com/open-mmlab/mmclassification

# Copyright 2019 OpenMMLAB

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export of Classification model.
"""
import os
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.classification.models import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.heads import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.tools.onnx_utils import pytorch_to_onnx
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.mmlab.mmclassification.classification_default_config import ExperimentConfig
from nvidia_tao_pytorch.core.mmlab.mmclassification.utils import MMPretrainConfig
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.mmlab.mmclassification.utils import load_model


def run_experiment(experiment_config):
    """Start the Export."""
    results_dir = experiment_config.results_dir
    # log to file
    status_logger = status_logging.get_status_logger()
    status_logger.write(message="********************** Start logging for Export **********************.")
    mmpretrain_config = MMPretrainConfig(experiment_config, phase="evaluate")
    export_cfg = mmpretrain_config.updated_config
    model_path = experiment_config.export.checkpoint
    if not model_path:
        raise ValueError("You need to provide the model path for Export.")

    model_to_test = load_model(model_path, export_cfg)
    output_file = experiment_config.export.onnx_file
    if not output_file:
        onnx_name = model_path.split("/")[-1]
        onnx_name = onnx_name.replace(".pth", ".onnx")
        onnx_path = os.path.join(results_dir, onnx_name)
    else:
        onnx_path = output_file

    input_channel = experiment_config.export.input_channel
    input_height = experiment_config.export.input_height
    input_width = experiment_config.export.input_width

    input_shape = [1] + [input_channel, input_height, input_width]
    opset_version = experiment_config.export.opset_version

    logger = None
    # export
    pytorch_to_onnx(
        model_to_test,
        input_shape,
        opset_version=opset_version,
        show=False,
        output_file=onnx_path,
        verify=False,
        num_classes=export_cfg["model"]["head"]["num_classes"],
        logger=logger)

    status_logger.write(message="Completed Export.", status_level=status_logging.Status.RUNNING)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="test_cats_and_dogs", schema=ExperimentConfig
)
@monitor_status(name="Classification", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """Run the Export."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
