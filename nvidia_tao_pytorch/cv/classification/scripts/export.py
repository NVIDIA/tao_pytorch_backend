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
import datetime
import os
from mmcls.utils import get_root_logger

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.segformer.utils.common_utils import check_and_create

from nvidia_tao_pytorch.cv.classification.models import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.heads import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.tools.onnx_utils import pytorch_to_onnx

from nvidia_tao_pytorch.core.mmlab.mmclassification.classification_default_config import ExperimentConfig
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.mmlab.mmclassification.utils import MMClsConfig, load_model


def run_experiment(experiment_config, results_dir):
    """Start the Export."""
    check_and_create(results_dir)
    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            append=True
        )
    )
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting Classification Export"
    )
    status_logger = status_logging.get_status_logger()
    mmcls_config_obj = MMClsConfig(experiment_config, phase="eval")
    mmcls_config = mmcls_config_obj.config
    # Set the logger
    log_file = os.path.join(results_dir, 'log_export_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = get_root_logger(log_file=log_file, log_level="INFO")
    # log to file
    logger.info('********************** Start logging for Export **********************')
    status_logger.write(message="**********************Start logging for Export**********************.")
    model_path = mmcls_config["export"]["checkpoint"]
    if not model_path:
        raise ValueError("You need to provide the model path for Export.")

    model_to_test = load_model(model_path, mmcls_config)
    output_file = mmcls_config["export"]["onnx_file"]
    if not output_file:
        onnx_name = model_path.split("/")[-1]
        onnx_name = onnx_name.replace(".pth", ".onnx")
        onnx_path = os.path.join(results_dir, onnx_name)
    else:
        onnx_path = output_file

    input_channel = mmcls_config["export"]["input_channel"]
    input_height = mmcls_config["export"]["input_height"]
    input_width = mmcls_config["export"]["input_width"]

    input_shape = [1] + [input_channel, input_height, input_width]
    opset_version = mmcls_config["export"]["opset_version"]

    # export
    pytorch_to_onnx(
        model_to_test,
        input_shape,
        opset_version=opset_version,
        show=False,
        output_file=onnx_path,
        verify=mmcls_config["export"]["verify"],
        num_classes=mmcls_config["model"]["head"]["num_classes"],
        logger=logger)

    status_logger.write(message="Completed Export.", status_level=status_logging.Status.SUCCESS)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="test_cats_and_dogs", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the Export."""
    try:
        if cfg.export.results_dir is not None:
            results_dir = cfg.export.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "export")
        run_experiment(experiment_config=cfg,
                       results_dir=results_dir)
        status_logging.get_status_logger().write(status_level=status_logging.Status.SUCCESS,
                                                 message="Export finished successfully.")
    except Exception as e:
        status_logging.get_status_logger().write(message=str(e),
                                                 status_level=status_logging.Status.FAILURE)
        raise e


if __name__ == "__main__":
    main()
