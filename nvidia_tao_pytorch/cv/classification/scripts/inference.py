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

"""
Inference of Classification model.
"""
from nvidia_tao_pytorch.cv.segformer.utils.common_utils import check_and_create
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.mmlab.common.utils import set_env, set_distributed
from nvidia_tao_pytorch.cv.classification.models import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.heads import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.core.mmlab.mmclassification.classification_default_config import ExperimentConfig
from nvidia_tao_pytorch.core.mmlab.mmclassification.utils import MMClsConfig, multi_gpu_test, load_model
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import get_root_logger
from mmcv.parallel import MMDistributedDataParallel
import torch
import datetime
import os
import numpy as np
import pandas as pd


def run_experiment(experiment_config, results_dir):
    """Start the Inference."""
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
        message="Starting Classification inference"
    )
    status_logger = status_logging.get_status_logger()
    mmcls_config_obj = MMClsConfig(experiment_config, phase="eval")
    mmcls_config = mmcls_config_obj.config
    # Set the logger
    log_file = os.path.join(results_dir, 'log_evaluation_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = get_root_logger(log_file=log_file, log_level="INFO")
    # log to file
    logger.info('**********************Start logging for Evaluation**********************')
    status_logger.write(message="**********************Start logging for Inference**********************.")
    meta = set_env()
    set_distributed(mmcls_config, "inference")

    # set the encryption key:
    seed = mmcls_config["inference"]["exp_config"]["manual_seed"]
    meta['seed'] = seed

    test_dataset = build_dataset(mmcls_config["dataset"]["data"]["test"])
    # Dataloader building
    data_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=mmcls_config["dataset"]["data"]["samples_per_gpu"],
        workers_per_gpu=mmcls_config["dataset"]["data"]["workers_per_gpu"],
        dist=True,
        shuffle=False)
    model_path = experiment_config["inference"]["checkpoint"]
    if not model_path:
        raise ValueError("You need to provide the model path for Inference.")

    model_to_test = load_model(model_path, mmcls_config)
    model_to_test = MMDistributedDataParallel(
        model_to_test.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    outputs, img_names = multi_gpu_test(model_to_test, data_loader, None,
                                        gpu_collect=True)

    rank = os.environ['LOCAL_RANK']
    predictions = []
    if int(rank) == 0:
        for idx, img_name in enumerate(img_names):
            assert (len(outputs[idx]) == len(test_dataset.CLASSES)), "The number of classes in the prediction: {} \
                                                                         does not match with the number of classes in the test dataset: {}. Please ensure to provide \
                                                                         the classes text file in the dataset config.".format(len(outputs[idx]), len(test_dataset.CLASSES))
            class_index = np.argmax(outputs[idx])
            class_label = test_dataset.CLASSES[class_index]
            class_conf = outputs[idx][class_index]
            predictions.append((img_name, class_label, class_conf))
    result_csv_path = os.path.join(results_dir, 'result.csv')
    with open(result_csv_path, 'w', encoding='utf-8') as csv_f:
        # Write predictions to file
        df = pd.DataFrame(predictions)
        df.to_csv(csv_f, header=False, index=False)
    logger.info("The inference result is saved at: %s", result_csv_path)
    status_logger.write(message="Completed Inference.", status_level=status_logging.Status.SUCCESS)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="test_cats_and_dogs", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    try:
        if cfg.inference.results_dir is not None:
            results_dir = cfg.inference.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "inference")
        run_experiment(experiment_config=cfg,
                       results_dir=results_dir)
        status_logging.get_status_logger().write(status_level=status_logging.Status.SUCCESS,
                                                 message="Inference finished successfully.")
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(message="Inference was interrupted",
                                                 verbosity_level=status_logging.Verbosity.INFO,
                                                 status_level=status_logging.Status.FAILURE)
    except Exception as e:
        status_logging.get_status_logger().write(message=str(e),
                                                 status_level=status_logging.Status.FAILURE)
        raise e


if __name__ == "__main__":
    main()
