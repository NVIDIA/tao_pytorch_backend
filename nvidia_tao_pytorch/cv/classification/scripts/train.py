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

""" MMClassification Train Module """

from mmcls.utils import get_root_logger
from mmcls.models import build_classifier
from mmcls.datasets import build_dataset
from nvidia_tao_pytorch.cv.segformer.utils.common_utils import check_and_create
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.mmlab.mmclassification.classification_default_config import ExperimentConfig
from nvidia_tao_pytorch.core.mmlab.mmclassification.classification_trainer import MMClsTrainer
from nvidia_tao_pytorch.core.mmlab.mmclassification.utils import MMClsConfig
from nvidia_tao_pytorch.core.mmlab.common.utils import set_env, set_distributed, get_latest_pth_model
from nvidia_tao_pytorch.cv.classification.heads import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.models import *  # noqa pylint: disable=W0401, W0614
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
import warnings
import json
import time
import datetime
import os


def run_experiment(experiment_config, results_dir):
    """Start the training."""
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
        message="Starting Classification Train"
    )
    status_logger = status_logging.get_status_logger()
    mmcls_config_obj = MMClsConfig(experiment_config)
    mmcls_config = mmcls_config_obj.config
    resume_checkpoint_local = get_latest_pth_model(results_dir)
    resume_checkpoint_config = mmcls_config["train"]["train_config"]["resume_training_checkpoint_path"]
    if not resume_checkpoint_config:  # If no resume ckpt was provided in the config
        mmcls_config["train"]["train_config"]["resume_training_checkpoint_path"] = resume_checkpoint_local

    # Set the logger
    log_file = os.path.join(results_dir, 'log_train_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = get_root_logger(log_file=log_file, log_level="INFO")
    # log to file
    logger.info('**********************Start logging for Training**********************')
    status_logger.write(message="**********************Start logging for Training**********************.")
    meta = set_env()
    set_distributed(mmcls_config)

    # set the encryption key:
    seed = mmcls_config["train"]["exp_config"]["manual_seed"]
    meta['seed'] = seed

    datasets = [build_dataset(mmcls_config["dataset"]["data"]["train"])]
    status_logger.write(message="Completed Data Module Construction", status_level=status_logging.Status.RUNNING)

    model = build_classifier(
        mmcls_config["model"])
    model.init_weights()
    status_logger.write(message="Model Classifier Construction", status_level=status_logging.Status.RUNNING)
    exp_params_file = os.path.join(results_dir, "experiment_params.json")
    try:
        with open(exp_params_file, 'w') as fp:
            json.dump(mmcls_config, fp)
    except Exception as e:
        logger.info(e)
        warnings.warn("The expeirment spec paras could not be dumped into file.")

    meta["CLASSES"] = datasets[0].CLASSES
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cls_trainer = MMClsTrainer(
        datasets,
        model,
        timestamp=timestamp,
        meta=meta,
        result_dir=results_dir,
        experiment_spec=mmcls_config)
    cls_trainer.set_up_trainer()  # This will setup dataloader, model, runner
    cls_trainer.fit()
    status_logger.write(message="Completed Train.", status_level=status_logging.Status.SUCCESS)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train_cats_dogs_new_fan", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        if cfg.train.results_dir is not None:
            results_dir = cfg.train.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "train")

        run_experiment(cfg, results_dir=results_dir)
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Training was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


if __name__ == "__main__":
    main()
