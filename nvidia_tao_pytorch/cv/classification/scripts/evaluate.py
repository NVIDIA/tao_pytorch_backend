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
Evaluation of Classification model.
"""
from nvidia_tao_pytorch.cv.segformer.utils.common_utils import check_and_create
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.mmlab.common.utils import set_env, set_distributed
from nvidia_tao_pytorch.cv.classification.models import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.heads import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.core.mmlab.mmclassification.classification_default_config import ExperimentConfig
from nvidia_tao_pytorch.core.mmlab.mmclassification.utils import MMClsConfig, load_model
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.apis import multi_gpu_test
from mmcls.utils import get_root_logger
from mmcv.parallel import MMDistributedDataParallel
import torch
import datetime
import os
import numpy as np
from numbers import Number


def run_experiment(experiment_config, results_dir):
    """Start the Evaluation."""
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
        message="Starting Classification evaluation"
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
    set_distributed(mmcls_config, "evaluate")

    # set the encryption key:
    seed = mmcls_config["evaluate"]["exp_config"]["manual_seed"]
    meta['seed'] = seed

    test_dataset = build_dataset(mmcls_config["dataset"]["data"]["test"])
    # Dataloader building
    data_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=mmcls_config["dataset"]["data"]["samples_per_gpu"],
        workers_per_gpu=mmcls_config["dataset"]["data"]["workers_per_gpu"],
        dist=True,
        shuffle=False)
    model_path = experiment_config["evaluate"]["checkpoint"]
    if not model_path:
        raise ValueError("You need to provide the model path for Evaluation.")

    model_to_test = load_model(model_path, mmcls_config)

    model_to_test = MMDistributedDataParallel(
        model_to_test.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    outputs = multi_gpu_test(model_to_test, data_loader, None,
                             gpu_collect=True)
    rank = os.environ['LOCAL_RANK']
    if int(rank) == 0:
        results = {}
        logger = get_root_logger()
        eval_results = test_dataset.evaluate(
            results=outputs,
            metric=["accuracy", "precision", "recall"],
            metric_options={"topk": mmcls_config["evaluate"]["topk"]},
            logger=logger)
        results.update(eval_results)
        for k, v in eval_results.items():
            if isinstance(v, np.ndarray):
                v = [round(out, 2) for out in v.tolist()]
            elif isinstance(v, Number):
                v = round(v, 2)
            else:
                raise ValueError(f'Unsupport metric type: {type(v)}')
            print(f'\n{k} : {v}')
    status_logger.write(message="Completed Evaluation.", status_level=status_logging.Status.SUCCESS)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="test_cats_and_dogs", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the Evaluate process."""
    try:
        if cfg.evaluate.results_dir is not None:
            results_dir = cfg.evaluate.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "evaluate")
        run_experiment(experiment_config=cfg,
                       results_dir=results_dir)
        status_logging.get_status_logger().write(status_level=status_logging.Status.SUCCESS,
                                                 message="Evaluation finished successfully.")
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(message="Evaluation was interrupted",
                                                 verbosity_level=status_logging.Verbosity.INFO,
                                                 status_level=status_logging.Status.FAILURE)
    except Exception as e:
        status_logging.get_status_logger().write(message=str(e),
                                                 status_level=status_logging.Status.FAILURE)
        raise e


if __name__ == "__main__":
    main()
