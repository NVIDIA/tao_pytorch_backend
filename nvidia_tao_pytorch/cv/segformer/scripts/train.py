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

"""Train Segformer model."""

import os
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.segformer.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.utils.common_utils import check_and_create
from nvidia_tao_pytorch.cv.segformer.model.sf_model import SFModel
from nvidia_tao_pytorch.cv.segformer.dataloader.segformer_dm import SFDataModule
from nvidia_tao_pytorch.cv.segformer.utils import collect_env, get_root_logger
from nvidia_tao_pytorch.cv.segformer.trainer.trainer import train_segmentor, set_random_seed
from nvidia_tao_pytorch.cv.segformer.dataloader.data_utils import build_dataset
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.segformer.model.builder import build_segmentor
from nvidia_tao_pytorch.cv.segformer.hooks.tao_status_logger import TaoTextLoggerHook # noqa pylint: disable=W0611
from mmcv.runner import init_dist, get_dist_info
from omegaconf import OmegaConf
import warnings
import datetime
import json
import time


def get_latest_pth_model(results_dir):
    """Utility function to return the latest pth model in a dir.
    Args:
        results_dir (str): Results dir to save the checkpoints.

    """
    trainable_ckpts = [int(item.split('.')[0].split('_')[1]) for item in os.listdir(results_dir)
                       if item.endswith(".pth")]
    num_ckpts = len(trainable_ckpts)
    if num_ckpts == 0:
        return None
    latest_step = sorted(trainable_ckpts, reverse=True)[0]
    latest_checkpoint = os.path.join(results_dir, "iter_{}.pth".format(latest_step))
    if not os.path.isfile(latest_checkpoint):
        raise FileNotFoundError("Checkpoint file not found at {}")
    return latest_checkpoint


def run_experiment(experiment_config, results_dir):
    """Start the training.
    Args:
        experiment_config (Dict): Config dictionary containing epxeriment parameters
        results_dir (str): Results dir to save the trained checkpoints.

    """
    check_and_create(results_dir)
    # Set the logger
    log_file = os.path.join(results_dir, 'log_train_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = get_root_logger(log_file=log_file, log_level="INFO")

    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    exp_params_file = os.path.join(results_dir, "experiment_params.json")
    try:
        with open(exp_params_file, 'w') as fp:
            exp_cfg_dict = OmegaConf.to_container(experiment_config)
            json.dump(exp_cfg_dict, fp)
    except Exception as e:
        warnings.warn("The expeirment spec paras could not be dumped into file due to {}.".format(e))
    num_gpus = experiment_config["train"]["num_gpus"]
    seed = experiment_config["train"]["exp_config"]["manual_seed"]
    # Need to change this
    rank, world_size = get_dist_info()
    # If distributed these env variables are set by torchrun
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
    if "RANK" not in os.environ:
        os.environ['RANK'] = str(rank)
    if "WORLD_SIZE" not in os.environ:
        os.environ['WORLD_SIZE'] = str(world_size)
    if "MASTER_PORT" not in os.environ:
        os.environ['MASTER_PORT'] = str(experiment_config["train"]["exp_config"]["MASTER_PORT"])
    if "MASTER_ADDR" not in os.environ:
        os.environ['MASTER_ADDR'] = experiment_config["train"]["exp_config"]["MASTER_ADDR"]

    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logger = status_logging.get_status_logger()
    # log to file
    logger.info('**********************Start logging for Training**********************')
    status_logger.write(message="**********************Start logging for Training**********************.")
    distributed = experiment_config["train"]["exp_config"]["distributed"]
    max_iters = experiment_config["train"]['max_iters']
    resume_ckpt = experiment_config["train"]['resume_training_checkpoint_path']
    init_dist(launcher="pytorch", backend="nccl")
    dm = SFDataModule(experiment_config["dataset"], num_gpus, seed, logger, "train", experiment_config["model"]["input_height"], experiment_config["model"]["input_width"])
    set_random_seed(seed, deterministic=False)
    with open(os.path.join(results_dir, 'target_class_id_mapping.json'), 'w') as fp:
        json.dump(dm.target_classes_train_mapping, fp)
    logger.info("Completed Data Module Construction")
    status_logger.write(message="Completed Data Module Construction", status_level=status_logging.Status.RUNNING)

    sf_model = SFModel(experiment_config, phase="train", num_classes=dm.num_classes)
    dm.setup()
    sf_model.max_iters = max_iters
    if not resume_ckpt:
        resume_ckpt = get_latest_pth_model(results_dir)
    sf_model.resume_ckpt = resume_ckpt
    sf_model.checkpoint_interval = experiment_config["train"]["checkpoint_interval"]
    dm.log_interval = experiment_config["train"]["logging_interval"]
    datasets = [build_dataset(dm.train_data, dm.default_args)]
    model = build_segmentor(
        sf_model.model_cfg,
        train_cfg=None,
        test_cfg=None)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    model.CLASSES = dm.CLASSES
    train_segmentor(
        model,
        datasets,
        distributed=distributed,
        validate=experiment_config["train"]["validate"],
        timestamp=timestamp,
        meta=meta,
        result_dir=results_dir,
        dm=dm,
        sf_model=sf_model)
    status_logger.write(message="Completed Training.", status_level=status_logging.Status.SUCCESS)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train_isbi", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        if cfg.train.results_dir is not None:
            results_dir = cfg.train.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "train")

        run_experiment(experiment_config=cfg,
                       results_dir=results_dir)
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Train was interrupted",
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
