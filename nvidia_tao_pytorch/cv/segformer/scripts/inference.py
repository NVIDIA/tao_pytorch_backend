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
Evaluation of Segformer model.
"""
import os
import torch
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.segformer.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.utils.common_utils import check_and_create
from nvidia_tao_pytorch.cv.segformer.inference.inferencer import multi_gpu_test, single_gpu_test
from nvidia_tao_pytorch.cv.segformer.model.sf_model import SFModel
from nvidia_tao_pytorch.cv.segformer.dataloader.segformer_dm import SFDataModule
from nvidia_tao_pytorch.cv.segformer.dataloader.data_utils import build_dataloader
from nvidia_tao_pytorch.cv.segformer.model.builder import build_segmentor
from nvidia_tao_pytorch.cv.segformer.dataloader.data_utils import build_dataset
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils
from omegaconf import OmegaConf
from mmcv.runner import get_dist_info, init_dist
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner.checkpoint import load_checkpoint
import datetime


def run_experiment(experiment_config, results_dir):
    """Start the inference.
    Args:
        experiment_config (Dict): Config dictionary containing epxeriment parameters
        results_dir (str): Results dir to save the inference images

    """
    check_and_create(results_dir)
    # Set the logger
    log_file = os.path.join(results_dir, 'log_inference_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

    logger = common_utils.create_logger(log_file, rank=0)
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logger = status_logging.get_status_logger()
    # log to file
    logger.info('********************** Start logging for Inference.**********************')
    status_logger.write(message="**********************Start logging for Inference**********************.")
    num_gpus = experiment_config["inference"]["num_gpus"]
    seed = experiment_config["inference"]["exp_config"]["manual_seed"]
    # Need to change this
    rank, world_size = get_dist_info()
    # If distributed these are set by torchrun
    if "LOCAL_RANK" not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
    if "RANK" not in os.environ:
        os.environ['RANK'] = str(rank)
    if "WORLD_SIZE" not in os.environ:
        os.environ['WORLD_SIZE'] = str(world_size)
    if "MASTER_PORT" not in os.environ:
        os.environ['MASTER_PORT'] = str(experiment_config["inference"]["exp_config"]["MASTER_PORT"])
    if "MASTER_ADDR" not in os.environ:
        os.environ['MASTER_ADDR'] = experiment_config["inference"]["exp_config"]["MASTER_ADDR"]

    init_dist(launcher="pytorch", backend="nccl")
    dm = SFDataModule(experiment_config["dataset"], num_gpus, seed, logger, "infer", experiment_config["model"]["input_height"], experiment_config["model"]["input_width"])
    if experiment_config["dataset"]["palette"]:
        pallete_colors = OmegaConf.to_container(experiment_config["dataset"]["palette"])
    else:
        pallete_colors
    dm.setup()
    # test_dataset = dm.test_dataset
    sf_model = SFModel(experiment_config, phase="infer", num_classes=dm.num_classes)
    test_dataset = build_dataset(dm.test_data, dm.default_args)
    data_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=dm.samples_per_gpu,
        workers_per_gpu=dm.workers_per_gpu,
        dist=True,
        shuffle=False)
    CLASSES = dm.CLASSES
    PALETTE = dm.PALETTE

    model_path = experiment_config["inference"]["checkpoint"]
    if not model_path:
        raise ValueError("You need to provide the model path for Evaluation.")

    model_to_test = build_segmentor(sf_model.model_cfg, test_cfg=None)
    _ = load_checkpoint(model_to_test, model_path, map_location='cpu')

    model_to_test.CLASSES = CLASSES
    model_to_test.PALETTE = PALETTE
    efficient_test = True  # False

    distributed = True
    if not distributed:
        model_to_test = MMDataParallel(model_to_test, device_ids=[0])
        outputs = single_gpu_test(model_to_test, data_loader, False, results_dir,
                                  efficient_test)
    else:
        model_to_test = MMDistributedDataParallel(
            model_to_test.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model_to_test, data_loader, None,
                                 gpu_collect=True, efficient_test=True)

    rank, _ = get_dist_info()
    kwargs = {}
    kwargs["imgfile_prefix"] = results_dir
    if rank == 0:
        test_dataset.format_results(outputs, **kwargs)

    status_logger.write(message="Completed Inference.", status_level=status_logging.Status.SUCCESS)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="test_isbi", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the Inference process."""
    try:
        # Obfuscate logs.
        if cfg.inference.results_dir is not None:
            results_dir = cfg.inference.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "inference")
        run_experiment(experiment_config=cfg,
                       results_dir=results_dir)
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Inference was interrupted",
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
