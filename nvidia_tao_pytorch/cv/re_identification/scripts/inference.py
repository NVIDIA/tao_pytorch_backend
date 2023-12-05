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

"""Inference on single patch."""
import os
import torch

from tqdm import tqdm
import json

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.re_identification.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.re_identification.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.re_identification.inference.inferencer import Inferencer
from nvidia_tao_pytorch.cv.re_identification.model.pl_reid_model import ReIdentificationModel
from nvidia_tao_pytorch.cv.re_identification.utils.common_utils import check_and_create
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import update_results_dir


def run_experiment(experiment_config, results_dir, key):
    """
    Start the inference process.

    This function initializes the necessary components for inference, including the model, data loader,
    and inferencer. It performs inference on the provided data and saves the results in the specified output file.

    Args:
        experiment_config (dict): The experiment configuration containing the model and inference parameters.
        results_dir (str): The directory to save the status and log files.
        key (str): The encryption key for intermediate checkpoints.

    Raises:
        Exception: If any error occurs during the inference process.
    """
    results_dir = experiment_config.inference.results_dir
    check_and_create(results_dir)
    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting Re-identification inference"
    )

    gpu_id = experiment_config.inference.gpu_id
    torch.cuda.set_device(gpu_id)
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    # build dataloader
    _, dataloader, _, _ = build_dataloader(experiment_config, is_train=False)

    # build inferencer @TODO TRT support
    model = ReIdentificationModel.load_from_checkpoint(experiment_config["inference"]["checkpoint"],
                                                       map_location="cpu",
                                                       experiment_spec=experiment_config,
                                                       prepare_for_training=False)

    if "swin" in experiment_config.model.backbone:
        model.model.load_param(experiment_config["inference"]["checkpoint"])

    infer = Inferencer(model)

    # do inference
    progress = tqdm(dataloader)
    results = []

    if "swin" in experiment_config.model.backbone:
        with torch.no_grad():
            for data, _, _, img_paths in progress:
                feats, _ = infer.inference(data)
                for img_path, feat in zip(img_paths, feats):
                    result = {"img_path": img_path, "embedding": feat.cpu().numpy().tolist()}
                    results.append(result)
    elif "resnet" in experiment_config.model.backbone:
        with torch.no_grad():
            for data, _, _, img_paths in progress:
                feats = infer.inference(data)
                for img_path, feat in zip(img_paths, feats):
                    result = {"img_path": img_path, "embedding": feat.cpu().numpy().tolist()}
                    results.append(result)

    # save the output
    output_file = open(experiment_config["inference"]["output_file"], "w")
    results = json.dumps(results, indent=4)
    output_file.write(results)
    output_file.close()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """
    Run the inference process.

    This function initializes the experiment and sets up logging. It calls run_experiment
    to perform inference on the data according to the experiment configuration, and handles
    any exceptions that occur during the process.

    Args:
        cfg (DictConfig): Configuration file.
    """
    try:
        cfg = update_results_dir(cfg, task="inference")
        run_experiment(experiment_config=cfg,
                       results_dir=cfg.results_dir,
                       key=cfg.encryption_key)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Inference finished successfully."
        )
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
