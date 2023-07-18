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

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.pose_classification.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.pose_classification.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.pose_classification.inference.inferencer import Inferencer
from nvidia_tao_pytorch.cv.pose_classification.model.pl_pc_model import PoseClassificationModel
from nvidia_tao_pytorch.cv.pose_classification.utils.common_utils import check_and_create
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import update_results_dir


def run_experiment(experiment_config, results_dir, key, model_path, data_path):
    """
    Start the inference process.

    This function initializes the necessary components for inference, including the model, data loader,
    and inferencer. It performs inference on the provided data and saves the results in the specified output file.

    Args:
        experiment_config (dict): The experiment configuration containing the model and inference parameters.
        results_dir (str): The directory to save the status and log files.
        key (str): The encryption key for intermediate checkpoints.
        model_path (str): The path to the pre-trained model checkpoint.
        data_path (str): The path to the test dataset.

    Raises:
        Exception: If any error occurs during the inference process.
    """
    check_and_create(results_dir)
    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting Pose classification inference"
    )

    gpu_id = experiment_config.inference.gpu_id
    torch.cuda.set_device(gpu_id)
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    # build dataloader
    label_map = experiment_config["dataset"]["label_map"]
    batch_size = experiment_config["dataset"]["batch_size"]
    num_workers = experiment_config["dataset"]["num_workers"]
    dataloader = build_dataloader(data_path=data_path,
                                  label_map=label_map,
                                  mmap=True,
                                  batch_size=batch_size,
                                  num_workers=num_workers)

    # build inferencer
    model = PoseClassificationModel.load_from_checkpoint(model_path,
                                                         map_location="cpu",
                                                         experiment_spec=experiment_config)
    infer = Inferencer(model, ret_prob=False)

    # do inference
    progress = tqdm(dataloader)
    id2name = {v: k for k, v in label_map.items()}
    results = []
    for data_label in progress:
        data = data_label[0]
        batch_size = len(data)
        pred_id = infer.inference(data)
        pred_name = []
        for label_idx in pred_id:
            pred_name.append(id2name[label_idx])
        results.extend(pred_name)

    # save the output
    output_file = open(experiment_config["inference"]["output_file"], "w")
    for idx in range(len(results)):
        output_file.write("{}\n".format(results[idx]))
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

    This function serves as the entry point for the inference script.
    It loads the experiment specification, obfuscates logs, updates the results directory, and calls the 'run_experiment' function.

    Args:
        cfg (ExperimentConfig): The experiment configuration retrieved from the Hydra configuration files.
    """
    # Obfuscate logs.
    try:
        cfg = update_results_dir(cfg, task="inference")
        run_experiment(experiment_config=cfg,
                       results_dir=cfg.results_dir,
                       key=cfg.encryption_key,
                       model_path=cfg.inference.checkpoint,
                       data_path=cfg.inference.test_dataset.data_path)
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
