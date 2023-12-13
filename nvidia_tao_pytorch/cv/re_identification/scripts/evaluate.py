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

"""Evaluate a trained re-identification model."""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from tabulate import tabulate

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.re_identification.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.re_identification.dataloader.build_data_loader import build_dataloader, list_dataset
from nvidia_tao_pytorch.cv.re_identification.inference.inferencer import Inferencer
from nvidia_tao_pytorch.cv.re_identification.model.pl_reid_model import ReIdentificationModel
from nvidia_tao_pytorch.cv.re_identification.utils.common_utils import check_and_create
from nvidia_tao_pytorch.cv.re_identification.utils.reid_metric import R1_mAP, R1_mAP_reranking
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import update_results_dir


def run_experiment(experiment_config, results_dir, key):
    """
    Run the evaluation process.

    This function initializes the necessary components for evaluation, including the model, data loader,
    and inferencer. It performs evaluation on the test dataset and computes evaluation metrics.

    Args:
        experiment_config (dict): The experiment configuration containing the model and evaluation parameters.
        results_dir (str): The directory to save the evaluation results.
        key (str): The encryption key for intermediate checkpoints.

    Raises:
        Exception: If any error occurs during the evaluation process.
    """
    results_dir = experiment_config.evaluate.results_dir
    check_and_create(results_dir)

    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting Re-Identification evaluation")

    gpu_id = experiment_config.evaluate.gpu_id
    torch.cuda.set_device(gpu_id)
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    # build dataloader
    _, dataloader, _, _ = build_dataloader(experiment_config, is_train=False)

    model = ReIdentificationModel.load_from_checkpoint(experiment_config["evaluate"]["checkpoint"],
                                                       map_location="cpu",
                                                       experiment_spec=experiment_config,
                                                       prepare_for_training=False)

    if "swin" in experiment_config.model.backbone:
        model.model.load_param(experiment_config["evaluate"]["checkpoint"])

    infer = Inferencer(model)
    # do inference
    progress = tqdm(dataloader)
    query_top_dir = experiment_config["evaluate"]["query_dataset"]
    query_dict = list_dataset(query_top_dir)

    if experiment_config["re_ranking"]["re_ranking"]:
        metrics = R1_mAP_reranking(len(query_dict), experiment_config, False, feat_norm=True)
    else:
        metrics = R1_mAP(len(query_dict), experiment_config, False, feat_norm=True)
    metrics.reset()

    if "swin" in experiment_config.model.backbone:
        for data, pids, camids, img_paths in progress:
            with torch.no_grad():
                output, _ = infer.inference(data)
                metrics.update(output, pids, camids, img_paths)
    elif "resnet" in experiment_config.model.backbone:
        for data, pids, camids, img_paths in progress:
            with torch.no_grad():
                output = infer.inference(data)
                metrics.update(output, pids, camids, img_paths)
    cmc, mAP = metrics.compute()

    table = []
    table.append(["mAP", "{:.1%}".format(mAP)])
    status_logging.get_status_logger().kpi = {"mAP": round(mAP, 1)}
    status_logging.get_status_logger().write(message="Evaluation metrics generated.", status_level=status_logging.Status.RUNNING)

    for r in [1, 5, 10]:
        # print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        table.append(["CMC curve, Rank-" + "{:<3}".format(r), "{:.1%}".format(cmc[r - 1])])
    print(tabulate(table, headers=["Name", "Score"], floatfmt=".4f", tablefmt="fancy_grid"))

    plt.figure()
    cmc_percentages = [value * 100 for value in cmc]
    plt.xticks(np.arange(len(cmc_percentages)), np.arange(1, len(cmc_percentages) + 1))
    plt.plot(cmc_percentages, marker="*")
    plt.title('Cumulative Matching Characteristics (CMC) Curve')
    plt.grid()
    plt.ylabel('Matching Rate[%]')
    plt.xlabel('Rank')
    plt.savefig(experiment_config["evaluate"]["output_cmc_curve_plot"])


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """
    Run the evaluation process.

    This function serves as the entry point for the evaluation script.
    It loads the experiment specification, updates the results directory, and calls the 'run_experiment' function.

    Args:
        cfg (ExperimentConfig): The experiment configuration retrieved from the Hydra configuration files.
    """
    try:
        cfg = update_results_dir(cfg, task="evaluate")
        run_experiment(experiment_config=cfg,
                       results_dir=cfg.results_dir,
                       key=cfg.encryption_key)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Evaluation finished successfully"
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Evaluation was interrupted",
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
