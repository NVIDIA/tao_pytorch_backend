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

"""Evaluate a trained Metric Learning Recognition model."""

import os

import torch

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.metric_learning_recognition.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.metric_learning_recognition.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.metric_learning_recognition.model.pl_ml_recog_model import MLRecogModel
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.cv.metric_learning_recognition.utils.decorators import monitor_status
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


def run_experiment(experiment_config):
    """Starts the evaluate.

    Args:
        experiment_config (DictConfig): Configuration dictionary

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_id = experiment_config["evaluate"].gpu_id
        torch.cuda.set_device(gpu_id)

    # no need to check `else` as it's verified in the decorator already
    if experiment_config['evaluate']["results_dir"]:
        results_dir = experiment_config['evaluate']["results_dir"]
    elif experiment_config["results_dir"]:
        results_dir = os.path.join(experiment_config["results_dir"], "evaluate")

    # get datasets
    _, _, _, dataset_dict = build_dataloader(experiment_config, mode="eval")

    status_logging.get_status_logger().write(
        message=f"Loading checkpoint: {experiment_config['evaluate']['checkpoint']}",
        status_level=status_logging.Status.STARTED)
    metric_learning_recognition = MLRecogModel.load_from_checkpoint(
        experiment_config["evaluate"]["checkpoint"],
        map_location="cpu",
        experiment_spec=experiment_config,
        results_dir=results_dir,
        subtask="evaluate")
    metric_learning_recognition.load_tester()
    metric_learning_recognition.dataset_dict = dataset_dict
    metric_learning_recognition.class_dict = dataset_dict["query"].class_dict

    metric_learning_recognition.to(torch.device(device))
    all_acc = metric_learning_recognition.get_query_accuracy()
    # df = metric_learning_recognition.report_accuracies(all_acc, save_results=True)
    print("******************* Evaluation results **********************")
    ami = all_acc['query']['AMI_level0']
    nmi = all_acc['query']['NMI_level0']
    mean_avg_prec = all_acc['query']['mean_average_precision_level0']
    mean_reciprocal_rank = all_acc['query']['mean_reciprocal_rank_level0']
    mean_r_precision = all_acc['query']['r_precision_level0']
    val_accuracy = all_acc['query']['precision_at_1_level0']

    status_logging_dict = {}
    status_logging_dict['AMI'] = ami
    status_logging_dict['NMI'] = nmi

    if experiment_config["evaluate"]["report_accuracy_per_class"]:
        status_logging_dict['Mean Average Precision'] = sum(mean_avg_prec) / len(mean_avg_prec)
        status_logging_dict['Mean Reciprocal Rank'] = sum(mean_reciprocal_rank) / len(mean_reciprocal_rank)
        status_logging_dict['r-Precision'] = sum(mean_r_precision) / len(mean_r_precision)
        status_logging_dict['Precision at Rank 1'] = sum(val_accuracy) / len(val_accuracy)
    else:
        status_logging_dict['Mean Average Precision'] = mean_avg_prec
        status_logging_dict['Mean Reciprocal Rank'] = mean_reciprocal_rank
        status_logging_dict['r-Precision'] = mean_r_precision
        status_logging_dict['Precision at Rank 1'] = val_accuracy

    status_logging.get_status_logger().kpi = status_logging_dict

    for metric in status_logging_dict:
        print(f"{metric}: {status_logging_dict[metric]:.4f}")

    if experiment_config["evaluate"]["report_accuracy_per_class"]:
        print("\n******************* Accuracy per class **********************")
        for k, v in all_acc['query'].items():
            if "level0" in k:
                if isinstance(v, list):
                    print(f"{k[:-7]}:")
                    for i, acc in enumerate(v):
                        print(f"  {metric_learning_recognition.class_dict[i]}: {acc:.4f}")
    print("*************************************************************")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process.

    Args:
        cfg (DictConfig): Hydra config object.

    """
    obfuscate_logs(cfg)

    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
