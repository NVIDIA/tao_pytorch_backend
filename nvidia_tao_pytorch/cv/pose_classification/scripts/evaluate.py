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

"""Evaluate a trained pose classification model."""
import csv
import os
import numpy as np
import torch
from tqdm import tqdm
from tabulate import tabulate
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.pose_classification.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.pose_classification.model.pl_pc_model import PoseClassificationModel
from nvidia_tao_pytorch.cv.pose_classification.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.pose_classification.inference.inferencer import Inferencer
from nvidia_tao_pytorch.cv.pose_classification.utils.common_utils import check_and_create
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.utilities import update_results_dir


def compute_metrics(confusion_matrix):
    """
    Compute evaluation metrics based on the confusion matrix.

    This function computes the percentage confusion matrix, accuracy, and average class accuracy
    from the provided confusion matrix.

    Args:
        confusion_matrix (np.ndarray): The confusion matrix of shape (num_classes, num_classes).

    Returns:
        np.ndarray: The percentage confusion matrix of the same shape as the input matrix.
        float: The overall accuracy.
        float: The average class accuracy.
    """
    row_sum = np.sum(confusion_matrix, axis=1)
    _shape = confusion_matrix.shape
    percentage_confusion_matrix = np.zeros(
        _shape, dtype=np.float32)
    for x in range(_shape[0]):
        for y in range(_shape[1]):
            if not row_sum[x] == 0:
                percentage_confusion_matrix[x][y] = np.float32(confusion_matrix[x][y]) / \
                    row_sum[x] * 100.0

    trace = np.trace(confusion_matrix)
    percent_trace = np.trace(percentage_confusion_matrix)

    accuracy = float(trace) / np.sum(confusion_matrix) * 100.0
    m_accuracy = percent_trace / _shape[0]

    return percentage_confusion_matrix, accuracy, m_accuracy


def dump_cm(csv_path, cm, id2name):
    """
    Dump the confusion matrix to a CSV file.

    This function saves the confusion matrix to a CSV file, where each row and column represent a class,
    and the cell values represent the counts.

    Args:
        csv_path (str): The path to the output CSV file.
        cm (np.ndarray): The confusion matrix of shape (num_classes, num_classes).
        id2name (dict): A dictionary mapping class IDs to class names.
    """
    n_class = len(id2name.keys())
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        label_list = ["class"]
        for idx in range(n_class):
            label_list.append(id2name[idx])
        writer.writerow(label_list)
        for row_id in range(n_class):
            row = [id2name[row_id]]
            for col_id in range(n_class):
                row.append(cm[row_id][col_id])
            writer.writerow(row)


def run_experiment(experiment_config, results_dir, key, model_path, data_path, label_path):
    """
    Run the evaluation process.

    This function initializes the necessary components for evaluation, including the model, data loader,
    and inferencer. It performs evaluation on the test dataset and computes evaluation metrics and the confusion matrix.

    Args:
        experiment_config (dict): The experiment configuration containing the model and evaluation parameters.
        results_dir (str): The directory to save the evaluation results.
        key (str): The encryption key for intermediate checkpoints.
        model_path (str): The path to the trained model checkpoint.
        data_path (str): The path to the test dataset.
        label_path (str): The path to the label data.

    Raises:
        Exception: If any error occurs during the evaluation process.
    """
    check_and_create(results_dir)

    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting Pose classification evaluation")

    gpu_id = experiment_config.evaluate.gpu_id
    torch.cuda.set_device(gpu_id)
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    # build dataloader
    label_map = experiment_config["dataset"]["label_map"]
    batch_size = experiment_config["dataset"]["batch_size"]
    num_workers = experiment_config["dataset"]["num_workers"]
    dataloader = build_dataloader(data_path=data_path,
                                  label_path=label_path,
                                  label_map=label_map,
                                  mmap=True,
                                  batch_size=batch_size,
                                  num_workers=num_workers)

    # build inferencer
    model = PoseClassificationModel.load_from_checkpoint(model_path,
                                                         map_location="cpu",
                                                         experiment_spec=experiment_config)
    infer = Inferencer(model, ret_prob=False)

    # do evaluation
    num_classes = len(label_map.keys())
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    progress = tqdm(dataloader)
    for data, label in progress:
        batch_size = len(data)
        pred_id = infer.inference(data)
        for idx in range(batch_size):
            confusion_matrix[label[idx].item(), pred_id[idx]] += 1

    percentage_confusion_matrix, accuracy, m_accuracy = compute_metrics(confusion_matrix)

    table = []
    id2name = {v: k for k, v in label_map.items()}
    for idx in range(len(label_map)):
        cls_acc = percentage_confusion_matrix[idx][idx]
        table.append(["Class accuracy: " + id2name[idx], cls_acc])
    table.append(["Total accuracy", accuracy])
    table.append(["Average class accuracy", m_accuracy])
    status_logging.get_status_logger().kpi = {"accuracy": round(accuracy, 2), "avg_accuracy": round(m_accuracy, 2)}
    status_logging.get_status_logger().write(message="Evaluation metrics generated.", status_level=status_logging.Status.RUNNING)
    print(tabulate(table, headers=["Name", "Score"], floatfmt=".4f", tablefmt="fancy_grid"))


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
                       key=cfg.encryption_key,
                       model_path=cfg.evaluate.checkpoint,
                       data_path=cfg.evaluate.test_dataset.data_path,
                       label_path=cfg.evaluate.test_dataset.label_path)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Evaluation finished successfully."
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
