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

"""Evaluate a trained action recognition model."""
import csv
import os
import numpy as np
import torch
from tqdm import tqdm

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.action_recognition.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.action_recognition.model.pl_ar_model import ActionRecognitionModel
from nvidia_tao_pytorch.cv.action_recognition.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.action_recognition.inference.inferencer import Inferencer
from nvidia_tao_pytorch.cv.action_recognition.utils.common_utils import check_and_create
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook


def compute_metrics(confusion_matrix):
    """Computes evaluation metrics.

    Args:
        confusion_matrix (numpy.ndarray): The confusion matrix.

    Returns:
        dict: A dictionary containing the evaluation metrics.
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
    """Dumps the confusion matrix to a CSV file.

    Args:
        csv_path (str): The path to the CSV file.
        cm (numpy.ndarray): The confusion matrix.
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


def run_experiment(experiment_config, model_path, key, output_dir,
                   batch_size=1, test_dataset_dir=None):
    """Run experiment."""
    check_and_create(output_dir)

    # Set status logging
    status_file = os.path.join(output_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting Action recognition evaluation"
    )

    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    # build dataloader
    model_config = experiment_config["model"]
    label_map = experiment_config["dataset"]["label_map"]
    gpu_id = experiment_config.evaluate.gpu_id
    torch.cuda.set_device(gpu_id)
    num_classes = len(label_map.keys())
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    output_shape = [experiment_config["model"]["input_height"],
                    experiment_config["model"]["input_width"]]
    action_set = os.listdir(test_dataset_dir)
    sample_dict = {}
    for action in action_set:
        action_root_path = os.path.join(test_dataset_dir, action)
        for video in os.listdir(action_root_path):
            video_path = os.path.join(action_root_path, video)
            sample_dict[video_path] = action

    aug_config = experiment_config["dataset"]["augmentation_config"]

    dataloader = build_dataloader(sample_dict=sample_dict,
                                  model_config=model_config,
                                  dataset_mode="val",
                                  output_shape=output_shape,
                                  input_type=model_config["input_type"],
                                  label_map=label_map,
                                  batch_size=batch_size,
                                  workers=experiment_config["dataset"]["workers"],
                                  eval_mode=experiment_config["evaluate"]["video_eval_mode"],
                                  augmentation_config=aug_config,
                                  num_segments=experiment_config["evaluate"]["video_num_segments"])

    model = ActionRecognitionModel.load_from_checkpoint(model_path,
                                                        map_location="cpu",
                                                        experiment_spec=experiment_config)
    # build inferencer @TODO TRT support
    eval_mode_flag = experiment_config["evaluate"]["video_eval_mode"] == "conv"

    if eval_mode_flag:
        infer = Inferencer(model, ret_prob=True)
    else:
        infer = Inferencer(model)

    # do evaluation
    progress = tqdm(dataloader)
    sample_pred_dict = {}
    with torch.no_grad():
        if eval_mode_flag:
            for sample_path, data, action_label in progress:
                batch_size = len(sample_path)
                prob = infer.inference(data)
                for idx in range(batch_size):
                    if sample_path[idx] not in sample_pred_dict:
                        sample_pred_dict[sample_path[idx]] = prob[idx]
                    sample_pred_dict[sample_path[idx]] += prob[idx]
            for k, v in sample_pred_dict.items():
                pred_id = np.argmax(v)
                action = sample_dict[k]
                confusion_matrix[label_map[action], pred_id] += 1
        else:
            for sample_path, data, action_label in progress:
                batch_size = len(sample_path)
                pred_id = infer.inference(data)
                for idx in range(batch_size):
                    confusion_matrix[action_label[idx], pred_id[idx]] += 1

    percentage_confusion_matrix, accuracy, m_accuracy = compute_metrics(confusion_matrix)

    id2name = {v: k for k, v in label_map.items()}
    print("*******************************")
    for idx in range(len(label_map)):
        cls_acc = percentage_confusion_matrix[idx][idx]
        print("{:<14}{:.4}".format(
            id2name[idx], cls_acc))

    print("*******************************")
    print("Total accuracy: {}".format(round(accuracy, 3)))
    print("Average class accuracy: {}".format(round(m_accuracy, 3)))

    status_logging.get_status_logger().kpi = {"accuracy": round(accuracy, 3),
                                              "m_accuracy": round(m_accuracy, 3)}
    status_logging.get_status_logger().write(
        message="Evaluation metrics generated.",
        status_level=status_logging.Status.RUNNING
    )


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        if cfg.evaluate.results_dir is not None:
            results_dir = cfg.evaluate.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "evaluate")
        run_experiment(experiment_config=cfg,
                       output_dir=results_dir,
                       key=cfg.encryption_key,
                       model_path=cfg.evaluate.checkpoint,
                       batch_size=cfg.evaluate.batch_size,
                       test_dataset_dir=cfg.evaluate.test_dataset_dir)
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
