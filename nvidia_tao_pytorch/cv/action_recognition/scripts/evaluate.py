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
from pytorch_lightning import Trainer

from nvidia_tao_core.config.action_recognition.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.cv.action_recognition.dataloader.pl_ar_data_module import ARDataModule
from nvidia_tao_pytorch.cv.action_recognition.model.pl_ar_model import ActionRecognitionModel


# TODO @seanf: cc reports this is never used
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


def run_experiment(experiment_config, key):
    """Run experiment."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config, key)

    dm = ARDataModule(experiment_config)
    model = ActionRecognitionModel.load_from_checkpoint(model_path,
                                                        map_location="cpu",
                                                        experiment_spec=experiment_config,
                                                        dm=dm)

    trainer = Trainer(**trainer_kwargs)

    trainer.test(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="Action Recognition", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
