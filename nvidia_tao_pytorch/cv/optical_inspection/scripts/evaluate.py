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

"""Evaluate a trained Optical Inspection model."""

import os
import torch
import pandas as pd
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.optical_inspection.config.default_config import OIExperimentConfig
from nvidia_tao_pytorch.cv.optical_inspection.model.pl_oi_model import OpticalInspectionModel
from nvidia_tao_pytorch.cv.optical_inspection.inference.inferencer import Inferencer
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.optical_inspection.model.build_nn_model import AOIMetrics
from nvidia_tao_pytorch.cv.optical_inspection.utils.common_utils import check_and_create
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


def run_experiment(experiment_config, model_path, key, results_dir):
    """Run experiment."""
    check_and_create(results_dir)
    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting Optical Inspection evaluate"
    )

    gpu_id = experiment_config.evaluate.gpu_id
    torch.cuda.set_device(gpu_id)

    # Set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    eval_data_path = experiment_config["dataset"]["test_dataset"]["csv_path"]
    margin = experiment_config["model"]["margin"]
    df = pd.read_csv(eval_data_path)

    # infer_results_path = experiment_config['inference']["results_dir"]
    logging.info("test_csv_path {}".format(eval_data_path))

    # build inferencer @TODO TRT support

    model = OpticalInspectionModel.load_from_checkpoint(
        model_path,
        map_location="cpu",
        experiment_spec=experiment_config
    )

    infer = Inferencer(model, ret_prob=False)
    with torch.no_grad():

        dataloader = build_dataloader(
            df=df,
            weightedsampling=True,
            split='test',
            data_config=experiment_config["dataset"]
        )

        valid_AOIMetrics = AOIMetrics(margin)

        for i, data in enumerate(dataloader, 0):

            euclidean_distance = infer.inference(data)
            valid_AOIMetrics.update(euclidean_distance, data[2])
            if i == 0:
                euclid = euclidean_distance
            else:
                euclid = torch.cat((euclid, euclidean_distance), 0)

        total_accuracy = valid_AOIMetrics.compute()['total_accuracy'].item()
        false_alarm = valid_AOIMetrics.compute()['false_alarm'].item()
        defect_accuracy = valid_AOIMetrics.compute()['defect_accuracy'].item()
        false_negative = valid_AOIMetrics.compute()['false_negative'].item()

        logging.info(
            "Tot Comp {} Total Accuracy {} False Negative {} False Alarm {} Defect Correctly Captured {} for Margin {}".format(
                len(euclid),
                round(total_accuracy, 2),
                round(false_negative, 2),
                round(false_alarm, 2),
                round(defect_accuracy, 2),
                margin
            )
        )


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment", schema=OIExperimentConfig
)
def main(cfg: OIExperimentConfig) -> None:
    """Run the Evaluate process."""
    if cfg.evaluate.results_dir is not None:
        results_dir = cfg.evaluate.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "evaluate")
    try:
        run_experiment(experiment_config=cfg,
                       key=cfg.encryption_key,
                       model_path=cfg.evaluate.checkpoint,
                       results_dir=results_dir)
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
