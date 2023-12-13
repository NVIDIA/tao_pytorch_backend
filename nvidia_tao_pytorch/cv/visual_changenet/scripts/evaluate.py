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
Evaluation of Visual ChangeNet model.
"""
import os
import torch
import pandas as pd

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.core.utilities import update_results_dir

from nvidia_tao_pytorch.cv.visual_changenet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.visual_changenet.utils.common_utils import check_and_create
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.dataloader.changenet_dm import CNDataModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlSegment
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlClassifier
from nvidia_tao_pytorch.core.path_utils import expand_path

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.optical_inspection.model.build_nn_model import AOIMetrics
from nvidia_tao_pytorch.cv.visual_changenet.classification.inference.inferencer import Inferencer as ClassificationInferencer
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.build_data_loader import build_dataloader

from pytorch_lightning import Trainer


def run_classifier_evaluate(experiment_config, results_dir):
    """Helper evaluate function for Visual ChangeNet Classiffier evaluation pipeline"""
    eval_data_path = experiment_config["dataset"]["classify"]["test_dataset"]["csv_path"]
    margin = experiment_config["model"]["classify"]["eval_margin"]
    pretrained_path = experiment_config.evaluate.checkpoint
    df = pd.read_csv(eval_data_path)

    logging.info("test_csv_path {}".format(eval_data_path))
    # build inferencer @TODO TRT support
    model = ChangeNetPlClassifier.load_from_checkpoint(pretrained_path,
                                                       map_location="cpu",
                                                       experiment_spec=experiment_config
                                                       )

    infer = ClassificationInferencer(model, difference_module=experiment_config.model.classify.difference_module)
    with torch.no_grad():

        dataloader = build_dataloader(
            df=df,
            weightedsampling=True,
            split='test',
            data_config=experiment_config["dataset"]["classify"]
        )

        valid_metrics = AOIMetrics(margin)

        for i, data in enumerate(dataloader, 0):

            siam_score = infer.inference(data)
            valid_metrics.update(siam_score, data[2])
            if i == 0:
                euclid = siam_score
            else:
                euclid = torch.cat((euclid, siam_score), 0)

        total_accuracy = valid_metrics.compute()['total_accuracy'].item()
        false_alarm = valid_metrics.compute()['false_alarm'].item()
        defect_accuracy = valid_metrics.compute()['defect_accuracy'].item()
        false_negative = valid_metrics.compute()['false_negative'].item()

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


def run_experiment(experiment_config, key, results_dir=None):
    """Run experiment."""
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    check_and_create(results_dir)

    # Set status logging
    status_file = expand_path(os.path.join(results_dir, "status.json"))
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            append=True
        )
    )
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting Visual ChangeNet evaluation"
    )

    task = experiment_config.task
    num_gpus = experiment_config["num_gpus"]
    # tlt inference
    pretrained_path = experiment_config.evaluate.checkpoint

    assert task in ['segment', 'classify'], "Visual ChangeNet only supports 'segment' and 'classify' tasks."
    if task == 'segment':
        if pretrained_path.endswith('.tlt') or pretrained_path.endswith('.pth'):
            # build dataloader
            dm = CNDataModule(experiment_config.dataset.segment)
            dm.setup(stage="test")

            # build model and load from the given checkpoint
            model = ChangeNetPlSegment.load_from_checkpoint(pretrained_path,
                                                            map_location="cpu",
                                                            experiment_spec=experiment_config
                                                            )

            acc_flag = None
            if num_gpus > 1:
                acc_flag = "ddp"

            trainer = Trainer(gpus=num_gpus,
                              default_root_dir=results_dir,
                              accelerator=acc_flag)

            trainer.test(model, datamodule=dm)

        elif pretrained_path.endswith('.engine'):
            raise NotImplementedError("TensorRT evaluation is supported through tao-deploy. "
                                      "Please use tao-deploy to generate TensorRT engine and run evaluation.")
        else:
            raise NotImplementedError("Model path format is only supported for .tlt or .pth")

    elif task == 'classify':
        run_classifier_evaluate(experiment_config, results_dir)

    else:
        raise NotImplementedError('Only tasks supported by Visual ChangeNet are: "segment" and "classify"')


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        cfg = update_results_dir(cfg, task="evaluate")
        # Obfuscate logs.
        obfuscate_logs(cfg)
        run_experiment(experiment_config=cfg,
                       key=cfg.encryption_key,
                       results_dir=cfg.results_dir)
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
