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

"""Evaluate a trained deformable detr model."""
import os
from pytorch_lightning import Trainer

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.utilities import update_results_dir
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.deformable_detr.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.deformable_detr.model.pl_dd_model import DeformableDETRModel
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import check_and_create

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook


def run_experiment(experiment_config, model_path, key, results_dir=None):
    """Run experiment."""
    if not model_path:
        raise FileNotFoundError("evaluate.checkpoint is not set!")

    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    check_and_create(results_dir)

    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            append=True
        )
    )
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting DDETR evaluation"
    )

    # tlt inference
    if model_path.endswith('.tlt') or model_path.endswith('.pth'):
        # build dataloader
        dm = ODDataModule(experiment_config.dataset, subtask_config=experiment_config.evaluate)
        dm.setup(stage="test")

        # build model and load from the given checkpoint
        model = DeformableDETRModel.load_from_checkpoint(model_path,
                                                         map_location="cpu",
                                                         experiment_spec=experiment_config)

        num_gpus = experiment_config.evaluate.num_gpus
        acc_flag = None
        if num_gpus > 1:
            acc_flag = "ddp"

        trainer = Trainer(devices=num_gpus,
                          default_root_dir=results_dir,
                          accelerator='gpu',
                          strategy=acc_flag)

        trainer.test(model, datamodule=dm)

    elif model_path.endswith('.engine'):
        raise NotImplementedError("TensorRT evaluation is supported through tao-deploy. "
                                  "Please use tao-deploy to generate TensorRT enigne and run evaluation.")
    else:
        raise NotImplementedError("Model path format is only supported for .tlt or .pth")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="evaluate", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluate process."""
    try:
        cfg = update_results_dir(cfg, task="evaluate")

        run_experiment(experiment_config=cfg,
                       key=cfg.encryption_key,
                       model_path=cfg.evaluate.checkpoint,
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
