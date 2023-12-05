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

""" Inference on single patch. """

import os

from pytorch_lightning import Trainer

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.utilities import update_results_dir
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook

from nvidia_tao_pytorch.cv.centerpose.dataloader.build_data_loader import CPDataModule
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import check_and_create

from nvidia_tao_pytorch.cv.centerpose.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.centerpose.model.pl_centerpose_model import CenterPosePlModel


def run_experiment(experiment_config, model_path, key, results_dir=None):
    """Start the inference."""
    if not model_path:
        raise FileNotFoundError("inference.checkpoint is not set!")

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
        message="Starting CenterPose inference"
    )

    # tlt inference
    if model_path.endswith('.tlt') or model_path.endswith('.pth'):
        num_gpus = experiment_config.inference.num_gpus

        # build data module
        dm = CPDataModule(experiment_config.dataset)
        dm.setup(stage="predict")

        # Run inference using tlt model
        acc_flag = None
        if num_gpus > 1:
            acc_flag = "ddp"

        model = CenterPosePlModel.load_from_checkpoint(model_path,
                                                       map_location="cpu",
                                                       experiment_spec=experiment_config)

        trainer = Trainer(devices=num_gpus,
                          default_root_dir=results_dir,
                          accelerator='gpu',
                          strategy=acc_flag)

        trainer.predict(model, datamodule=dm)
    elif model_path.endswith('.engine'):
        raise NotImplementedError("TensorRT inference is supported through tao-deploy. "
                                  "Please use tao-deploy to generate TensorRT enigne and run inference.")
    else:
        raise NotImplementedError("Model path format is only supported for .tlt or .pth")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="infer", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    try:
        cfg = update_results_dir(cfg, task="inference")

        run_experiment(experiment_config=cfg,
                       key=cfg.encryption_key,
                       model_path=cfg.inference.checkpoint,
                       results_dir=cfg.results_dir)
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
