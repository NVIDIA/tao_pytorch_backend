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
Evaluate OCRNet script.
"""
import logging
import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.ocrnet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocrnet.dataloader.pl_ocr_data_module import OCRDataModule
from nvidia_tao_pytorch.cv.ocrnet.model.pl_ocrnet import OCRNetModel
from nvidia_tao_pytorch.cv.ocrnet.utils.utils import load_checkpoint


def run_experiment(experiment_spec, key):
    """run experiment."""
    if "train_gt_file" in experiment_spec["dataset"]:
        if experiment_spec["dataset"]["train_gt_file"] == "":
            experiment_spec["dataset"]["train_gt_file"] = None
    if "val_gt_file" in experiment_spec["dataset"]:
        if experiment_spec["dataset"]["val_gt_file"] == "":
            experiment_spec["dataset"]["val_gt_file"] = None
    results_dir, model_path, gpus = initialize_evaluation_experiment(experiment_spec, key)
    if len(gpus) > 1:
        gpus = [gpus[0]]
        logging.log(f"OCRNet does not support multi-GPU evaluation at this time. Using only GPU {gpus}")

    dm = OCRDataModule(experiment_spec)
    dm.setup(stage='test')

    # If pruned, will load the pruned model graph during construction
    model = OCRNetModel(experiment_spec, dm)
    # load model
    print('loading pretrained model from %s' % model_path)
    ckpt = load_checkpoint(model_path,
                           key=key,
                           to_cpu=True)

    model.model.load_state_dict(ckpt.state_dict(), strict=True)

    trainer = Trainer(devices=gpus,
                      default_root_dir=results_dir,
                      accelerator='gpu',
                      strategy='auto')

    trainer.test(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="OCRNet", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_spec=cfg,
                   key=cfg.encryption_key)


if __name__ == '__main__':
    main()
