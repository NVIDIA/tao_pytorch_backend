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
Train OCRNet script.
"""
import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.ocrnet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocrnet.dataloader.pl_ocr_data_module import OCRDataModule
from nvidia_tao_pytorch.cv.ocrnet.model.pl_ocrnet import OCRNetModel


def run_experiment(experiment_spec: ExperimentConfig):
    """run experiment."""
    if "train_gt_file" in experiment_spec["dataset"]:
        if experiment_spec["dataset"]["train_gt_file"] == "":
            experiment_spec["dataset"]["train_gt_file"] = None
    if "val_gt_file" in experiment_spec["dataset"]:
        if experiment_spec["dataset"]["val_gt_file"] == "":
            experiment_spec["dataset"]["val_gt_file"] = None
    results_dir, resume_ckpt, gpus, ptl_loggers = initialize_train_experiment(experiment_spec)

    total_epochs = experiment_spec.train.num_epochs

    dm = OCRDataModule(experiment_spec)
    dm.setup(stage='fit')
    ocrnet_model = OCRNetModel(experiment_spec, dm)
    clip_grad = experiment_spec.train.clip_grad_norm
    distributed_strategy = 'auto'
    if len(gpus) > 1:
        distributed_strategy = experiment_spec.train.distributed_strategy
    val_inter = experiment_spec.train.validation_interval

    trainer = Trainer(logger=ptl_loggers,
                      devices=gpus,
                      max_epochs=total_epochs,
                      check_val_every_n_epoch=val_inter,
                      default_root_dir=results_dir,
                      enable_checkpointing=False,
                      strategy=distributed_strategy,
                      accelerator='gpu',
                      num_sanity_val_steps=0,
                      gradient_clip_val=clip_grad)

    trainer.fit(ocrnet_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="OCRNet", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_spec=cfg)


if __name__ == '__main__':
    main()
