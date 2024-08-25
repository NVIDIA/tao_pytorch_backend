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

"""This script responsible for copying file specs to the folder pointed by the user."""

from os import makedirs, listdir
from os.path import abspath, dirname, exists, join
import shutil

from omegaconf import MISSING
from dataclasses import dataclass

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.exp_manager import minimal_exp_manager, MinimalExpManagerConfig
from nvidia_tao_pytorch.core.loggers.api_logging import Status, StatusLogger
from nvidia_tao_pytorch.core.tlt_logging import logging, obfuscate_logs

# Usage example - for ASR:
# ==============
"""
python download_specs \
    exp_manager.explicit_log_dir=/results/speech_to_text/download_specs/ \
    source_data_dir=/home/tkornuta/workspace/tlt-pytorch/asr/experiment_specs \
    workflow=asr \
    target_data_dir=/specs/asr \
"""


@dataclass
class DefaultConfig:
    """This is a structured config for ASR dataset processing."""

    # Minimalistic experiment manager.
    exp_manager: MinimalExpManagerConfig = MinimalExpManagerConfig(task_name="download_specs")

    # Input folder where the default configs are.
    source_data_dir: str = MISSING

    # Output folder path
    target_data_dir: str = MISSING

    # Name of the worflow.
    workflow: str = MISSING


spec_path = dirname(abspath(__file__))


@hydra_runner(config_path=spec_path, config_name="download_specs", schema=DefaultConfig)
def main(cfg: DefaultConfig) -> None:
    """Script to run dataset convert.

    Args:
        cfg (OmegaConf.DictConf): Hydra parsed config object.
    """
    # Obfuscate logs.
    obfuscate_logs(cfg)

    # Initialize export manager (simple logging).
    log_dir = minimal_exp_manager(cfg.exp_manager)
    status_logger = StatusLogger(
        filename=join(log_dir, "status.json"),
        append=True
    )
    status_logger.write(
        message=f"Downloading default specs for {cfg.workflow}",
        status_level=Status.STARTED
    )

    if exists(cfg.target_data_dir):
        if listdir(cfg.target_data_dir):
            raise FileExistsError(f"The target directory `{cfg.target_data_dir}` is not empty!\n"
                                  "In order to avoid overriding the existing spec files please point to a different folder.")
    else:
        # Create a new folder.
        makedirs(cfg.target_data_dir, exist_ok=True)

    # Copy files from source to target.
    names = [item for item in listdir(cfg.source_data_dir) if item.endswith("yaml")]
    for name in names:
        srcname = join(cfg.source_data_dir, name)
        dstname = join(cfg.target_data_dir, name)
        shutil.copy2(srcname, dstname)

    # Inform where the logs are.
    logging.info(f"Default specification files for {cfg.workflow} downloaded to '{cfg.target_data_dir}'")
    status_message = f"Default specification files for {cfg.workflow} downloaded."\
        f"List of files: {names}"
    status_logger.write(
        message=status_message,
        status_level=Status.RUNNING
    )
    logging.info(f"Experiment logs saved to '{log_dir}'")


if __name__ == "__main__":
    main()
