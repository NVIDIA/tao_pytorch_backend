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

"""File containing a minimalistic "export manager" for TLT scripts not using PTL trainer/NeMo ExpManager."""


import os
from pathlib import Path

from dataclasses import dataclass

from nvidia_tao_pytorch.core.tlt_logging import logging


@dataclass
class MinimalExpManagerConfig:
    """Minimalistic config enabling setup of log dir and task name."""

    task_name: str = "task"
    # Log dir creation parameters
    explicit_log_dir: str = "./"


def minimal_exp_manager(cfg: MinimalExpManagerConfig) -> Path:
    """Minimalistic experiment manager - sets logging and returns log_dir.

    Args:
        cfg (MinimalExpManagerConfig): Omega config dictionary for Minimal experiment manager.

    Returns:
        log_dir(str): String path to the logs.
    """
    if Path(cfg.explicit_log_dir).exists():
        logging.warning(f"Exp_manager is logging to `{cfg.explicit_log_dir}``, but it already exists.")

    # Shortcut.
    log_dir = Path(cfg.explicit_log_dir)

    # Create the logging directory if it does not exist
    os.makedirs(log_dir, exist_ok=True)

    # Set output log file.
    logging.add_file_handler(os.path.join(log_dir, cfg.task_name + ".log"))

    return log_dir
