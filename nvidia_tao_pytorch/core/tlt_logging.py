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

"""Common utilities useful for logging."""

import logging as _logging
import os

from random import randint
from omegaconf import OmegaConf


class MessageFormatter(_logging.Formatter):
    """Formatter that supports colored logs."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        _logging.DEBUG: grey + fmt + reset,
        _logging.INFO: grey + fmt + reset,
        _logging.WARNING: yellow + fmt + reset,
        _logging.ERROR: red + fmt + reset,
        _logging.CRITICAL: bold_red + fmt + reset
    }

    def format(self, record):
        """Format the log message."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = _logging.Formatter(log_fmt)
        return formatter.format(record)


logger = _logging.getLogger('TAO Toolkit')
logger.setLevel(_logging.DEBUG)
ch = _logging.StreamHandler()
ch.setLevel(_logging.DEBUG)
ch.setFormatter(MessageFormatter())
logger.addHandler(ch)
logging = logger


def obfuscate_logs(cfg):
    """Function obfuscates encryption key if exposed/present in args.

    Args:
        cfg(OmegaConf.DictConfig): Function to obfuscate key from the log.
    """
    # First obfuscate what is being shown as configuration.
    config = OmegaConf.to_container(cfg)
    if "encryption_key" in config.keys():
        config["encryption_key"] = '*' * randint(3, 10)

    # Show the experiment configuration.
    logging.info(f'Experiment configuration:\n{OmegaConf.to_yaml(config)}')


def remove_logs(log_dir):
    """Function removes the cmd-args and git-info log files from log_dir.

    Args:
        log_dir(str): Path to the results directory containing the logs.
    """
    log_files = ["cmd-args.log", "git-info.log"]
    for log in log_files:
        logfile = os.path.join(log_dir, log)
        if os.path.exists(logfile):
            os.remove(logfile)
