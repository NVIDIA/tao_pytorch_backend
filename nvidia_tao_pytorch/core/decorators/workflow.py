# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Status logging decorators."""

from functools import wraps
from omegaconf import OmegaConf
import os

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.utilities import update_results_dir


def monitor_status(name='module name', mode='train'):
    """Status monitoring decorator."""
    def inner(runner):
        @wraps(runner)
        def _func(cfg, **kwargs):

            cfg = update_results_dir(cfg, task=mode)
            os.makedirs(cfg["results_dir"], exist_ok=True)
            results_dir = cfg["results_dir"]

            OmegaConf.save(cfg, os.path.join(results_dir, "experiment.yaml"))

            status_file = os.path.join(results_dir, "status.json")
            status_logging.set_status_logger(
                status_logging.StatusLogger(
                    filename=status_file,
                    verbosity=1,
                    append=True
                )
            )
            s_logger = status_logging.get_status_logger()
            try:
                s_logger.write(
                    status_level=status_logging.Status.STARTED,
                    message=f"Starting {name} {mode}."
                )
                runner(cfg, **kwargs)
                s_logger.write(
                    status_level=status_logging.Status.RUNNING,
                    message=f"{mode.capitalize()} finished successfully."
                )
                if os.getenv("CLOUD_BASED") == "True":
                    s_logger.write(
                        status_level=status_logging.Status.RUNNING,
                        message="Job artifacts in results dir are being uploaded to the cloud"
                    )
            except (KeyboardInterrupt, SystemError):
                s_logger.write(
                    message=f"{mode.capitalize()} was interrupted",
                    verbosity_level=status_logging.Verbosity.INFO,
                    status_level=status_logging.Status.FAILURE
                )
            except Exception as e:
                s_logger.write(
                    message=str(e),
                    status_level=status_logging.Status.FAILURE
                )
                raise e

        return _func
    return inner
