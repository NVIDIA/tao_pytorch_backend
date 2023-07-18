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

"""Auto logging for Metric Learning Recognition subtasks."""

import os
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from functools import wraps


def monitor_status(name='Metric Learning Recognition', mode='train'):
    """Status monitoring decorator."""
    def inner(runner):
        @wraps(runner)
        def _func(cfg, **kwargs):

            if cfg[mode]["results_dir"]:
                results_dir = cfg[mode]["results_dir"]
            elif cfg.results_dir:
                results_dir = os.path.join(cfg.results_dir, mode)
            else:
                raise ValueError("You need to set at least one of following fields: results_dir, {mode}.results_dir")
            os.makedirs(results_dir, exist_ok=True)
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
                    status_level=status_logging.Status.SUCCESS,
                    message=f"{mode.capitalize()} finished successfully."
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
