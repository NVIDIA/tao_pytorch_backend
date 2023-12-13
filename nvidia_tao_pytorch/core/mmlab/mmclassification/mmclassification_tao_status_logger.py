# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Original source taken from https://github.com/open-mmlab/mmsegmentation

# Copyright 2019 OpenMMLAB

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Tao status logger for segformer """

import os
from typing import Dict
from mmcv.runner.hooks import HOOKS
from nvidia_tao_pytorch.core.mmlab.common.base_tao_status_logger import BaseTaoTextLoggerHook
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
import json
from time import strftime, gmtime

STATUS_JSON_FILENAME = "status.json"


@HOOKS.register_module()
class MMClsTaoTextLoggerHook(BaseTaoTextLoggerHook):
    """TAO Epoch based runner.

    Overrides  mmcv.runner.epoch_based_runner.EpochBaseRunner to save checkpoints
    without symlinks which requires root access.
    """

    def _status_log(self, log_dict: Dict, runner) -> None:
        """ status_log
        Args:
            log_dict (Dict): Contains the parameters for experiment logging.
            runner (Class): Object of TAO Runner

        """
        self.monitor_data["mode"] = log_dict["mode"]
        if log_dict['mode'] == 'val':
            self.monitor_data["accuracy_top-1"] = log_dict["accuracy_top-1"]
            self.s_logger.kpi = {
                "accuracy_top-1": log_dict["accuracy_top-1"],
            }
        if log_dict['mode'] == 'train':
            running_avg_loss = log_dict["loss"]
            self.monitor_data["epoch"] = log_dict["epoch"]
            self.monitor_data["loss"] = running_avg_loss
            time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)  # Per iter
            time_sec_avg_epoch = len(runner.data_loader) * time_sec_avg
            self.monitor_data["time_per_epoch"] = strftime("%H:%M:%S", gmtime(time_sec_avg_epoch))
            self.monitor_data["lr"] = log_dict["lr"]
            self.s_logger.graphical = {
                "loss": running_avg_loss,
            }
        try:
            self.s_logger.write(
                data=self.monitor_data,
                status_level=status_logging.Status.RUNNING)
        except IOError:
            # We let this pass because we do not want the json file writing to crash the whole job.
            pass
        # Save the json file.
        filename = os.path.join(runner.work_dir, STATUS_JSON_FILENAME)
        try:
            with open(filename, "a+") as f:
                json.dump(self.monitor_data, f)
                f.write('\n')
        except IOError:
            # We let this pass because we do not want the json file writing to crash the whole job.
            pass
