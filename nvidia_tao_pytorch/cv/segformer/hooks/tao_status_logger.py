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
from collections import OrderedDict
from typing import Dict
import torch
from mmcv.runner.hooks import HOOKS, TextLoggerHook
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
import json
from time import strftime, gmtime

STATUS_JSON_FILENAME = "status.json"


@HOOKS.register_module()
class TaoTextLoggerHook(TextLoggerHook):
    """ Logger hook in text. """

    def __init__(self, *args, **kwargs):
        """ init """
        self.s_logger = status_logging.get_status_logger()
        super(TaoTextLoggerHook, self).__init__(*args, **kwargs)

    def _status_log(self, log_dict: Dict, runner) -> None:
        """ Status Logging
        Args:
            log_dict (Dict): Dictionary with logging values
            runner (class): MMLab trainer instance
        """
        monitor_data = {}
        monitor_data["mode"] = log_dict["mode"]
        if self.by_epoch:
            monitor_data["cur_epoch"] = log_dict["epoch"]
        else:
            monitor_data["cur_iter"] = log_dict["iter"]
        if log_dict['mode'] == 'val':
            if int(os.environ['LOCAL_RANK']) == 0:  # In multi-GPU setting
                monitor_data["mIoU"] = log_dict["mIoU"]
                monitor_data["mAcc"] = log_dict["mAcc"]
                self.s_logger.kpi = {
                    "Mean IOU": log_dict["mIoU"],
                    "mAcc": log_dict["mAcc"]
                }
        if log_dict['mode'] == 'train':
            running_avg_loss = log_dict["loss"]
            monitor_data["loss"] = running_avg_loss
            time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)  # Per iter
            monitor_data["time_per_iter"] = strftime("%H:%M:%S", gmtime(time_sec_avg))
            monitor_data["train_accuracy"] = log_dict["decode.acc_seg"]
            self.s_logger.graphical = {
                "loss": running_avg_loss,
                "train_accuracy": log_dict["decode.acc_seg"]
            }
        try:
            self.s_logger.write(
                data=monitor_data,
                status_level=status_logging.Status.RUNNING)
        except IOError:
            # We let this pass because we do not want the json file writing to crash the whole job.
            pass
        # Save the json file.
        filename = os.path.join(runner.work_dir, STATUS_JSON_FILENAME)
        try:
            with open(filename, "a+") as f:
                json.dump(monitor_data, f)
                f.write('\n')
        except IOError:
            # We let this pass because we do not want the json file writing to crash the whole job.
            pass

    def log(self, runner) -> OrderedDict:
        """ log runner """
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = OrderedDict(
            mode=self.get_mode(runner),
            epoch=self.get_epoch(runner),
            iter=cur_iter)

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        if 'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)

        log_dict = dict(log_dict, **runner.log_buffer.output)  # type: ignore

        self._log_info(log_dict, runner)
        self._dump_log(log_dict, runner)
        self._status_log(log_dict, runner)
        return log_dict
