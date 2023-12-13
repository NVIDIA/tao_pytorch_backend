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

"""MMCV Base Tao status logger for segformer """

from collections import OrderedDict
from typing import Dict
import torch
from mmcv.runner.hooks import HOOKS, TextLoggerHook
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from abc import abstractmethod


@HOOKS.register_module()
class BaseTaoTextLoggerHook(TextLoggerHook):
    """ Logger hook in text. """

    def __init__(self, *args, **kwargs):
        """ init """
        self.s_logger = status_logging.get_status_logger()
        super(BaseTaoTextLoggerHook, self).__init__(*args, **kwargs)
        self.monitor_data = {}

    @abstractmethod
    def _status_log(self, log_dict: Dict, runner) -> None:
        """Function to generate dataloaders."""
        raise NotImplementedError(
            "Base Trainer doesn't implement data loader instantiation."
        )

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
