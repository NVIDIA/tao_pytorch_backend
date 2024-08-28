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

# Copyright (c) OpenMMLab. All rights reserved.

"""MMCV Base Tao status logger for segformer """

import os
from typing import Dict, Optional, Sequence, Union
import json
from time import strftime, gmtime

from mmengine.registry import HOOKS
from mmengine.hooks import LoggerHook

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


DATA_BATCH = Optional[Union[dict, tuple, list]]
SUFFIX_TYPE = Union[Sequence[str], str]
STATUS_JSON_FILENAME = "status.json"


@HOOKS.register_module()
class TaoTextLoggerHook(LoggerHook):
    """ Logger hook in text. """

    def __init__(self, *args, **kwargs):
        """ init """
        self.s_logger = status_logging.get_status_logger()
        super(TaoTextLoggerHook, self).__init__(*args, **kwargs)
        self.monitor_data = {}

    def after_train_iter(self, runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Record logs after training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        # Print experiment name every n iterations.
        if self.every_n_train_iters(
                runner, self.interval_exp_name) or (self.end_of_epoch(
                    runner.train_dataloader, batch_idx)):
            exp_info = f'Exp name: {runner.experiment_name}'
            runner.logger.info(exp_info)
        if self.every_n_inner_iters(batch_idx, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        elif (self.end_of_epoch(runner.train_dataloader,
                                batch_idx) and (not self.ignore_last or len(runner.train_dataloader) <= self.interval)):
            # `runner.max_iters` may not be divisible by `self.interval`. if
            # `self.ignore_last==True`, the log of remaining iterations will
            # be recorded (Epoch [4][1000/1007], the logs of 998-1007
            # iterations will be recorded).
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        else:
            return

        # log_str looks like:
        # Epoch(train) [1][500/801]  lr: 1.0000e-03  eta: 0:04:24  time: 0.1406  data_time: 0.0273  memory: 2392  loss: 0.6949

        self.monitor_data["epoch"] = int(runner.log_processor._get_epoch(runner, "train"))
        # cur_iter = runner.log_processor._get_iter(runner, batch_idx)

        lr = log_str.split("lr: ")[1].split(" ")[0]
        self.monitor_data["lr"] = float(lr)

        running_avg_loss = float(log_str.split("loss: ")[1].split(" ")[0])
        self.monitor_data["loss"] = running_avg_loss

        dataloader_len = runner.log_processor._get_dataloader_size(runner, "train")
        time = float(log_str.split("time: ")[1].split(" ")[0])
        # Bug for reference for time: https://github.com/open-mmlab/mmdetection/issues/10795
        time_sec_avg = time  # Per iter
        time_sec_avg_epoch = dataloader_len * time_sec_avg
        self.monitor_data["time_per_epoch"] = strftime("%H:%M:%S", gmtime(time_sec_avg_epoch))

        self.monitor_data["eta"] = log_str.split(" eta: ")[1].split(" ")[0]

        self.s_logger.graphical = {
            "loss": running_avg_loss}

        runner.logger.info(log_str)
        runner.visualizer.add_scalars(
            tag, step=runner.iter + 1, file_path=self.json_log_path)
        self.write_to_status(runner)

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        """Record logs after validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the validation
                loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
                Defaults to None.
            outputs (sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            _, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'val')
            runner.logger.info(log_str)

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """Record logs after testing iteration.

        Args:
            runner (Runner): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            _, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'test')
            runner.logger.info(log_str)
            self.monitor_data["phase"] = "test"
            acc = float(log_str.split("accuracy/top1: ")[1].split(" ")[0])
            self.monitor_data["accuracy_top-1"] = acc
            self.s_logger.kpi = {
                "accuracy_top-1": acc,
            }
            self.write_to_status(runner)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        runner.logger.info(log_str)
        self.monitor_data["phase"] = "val"
        acc = float(log_str.split("accuracy/top1: ")[1].split(" ")[0])
        self.monitor_data["accuracy_top-1"] = acc
        self.s_logger.kpi = {
            "accuracy_top-1": acc,
        }
        self.write_to_status(runner)
        if self.log_metric_by_epoch:
            # Accessing the epoch attribute of the runner will trigger
            # the construction of the train_loop. Therefore, to avoid
            # triggering the construction of the train_loop during
            # validation, check before accessing the epoch.
            if (isinstance(runner._train_loop, dict) or runner._train_loop is None):
                epoch = 0
            else:
                epoch = runner.epoch
            runner.visualizer.add_scalars(
                tag, step=epoch, file_path=self.json_log_path)
        else:
            if (isinstance(runner._train_loop, dict) or runner._train_loop is None):
                iter = 0  # noqa pylint: disable=W0622
            else:
                iter = runner.iter
            runner.visualizer.add_scalars(
                tag, step=iter, file_path=self.json_log_path)

    def write_to_status(self, runner):
        """Write the monitor data to the status logger."""
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
