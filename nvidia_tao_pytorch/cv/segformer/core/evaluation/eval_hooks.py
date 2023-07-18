# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Original source taken from https://github.com/open-mmlab/mmskeleton

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

"""Eval Hooks Module."""

import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=False, **eval_kwargs):
        """Init Module."""
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from nvidia_tao_pytorch.cv.segformer.inference.inferencer import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from nvidia_tao_pytorch.cv.segformer.inference.inferencer import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Call evaluate function of dataset."""
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 **eval_kwargs):
        """Init Module."""
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from nvidia_tao_pytorch.cv.segformer.inference.inferencer import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from nvidia_tao_pytorch.cv.segformer.inference.inferencer import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
