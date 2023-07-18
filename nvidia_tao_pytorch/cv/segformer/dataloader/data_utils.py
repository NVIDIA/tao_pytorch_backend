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

"""Data Utils Module."""

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from nvidia_tao_pytorch.cv.segformer.dataloader.builder import DATASETS
import copy
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import build_from_cfg
from mmcv.utils.parrots_wrapper import DataLoader, PoolDataLoader


def _concat_dataset(cfg, default_args=None):
    """Build :obj:`ConcatDataset by."""
    img_dir = cfg['img_dir']
    ann_dir = cfg.get('ann_dir', None)
    split = cfg.get('split', None)
    num_img_dir = len(img_dir) if isinstance(img_dir, (list, tuple)) else 1
    if ann_dir is not None:
        num_ann_dir = len(ann_dir) if isinstance(ann_dir, (list, tuple)) else 1
    else:
        num_ann_dir = 0
    if split is not None:
        num_split = len(split) if isinstance(split, (list, tuple)) else 1
    else:
        num_split = 0
    if num_img_dir > 1:
        assert num_ann_dir in (num_img_dir, 0)
        assert num_split in (num_img_dir, 0)
    else:
        assert num_split == num_ann_dir or num_ann_dir <= 1
    num_dset = max(num_split, num_img_dir)

    datasets = []
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        if isinstance(img_dir, (list, tuple)):
            data_cfg['img_dir'] = img_dir[i]
        if isinstance(ann_dir, (list, tuple)):
            data_cfg['ann_dir'] = ann_dir[i]
        if isinstance(split, (list, tuple)):
            data_cfg['split'] = split[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


class TargetClass(object):
    """Target class parameters."""

    def __init__(self, name, label_id, train_id=None, color=None, train_name=None):
        """Constructor.

        Args:
            name (str): Name of the target class.
            label_id (str):original label id of every pixel of the mask
            train_id (str): The mapped train id of every pixel in the mask
        Raises:
            ValueError: On invalid input args.
        """
        self.name = name
        self.train_id = train_id
        self.label_id = label_id
        self.color = color
        self.train_name = train_name


def get_train_class_mapping(target_classes):
    """Utility function that returns the mapping of the train id to orig class."""
    train_id_name_mapping = {}
    for target_class in target_classes:
        if target_class.train_id not in train_id_name_mapping.keys():
            train_id_name_mapping[target_class.train_id] = [target_class.name]
        else:
            train_id_name_mapping[target_class.train_id].append(target_class.name)
    return train_id_name_mapping


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.
    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.
    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        """Init Module."""
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE


@DATASETS.register_module()
class RepeatDataset(object):
    """A wrapper of repeated dataset.
    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.
    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        """Init Module."""
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item from original dataset."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """The length is multiplied by ``times``"""
        return self.times * self._ori_len


def is_dist_avail_and_initialized():
    """is dist initialized"""
    is_dist = True
    if not dist.is_available():
        is_dist = False
    else:
        is_dist = dist.is_initialized() or False
    return is_dist


def build_dataset(cfg, default_args=None):
    """Build datasets."""
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg.get('img_dir'), (list, tuple)) or isinstance(
            cfg.get('split', None), (list, tuple)):
        print("Concatenating datasets")

        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     dataloader_type='PoolDataLoader',
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        dataloader_type (str): Type of dataloader. Default: 'PoolDataLoader'
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    assert dataloader_type in (
        'DataLoader',
        'PoolDataLoader'), f'unsupported dataloader {dataloader_type}'

    if dataloader_type == 'PoolDataLoader':
        dataloader = PoolDataLoader
    elif dataloader_type == 'DataLoader':
        dataloader = DataLoader

    data_loader = dataloader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.
    The seed of each worker equals to num_worker * rank + worker_id + user_seed
    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
