# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MAL/blob/main/LICENSE
"""Custom LightningDataModule for MAL."""

import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from nvidia_tao_pytorch.cv.mal.datasets.coco import BoxLabelCOCO, InstSegCOCO, InstSegCOCOwithBoxInput
from nvidia_tao_pytorch.cv.mal.datasets.data_aug import data_aug_pipelines, custom_collate_fn
logger = logging.getLogger(__name__)


class WSISDataModule(pl.LightningDataModule):
    """Weakly supervised instance segmentation data module."""

    def __init__(self,
                 num_workers,
                 load_train=False,
                 load_val=False,
                 cfg=None):
        """Initialize train/val dataset and dataloader.

        Args:
            num_workers (int): Total number of workers
            load_train (bool): Whether to load training set
            load_val (bool): Whether to load validation set
            cfg (OmegaConf): Hydra config
        """
        super().__init__()
        self.cfg = cfg
        self.num_workers = num_workers
        self.train_transform = data_aug_pipelines['train'](cfg)
        self.test_transform = data_aug_pipelines['test'](cfg)
        assert self.cfg.dataset.type == 'coco', 'only COCO format is supported.'
        self._train_data_loader = None
        self._val_data_loader = None
        self.box_inputs = None

        if load_train:
            logger.info("Loading train set...")
            dataset = BoxLabelCOCO(
                self.cfg.dataset.train_ann_path,
                self.cfg.dataset.train_img_dir,
                min_obj_size=self.cfg.dataset.min_obj_size,
                max_obj_size=self.cfg.dataset.max_obj_size,
                transform=self.train_transform, cfg=cfg)
            data_loader = DataLoader(
                dataset, batch_size=self.cfg.train.batch_size, shuffle=True,
                num_workers=self.num_workers)
            self._train_data_loader = data_loader
            logger.info("Train set is loaded successfully.")

        if load_val:
            logger.info("Loading validation set...")
            build_dataset = InstSegCOCOwithBoxInput if self.box_inputs else InstSegCOCO

            dataset = build_dataset(
                self.cfg.dataset.val_ann_path,
                self.cfg.dataset.val_img_dir,
                min_obj_size=0,
                max_obj_size=1e9,
                load_mask=self.cfg.dataset.load_mask,
                transform=self.test_transform,
                box_inputs=self.box_inputs
            )
            data_loader = DataLoader(
                dataset, collate_fn=custom_collate_fn,
                batch_size=self.cfg.train.batch_size, num_workers=self.num_workers)
            self._val_data_loader = data_loader
            logger.info("Validation set is loaded successfully.")

    def train_dataloader(self):
        """Set train dataloader."""
        return self._train_data_loader

    def val_dataloader(self):
        """Set val dataloader."""
        return self._val_data_loader
