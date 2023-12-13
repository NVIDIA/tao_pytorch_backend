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

"""PNG mask dataset."""

import os
from nvidia_tao_pytorch.cv.segformer.dataloader.data_utils import TargetClass, get_train_class_mapping
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import numpy as np


class SFDataModule(object):
    """ DataModule for Segformer."""

    def __init__(self, dataset_config, num_gpus, seed, logger, phase="train", input_height=512, input_width=512):
        """ DataModule Initialization
        Args:
            dataset_config (OmegaConfig): dataset configuration
            num_gpus (int): num of GPUs
            seed (int): random init seed
            logger (class): logger object
            phase (str): phase of task
            input_height (int): Input height of the model
            input_width (int): Input width of the model
        """
        self.phase = phase
        self.input_height = input_height
        self.input_width = input_width
        self.dataset_config = dataset_config
        self.samples_per_gpu = self.dataset_config["batch_size"]
        self.dataloader = self.dataset_config["dataloader"]
        self.shuffle = self.dataset_config["shuffle"]
        self.num_gpus = num_gpus
        self.seed = seed
        self.input_type = self.dataset_config["input_type"]
        # Paths to the dataset
        self.data_root = self.dataset_config["data_root"]
        if self.phase == "train":
            self.repeat_times = self.dataset_config["repeat_data_times"]
            self.train_img_dirs = self.dataset_config["train_dataset"]["img_dir"]
            self.train_ann_dirs = self.dataset_config["train_dataset"]["ann_dir"]
            self.val_img_dir = self.dataset_config["val_dataset"]["img_dir"]
            self.val_ann_dir = self.dataset_config["val_dataset"]["ann_dir"]
            assert (type(self.train_img_dirs) == ListConfig), "Image Directories should be list of directories."
            assert (type(self.train_ann_dirs) == ListConfig), "Image annotation directories should be a list."
            assert (type(self.val_img_dir) == str), "Currently Segformer supports only 1 validation directory."
            assert (type(self.val_ann_dir) == str), "Currently Segformer supports only 1 validation directory."
            # Setting up pipeline
            self.train_pipeline = self.dataset_config["train_dataset"]["pipeline"]
            self.train_pipeline["CollectKeys"] = ["img", "gt_semantic_seg"]
            self.val_pipeline = self.dataset_config["val_dataset"]["pipeline"]
            self.train_pipeline["img_norm_cfg"] = self.dataset_config["img_norm_cfg"]
            self.val_pipeline["img_norm_cfg"] = self.dataset_config["img_norm_cfg"]
            if self.dataset_config["seg_map_suffix"] and self.dataset_config["img_suffix"]:
                self.img_suffix = self.dataset_config["img_suffix"]
                # This allows provide suffixes that are not .png. For e.g., cityscapes
                self.seg_map_suffix = self.dataset_config["seg_map_suffix"]
            else:
                self.img_suffix, self.seg_map_suffix = self.get_extensions(self.train_img_dirs[0], self.train_ann_dirs[0])
        else:
            # Eval / Inference
            self.test_img_dir = self.dataset_config["test_dataset"]["img_dir"]
            assert (type(self.test_img_dir) == str), "Currently Segformer supports only 1 test directory."
            # It is not mandatory to provide the mask path for inference
            if self.phase == "eval":
                try:
                    self.test_ann_dir = self.dataset_config["test_dataset"]["ann_dir"]
                except Exception as e:
                    raise ValueError("Test Annotation dir should be provided for evaluation {}".format(e))
            else:
                self.test_ann_dir = self.test_img_dir
            self.test_pipeline = self.dataset_config["test_dataset"]["pipeline"]
            self.test_pipeline["img_norm_cfg"] = self.dataset_config["img_norm_cfg"]
            self.img_suffix, self.seg_map_suffix = self.get_extensions(self.test_img_dir, self.test_ann_dir)
        self.train_dataset = None
        self.val_dataset = None
        self.workers_per_gpu = self.dataset_config["workers_per_gpu"]
        self.num_workers = None
        self.sampler_train = None
        self.sampler_test = None
        self.sampler_val = None
        self.batch_size = self.num_gpus * self.samples_per_gpu
        self.num_workers = self.num_gpus * self.workers_per_gpu
        self.logger = logger
        self.target_classes = self.build_target_class_list()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.log_interval = 50

        # This needs to be implemented
        self.PALETTE, self.CLASSES, self.label_map, self.id_color_map = self.build_palette()
        self.default_args = {"classes": tuple(self.CLASSES), "palette": self.PALETTE, "label_map": self.label_map,
                             "img_suffix": self.img_suffix, "seg_map_suffix": self.seg_map_suffix, "id_color_map": self.id_color_map,
                             "input_type": self.input_type, "logger": self.logger}
        self.target_classes_train_mapping = get_train_class_mapping(self.target_classes)
        self.num_classes = self.get_num_unique_train_ids()

    def setup(self):
        """ Function to initlaize the samplers and datasets. """
        if self.phase == "train":
            train_dic = {}
            train_dic["type"] = "SFDataset"
            train_dic["data_root"] = self.data_root
            train_dic["img_dir"] = OmegaConf.to_container(self.train_img_dirs)
            train_dic["ann_dir"] = OmegaConf.to_container(self.train_ann_dirs)
            updated_train_pipeline = self.build_train_pipeline(self.train_pipeline)
            train_dic["pipeline"] = updated_train_pipeline
            train_data = {}
            train_data["type"] = "RepeatDataset"
            train_data["times"] = self.repeat_times
            train_data["dataset"] = train_dic
            train_data["img_suffix"] = self.img_suffix
            train_data["seg_map_suffix"] = self.seg_map_suffix

            # Val Dictionary
            val_dic = {}
            val_dic["type"] = "SFDataset"
            val_dic["data_root"] = self.data_root
            val_dic["img_dir"] = self.val_img_dir
            val_dic["ann_dir"] = self.val_ann_dir
            updated_val_pipeline = self.build_test_pipeline(self.val_pipeline)
            val_dic["pipeline"] = updated_val_pipeline
            val_data = val_dic

            self.train_data = train_data
            self.val_data = val_data
        else:
            # Test Dictionary
            test_dic = {}
            test_dic["type"] = "SFDataset"
            test_dic["data_root"] = self.data_root
            test_dic["img_dir"] = self.test_img_dir
            test_dic["ann_dir"] = self.test_ann_dir
            updated_test_pipeline = self.build_test_pipeline(self.test_pipeline)
            test_dic["pipeline"] = updated_test_pipeline
            test_data = test_dic
            self.test_data = test_data

    def get_extensions(self, img_dir, ann_dir):
        """ Function to automatically get the image and mask extensions. """
        img_suffix = os.listdir(img_dir)[0].split(".")[-1]
        seg_map_suffix = os.listdir(ann_dir)[0].split(".")[-1]
        return img_suffix, seg_map_suffix

    def build_target_class_list(self):
        """Build a list of TargetClasses based on proto."""
        target_classes = []
        orig_class_label_id_map = {}
        color_mapping = {}
        for target_class in self.dataset_config.palette:
            orig_class_label_id_map[target_class.seg_class] = target_class.label_id
            color_mapping[target_class.seg_class] = target_class.rgb
        class_label_id_calibrated_map = orig_class_label_id_map.copy()
        for target_class in self.dataset_config.palette:
            label_name = target_class.seg_class
            train_name = target_class.mapping_class

            class_label_id_calibrated_map[label_name] = orig_class_label_id_map[train_name]

        train_ids = sorted(list(set(class_label_id_calibrated_map.values())))
        train_id_calibrated_map = {}
        for idx, tr_id in enumerate(train_ids):
            train_id_calibrated_map[tr_id] = idx

        class_train_id_calibrated_map = {}
        for label_name, train_id in class_label_id_calibrated_map.items():
            class_train_id_calibrated_map[label_name] = train_id_calibrated_map[train_id]

        for target_class in self.dataset_config.palette:
            target_classes.append(
                TargetClass(target_class.seg_class, label_id=target_class.label_id,
                            train_id=class_train_id_calibrated_map[target_class.seg_class],
                            color=color_mapping[target_class.mapping_class],
                            train_name=target_class.mapping_class
                            ))
        for target_class in target_classes:
            self.logger.info("Label Id {}: Train Id {}".format(target_class.label_id, target_class.train_id))

        return target_classes

    def build_palette(self):
        """Build palette, classes and label_map."""
        label_map = {}
        classes_color = {}
        id_color_map = {}
        classes = []
        palette = []
        for target_class in self.target_classes:
            label_map[target_class.label_id] = target_class.train_id
            if target_class.train_name not in classes_color.keys():
                classes_color[target_class.train_id] = (target_class.train_name, target_class.color)
                id_color_map[target_class.train_id] = target_class.color
        keylist = list(classes_color.keys())
        keylist.sort()
        for train_id in keylist:
            classes.append(classes_color[train_id][0])
            palette.append(classes_color[train_id][1])

        return palette, classes, label_map, id_color_map

    def get_num_unique_train_ids(self):
        """Return the final number classes used for training.

        Arguments:
            target_classes: The target classes object that contain the train_id and
            label_id.
        Returns:
            Number of classes to be segmented.
        """
        train_ids = [target.train_id for target in self.target_classes]
        train_ids = np.array(train_ids)
        train_ids_unique = np.unique(train_ids)
        return len(train_ids_unique)

    def build_train_pipeline(self, train_pipeline):
        """ Function to Build Train Pipeline.
        Args:
            train_pipeline (Dict): dictionary having the parameters for training augmentation

        """
        augmentation_config = train_pipeline["augmentation_config"]
        if not augmentation_config["resize"]["img_scale"]:
            img_scale_min = min(self.input_height, self.input_width)
            img_scale_max = 1024 if img_scale_min < 1024 else 2048
            augmentation_config["resize"]["img_scale"] = [img_scale_min, img_scale_max]
        updated_train_pipeline = [dict(type="LoadImageFromFile"),
                                  dict(type="LoadAnnotations", input_type=self.input_type),
                                  dict(type="Resize", img_scale=tuple(augmentation_config["resize"]["img_scale"]), ratio_range=tuple(augmentation_config["resize"]["ratio_range"])),
                                  dict(type="RandomCrop", crop_size=tuple([self.input_height, self.input_width]), cat_max_ratio=augmentation_config["random_crop"]["cat_max_ratio"]),
                                  dict(type="RandomFlip", prob=augmentation_config["random_flip"]["prob"]),
                                  dict(type='PhotoMetricDistortion'),
                                  dict(type="Normalize", mean=train_pipeline["img_norm_cfg"]["mean"],
                                       std=train_pipeline["img_norm_cfg"]["std"],
                                       to_rgb=train_pipeline["img_norm_cfg"]["to_rgb"]),
                                  dict(type="Pad", size=(self.input_height, self.input_width), pad_val=0, seg_pad_val=255),
                                  dict(type='DefaultFormatBundle'),
                                  dict(type='Collect', keys=train_pipeline["CollectKeys"]),
                                  ]

        return updated_train_pipeline

    def build_test_pipeline(self, test_pipeline):
        """ Function to Build Test Pipeline.
        Args:
            test_pipeline (Dict): dictionary having the parameters for testing parameters

        """
        augmentation_config = test_pipeline["augmentation_config"]
        keep_ar = augmentation_config["resize"]["keep_ratio"]
        if not test_pipeline["multi_scale"]:
            test_pipeline["multi_scale"] = [self.input_height, 2048]

        transforms = [
            dict(type='Resize', keep_ratio=keep_ar),
            dict(type='RandomFlip'),
            dict(type="Normalize", mean=test_pipeline["img_norm_cfg"]["mean"],
                 std=test_pipeline["img_norm_cfg"]["std"], to_rgb=test_pipeline["img_norm_cfg"]["to_rgb"]),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
        updated_test_pipeline = [dict(type="LoadImageFromFile"),
                                 dict(type='MultiScaleFlipAug', img_scale=tuple(test_pipeline["multi_scale"]),
                                      flip=False,
                                      transforms=transforms)]

        return updated_test_pipeline
