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

"""Trainer Class for Classification."""

from abc import abstractmethod
from tqdm import tqdm
from typing import Optional
import dataclasses
from omegaconf import OmegaConf
import time
from datetime import timedelta
from collections import defaultdict

from mmengine import Config
from mmpretrain import FeatureExtractor, get_model

import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np

from nvidia_tao_pytorch.core.mmlab.mmclassification.logistic_regression_dataset import LRDataset
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


class LogisticRegressionTrainer(object):
    """MMClassification Trainer."""

    def __init__(self, train_cfg=None, updated_config=None, status_logger=None):
        """Initialize LogisticRegressionTrainer.

        Args:
            train_cfg (Any): Configuration for training. Defaults to None.
            updated_config (Any): Updated configuration. Defaults to None.
            status_logger (StatusLogger): Logger for tracking status.
        """
        self.train_cfg = dataclasses.asdict(OmegaConf.to_object(train_cfg))
        self.classifier = None
        self.train_features = None
        self.train_labels = None
        self.dataloader = None
        self.val_dataloader = None
        self.model = None
        self.updated_config = updated_config
        self.status_logger = status_logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.build_dataloader()
        self.get_model()
        self.build_classifier()
        self.get_features()

    def build_dataloader(self):
        """Builds dataloaders for training and validation."""
        train_dataset = LRDataset(data_prefix=self.train_cfg["dataset"]["data"]["train"]["data_prefix"],
                                  classes=self.train_cfg["dataset"]["data"]["train"]["classes"])
        self.dataloader = DataLoader(train_dataset, batch_size=self.train_cfg["dataset"]["data"]["samples_per_gpu"],
                                     shuffle=True, num_workers=4, pin_memory=True)
        val_dataset = LRDataset(data_prefix=self.train_cfg["dataset"]["data"]["val"]["data_prefix"],
                                classes=self.train_cfg["dataset"]["data"]["val"]["classes"])
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.train_cfg["dataset"]["data"]["samples_per_gpu"],
                                         shuffle=False, num_workers=4, pin_memory=True)

    def build_classifier(self, c: Optional[float] = None):
        """Builds the logistic regression classifier.
        Args:
            c (float): Inverse of regularization strength
        """
        lr_head_config = self.train_cfg['model']['head']['lr_head']
        self.classifier = LogisticRegression(random_state=0,
                                             C=c if c else lr_head_config['C'],
                                             max_iter=lr_head_config['max_iter'],
                                             verbose=1,
                                             class_weight=lr_head_config['class_weight'],
                                             multi_class='multinomial')

    def get_model(self):
        """Get Model."""
        model_config = Config(self.updated_config)
        self.model = get_model(model_config)

    def get_features(self):
        """Extracts features from the dataset."""
        all_features = []
        all_labels = []
        with torch.no_grad():
            extractor = FeatureExtractor(self.model, device=self.device)
            for images, labels in tqdm(self.dataloader):
                features = extractor(images)
                features_cat = torch.cat([torch.unsqueeze(f, dim=0) for f in features], dim=0)
                all_features.append(features_cat.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        self.train_features = np.concatenate(all_features)
        self.train_labels = np.concatenate(all_labels)

    def get_validation_metrics(self):
        """Calculates validation accuracy and loss."""
        all_features = []
        all_labels = []
        with torch.no_grad():
            total, val_loss = 0, 0
            extractor = FeatureExtractor(self.model, device=self.device)
            for images, labels in tqdm(self.val_dataloader):
                features = extractor(images)
                features = torch.cat([torch.unsqueeze(f, dim=0) for f in features], dim=0)
                predictions = self.classifier.predict(features.cpu().numpy())
                correct_pred = np.sum((labels.cpu().numpy() == predictions).astype(float))
                total += correct_pred
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            val_accuracy = total / (len(self.val_dataloader.dataset))
        self.val_features = np.concatenate(all_features)
        self.val_labels = np.concatenate(all_labels)
        pred_probs = self.classifier.predict_proba(self.val_features)
        val_loss = log_loss(self.val_labels, pred_probs)
        print(f"[validation accuracy = {val_accuracy:.4f}]")
        return val_accuracy, val_loss

    def hyperparams_search(self):
        """Grid search for hyperparameters. Currently only support C of LogisticRegression head"""
        results = defaultdict(dict)
        criteria = self.train_cfg["model"]["head"]["lr_head"]["criteria"]
        cs_tune = self.train_cfg["model"]["head"]["lr_head"]["cs_tune"]

        assert criteria in ["accuracy", "loss"], "Only suport accuracy and loss criteria"

        if not cs_tune:
            raise ValueError("cs_tune must be a list of float for hpo.")

        self.status_logger.write(message="********************** Start logging for Hyperparameter C tuning **********************.")
        for c_val in cs_tune:
            print(f"Hyperparameters search: C={c_val}")
            # train classifier for specific c_val
            self.build_classifier(c_val)
            self.classifier.fit(self.train_features, self.train_labels)

            val_accuracy, val_loss = self.get_validation_metrics()

            # status logging
            self.status_logger.kpi = {"C": c_val, "val_loss": val_loss, "val_acc": val_accuracy}
            self.status_logger.write(
                message="Hyperparameters search val metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

            results["accuracy"][str(c_val)] = val_accuracy
            results["loss"][str(c_val)] = val_loss

        # get the best c by criteria
        metrics = results[criteria]
        best_c = min(metrics, key=metrics.get) if criteria == 'loss' else max(metrics, key=metrics.get)

        return float(best_c)

    @abstractmethod
    def fit(self):
        """Fit Function."""
        if self.train_cfg["model"]["head"]["lr_head"]["hpo"]:
            c_val = self.hyperparams_search()
            self.status_logger.kpi = {"Best C": c_val}
            self.status_logger.write(message="Finish Hyperparameter C tuning.")

            self.build_classifier(c=c_val)

        start_time = time.time()
        self.classifier.fit(self.train_features, self.train_labels)
        end_time = time.time()
        time_per_iter = (end_time - start_time) / self.classifier.n_iter_[0]
        if time_per_iter == float("inf"):
            time_per_iter = 0.0

        pred_probs = self.classifier.predict_proba(self.train_features)
        train_loss = log_loss(self.train_labels, pred_probs)
        val_accuracy, val_loss = self.get_validation_metrics()

        # Status logging
        logging_dict = {}
        logging_dict["cur_iter"] = 1
        logging_dict["max_iters"] = float(self.classifier.n_iter_[0])
        logging_dict["time_per_iter"] = str(timedelta(seconds=time_per_iter))
        logging_dict["eta"] = "0:00:00"

        self.status_logger.data = logging_dict
        self.status_logger.kpi = {"loss": train_loss, "val_loss": val_loss, "val_acc": val_accuracy}
        self.status_logger.write(
            message="Train and Val metrics generated.",
            data=logging_dict,
            status_level=status_logging.Status.RUNNING
        )
