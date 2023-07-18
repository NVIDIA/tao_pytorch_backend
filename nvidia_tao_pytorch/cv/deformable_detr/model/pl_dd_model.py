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

""" Main PTL model file for deformable detr. """

import datetime
import os
import json
from typing import Any, Dict

import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from fairscale.optim import OSS

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging

from nvidia_tao_pytorch.cv.action_recognition.utils.common_utils import patch_decrypt_checkpoint
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.od_dataset import CoCoDataMerge
from nvidia_tao_pytorch.cv.deformable_detr.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.deformable_detr.model.matcher import HungarianMatcher
from nvidia_tao_pytorch.cv.deformable_detr.model.criterion import SetCriterion
from nvidia_tao_pytorch.cv.deformable_detr.model.post_process import PostProcess, save_inference_prediction, threshold_predictions
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import match_name_keywords
from nvidia_tao_pytorch.cv.deformable_detr.utils.coco import COCO
from nvidia_tao_pytorch.cv.deformable_detr.utils.coco_eval import CocoEvaluator


# pylint:disable=too-many-ancestors
class DeformableDETRModel(pl.LightningModule):
    """PTL module for Deformable DETR Object Detection Model."""

    def __init__(self, experiment_spec, export=False):
        """Init training for Deformable DETR Model."""
        super().__init__()
        self.experiment_spec = experiment_spec
        self.dataset_config = experiment_spec.dataset
        self.model_config = experiment_spec.model
        self.eval_class_ids = self.dataset_config["eval_class_ids"]
        self.dataset_type = self.dataset_config["dataset_type"]
        if self.dataset_type not in ("serialized", "default"):
            raise ValueError(f"{self.dataset_type} is not supported. Only serialized and default are supported.")

        # init the model
        self._build_model(export)
        self._build_criterion()

        self.status_logging_dict = {}

    def _build_model(self, export):
        """Internal function to build the model."""
        self.model = build_model(experiment_config=self.experiment_spec, export=export)

    def _build_criterion(self):
        """Internal function to build the loss function."""
        self.matcher = HungarianMatcher(cost_class=self.model_config["cls_loss_coef"], cost_bbox=self.model_config["bbox_loss_coef"], cost_giou=self.model_config["giou_loss_coef"])
        self.weight_dict = {'loss_ce': self.model_config["cls_loss_coef"], 'loss_bbox': self.model_config["bbox_loss_coef"], 'loss_giou': self.model_config["giou_loss_coef"]}
        if self.model_config["aux_loss"]:
            aux_weight_dict = {}
            for i in range(self.model_config["dec_layers"] - 1):
                aux_weight_dict.update({f'{k}_{i}': v for k, v in self.weight_dict.items()})
            aux_weight_dict.update({f'{k}_enc': v for k, v in self.weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)
        self.weight_dict = self.weight_dict
        loss_types = self.model_config["loss_types"]  # ['labels', 'boxes']
        self.criterion = SetCriterion(self.dataset_config["num_classes"], self.matcher, loss_types, focal_alpha=self.model_config["focal_alpha"])
        self.box_processors = PostProcess()

    def configure_optimizers(self):
        """Configure optimizers for training."""
        self.train_config = self.experiment_spec.train
        param_dicts = [
            {
                "params":
                    [p for n, p in self.model.named_parameters()
                     if not match_name_keywords(n, self.model_config["backbone_names"]) and
                     not match_name_keywords(n, self.model_config["linear_proj_names"]) and
                     p.requires_grad],
                "lr": self.train_config['optim']['lr'],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if match_name_keywords(n, self.model_config["backbone_names"]) and p.requires_grad],
                "lr": self.train_config['optim']['lr_backbone'],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if match_name_keywords(n, self.model_config["linear_proj_names"]) and p.requires_grad],
                "lr": self.train_config['optim']['lr'] * self.train_config['optim']['lr_linear_proj_mult'],
            }
        ]

        if self.train_config.optim.optimizer == 'SGD':
            base_optimizer = torch.optim.SGD(params=param_dicts,
                                             lr=self.train_config.optim.lr,
                                             momentum=self.train_config.optim.momentum,
                                             weight_decay=self.train_config.optim.weight_decay)
        elif self.train_config.optim.optimizer == 'AdamW':
            base_optimizer = torch.optim.AdamW(params=param_dicts,
                                               lr=self.train_config.optim.lr,
                                               weight_decay=self.train_config.optim.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.train_config.optim.optimizer} is not implemented")

        if self.train_config.distributed_strategy == "ddp_sharded":
            # Override force_broadcast_object=False in PTL
            optim = OSS(params=base_optimizer.param_groups, optim=type(base_optimizer), force_broadcast_object=True, **base_optimizer.defaults)
        else:
            optim = base_optimizer

        optim_dict = {}
        optim_dict["optimizer"] = optim
        scheduler_type = self.train_config['optim']['lr_scheduler']

        if scheduler_type == "MultiStep":
            lr_scheduler = MultiStepLR(optimizer=optim,
                                       milestones=self.train_config['optim']["lr_steps"],
                                       gamma=self.train_config['optim']["lr_decay"],
                                       verbose=True)
        elif scheduler_type == "StepLR":
            lr_scheduler = StepLR(optimizer=optim,
                                  step_size=self.train_config['optim']["lr_step_size"],
                                  gamma=self.train_config['optim']["lr_decay"],
                                  verbose=True)
        else:
            raise NotImplementedError("LR Scheduler {} is not implemented".format(scheduler_type))

        optim_dict["lr_scheduler"] = lr_scheduler
        optim_dict['monitor'] = self.train_config['optim']['monitor_name']
        return optim_dict

    def training_step(self, batch, batch_idx):
        """Training step."""
        data, targets, _ = batch
        batch_size = data.shape[0]
        outputs = self.model(data)
        # loss
        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        self.log("train_loss", losses, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_class_error", loss_dict['class_error'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("train_loss_ce", loss_dict['loss_ce'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("train_loss_bbox", loss_dict['loss_bbox'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("train_loss_giou", loss_dict['loss_giou'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)

        return {'loss': losses}

    def training_epoch_end(self, training_step_outputs):
        """Log Training metrics to status.json"""
        average_train_loss = 0.0
        for out in training_step_outputs:
            average_train_loss += out['loss'].item()
        average_train_loss /= len(training_step_outputs)

        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train and Val metrics generated.",
            status_level=status_logging.Status.RUNNING
        )
        training_step_outputs.clear()

    def on_validation_epoch_start(self) -> None:
        """
        Validation epoch start.
        Reset coco evaluator for each epoch.
        """
        if self.dataset_type == "serialized":
            # Load from scratch since COCO object is not instantiated for SerializedDatasetFromList
            coco_lists = []
            for source in self.dataset_config["val_data_sources"]:
                with open(source["json_file"], "r") as f:
                    tmp = json.load(f)
                coco_lists.append(COCO(tmp))
            coco = COCO(CoCoDataMerge(coco_lists))
            self.val_coco_evaluator = CocoEvaluator(coco, iou_types=['bbox'], eval_class_ids=self.eval_class_ids)
        else:
            self.val_coco_evaluator = CocoEvaluator(self.trainer.datamodule.val_dataset.coco, iou_types=['bbox'], eval_class_ids=self.eval_class_ids)

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        data, targets, image_names = batch
        outputs = self.model(data)
        batch_size = data.shape[0]

        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.box_processors(outputs, orig_target_sizes, image_names)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        self.val_coco_evaluator.update(res)

        self.log("val_loss", losses, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_class_error", loss_dict['class_error'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("val_loss_ce", loss_dict['loss_ce'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("val_loss_bbox", loss_dict['loss_bbox'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("val_loss_giou", loss_dict['loss_giou'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        return losses

    def validation_epoch_end(self, outputs):
        """
        Validation epoch end.
        Compute mAP at the end of epoch.
        """
        self.val_coco_evaluator.synchronize_between_processes()
        self.val_coco_evaluator.overall_accumulate()
        self.val_coco_evaluator.overall_summarize(is_print=False)
        mAP = self.val_coco_evaluator.coco_eval['bbox'].stats[0]
        mAP50 = self.val_coco_evaluator.coco_eval['bbox'].stats[1]
        if self.trainer.is_global_zero:
            print("\n Validation mAP : {}\n".format(mAP))
            print("\n Validation mAP50 : {}\n".format(mAP50))
        self.log("val_mAP", mAP, rank_zero_only=True, sync_dist=True)
        self.log("val_mAP50", mAP50, rank_zero_only=True, sync_dist=True)
        self.status_logging_dict["val_mAP"] = str(mAP)
        self.status_logging_dict["val_mAP50"] = str(mAP50)

        average_val_loss = 0.0
        for out in outputs:
            average_val_loss += out.item()
        average_val_loss /= len(outputs)

        self.status_logging_dict["val_loss"] = average_val_loss
        outputs.clear()

    def on_test_epoch_start(self) -> None:
        """
        Test epoch start.
        Reset coco evaluator at start.
        """
        if self.dataset_type == "serialized":
            # Load from scratch since COCO object is not instantiated for SerializedDatasetFromList
            with open(self.dataset_config["test_data_sources"]["json_file"], "r") as f:
                tmp = json.load(f)
            coco = COCO(tmp)
            self.test_coco_evaluator = CocoEvaluator(coco, iou_types=['bbox'], eval_class_ids=self.eval_class_ids)
        else:
            self.test_coco_evaluator = CocoEvaluator(self.trainer.datamodule.test_dataset.coco, iou_types=['bbox'], eval_class_ids=self.eval_class_ids)

    def test_step(self, batch, batch_idx):
        """Test step. Evaluate."""
        data, targets, image_names = batch
        outputs = self.model(data)
        batch_size = data.shape[0]
        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.box_processors(outputs, orig_target_sizes, image_names)
        if self.experiment_spec.evaluate.conf_threshold > 0:
            filtered_res = threshold_predictions(results, self.experiment_spec.evaluate.conf_threshold)
        else:
            filtered_res = results
        res = {target['image_id'].item(): output for target, output in zip(targets, filtered_res)}
        self.test_coco_evaluator.update(res)

        self.log("test_loss", losses, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("test_class_error", loss_dict['class_error'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("test_loss_ce", loss_dict['loss_ce'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("test_loss_bbox", loss_dict['loss_bbox'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log("test_loss_giou", loss_dict['loss_giou'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)

    def test_epoch_end(self, outputs):
        """
        Test epoch end.
        Compute mAP at the end of epoch.
        """
        self.test_coco_evaluator.synchronize_between_processes()
        self.test_coco_evaluator.overall_accumulate()
        self.test_coco_evaluator.overall_summarize(is_print=True)
        mAP = self.test_coco_evaluator.coco_eval['bbox'].stats[0]
        mAP50 = self.test_coco_evaluator.coco_eval['bbox'].stats[1]
        self.log("test_mAP", mAP, rank_zero_only=True)
        self.log("test_mAP50", mAP50, rank_zero_only=True)

        # Log the evaluation results to a file
        log_file = os.path.join(self.experiment_spec.results_dir, 'log_eval_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
        logger = common_utils.create_logger(log_file, rank=0)
        if self.trainer.is_global_zero:
            logger.info('**********************Start logging Evaluation Results **********************')
            logger.info('*************** mAP *****************')
            logger.info('mAP : %2.2f' % mAP)
            logger.info('*************** mAP50 *****************')
            logger.info('mAP50 : %2.2f' % mAP50)
        self.status_logging_dict["test_mAP"] = str(mAP)
        self.status_logging_dict["test_mAP50"] = str(mAP50)
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Evaluation metrics generated.",
            status_level=status_logging.Status.RUNNING
        )
        outputs.clear()

    def predict_step(self, batch, batch_idx):
        """Predict step. Inference."""
        data, targets, image_names = batch
        outputs = self.model(data)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        pred_results = self.box_processors(outputs, orig_target_sizes, image_names)
        return pred_results

    @rank_zero_only
    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """
        Predict batch end.
        Save the result inferences at the end of batch.
        """
        output_dir = self.experiment_spec.results_dir
        label_map = self.trainer.datamodule.pred_dataset.label_map
        color_map = self.experiment_spec.inference.color_map
        conf_threshold = self.experiment_spec.inference.conf_threshold
        is_internal = self.experiment_spec.inference.is_internal
        save_inference_prediction(outputs, output_dir, conf_threshold, label_map, color_map, is_internal)

    def forward(self, x):
        """Forward of the deformable detr model."""
        outputs = self.model(x)
        return outputs

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Encrpyt the checkpoint. The encryption is done in TLTCheckpointConnector."""
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Decrpyt the checkpoint."""
        if checkpoint.get("state_dict_encrypted", False):
            # Retrieve encryption key from TLTPyTorchCookbook.
            key = TLTPyTorchCookbook.get_passphrase()
            if key is None:
                raise PermissionError("Cannot access model state dict without the encryption key")
            checkpoint = patch_decrypt_checkpoint(checkpoint, key)
