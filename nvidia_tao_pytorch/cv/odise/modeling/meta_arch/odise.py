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

import logging
import numpy as np
import operator
from collections import OrderedDict
from typing import Any, Mapping
import diffdist.functional as diff_dist
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.memory import retry_if_cuda_oom

from nvidia_tao_pytorch.cv.odise.data.build import get_openseg_labels, prompt_labels
from nvidia_tao_pytorch.cv.odise.modeling.meta_arch.clip import ClipAdapter, build_clip_text_embed
from nvidia_tao_pytorch.cv.odise.modeling.meta_arch.helper import (
    ensemble_logits_with_labels,
    ensemble_logits_with_labels_legacy,
    MaskPooling, to_tuple
)
from nvidia_tao_pytorch.cv.odise.modeling.meta_arch.maskformer_model import MaskFormer
from nvidia_tao_pytorch.cv.odise.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MLP,
    MultiScaleMaskedTransformerDecoder,
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def _concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_world_batch_sizes(batch_size: int, device):
    batch_size = torch.as_tensor([batch_size], dtype=torch.long, device=device)
    global_batch_sizes = _concat_all_gather(batch_size)
    return global_batch_sizes


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        for _ in range(comm.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, padded_tensor, async_op=False)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


def dist_collect(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    Use diff_dist to get gradient
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        for _ in range(comm.get_world_size())
    ]
    tensors_gather = diff_dist.all_gather(tensors_gather, padded_tensor)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


class ODISE(MaskFormer):
    def get_mask_pred(self, mask_embed, text_embed, labels):
        """
        mask_embed: torch.Size([1, 100, 768]) = B, Q, proj_dim
        text_embed: torch.Size([254, 768]) = B, text_dim
        """
        logit_per_mask = (
            torch.einsum(
                "bqc,nc->bqn", F.normalize(mask_embed.float(), dim=-1),
                F.normalize(text_embed, dim=-1)
            )
            * self.logit_scale
        )
        if not self.is_inference:
            logit_per_mask = ensemble_logits_with_labels_legacy(logit_per_mask, labels)
        else:
            logit_per_mask = ensemble_logits_with_labels(logit_per_mask, self.chunk_sizes_pyt, self.chunk_start_idx_pyt)
        return logit_per_mask

    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return destination


class CategoryODISE(ODISE):
    def __init__(
        self,
        category_head=None,
        precision='fp32',
        is_inference=False,
        alpha=0.4,
        beta=0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.category_head = category_head
        self.logit_scale = torch.clamp(self.backbone.clip_model.logit_scale.exp(), max=100)
        self.mask_pooling = MaskPooling()
        self.alpha = alpha
        self.beta = beta
        self.train_labels = category_head.labels
        self.is_fp16 = precision == 'fp16'
        self.category_overlapping_mask = None
        self.chunk_sizes_pyt = None
        self.chunk_start_idx_pyt = None
        self.is_inference = is_inference

    def cal_pred_logits(self, outputs):
        # [B, Q, C]
        mask_embed = outputs["mask_embed"]
        # [K, C]
        text_embed = outputs["text_embed"]
        # [1, C]
        null_embed = outputs["null_embed"]

        labels = outputs["labels"]

        mask_embed = F.normalize(mask_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        logit_scale = outputs["logit_scale"]

        # [B, Q, K]
        pred = logit_scale * (mask_embed @ text_embed.t())

        if not self.is_inference:
            pred = ensemble_logits_with_labels_legacy(pred, labels, ensemble_method="max")
        else:
            pred = ensemble_logits_with_labels(pred, self.chunk_sizes_pyt, self.chunk_start_idx_pyt, ensemble_method="max")

        null_embed = F.normalize(null_embed, dim=-1)
        null_pred = logit_scale * (mask_embed @ null_embed.t())

        # [B, Q, K+1]
        pred = torch.cat([pred, null_pred], dim=-1)

        return pred

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the
                        values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        denormalized_images = ImageList.from_tensors(
            [x["image"].to(self.device) / 255.0 for x in batched_inputs]
        )

        if self.is_fp16:
            self.backbone.half()
            features = self.backbone(images.tensor.half())
        else:
            features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        outputs["images"] = denormalized_images.tensor

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            if self.category_head is not None:
                category_head_outputs = self.category_head(outputs, targets)
                outputs.update(category_head_outputs)
                # inplace change pred_logits
                outputs["pred_logits"] = self.cal_pred_logits(outputs)
                if "aux_outputs" in outputs:
                    for aux_outputs in outputs["aux_outputs"]:
                        aux_outputs.update(category_head_outputs)
                        # inplace change pred_logits
                        aux_outputs["pred_logits"] = self.cal_pred_logits(aux_outputs)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:

            # get text_embeddings
            outputs.update(self.category_head(outputs))
            if self.is_fp16:
                outputs['text_embed'] = outputs['text_embed'].float()
                outputs['null_embed'] = outputs['null_embed'].float()
            outputs["pred_logits"] = self.cal_pred_logits(outputs)

            mask_pred_results = outputs["pred_masks"]
            mask_cls_results = outputs["pred_logits"] # in-vocab 
            clip_feature = features["clip_vis_dense"] # torch.Size([1, 1536, 32, 43])
            mask_for_pooling = F.interpolate(mask_pred_results, size=clip_feature.shape[-2:], # torch.Size([1, 100, 32, 43])  
                                             mode='bilinear', align_corners=False)
            if self.is_fp16:
                mask_for_pooling = mask_for_pooling.half()
            if "convnext" in self.backbone.model_name.lower():
                pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling)
                pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
            else:
                raise NotImplementedError
            outputs['mask_embed'] = pooled_clip_feature

            outputs["mask_pred_open_logits"] = self.get_mask_pred(
                pooled_clip_feature, outputs['text_embed'], outputs['labels'])
            # [B, Q, K]
            in_vocab_cls_results = mask_cls_results[..., :-1]
            out_vocab_cls_results = outputs["mask_pred_open_logits"]

            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_results = in_vocab_cls_results.softmax(-1)

            cls_logits_seen = (
                (in_vocab_cls_results ** (1 - self.alpha) * out_vocab_cls_probs**self.alpha).log()
                * self.category_overlapping_mask
            )
            cls_logits_unseen = (
                (in_vocab_cls_results ** (1 - self.beta) * out_vocab_cls_probs**self.beta).log()
                * (1 - self.category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen

            # This is used to filtering void predictions.
            is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob], dim=-1)
            mask_cls_results = torch.log(mask_cls_probs + 1e-8)

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["instances"] = instance_r

            return processed_results


class CaptionODISE(ODISE):
    def __init__(
        self,
        word_head=None,
        grounding_criterion=None,
        precision='fp32',
        is_inference=False,
        alpha=0.1,
        beta=0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.word_head = word_head
        self.train_labels = word_head.labels
        self.grounding_criterion = grounding_criterion
        self.mask_pooling = MaskPooling()
        self.logit_scale = torch.clamp(self.backbone.clip_model.logit_scale.exp(), max=100)
        self.alpha = alpha
        self.beta = beta
        self.category_overlapping_mask = None
        self.chunk_sizes_pyt = None
        self.chunk_start_idx_pyt = None
        self.is_fp16 = precision == 'fp16'
        self.is_inference = is_inference

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )

            if targets_per_image.has("original_gt_classes"):
                # "labels" maybe binary, store original labels in as well
                new_targets[-1]["original_labels"] = targets_per_image.original_gt_classes

        return new_targets

    def prepare_pseudo_targets(self, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for _ in range(len(images)):
            # pad gt
            padded_masks = torch.zeros((0, h_pad, w_pad), dtype=torch.bool, device=images.device)
            new_targets.append(
                {
                    "labels": torch.zeros(0, dtype=torch.long, device=images.device),
                    "masks": padded_masks,
                }
            )

        return new_targets

    @property
    def binary_classification(self):
        return self.sem_seg_head.num_classes == 1

    def cal_pred_open_logits(self, outputs):
        # [B, Q, C]
        mask_embed = outputs["mask_embed"]
        # [K, C]
        text_embed = outputs["text_embed"]

        labels = outputs["labels"]

        mask_embed = F.normalize(mask_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        logit_scale = outputs["logit_scale"]

        # [B, Q, K]
        pred = logit_scale * (mask_embed @ text_embed.t())

        pred = ensemble_logits_with_labels_legacy(pred, labels, ensemble_method="max")

        return pred

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the
                        values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        denormalized_images = ImageList.from_tensors(
            [x["image"].to(self.device) / 255.0 for x in batched_inputs]
        )

        if self.is_fp16:
            self.backbone.half()
            features = self.backbone(images.tensor.half())
        else:
            features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        outputs["images"] = denormalized_images.tensor

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

                if self.binary_classification:
                    # NOTE: convert to binary classification target
                    for i in range(len(gt_instances)):
                        gt_instances[i].original_gt_classes = gt_instances[i].gt_classes.clone()
                        gt_instances[i].gt_classes = torch.zeros_like(gt_instances[i].gt_classes)

                targets = self.prepare_targets(gt_instances, images)
                has_anno = True
            else:
                targets = self.prepare_pseudo_targets(images)
                has_anno = False

            if "captions" in batched_inputs[0]:
                gt_captions = [x["captions"] for x in batched_inputs]
                targets = self.word_head.prepare_targets(gt_captions, targets)

            if self.word_head is not None:
                word_head_outputs = self.word_head(outputs, targets)
                outputs.update(word_head_outputs)
                if "aux_outputs" in outputs:
                    for aux_outputs in outputs["aux_outputs"]:
                        aux_outputs.update(word_head_outputs)

            # CLIP head needs output to prepare targets
            # disable for now
            # targets = self.clip_head.prepare_targets(outputs, targets)

            if self.criterion is not None:
                # bipartite matching-based loss
                losses = self.criterion(outputs, targets)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)

                # multiple by 0 to avoid gradient but make sure the param is used
                if not has_anno:
                    for k in list(losses.keys()):
                        losses[k] *= 0
            else:
                losses = {}

            if self.grounding_criterion is not None:
                grounding_losses = self.grounding_criterion(outputs, targets)
                losses.update(grounding_losses)

            return losses
        else:

            mask_cls_results = outputs["pred_logits"]
            assert mask_cls_results.shape[-1] == 2
            mask_pred_results = outputs["pred_masks"]

            if self.word_head is not None:
                outputs.update(self.word_head(outputs))
            outputs["pred_open_logits"] = self.cal_pred_open_logits(outputs)

            clip_feature = features["clip_vis_dense"] # torch.Size([1, 1536, 32, 43])
            mask_for_pooling = F.interpolate(mask_pred_results, size=clip_feature.shape[-2:], # torch.Size([1, 100, 32, 43])
                                             mode='bilinear', align_corners=False)
            if self.is_fp16:
                mask_for_pooling = mask_for_pooling.half()
            if "convnext" in self.backbone.model_name.lower():
                pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling)
                pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature) # (torch.Size([1, 100, 768]))
            else:
                raise NotImplementedError
            outputs['mask_embed'] = pooled_clip_feature

            outputs["mask_pred_open_logits"] = self.get_mask_pred(
                pooled_clip_feature, outputs['text_embed'], outputs['labels'])

            # [B, Q, K]
            in_vocab_cls_results = outputs["pred_open_logits"]
            out_vocab_cls_results = outputs["mask_pred_open_logits"]

            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_results = in_vocab_cls_results.softmax(-1)

            cls_logits_seen = (
                (in_vocab_cls_results ** (1 - self.alpha) * out_vocab_cls_probs**self.alpha).log()
                * self.category_overlapping_mask
            )
            cls_logits_unseen = (
                (in_vocab_cls_results ** (1 - self.beta) * out_vocab_cls_probs**self.beta).log()
                * (1 - self.category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen

            # This is used to filtering void predictions.
            is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob], dim=-1)

            mask_cls_results = torch.log(mask_cls_probs + 1e-8)
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["instances"] = instance_r

            return processed_results


class ODISEMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    def __init__(
        self,
        *,
        precision='fp32',
        class_embed=None,
        mask_embed=None,
        post_mask_embed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.mask_classification
        self.is_fp16 = precision == 'fp16'
        if class_embed is not None:
            self.class_embed = class_embed
        if mask_embed is not None:
            self.mask_embed = mask_embed
        if post_mask_embed is not None:
            assert mask_embed is None
        self.post_mask_embed = post_mask_embed

    def forward(self, x, mask_features, mask=None, *, inputs_dict=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(
                self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]
            )

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_extra_results = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, extra_results = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0], inputs_dict=inputs_dict
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_extra_results.append(extra_results)

        if self.is_fp16:
            query_embed = query_embed.half()

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            if self.is_fp16:
                self.transformer_self_attention_layers[i] = self.transformer_self_attention_layers[i].half()
                output = output.half()

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            if self.is_fp16:
                self.transformer_ffn_layers[i] = self.transformer_ffn_layers[i].half()
            output = self.transformer_ffn_layers[i](output)
            output = output.float()

            outputs_class, outputs_mask, attn_mask, extra_results = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                inputs_dict=inputs_dict,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_extra_results.append(extra_results)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
        }

        # adding extra_results to out and out["aux_outputs"]
        for k in predictions_extra_results[-1].keys():
            out[k] = predictions_extra_results[-1][k]
            for i in range(len(predictions_extra_results) - 1):
                out["aux_outputs"][i][k] = predictions_extra_results[i][k]

        return out

    def forward_prediction_heads(
        self, output, mask_features, attn_mask_target_size, *, inputs_dict=None
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)

        extra_results = dict()

        mask_embed_results = self.mask_embed(decoder_output)
        if isinstance(mask_embed_results, dict):
            mask_embed = mask_embed_results.pop("mask_embed")
            extra_results.update(mask_embed_results)
        # BC
        else:
            mask_embed = mask_embed_results

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        if self.post_mask_embed is not None:
            post_mask_embed_results = self.post_mask_embed(
                decoder_output, mask_embed, mask_features, outputs_class, outputs_mask
            )

            if "outputs_mask" in post_mask_embed_results:
                outputs_mask = post_mask_embed_results.pop("outputs_mask")

            extra_results.update(post_mask_embed_results)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(
            outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False
        )
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend,
        # while ``False`` values will be unchanged.
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, extra_results


class MaskGroundingCriterion(nn.Module):
    def __init__(
        self,
        collect_mode: str = "concat",
        loss_weight=1.0,
    ):
        super().__init__()

        self.collect_mode = collect_mode
        self.loss_weight = loss_weight

        if collect_mode == "diff":
            self.collect_func = dist_collect
        elif collect_mode == "concat":
            self.collect_func = concat_all_gather
        elif collect_mode is None:
            self.collect_func = lambda x: x
        else:
            raise ValueError(f"collect_mode {collect_mode} is not supported")

    def extra_repr(self) -> str:
        return f"collect_mode={self.collect_mode}, \n" f"loss_weight={self.loss_weight} \n"

    def forward(self, outputs, targets):

        losses = {}
        losses.update(self.get_loss(outputs, targets))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                l_dict = self.get_loss(aux_outputs, targets)
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def get_loss(self, outputs, targets):

        logit_scale = outputs["logit_scale"]

        rank = comm.get_rank() if self.collect_mode is not None else 0

        # normalized embeds
        # [B, Q, C]
        mask_embed = outputs["mask_embed"]
        # [B, K, C]
        word_embed = outputs["word_embed"]
        # [B, K]
        word_valid_mask = torch.stack([t["word_valid_mask"] for t in targets], dim=0)

        mask_embed = F.normalize(mask_embed, dim=-1)
        word_embed = F.normalize(word_embed, dim=-1)

        batch_size, num_queries, embed_dim = mask_embed.shape
        assert batch_size == word_embed.shape[0], f"{batch_size} != {word_embed.shape[0]}"
        assert embed_dim == word_embed.shape[2], f"{embed_dim} != {word_embed.shape[2]}"
        num_words = word_embed.shape[1]

        # [B*Q, C]
        mask_embed = mask_embed.reshape(batch_size * num_queries, embed_dim)
        # [B*K, C]
        word_embed = word_embed.reshape(batch_size * num_words, embed_dim)

        if self.collect_mode is not None and comm.get_world_size() > 1:
            global_batch_sizes = get_world_batch_sizes(batch_size, device=mask_embed.device)
            global_batch_size = global_batch_sizes.sum().item()
        else:
            global_batch_sizes = None
            global_batch_size = batch_size

        # [W*B*Q, B*K]
        sim_global_mask_word = self.collect_func(mask_embed) @ word_embed.t() * logit_scale

        # [W*B, Q, B, K]
        sim_global_mask_word = sim_global_mask_word.view(
            global_batch_size, num_queries, batch_size, num_words
        )

        # [W*B, B]
        sim_global_img_txt = (
            (sim_global_mask_word.softmax(dim=1) * sim_global_mask_word).sum(dim=1).mean(dim=-1)
        )

        # [B*Q, W*B*K]
        sim_mask_global_word = mask_embed @ self.collect_func(word_embed).t() * logit_scale

        # [B, Q, W*B, K]
        sim_mask_global_word = sim_mask_global_word.view(
            batch_size, num_queries, global_batch_size, num_words
        )

        # [B, W*B]
        sim_img_global_txt = (
            (sim_mask_global_word.softmax(dim=1) * sim_mask_global_word).sum(dim=1).mean(dim=-1)
        )

        if global_batch_sizes is None:
            # get label globally
            labels = (
                torch.arange(batch_size, dtype=torch.long, device=mask_embed.device)
                + batch_size * rank
            )
        else:
            # get label globally and dynamically
            labels = (
                torch.arange(batch_size, dtype=torch.long, device=mask_embed.device)
                + global_batch_sizes[:rank].sum()
            )

        # [B]
        valid_mask = word_valid_mask.any(dim=-1)
        # [W*B]
        global_valid_mask = self.collect_func(valid_mask)

        # [WxB, B] -> [B, WXB] -> [B]
        loss_global_img_txt = F.cross_entropy(sim_global_img_txt.t(), labels, reduction="none")
        loss_global_img_txt = (loss_global_img_txt * valid_mask).mean()

        # [B, WXB] -> [B]
        loss_img_global_txt = F.cross_entropy(
            sim_img_global_txt, labels, weight=global_valid_mask.float()
        )
        if not torch.isfinite(loss_img_global_txt).all():
            # TODO: find reason. Not using vaid mask if NaN as temporary solution
            loss_img_global_txt = F.cross_entropy(sim_img_global_txt, labels)

        loss = 0.5 * (loss_global_img_txt + loss_img_global_txt)

        return {"loss_mask_word": loss * self.loss_weight}


class PseudoClassEmbed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # predict as foreground only
        fg_logits = torch.ones((*x.shape[:-1], self.num_classes), dtype=x.dtype, device=x.device)
        bg_logits = torch.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
        logits = torch.cat([fg_logits, bg_logits], dim=-1)
        return logits


class PooledMaskEmbed(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mask_dim,
        projection_dim,
        temperature=0.07,
    ):
        super().__init__()
        self.pool_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.mask_embed = nn.Sequential(
            nn.LayerNorm(mask_dim), MLP(mask_dim, hidden_dim, projection_dim, 3)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.mask_pooling = MaskPooling()

    def forward(self, decoder_output, input_mask_embed, mask_features, pred_logits, pred_masks):
        """
        Args:
            decoder_output: [B, Q, C]
            input_mask_embed: [B, Q, C]
            mask_features: [B, C, H, W]
            pred_logits: [B, Q, K+1]
            pred_masks: [B, Q, H, W]
        """
        mask_pooled_x = self.mask_pooling(mask_features, pred_masks)
        # mask_pooled_x = mask_pooled_results["mask_pooled_features"]
        # outputs_mask = mask_pooled_results.get("outputs_mask", None)
        mask_pooled_x = self.pool_proj(mask_pooled_x)
        mask_pooled_x += decoder_output
        mask_embed = self.mask_embed(mask_pooled_x)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        output = {
            "mask_embed": mask_embed,
            "mask_pooled_features": mask_pooled_x,
            "logit_scale": logit_scale,
        }
        # if outputs_mask is not None:
        #     output["outputs_mask"] = outputs_mask
        return output


class WordEmbed(nn.Module):
    def __init__(
        self,
        projection_dim,
        clip_model_name="ViT-L-14",
        word_dropout=0.0,
        word_tags="noun_phrase",
        num_words=8,
        prompt="photo",
        pretrained="laion2b_s29b_b131k_ft_soup",
        precision='fp32',
        labels=None,
    ):
        super().__init__()
        self.labels = labels
        self.clip_model_name = clip_model_name
        self.clip = ClipAdapter(name=self.clip_model_name, normalize=False, pretrained=pretrained, precision=precision)

        if projection_dim < 0:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(self.clip.dim_latent, projection_dim)

        self.test_labels = None
        self._test_text_embed_dict = OrderedDict()

        import nltk

        if comm.get_local_rank() == 0:
            nltk.download("popular", quiet=True)
            nltk.download("universal_tagset", quiet=True)
        comm.synchronize()
        self.nltk = nltk

        self.word_dropout = word_dropout
        self.word_tags = word_tags
        self.num_words = num_words
        self.prompt = prompt

    def extra_repr(self) -> str:
        return (
            f"clip_model_name={self.clip_model_name},\n"
            f"word_dropout={self.word_dropout},\n"
            f"word_tags={self.word_tags},\n"
            f"num_words={self.num_words}"
        )

    @property
    def device(self):
        return self.clip.device

    def _open_state_dict(self):
        return {"test_labels": self.test_labels}

    @torch.no_grad()
    def build_text_embed(self, labels, verbose=False):
        return build_clip_text_embed(
            clip_model_name=self.clip.clip,
            labels=labels,
            verbose=verbose,
        )

    def get_and_cache_test_text_embed(self, labels):
        if isinstance(labels, list):
            labels = to_tuple(labels)
        if labels not in self._test_text_embed_dict:
            text_embed = self.build_text_embed(labels, verbose=True)
            if len(self._test_text_embed_dict) > 3:
                # pop the first element, only caching 3 elements
                self._test_text_embed_dict.pop(list(self._test_text_embed_dict.keys())[0])
            self._test_text_embed_dict[labels] = text_embed.cpu()
        else:
            text_embed = self._test_text_embed_dict[labels].to(self.device)
        return text_embed

    def get_tag(self, caption, tags):
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        for (word, pos) in self.nltk.pos_tag(self.nltk.word_tokenize(caption), tagset="universal"):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
        return ret

    def _get_phrase(self, caption, with_preposition):
        if with_preposition:
            # Taken from Su Nam Kim Paper...
            grammar = r"""
                NBAR:
                    {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                NP:
                    {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                    {<NBAR>} # If pattern is not found, just a single NBAR is ok
            """
        else:
            # Taken from Su Nam Kim Paper...
            grammar = r"""
                NBAR:
                    {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                NP:
                    {<NBAR>} # If pattern is not found, just a single NBAR is ok
            """
        tokenized = self.nltk.word_tokenize(caption)
        chunker = self.nltk.RegexpParser(grammar)

        chunked = chunker.parse(self.nltk.pos_tag(tokenized))
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if isinstance(subtree, self.nltk.Tree):
                current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk

    def get_noun_phrase(self, caption):
        noun_phrase = []
        noun_phrase.extend(self._get_phrase(caption, with_preposition=False))
        noun_phrase.extend(self._get_phrase(caption, with_preposition=True))

        return list(set(noun_phrase))

    def prepare_targets(self, captions, targets):

        if targets is None:
            targets = [{} for _ in range(len(captions))]

        for caption, target in zip(captions, targets):
            caption = np.random.choice(caption)
            if self.word_tags == "noun_phrase":
                words = self.get_noun_phrase(caption)
            elif "noun_phrase" in self.word_tags:
                words = []
                words.extend(self.get_noun_phrase(caption))
                words.extend(self.get_tag(caption, tuple(set(self.word_tags) - set("noun_phrase"))))
                words = list(set(words))
            else:
                words = self.get_tag(caption, self.word_tags)
            if not len(words):
                words = [""]
            # drop with probability
            words_after_drop = [w for w in words if np.random.rand() > self.word_dropout]
            if len(words_after_drop) == 0:
                # Fall back to no drop if all words are dropped
                words_after_drop = words
            words = np.random.choice(words_after_drop, size=self.num_words).tolist()
            target["words"] = words

            valid_mask = [len(w) > 0 for w in words]
            valid_mask = torch.tensor(valid_mask, device=self.device, dtype=torch.bool)
            target["word_valid_mask"] = valid_mask

        return targets

    def forward(self, outputs, targets=None):
        if self.training:
            words = [x["words"] for x in targets]

            words = prompt_labels(words, self.prompt)

            word_embed = self.build_text_embed(words)
            # [B, K, C]
            word_embed = torch.stack(word_embed.split([len(w) for w in words]), dim=0)

            word_embed = self.text_proj(word_embed)

            return {"word_embed": word_embed}
        else:
            assert targets is None
            assert self.test_labels is not None
            labels = self.test_labels

            labels = prompt_labels(list(labels), self.prompt)

            text_embed = self.get_and_cache_test_text_embed(labels)

            text_embed = self.text_proj(text_embed)
            return {"text_embed": text_embed, "labels": labels}


class CategoryEmbed(nn.Module):
    def __init__(
        self,
        labels,
        projection_dim,
        clip_model_name="convnext_large_d_320",
        prompt=None,
        pretrained=None,
        precision='fp32'
    ):
        super().__init__()
        self.labels = labels
        self.precision = precision
        self.clip_model_name = clip_model_name
        self.clip = ClipAdapter(
            name=self.clip_model_name, normalize=False, pretrained=pretrained, precision=precision)

        if projection_dim < 0:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(self.clip.dim_latent, projection_dim)

        self.register_buffer(
            "text_embed", self.build_text_embed(prompt_labels(labels, prompt), verbose=True), False
        )
        self.null_embed = nn.Parameter(self.build_text_embed(""))

        self.prompt = prompt

        self.test_labels = None
        self._test_text_embed_dict = dict()

    def extra_repr(self) -> str:
        return f"clip_model_name={self.clip_model_name},\n"

    @property
    def device(self):
        return self.clip.device

    @torch.no_grad()
    def build_text_embed(self, labels, verbose=False):
        return build_clip_text_embed(
            clip_model_name=self.clip.clip,
            labels=labels,
            verbose=verbose,
        )

    def get_and_cache_test_text_embed(self, labels):
        if isinstance(labels, list):
            labels = to_tuple(labels)
        text_embed = self._test_text_embed_dict.get(labels, None)
        if text_embed is None:
            text_embed = self.build_text_embed(labels, verbose=True)
            self._test_text_embed_dict[labels] = text_embed

        return text_embed


    def forward(self, outputs, targets=None):
        if self.training:

            text_embed = self.text_proj(self.text_embed) # (254, 768)
            null_embed = self.text_proj(self.null_embed)

            return {"text_embed": text_embed, "null_embed": null_embed, "labels": self.labels}

        else:
            assert targets is None
            assert self.test_labels is not None
            labels = self.test_labels
            text_embed = self.get_and_cache_test_text_embed(prompt_labels(labels, self.prompt))

            text_embed = self.text_proj(text_embed)
            null_embed = self.text_proj(self.null_embed)

            return {"text_embed": text_embed, "null_embed": null_embed, "labels": labels}
