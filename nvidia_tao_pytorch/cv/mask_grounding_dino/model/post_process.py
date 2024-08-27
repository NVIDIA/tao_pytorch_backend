# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

""" Post processing for inference. """

import torch
from torch import nn
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.deformable_detr.utils import box_ops
from nvidia_tao_pytorch.cv.grounding_dino.utils.vl_utils import create_positive_map


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, tokenizer, cat_list, num_select=100, has_mask=True) -> None:
        """Initialize GDINO PostProcess."""
        super().__init__()
        self.num_select = num_select
        self.tokenizer = tokenizer
        self.has_mask = has_mask

        caption = " . ".join(cat_list) + ' .'
        tokenized = self.tokenizer(caption, padding="longest", return_tensors="pt")
        label_list = torch.arange(len(cat_list))
        pos_map = create_positive_map(tokenized, label_list, cat_list, caption)

        self.positive_map = pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, image_names, input_sizes=None):
        """ Perform the computation

        Args:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        img_h, img_w = target_sizes.unbind(1)
        if input_sizes is not None:
            inp_h, inp_w = input_sizes.unbind(1)
        else:
            self.has_mask = False

        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        for label_ind in range(len(pos_maps)):
            if pos_maps[label_ind].sum() != 0:
                pos_maps[label_ind] = pos_maps[label_ind] / pos_maps[label_ind].sum()

        prob_to_label = prob_to_token @ pos_maps.T

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(prob.view(prob.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob.shape[2], rounding_mode='trunc')
        labels = topk_indexes % prob.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        masks = []
        if self.has_mask:
            pred_masks = outputs["pred_masks"].squeeze(2)
            b, _, mh, mw = pred_masks.shape
            pred_masks = torch.gather(pred_masks, 1, topk_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mh, mw).to(torch.int64))
            for i in range(b):
                mask_i = F.interpolate(pred_masks[i: i + 1], size=(mh * 4, mw * 4), mode='bilinear', align_corners=False)
                mask_i = mask_i[:, :, 0:inp_h[i], 0:inp_w[i]]
                mask_i = F.interpolate(mask_i, size=(img_h[i], img_w[i]), mode='bilinear', align_corners=False)
                masks.append(mask_i.sigmoid()[0])

        # and from relative [0, 1] to absolute [0, height] coordinates
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        if self.has_mask:
            results = [{'scores': s, 'labels': l, 'boxes': b, 'masks': m, 'image_names': n, 'image_size': i}
                       for s, l, b, m, n, i in zip(scores, labels, boxes, masks, image_names, target_sizes)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b, 'image_names': n, 'image_size': i}
                       for s, l, b, n, i in zip(scores, labels, boxes, image_names, target_sizes)]
        return results
