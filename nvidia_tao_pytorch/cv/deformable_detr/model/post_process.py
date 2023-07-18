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

""" Post processing for inference. """

import torch
from torch import nn
import os
from PIL import Image, ImageDraw

from nvidia_tao_pytorch.cv.deformable_detr.utils import box_ops


def get_key(label_map, val):
    """get_key for class label."""
    for label in label_map:
        if label['id'] == val:
            return label['name']
    return None


def check_key(my_dict, key):
    """check_key for classes."""
    return bool(key in my_dict.keys())


def save_inference_prediction(predictions, output_dir, conf_threshold, label_map, color_map, is_internal=False):
    """Save the annotated images and label file to the output directory.

    Args:
        predictions (List): List of predictions from the model.
        output_dir (str) : Output directory to save predictions.
        conf_threshold (float) : Confidence Score Threshold value.
        label_map(Dict): Dictonary for the class lables.
        color_map(Dict): Dictonary for the color mapping to annotate the bounding box per class.
        is_internal(Bool) : To save the inference results in format of output_dir/sequence/image_name.
    """
    for pred in predictions:

        image_name = pred['image_names']
        image_size = pred['image_size']
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        assert pred_boxes.shape[0] == pred_labels.shape[0] == pred_scores.shape[0]

        path_list = image_name.split(os.sep)
        basename, extension = os.path.splitext(path_list[-1])
        if is_internal:
            folder_name = path_list[-3]

            output_label_root = os.path.join(output_dir, folder_name, 'labels')
            output_label_name = os.path.join(output_label_root, basename + '.txt')

            output_annotate_root = os.path.join(output_dir, folder_name, 'images_annotated')
            output_image_name = os.path.join(output_annotate_root, basename + extension)
        else:
            output_label_root = os.path.join(output_dir, 'labels')
            output_label_name = os.path.join(output_label_root, basename + '.txt')

            output_annotate_root = os.path.join(output_dir, 'images_annotated')
            output_image_name = os.path.join(output_annotate_root, basename + extension)

        if not os.path.exists(output_label_root):
            os.makedirs(output_label_root)

        if not os.path.exists(output_annotate_root):
            os.makedirs(output_annotate_root)

        pil_input = Image.open(image_name)
        pil_input = pil_input.resize((image_size[1], image_size[0]))
        im1 = ImageDraw.Draw(pil_input)

        with open(output_label_name, 'w') as f:
            pred_boxes = pred_boxes.tolist()
            scores = pred_scores.tolist()
            labels = pred_labels.tolist()
            for k, box in enumerate(pred_boxes):
                class_key = get_key(label_map, labels[k])
                if class_key is None:
                    continue
                else:
                    class_name = class_key

                # Conf score Thresholding
                if scores[k] < conf_threshold:
                    continue

                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])

                label_head = class_name + " 0.00 0 0.00 "
                bbox_string = f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}"
                label_tail = f" 0.00 0.00 0.00 0.00 0.00 0.00 0.00 {scores[k]:.3f}\n"

                label_string = label_head + bbox_string + label_tail
                f.write(label_string)

                if check_key(color_map, class_name):
                    im1.rectangle([int(x1), int(y1), int(x2), int(y2)], fill=None, outline=color_map[class_name], width=1)

        pil_input.save(output_image_name)
        f.closed


def threshold_predictions(predictions, conf_threshold):
    """Thresholding the predctions based on the given confidence score threshold.

    Args:
        predictions (List): List of predictions from the model.
        conf_threshold (float) : Confidence Score Threshold value.

    Returns:
        filtered_predictions (List): List of thresholded predictions.
    """
    filtered_predictions = []

    for pred in predictions:
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        assert pred_boxes.shape[0] == pred_labels.shape[0] == pred_scores.shape[0]

        if len(pred_boxes) == 0:
            continue

        pred_boxes = pred_boxes.tolist()
        scores = pred_scores.tolist()
        labels = pred_labels.tolist()
        for k, _ in enumerate(pred_boxes):
            # Conf score Thresholding
            if scores[k] < conf_threshold:
                # remove from list
                scores.pop(k)
                labels.pop(k)
                pred_boxes.pop(k)

        filtered_predictions.extend(
            [
                {
                    'image_names': pred['image_names'],
                    'image_size': pred['image_size'],
                    'boxes': torch.Tensor(pred_boxes),
                    'scores': torch.Tensor(scores),
                    'labels': torch.Tensor(labels)
                }
            ]
        )

    return filtered_predictions


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api."""

    def __init__(self, num_select=100) -> None:
        """PostProcess constructor.

        Args:
            num_select (int): top K predictions to select from
        """
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes, image_names):
        """ Perform the post-processing. Scale back the boxes to the original size.

        Args:
            outputs (dict[torch.Tensor]): raw outputs of the model
            target_sizes (torch.Tensor): tensor of dimension [batch_size x 2] containing the size of each images of the batch.
                For evaluation, this must be the original image size (before any data augmentation).
                For visualization, this should be the image size after data augment, but before padding.

        Returns:
            results (List[dict]): final predictions compatible with COCOEval format.
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        labels = topk_indexes % out_logits.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b, 'image_names': n, 'image_size': i}
                   for s, l, b, n, i in zip(scores, labels, boxes, image_names, target_sizes)]

        return results
