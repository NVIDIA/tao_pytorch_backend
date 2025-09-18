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
import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import torch
import torch.nn.functional as F
from torch import nn

from nvidia_tao_pytorch.cv.deformable_detr.utils import box_ops
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import read_h5_image_from_path

# Referenced from mmdet visualization
METAINFO = {
    'classes': ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    # palette is a list of color tuples, which is used for visualization.
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
                (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
                (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
                (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
                (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
                (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
                (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
                (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
                (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
                (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
                (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
                (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
                (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
                (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
                (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
                (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
                (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
                (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
                (246, 0, 122), (191, 162, 208)]
}


def get_key(label_map, val):
    """get_key for class label."""
    for label in label_map:
        if label['id'] == val:
            return label['name']
    return None


def check_key(my_dict, key):
    """check_key for classes."""
    return bool(key in my_dict.keys())


def save_inference_prediction(predictions, output_dir, conf_threshold, label_map, color_map, is_internal=False, outline_width=3):
    """Save the annotated images and label file to the output directory.

    Args:
        predictions (List): List of predictions from the model.
        output_dir (str) : Output directory to save predictions.
        conf_threshold (float) : Confidence Score Threshold value.
        label_map(Dict): Dictonary for the class lables.
        color_map(Dict): Dictonary for the color mapping to annotate the bounding box per class.
        is_internal(Bool) : To save the inference results in format of output_dir/sequence/image_name.
        outline_width (int): Bbox outline width (default: 3)
    """
    # If not explicitly specified, use COCO classes as default color scheme.
    if color_map is None:
        color_map = {c: p for c, p in zip(METAINFO['classes'], METAINFO['palette'])}

    for pred in predictions:

        image_name = pred['image_names']
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        masks_filt = None
        pred_masks = None
        if "masks" in pred:
            pred_masks = pred["masks"]

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
            os.makedirs(output_label_root, exist_ok=True)

        if not os.path.exists(output_annotate_root):
            os.makedirs(output_annotate_root, exist_ok=True)

        if image_name.startswith("h5://"):
            pil_input, _ = read_h5_image_from_path(image_name)
        else:
            pil_input = Image.open(image_name).convert("RGB")

        pil_input = ImageOps.exif_transpose(pil_input)
        W, H = pil_input.size

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
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if is_internal:
                        # Don't put class name
                        im1.rectangle(((x1, y1, x2, y2)),
                                      fill=None,
                                      outline=color_map[class_name],
                                      width=outline_width)
                    else:
                        im1.rectangle(((x1, y1, x2, y2)),
                                      fill=None,
                                      outline=color_map[class_name],
                                      width=outline_width)
                        # text pad
                        im1.rectangle(((x1, y1 - 10), (x1 + (x2 - x1), y1)),
                                      fill=color_map[class_name])
                        im1.text((x1, y1 - 10), f"{class_name}: {scores[k]:.2f}")
                        if pred_masks is not None:
                            masks_filt = F.interpolate(pred_masks[None, ...], size=(H, W), mode='bilinear', align_corners=False)[0]
                            masks_filt = masks_filt > 0.5
                            masks_filt = masks_filt.cpu().numpy()
                            color = tuple(np.random.randint(0, 255, size=3).tolist())
                            pred_color = masks_filt[k][..., None].astype(np.uint8) * np.array([color])
                            pred_color = pred_color.astype(np.uint8)
                            pred_color = Image.fromarray(pred_color).convert('RGBA')
                            pred_color = pred_color.resize((W, H), resample=Image.NEAREST)
                            pil_input.paste(pred_color, (0, 0), Image.fromarray(masks_filt[k]).convert("L").resize((W, H), resample=Image.NEAREST))

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

        filtered = [
            (box, score, label)
            for box, score, label in zip(pred_boxes, scores, labels)
            if score >= conf_threshold
        ]

        if not filtered:
            continue

        pred_boxes, scores, labels = map(list, zip(*filtered))

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
