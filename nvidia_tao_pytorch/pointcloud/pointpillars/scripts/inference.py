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

"""Inference script for PointPillars."""
import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    import tensorrt as trt  # pylint: disable=unused-import  # noqa: F401
    from nvidia_tao_pytorch.pointcloud.pointpillars.tools.export.tensorrt_model import TrtModel
except:  # noqa: E722
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, "
        "inference with TensorRT engine will not be available."
    )
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.pointcloud.pointpillars.tools.eval_utils import eval_utils
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets import build_dataloader
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models import load_checkpoint
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models.model_utils import model_nms_utils
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils


def parse_config():
    """Argument Parser."""
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument("--output_dir", type=str, required=False, default=None, help="output checkpoint directory.")
    parser.add_argument(
        "--trt_engine",
        type=str,
        required=False,
        default=None,
        help="Path to the TensorRT engine to be used for inference"
    )
    parser.add_argument("--key", "-k", type=str, required=True, help="Encryption key")
    args = parser.parse_args()
    cfg_from_yaml_file(expand_path(args.cfg_file), cfg)
    np.random.seed(1024)
    return args, cfg


def parse_epoch_num(model_file):
    """Parse epoch number from model file."""
    model_base = os.path.basename(model_file)
    epoch_string = model_base[:-4].split("_")[-1]
    return int(epoch_string)


def infer_single_ckpt(
    model, test_loader, args,
    infer_output_dir, logger,
    cfg
):
    """Do inference with PyTorch model."""
    model.cuda()
    eval_utils.infer_one_epoch(
        cfg, model, test_loader, logger,
        result_dir=infer_output_dir, save_to_file=args.save_to_file
    )


def infer_single_ckpt_trt(
    model, test_loader, args,
    infer_output_dir, logger,
    cfg
):
    """Do inference with TensorRT engine."""
    eval_utils.infer_one_epoch_trt(
        cfg, model, test_loader, logger,
        result_dir=infer_output_dir, save_to_file=args.save_to_file
    )


class CustomNMS(nn.Module):
    """Customized NMS module."""

    def __init__(self, post_process_cfg):
        """Initialize."""
        super().__init__()
        self.post_process_cfg = post_process_cfg

    def forward(self, output_boxes, num_boxes):
        """Forward method."""
        batch_output = []
        for idx, box_per_frame in enumerate(output_boxes):
            num_box_per_frame = num_boxes[idx]
            box_per_frame = torch.from_numpy(box_per_frame).cuda()
            box_per_frame = box_per_frame[:num_box_per_frame, ...]
            box_preds = box_per_frame[:, 0:7]
            label_preds = box_per_frame[:, 7] + 1
            cls_preds = box_per_frame[:, 8]
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=cls_preds, box_preds=box_preds,
                nms_config=self.post_process_cfg.nms_config,
                score_thresh=self.post_process_cfg.score_thresh
            )
            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]
            final_output = torch.cat(
                [
                    final_boxes,
                    final_scores.view((-1, 1)),
                    final_labels.view((-1, 1))
                ],
                axis=-1
            )
            batch_output.append(final_output.cpu().numpy())
        return batch_output


class CustomPostProcessing(nn.Module):
    """Customized PostProcessing module."""

    def __init__(self, model, cfg):
        """Initialize."""
        super().__init__()
        self.model = model
        self.custom_nms = CustomNMS(cfg)

    def forward(self, output_boxes, num_boxes):
        """Forward method."""
        return self.custom_nms(
            output_boxes,
            num_boxes
        )


class TrtModelWrapper():
    """TensorRT engine wrapper."""

    def __init__(self, model, cfg, trt_model):
        """Initialize."""
        self.model = model
        self.cfg = cfg
        self.trt_model = trt_model
        self.post_processor = CustomPostProcessing(
            self.model,
            self.cfg.model.post_processing
        )

    def __call__(self, input_dict):
        """Call method."""
        trt_output = self.trt_model.predict(input_dict)
        return self.post_processor(
            trt_output["output_boxes"],
            trt_output["num_boxes"],
        )


def main():
    """Main function."""
    args, cfg = parse_config()
    args.batch_size = cfg.inference.batch_size
    args.workers = cfg.dataset.num_workers
    args.ckpt = cfg.inference.checkpoint
    if args.output_dir is None:
        if cfg.results_dir is None:
            raise OSError("Either provide results_dir in config file or provide output_dir as a CLI argument")
        else:
            args.output_dir = cfg.results_dir
    output_dir = Path(expand_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    infer_output_dir = output_dir / 'infer'
    infer_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = infer_output_dir / ('log_infer_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)
    # log to file
    logger.info('**********************Start logging**********************')
    # Set status logging
    status_file = os.path.join(str(infer_output_dir), "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting PointPillars inference")
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_loader = build_dataloader(
        dataset_cfg=cfg.dataset,
        class_names=cfg.class_names,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False
    )[1]
    model = load_checkpoint(
        args.ckpt,
        args.key
    )[0]
    # Try to load TRT engine if there is any
    if args.trt_engine is not None:
        trt_model = TrtModel(
            args.trt_engine,
            args.batch_size,
        )
        trt_model.build_or_load_trt_engine()
        # Check the batch size
        engine_batch_size = trt_model.engine._engine.get_binding_shape(0)[0]
        if engine_batch_size != args.batch_size:
            raise ValueError(f"TensorRT engine batch size: {engine_batch_size}, mismatch with "
                             f"batch size for evaluation: {args.batch_size}. "
                             "Please make sure they are the same by generating a new engine or "
                             f"modifying the evaluation batch size in spec file to {engine_batch_size}.")
        model_wrapper = TrtModelWrapper(
            model,
            cfg,
            trt_model
        )
        with torch.no_grad():
            infer_single_ckpt_trt(
                model_wrapper, test_loader, args,
                infer_output_dir, logger, cfg
            )
    else:
        # Load model from checkpoint
        with torch.no_grad():
            infer_single_ckpt(
                model, test_loader, args, infer_output_dir,
                logger, cfg
            )


if __name__ == '__main__':
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Inference finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Inference was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
