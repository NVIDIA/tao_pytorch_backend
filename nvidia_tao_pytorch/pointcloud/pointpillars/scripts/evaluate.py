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

"""Evaluation script for PointPillars."""
import datetime
import os
from pathlib import Path

import torch
from torch import nn

# force pycuda on primary context before using TensorRT
import pycuda
import pycuda.autoinit
try:
    import tensorrt as trt  # pylint: disable=unused-import  # noqa: F401
    from nvidia_tao_pytorch.pointcloud.pointpillars.tools.export.tensorrt_model import TrtModel
except:  # noqa: E722
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, "
        "evaluation with TensorRT engine will not be available."
    )
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.pointcloud.pointpillars.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.pointcloud.pointpillars.tools.eval_utils import eval_utils
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets import build_dataloader
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models import load_checkpoint
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models.model_utils import model_nms_utils
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils


pyc_dev = pycuda.autoinit.device
pyc_ctx = pyc_dev.retain_primary_context()


def parse_epoch_num(model_file):
    """Parse epoch number from model file."""
    model_base = os.path.basename(model_file)
    epoch_string = model_base[:-4].split("_")[-1]
    return int(epoch_string)


def eval_single_ckpt(
    model, test_loader, cfg,
    eval_output_dir, logger, epoch_id,
    dist_test=False
):
    """Evaluation with PyTorch model."""
    model.cuda()
    # start evaluation
    return eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=cfg.evaluate.save_to_file
    )


def eval_single_ckpt_trt(
    model, test_loader, cfg,
    eval_output_dir, logger,
    dist_test=False
):
    """Evaluation with TensorRT engine."""
    return eval_utils.eval_one_epoch_trt(
        cfg, model, test_loader, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=cfg.evaluate.save_to_file
    )


class CustomNMS(nn.Module):
    """NMS module."""

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
    """PostProcessing module."""

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
    """TensorRT model wrapper."""

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
        """call method."""
        trt_output = self.trt_model.predict(input_dict)
        return self.post_processor(
            trt_output["output_boxes"],
            trt_output["num_boxes"],
        )


spec_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "cfgs")


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=spec_root, config_name="pointpillar_general", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Main function."""
    if cfg.get("evaluate", {}).get("trt_engine", None) == "":
        cfg["evaluate"]["trt_engine"] = None
    if cfg.results_dir is None:
        raise OSError("Either provide output_dir in config file or provide output_dir as a CLI argument")
    output_dir = Path(expand_path(cfg.results_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / 'eval'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    # Set status logging
    status_file = os.path.join(str(eval_output_dir), "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting PointPillars evaluation"
    )
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)
    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    test_loader = build_dataloader(
        dataset_cfg=cfg.dataset,
        class_names=cfg.dataset.class_names,
        batch_size=cfg.evaluate.batch_size,
        dist=False,
        workers=cfg.dataset.num_workers,
        logger=logger,
        training=False,
        info_path=cfg.dataset.data_info_path
    )[1]
    checkpoint_loaded = load_checkpoint(
        cfg.evaluate.checkpoint,
        cfg.key
    )
    model = checkpoint_loaded[0]
    epoch_num = checkpoint_loaded[2]
    # Try to load TRT engine if there is any
    if cfg.evaluate.trt_engine is not None:
        trt_model = TrtModel(
            cfg.evaluate.trt_engine,
            cfg.evaluate.batch_size,
        )
        trt_model.build_or_load_trt_engine()
        # Check the batch size
        engine_batch_size = trt_model.engine._engine.get_binding_shape(0)[0]
        if engine_batch_size != cfg.evaluate.batch_size:
            raise ValueError(f"TensorRT engine batch size: {engine_batch_size}, mismatch with "
                             f"batch size for evaluation: {cfg.evaluate.batch_size}. "
                             "Please make sure they are the same by generating a new engine or "
                             f"modifying the evaluation batch size in spec file to {engine_batch_size}.")
        model_wrapper = TrtModelWrapper(
            model,
            cfg,
            trt_model
        )
        with torch.no_grad():
            ret_dict = eval_single_ckpt_trt(
                model_wrapper, test_loader, cfg,
                eval_output_dir, logger, dist_test=False
            )
        status_logging.get_status_logger().kpi = ret_dict
        status_logging.get_status_logger().write(
            message="Evaluation metrics generated.",
            status_level=status_logging.Status.RUNNING
        )
    else:
        # Load model from checkpoint
        with torch.no_grad():
            ret_dict = eval_single_ckpt(
                model, test_loader, cfg, eval_output_dir,
                logger, epoch_num, dist_test=False
            )
        status_logging.get_status_logger().kpi = ret_dict
        status_logging.get_status_logger().write(
            message="Evaluation metrics generated.",
            status_level=status_logging.Status.RUNNING
        )


if __name__ == '__main__':
    pyc_ctx.push()
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Evaluation finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Evaluation was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
    pyc_ctx.pop()
