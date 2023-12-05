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
#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
"""Inference module."""
import os
import pathlib
import time
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from nvidia_tao_pytorch.cv.ocdnet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocdnet.post_processing.seg_detector_representer import get_post_processing
from nvidia_tao_pytorch.cv.ocdnet.model.pl_ocd_model import OCDnetModel
from nvidia_tao_pytorch.cv.ocdnet.utils.util import show_img, draw_bbox, save_result, get_file_list, load_checkpoint
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from tqdm import tqdm
import matplotlib.pyplot as plt

import pycuda
import pycuda.autoinit
pyc_dev = pycuda.autoinit.device
pyc_ctx = pyc_dev.retain_primary_context()


def resize_image(img, image_size):
    """Resize image"""
    resized_img = cv2.resize(img, image_size)
    return resized_img


class Inference:
    """Infer class."""

    def __init__(self, model_path, config, post_p_thre=0.7, gpu_id=None):
        """Init model."""
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        self.post_process = get_post_processing(config['inference']['post_processing'])
        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['inference']['img_mode']
        config["dataset"]["train_dataset"]["data_path"] = [os.path.dirname(config["inference"]["input_folder"])]
        config["dataset"]["validate_dataset"]["data_path"] = [os.path.dirname(config["inference"]["input_folder"])]
        if model_path.split(".")[-1] in ["trt", "engine"]:
            raise Exception("Please use tao_deploy to run inference against tensorrt engine.")
        else:
            checkpoint = load_checkpoint(model_path, to_cpu=True)
            self.model = OCDnetModel(config)
            self.model.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self.is_trt = False

    def predict(self, img_path: str, image_size, is_output_polygon=False):
        """Run prediction."""
        assert os.path.exists(img_path), 'file is not exists'
        ori_img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0).astype(np.float32)
        if self.img_mode == 'RGB':
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        h, w = ori_img.shape[:2]
        ori_img = resize_image(ori_img, image_size)
        rgb_mean = np.array([122.67891434, 116.66876762, 104.00698793])
        image = ori_img
        image -= rgb_mean
        image /= 255.
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        # change (w,h) to (1,img_channel,h,w)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)
        batch = {'img': torch.Tensor(1, 3, h, w)}
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            if self.is_trt:
                tensor_np = tensor.detach().cpu().numpy()
                preds = torch.from_numpy(
                    self.model.predict({"input": tensor_np})["pred"]
                ).cuda()
            else:
                preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # filer bbox has all 0
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            t = time.time() - start
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t


def run_experiment(experiment_config, model_path, post_p_thre, input_folder,
                   width, height, polygon, show):
    """Run experiment."""
    gpu_id = experiment_config.inference.gpu_id
    torch.cuda.set_device(gpu_id)

    if experiment_config.inference.results_dir is not None:
        results_dir = experiment_config.inference.results_dir
    else:
        results_dir = os.path.join(experiment_config.results_dir, "inference")
        experiment_config.inference.results_dir = results_dir

    os.makedirs(results_dir, exist_ok=True)

    experiment_config = OmegaConf.to_container(experiment_config)

    experiment_config['model']['pretrained'] = False

    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            append=True
        )
    )
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting OCDNet inference"
    )

    # Init the network
    infer_model = Inference(
        model_path,
        experiment_config,
        post_p_thre,
        gpu_id=0
    )
    for img_path in tqdm(get_file_list(input_folder, p_postfix=['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp'])):
        preds, boxes_list, score_list, _ = infer_model.predict(
            img_path,
            (width, height),
            is_output_polygon=polygon
        )
        im = cv2.imread(img_path)
        img = draw_bbox(im[:, :, ::-1], boxes_list)
        if show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        # save result
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(results_dir, img_path.stem + '_result.jpg')
        pred_path = os.path.join(results_dir, img_path.stem + '_pred.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        cv2.imwrite(pred_path, preds * 255)
        save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, polygon)

    pyc_ctx.pop()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="inference", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)

    pyc_ctx.push()

    try:
        run_experiment(experiment_config=cfg,
                       model_path=cfg.inference.checkpoint,
                       post_p_thre=cfg.inference.post_processing.args.box_thresh,
                       input_folder=cfg.inference.input_folder,
                       width=cfg.inference.width,
                       height=cfg.inference.height,
                       polygon=cfg.inference.polygon,
                       show=cfg.inference.show
                       )
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


if __name__ == "__main__":
    main()
