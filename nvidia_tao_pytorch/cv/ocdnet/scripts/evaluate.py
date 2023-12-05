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
"""Evaluate a trained ocdnet model."""

import os
import time
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.ocdnet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocdnet.data_loader.build_dataloader import get_dataloader
from nvidia_tao_pytorch.cv.ocdnet.post_processing.seg_detector_representer import get_post_processing
from nvidia_tao_pytorch.cv.ocdnet.utils.ocr_metric.icdar2015.quad_metric import get_metric
from nvidia_tao_pytorch.cv.ocdnet.utils.util import load_checkpoint
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs

from nvidia_tao_pytorch.cv.ocdnet.model.pl_ocd_model import OCDnetModel

import pycuda
import pycuda.autoinit
pyc_dev = pycuda.autoinit.device
pyc_ctx = pyc_dev.retain_primary_context()


class Evaluate():
    """Eval class."""

    def __init__(self, model_path, config_file, gpu_id=0):
        """Initialize."""
        config = config_file
        config['model']['pretrained'] = False
        config["dataset"]["train_dataset"] = config["dataset"]["validate_dataset"]
        self.validate_loader = get_dataloader(config['dataset']['validate_dataset'], False)
        self.post_process = get_post_processing(config['evaluate']['post_processing'])
        self.metric_cls = get_metric(config['evaluate']['metric'])
        self.box_thresh = config['evaluate']['post_processing']["args"]["box_thresh"]
        self.model = None
        self.trt_model = None
        if model_path.split(".")[-1] in ["trt", "engine"]:
            raise Exception("Please use tao_deploy to run evaluation against tensorrt engine.")
        else:
            self.gpu_id = gpu_id
            if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
                self.device = torch.device("cuda:%s" % self.gpu_id)
                torch.backends.cudnn.benchmark = True
            else:
                self.device = torch.device("cpu")
            raw_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

            if not isinstance(raw_checkpoint, dict):
                self.model = raw_checkpoint
                self.model.to(self.device)
            else:
                checkpoint = load_checkpoint(model_path, to_cpu=True)
                self.model = OCDnetModel(config)
                layers = checkpoint.keys()
                ckpt = dict()
                # Support loading official pretrained weights for eval
                for layer in layers:
                    new_layer = layer
                    if new_layer.startswith("model.module."):
                        new_layer = new_layer[13:]
                    if new_layer == "decoder.in5.weight":
                        new_layer = "neck.in5.weight"
                    elif new_layer == "decoder.in4.weight":
                        new_layer = "neck.in4.weight"
                    elif new_layer == "decoder.in3.weight":
                        new_layer = "neck.in3.weight"
                    elif new_layer == "decoder.in2.weight":
                        new_layer = "neck.in2.weight"
                    elif new_layer == "decoder.out5.0.weight":
                        new_layer = "neck.out5.0.weight"
                    elif new_layer == "decoder.out4.0.weight":
                        new_layer = "neck.out4.0.weight"
                    elif new_layer == "decoder.out3.0.weight":
                        new_layer = "neck.out3.0.weight"
                    elif new_layer == "decoder.out2.weight":
                        new_layer = "neck.out2.weight"
                    elif new_layer == "decoder.binarize.0.weight":
                        new_layer = "head.binarize.0.weight"
                    elif new_layer == "decoder.binarize.1.weight":
                        new_layer = "head.binarize.1.weight"
                    elif new_layer == "decoder.binarize.1.bias":
                        new_layer = "head.binarize.1.bias"
                    elif new_layer == "decoder.binarize.1.running_mean":
                        new_layer = "head.binarize.1.running_mean"
                    elif new_layer == "decoder.binarize.1.running_var":
                        new_layer = "head.binarize.1.running_var"
                    elif new_layer == "decoder.binarize.3.weight":
                        new_layer = "head.binarize.3.weight"
                    elif new_layer == "decoder.binarize.3.bias":
                        new_layer = "head.binarize.3.bias"
                    elif new_layer == "decoder.binarize.4.weight":
                        new_layer = "head.binarize.4.weight"
                    elif new_layer == "decoder.binarize.4.bias":
                        new_layer = "head.binarize.4.bias"
                    elif new_layer == "decoder.binarize.4.running_mean":
                        new_layer = "head.binarize.4.running_mean"
                    elif new_layer == "decoder.binarize.4.running_var":
                        new_layer = "head.binarize.4.running_var"
                    elif new_layer == "decoder.binarize.6.weight":
                        new_layer = "head.binarize.6.weight"
                    elif new_layer == "decoder.binarize.6.bias":
                        new_layer = "head.binarize.6.bias"
                    elif new_layer == "decoder.thresh.0.weight":
                        new_layer = "head.thresh.0.weight"
                    elif new_layer == "decoder.thresh.1.weight":
                        new_layer = "head.thresh.1.weight"
                    elif new_layer == "decoder.thresh.1.bias":
                        new_layer = "head.thresh.1.bias"
                    elif new_layer == "decoder.thresh.1.running_mean":
                        new_layer = "head.thresh.1.running_mean"
                    elif new_layer == "decoder.thresh.1.running_var":
                        new_layer = "head.thresh.1.running_var"
                    elif new_layer == "decoder.thresh.3.weight":
                        new_layer = "head.thresh.3.weight"
                    elif new_layer == "decoder.thresh.3.bias":
                        new_layer = "head.thresh.3.bias"
                    elif new_layer == "decoder.thresh.4.weight":
                        new_layer = "head.thresh.4.weight"
                    elif new_layer == "decoder.thresh.4.bias":
                        new_layer = "head.thresh.4.bias"
                    elif new_layer == "decoder.thresh.4.running_mean":
                        new_layer = "head.thresh.4.running_mean"
                    elif new_layer == "decoder.thresh.4.running_var":
                        new_layer = "head.thresh.4.running_var"
                    elif new_layer == "decoder.thresh.6.weight":
                        new_layer = "head.thresh.6.weight"
                    elif new_layer == "decoder.thresh.6.bias":
                        new_layer = "head.thresh.6.bias"
                    elif "num_batches_tracked" in new_layer:
                        continue
                    elif "backbone.fc" in new_layer:
                        continue
                    elif "backbone.smooth" in new_layer:
                        continue
                    ckpt[new_layer] = checkpoint[layer]
                self.model.model.load_state_dict(ckpt)
                self.model.to(self.device)

    def eval(self):
        """eval function."""
        if self.model is not None:
            self.model.eval()
        thresh_range = [i * 0.1 for i in range(1, 10)]
        raw_metrics = {thresh: [] for thresh in thresh_range}
        metrics = {thresh: {} for thresh in thresh_range}
        total_frame = 0.0
        total_time = 0.0
        for _, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                if self.model is not None:
                    for key, value in batch.items():
                        if value is not None:
                            if isinstance(value, torch.Tensor):
                                batch[key] = value.to(self.device)
                start = time.time()
                if self.model is not None:
                    preds = self.model(batch['img'])
                else:
                    img = batch["img"].detach().cpu().numpy()
                    start = time.time()
                    preds = torch.from_numpy(
                        self.trt_model.predict({"input": img})["pred"]
                    ).cuda()
                for thresh in thresh_range:
                    self.post_process.thresh = thresh
                    boxes, scores = self.post_process(batch, preds, is_output_polygon=self.metric_cls.is_output_polygon)
                    total_frame += batch['img'].size()[0]
                    total_time += time.time() - start
                    raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores), box_thresh=self.box_thresh)
                    raw_metrics[thresh].append(raw_metric)
        best_hmean = 0
        for thresh in thresh_range:
            metric = self.metric_cls.gather_measure(raw_metrics[thresh])
            metrics[thresh] = {'recall': metric['recall'].avg, 'precision': metric['precision'].avg, 'hmean': metric['hmean'].avg}
            msg = f"thresh: {round(thresh, 1)}, recall: {metric['recall'].avg}, precision: {metric['precision'].avg}, hmean: {metric['hmean'].avg}"
            status_logging.get_status_logger().write(message=msg)
            if metric['hmean'].avg > best_hmean:
                best_hmean = metric['hmean'].avg
                metrics['best'] = {'Thresh': round(thresh, 1), 'Recall': metric['recall'].avg, 'Precision': metric['precision'].avg, 'Hmean': metric['hmean'].avg}
        return metrics


def run_experiment(experiment_config, model_path):
    """Run experiment."""
    gpu_id = experiment_config.evaluate.gpu_id
    torch.cuda.set_device(gpu_id)

    if experiment_config.evaluate.results_dir is not None:
        results_dir = experiment_config.evaluate.results_dir
    else:
        results_dir = os.path.join(experiment_config.results_dir, "evaluate")
        experiment_config.evaluate.results_dir = results_dir
    os.makedirs(results_dir, exist_ok=True)
    experiment_config = OmegaConf.to_container(experiment_config)

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
        message="Starting OCDNet evaluation"
    )

    evaluation = Evaluate(model_path, experiment_config)

    result = evaluation.eval()
    status_logging.get_status_logger().kpi = result['best']

    pyc_ctx.pop()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="evaluate", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluation process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    pyc_ctx.push()

    try:
        run_experiment(experiment_config=cfg,
                       model_path=cfg.evaluate.checkpoint)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
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


if __name__ == "__main__":
    main()
