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

"""Prune module."""
import os
import torch
from torch import nn
from typing import Sequence
from functools import reduce
from operator import mul
from omegaconf import OmegaConf

from torchvision.ops import DeformConv2d

from nvidia_tao_pytorch.cv.ocdnet.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocdnet.model.model import Model
from nvidia_tao_pytorch.cv.ocdnet.data_loader.build_dataloader import get_dataloader
from nvidia_tao_pytorch.cv.ocdnet.utils import mkdir
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
import nvidia_tao_pytorch.cv.ocdnet.pruning.torch_pruning as tp

# force pycuda on primary context before using TensorRT
import pycuda
import pycuda.autoinit
pyc_dev = pycuda.autoinit.device
pyc_ctx = pyc_dev.retain_primary_context()


class DCNv2OutputPruning(tp.functional.structured.BasePruner):
    """DCNv2 Pruning."""

    def prune(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        layer.out_channels = layer.out_channels - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * reduce(mul, layer.weight.shape[1:]) + (len(idxs) if layer.bias is not None else 0)
        return nparams_to_prune


class DCNv2InputPruning(tp.functional.structured.BasePruner):
    """DCNv2 Pruning."""

    def prune(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        layer.in_channels = layer.in_channels - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * layer.weight.shape[0] * reduce(mul, layer.weight.shape[2:])
        return nparams_to_prune


class Prune():
    """Prune."""

    def __init__(
        self,
        model_path,
        config,
        pruning_thresh,
        output_dir,
        gpu_id=0
    ):
        """Initialize."""
        config['model']['pretrained'] = False
        self.validate_loader = get_dataloader(config['dataset']['validate_dataset'], False)
        self.model = None
        self.pruning_thresh = pruning_thresh
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
            checkpoint = {key.replace("model.", ""): value for key, value in checkpoint.items()}
        self.model = Model(config['model'])
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
        self.model.load_state_dict(ckpt)
        self.model.to(self.device)

    def prune(self):
        """Prune function."""
        input_dict = next(iter(self.validate_loader))
        if self.model is not None:
            self.model.eval()
        print(self.model)
        with torch.no_grad():
            if self.model is not None:
                for key, value in input_dict.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            input_dict[key] = value.to(self.device)
        unpruned_total_params = sum(p.numel() for p in self.model.parameters())
        strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy()
        DG = tp.DependencyGraph()

        DG.register_customized_layer(
            DeformConv2d,
            in_ch_pruning_fn=DCNv2InputPruning(),  # A function to prune channels/dimensions of input tensor
            out_ch_pruning_fn=DCNv2OutputPruning(),  # A function to prune channels/dimensions of output tensor
            get_in_ch_fn=lambda n: n.in_channels,  # estimate the n_channel of layer input. Return None if the layer does not change tensor shape.
            get_out_ch_fn=lambda n: n.out_channels)  # estimate the n_channel of layer output. Return None if the layer does not change tensor shape.

        DG.build_dependency(self.model, example_inputs=input_dict["img"])
        for m in DG.module2node:
            _inputs = DG.module2node[m].inputs
            _deps = DG.module2node[m].dependencies
            if isinstance(m, DeformConv2d):
                DG.module2node[m].inputs = [_inputs[0]]
                DG.module2node[m].dependencies = [_deps[0], _deps[3]]
        # Prune Conv2d, DeformConv2d will be pruned indirectly by coupled pruning
        layers = [module for module in self.model.modules() if isinstance(module, torch.nn.Conv2d)]
        # Exclude DCNv2 conv2_offset layer
        black_list = []
        for layer in layers:
            if layer.out_channels == 27:
                black_list.append(layer)
        count = 0
        for layer in layers:
            # skip black list layers
            if layer in black_list:
                continue
            # Skip thresh module(not used in eval mode)
            if layer not in DG.module2node:
                continue
            threshold_run = self.pruning_thresh
            pruning_idxs = strategy(layer.weight, amount=threshold_run, round_to=64)
            pruning_plan = DG.get_pruning_plan(layer, tp.prune_conv_out_channel, idxs=pruning_idxs)
            if pruning_plan is not None:
                pruning_plan.exec()
            else:
                continue
            count += 1
        pruned_total_params = sum(p.numel() for p in self.model.parameters())
        print("Pruning ratio: {}".format(
            pruned_total_params / unpruned_total_params)
        )
        # Do inference to sanity check the pruned model
        self.model(input_dict["img"])
        # Save pruned model
        if not os.path.exists(self.output_dir):
            mkdir(self.output_dir)
        assert os.path.exists(self.output_dir) and os.path.isdir(self.output_dir), "The output_folder should exist."
        save_path = os.path.join(self.output_dir, f"pruned_{self.pruning_thresh}.pth")
        torch.save(self.model, save_path)


def run_experiment(experiment_config, model_path, pruning_thresh):
    """Run experiment."""
    gpu_id = experiment_config.prune.gpu_id
    torch.cuda.set_device(gpu_id)

    if experiment_config.prune.results_dir is not None:
        results_dir = experiment_config.prune.results_dir
    else:
        results_dir = os.path.join(experiment_config.results_dir, "prune")
        experiment_config.prune.results_dir = results_dir

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
        message="Starting OCDNet pruning"
    )

    pruner = Prune(
        model_path,
        experiment_config,
        pruning_thresh,
        output_dir=results_dir
    )
    pruner.prune()

    pyc_ctx.pop()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="prune", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the pruning process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)

    pyc_ctx.push()

    try:
        run_experiment(experiment_config=cfg,
                       model_path=cfg.prune.checkpoint,
                       pruning_thresh=cfg.prune.pruning_thresh
                       )
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Pruning finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Pruning was interrupted",
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
