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

"""
Prune OCRNet script.
"""
import os
import argparse

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.ocrnet.config.default_config import ExperimentConfig
import nvidia_tao_pytorch.pruning.torch_pruning as tp


def prune(opt):
    """Prune the the OCRNet according to option"""
    # @TODO(tylerz): Lazy import for correctly setting CUDA_VISIBLE_DEVICES
    import torch
    import torch.utils.data

    from nvidia_tao_pytorch.cv.ocrnet.utils.utils import (CTCLabelConverter,
                                                          AttnLabelConverter,
                                                          load_checkpoint)
    from nvidia_tao_pytorch.cv.ocrnet.model.model import Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    ckpt = load_checkpoint(opt.saved_model, key=opt.encryption_key, to_cpu=True)
    if not isinstance(ckpt, Model):
        model = Model(opt)
        state_dict = ckpt
        model.load_state_dict(state_dict)
    else:
        model = ckpt
        # print(model)

    model = model.to(device)

    model.eval()
    num_params_before_pruning = tp.utils.count_params(model)
    # 1. build dependency graph
    dep_graph = tp.DependencyGraph()
    dep_graph.build_dependency(model, example_inputs=(torch.randn([1, opt.input_channel, opt.imgH, opt.imgW]).to(device),
                                                      torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)))
    # 1.1 excluded layer @TODO(tylerz): only support for CTC now.
    excluded_layers = list(model.modules())[-11:]

    pruned_module = []
    prunable_list = [torch.nn.Conv2d]

    # 2. loop through the graph to execute the pruning:
    if opt.prune_mode in ["amount", "threshold"]:
        strategy = tp.strategy.LNStrategy(p=opt.p, mode=opt.prune_mode)
        if opt.prune_mode == "amount":
            th = opt.amount
        else:
            th = opt.threshold
        for _, m in model.named_modules():

            if isinstance(m, tuple(prunable_list)) and m not in excluded_layers and m not in pruned_module:
                pruned_idxs = strategy(m.weight, amount=th, round_to=opt.granularity)

                prune_func = tp.prune.prune_conv
                pruning_plan = dep_graph.get_pruning_plan(m, prune_func, idxs=pruned_idxs)
                if pruning_plan is not None:
                    pruning_plan.exec()
                else:
                    continue
    else:  # experimental hybrid path
        strategy = tp.strategy.CustomScoreStrategy()
        global_thresh, module2scores = tp.utils.get_global_thresh(model, prune_ratio=opt.amount)
        merged_sets = {}
        # 2.1 find the merged set:
        for _, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                prune_func = tp.prune.prune_conv
                merged_set = tp.dependency.find_merged_set(dep_graph.module_to_node[m], prune_func)
                merged_sets[m] = merged_set

        tp.utils.execute_custom_score_prune(model,
                                            global_thresh=global_thresh,
                                            module2scores=module2scores,
                                            dep_graph=dep_graph,
                                            granularity=opt.granularity,
                                            excluded_layers=excluded_layers,
                                            merged_sets=merged_sets)

    num_params_after_pruning = tp.utils.count_params(model)
    print("  Params: %s => %s" % (num_params_before_pruning, num_params_after_pruning))
    encoded_output_file = opt.output_file
    print(f"Pruned model is saved to {encoded_output_file}")
    torch.save(model, encoded_output_file)


def init_configs(experiment_spec: ExperimentConfig):
    """Pass the yaml config to argparse.Namespace"""
    parser = argparse.ArgumentParser()

    opt, _ = parser.parse_known_args()
    opt.encryption_key = experiment_spec.encryption_key
    opt.output_file = experiment_spec.prune.pruned_file

    # 1. Init dataset params
    dataset_config = experiment_spec.dataset
    model_config = experiment_spec.model
    opt.batch_max_length = dataset_config.max_label_length
    opt.imgH = model_config.input_height
    opt.imgW = model_config.input_width
    opt.input_channel = model_config.input_channel
    if model_config.input_channel == 3:
        opt.rgb = True
    else:
        opt.rgb = False

    # load character list:
    # Don't convert the characters to lower case
    with open(dataset_config.character_list_file, "r") as f:
        characters = "".join([ch.strip() for ch in f.readlines()])
    opt.character = characters

    # 2. Init Model params
    opt.saved_model = experiment_spec.prune.checkpoint
    if model_config.TPS:
        opt.Transformation = "TPS"
    else:
        opt.Transformation = "None"

    opt.FeatureExtraction = model_config.backbone
    opt.SequenceModeling = model_config.sequence
    opt.Prediction = model_config.prediction
    opt.num_fiducial = model_config.num_fiducial
    opt.output_channel = model_config.feature_channel
    opt.hidden_size = model_config.hidden_size

    opt.baiduCTC = False

    # 3. Init pruning params:
    prune_config = experiment_spec.prune.prune_setting
    opt.prune_mode = prune_config.mode
    if opt.prune_mode in ["amount", "experimental_hybrid"]:
        opt.amount = prune_config.amount
    elif opt.prune_mode in ["threshold"]:
        opt.threshold = prune_config.threshold
    else:
        raise ValueError("Only supports prune mode in [amount, threshold, \
            experimental_hybrid]")

    opt.granularity = prune_config.granularity
    if prune_config.raw_prune_score == "L2":
        opt.p = 2
    else:
        opt.p = 1

    # 4. Init for Device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(experiment_spec.prune.gpu_id)
    import torch
    opt.num_gpu = torch.cuda.device_count()

    return opt


def run_experiment(experiment_spec):
    """run experiment."""
    opt = init_configs(experiment_spec)
    if experiment_spec.prune.results_dir is not None:
        results_dir = experiment_spec.prune.results_dir
    else:
        results_dir = os.path.join(experiment_spec.results_dir, "prune")
        experiment_spec.prune.results_dir = results_dir
    os.makedirs(results_dir, exist_ok=True)
    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file,
                                                                 append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED,
                                             message="Starting OCRNet Prune")
    # Set default output filename if the filename
    # isn't provided over the command line.
    if opt.output_file is None:
        split_name = os.path.splitext(opt.saved_model)[0]
        opt.output_file = "pruned_{}.etlt".format(split_name)

    # Warn the user if an exported file already exists.
    assert not os.path.exists(opt.output_file), "Default output file {} already "\
        "exists".format(opt.output_file)

    prune(opt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        run_experiment(experiment_spec=cfg)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Prune finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Prune was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


if __name__ == '__main__':
    main()
