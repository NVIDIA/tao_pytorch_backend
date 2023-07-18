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
Export of Segformer model.
"""
import os
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.segformer.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.utils.common_utils import check_and_create
from nvidia_tao_pytorch.cv.segformer.model.sf_model import SFModel
from nvidia_tao_pytorch.cv.segformer.dataloader.segformer_dm import SFDataModule
from nvidia_tao_pytorch.cv.segformer.model.builder import build_segmentor
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils
import datetime
import numpy as np
from functools import partial
import onnxruntime as rt
import onnx
import torch
import torch._C
import torch.serialization
from mmcv.runner.checkpoint import load_checkpoint
from mmcv.runner import get_dist_info, init_dist
from mmcv.onnx import register_extra_symbolics


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    segs = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)
    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False}]
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs


def pytorch2onnx(model,
                 input_shape,
                 num_classes,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 test_cfg=None,
                 simplify=False,
                 logger=None):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.cuda()
    model.eval()
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # avoid input splits
    img_list = [imgs]
    img_meta_list = [img_metas]

    # replace original forward function
    origin_forward = model.forward_export
    model.forward = partial(
        model.forward_export, img_meta_list, False)

    register_extra_symbolics(opset_version)

    with torch.no_grad():
        torch.onnx.export(model,
                          (img_list, ),
                          output_file,
                          export_params=True,
                          input_names=['input'],
                          output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"}, 'output': {0: 'batch_size'}},
                          verbose=show,
                          opset_version=opset_version)
        logger.info('Successfully exported ONNX model.')
    model.forward = origin_forward
    final_tmp_onnx_path = output_file
    if simplify:
        logger.info('[INFO] Simplifying model...')
        from onnxsim import simplify
        onnx_model = onnx.load(output_file)
        # simplifying dynamic model
        _, C, H, W = imgs.shape
        simplified_model, _ = simplify(onnx_model,
                                       input_shapes={'input': (1, C, H, W)},  # test bz = 2
                                       dynamic_input_shape=True,
                                       check_n=3)
        simplified_path = output_file[:-5] + "_sim.onnx"
        onnx.save(simplified_model, simplified_path)
        final_tmp_onnx_path = simplified_path

    if verify:
        # check by onnx
        onnx_model = onnx.load(final_tmp_onnx_path)
        onnx.checker.check_model(onnx_model)
        # check the numerical value
        # get pytorch output
        pytorch_result = model(img_meta_list, False, img_list)[0]
        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1), "Input dimensions do not match."
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(None, {net_feed_input[0]: img_list[0].detach().numpy()})[0][0, :, :, 0]

        if not np.allclose(pytorch_result, onnx_result):
            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
        logger.info('The outputs are same between Pytorch and ONNX')


def run_experiment(experiment_config, results_dir):
    """Start the Export.
    Args:
        experiment_config (Dict): Config dictionary containing epxeriment parameters
        results_dir (str): Results dir to save the exported ONNX.

    """
    check_and_create(results_dir)
    # Set the logger
    log_file = os.path.join(results_dir, 'log_evaluate_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

    logger = common_utils.create_logger(log_file, rank=0)
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logger = status_logging.get_status_logger()
    # log to file
    logger.info('********************** Start logging for Export.**********************')
    status_logger.write(message="**********************Start logging for Export**********************.")
    num_gpus = 1
    seed = experiment_config["export"]["exp_config"]["manual_seed"]
    # Need to change this
    rank, world_size = get_dist_info()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
    if "RANK" not in os.environ:
        os.environ['RANK'] = str(rank)
    if "WORLD_SIZE" not in os.environ:
        os.environ['WORLD_SIZE'] = str(world_size)
    if "MASTER_PORT" not in os.environ:
        os.environ['MASTER_PORT'] = str(experiment_config["export"]["exp_config"]["MASTER_PORT"])
    if "MASTER_ADDR" not in os.environ:
        os.environ['MASTER_ADDR'] = experiment_config["export"]["exp_config"]["MASTER_ADDR"]
    init_dist(launcher="pytorch", backend="nccl")
    dm = SFDataModule(experiment_config["dataset"], num_gpus, seed, logger, "eval", experiment_config["model"]["input_height"], experiment_config["model"]["input_width"])
    dm.setup()
    sf_model = SFModel(experiment_config, phase="eval", num_classes=dm.num_classes)

    CLASSES = dm.CLASSES
    PALETTE = dm.PALETTE

    model_path = experiment_config["export"]["checkpoint"]
    if not model_path:
        raise ValueError("You need to provide the model path for Export.")

    model_to_test = build_segmentor(sf_model.model_cfg, test_cfg=None)
    model_to_test = sf_model._convert_batchnorm(model_to_test)

    _ = load_checkpoint(model_to_test, model_path, map_location='cpu')

    model_to_test.CLASSES = CLASSES
    model_to_test.PALETTE = PALETTE

    output_file = experiment_config["export"]["onnx_file"]
    if not output_file:
        onnx_path = model_path.replace(".pth", ".onnx")
    else:
        onnx_path = output_file

    input_channel = experiment_config["export"]["input_channel"]
    input_height = experiment_config["export"]["input_height"]
    input_width = experiment_config["export"]["input_width"]

    input_shape = [1] + [input_channel, input_height, input_width]
    pytorch2onnx(model_to_test,
                 input_shape,
                 opset_version=experiment_config["export"]["opset_version"],
                 show=False,
                 output_file=onnx_path,
                 verify=False,
                 num_classes=dm.num_classes,
                 test_cfg=sf_model.test_cfg,
                 simplify=experiment_config["export"]["simplify"],
                 logger=logger)

    status_logger.write(message="Completed Export.", status_level=status_logging.Status.SUCCESS)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export_fan", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the Export."""
    try:
        if cfg.export.results_dir is not None:
            results_dir = cfg.export.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "export")
        run_experiment(experiment_config=cfg,
                       results_dir=results_dir)
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Export was interrupted",
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
