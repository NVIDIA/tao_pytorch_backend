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
Evaluation of Segformer model.
"""
import os

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.segformer.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.utils.config import MMSegmentationConfig

# Triggers build of custom modules
from nvidia_tao_pytorch.cv.segformer.model import * # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.segformer.dataloader import * # noqa pylint: disable=W0401, W0614

from mmengine.config import Config
from mmseg.apis import MMSegInferencer


def run_experiment(experiment_config, results_dir):
    """Start the inference.
    Args:
        experiment_config (Dict): Config dictionary containing epxeriment parameters
        results_dir (str): Results dir to save the inference images

    """
    os.makedirs(results_dir, exist_ok=True)
    # Set the logger
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logger = status_logging.get_status_logger()
    # log to file
    status_logger.write(message="**********************Start logging for Inference**********************.")

    mmseg_config = MMSegmentationConfig(experiment_config, phase="inference")
    eval_cfg = mmseg_config.updated_config

    # This is provided in notebook
    checkpoint = experiment_config.inference.checkpoint

    eval_cfg["visualizer"]["save_dir"] = results_dir
    mmseg_model_config = Config(eval_cfg)

    # @sean: we manually delete our custom load annotations for inference visualization
    # This is done in MMSegInferencer, but only does it for "LoadAnnotations, our custom-named one is skipped
    pipeline_cfg = mmseg_model_config.test_dataloader.dataset.pipeline
    # Loading annotations is also not applicable
    for i, transform in enumerate(pipeline_cfg):
        if transform['type'] == "TAOLoadAnnotations":
            del pipeline_cfg[i]

    classes = mmseg_model_config.test_dataloader.dataset.metainfo.classes
    palette = mmseg_model_config.test_dataloader.dataset.metainfo.palette

    inferencer = MMSegInferencer(
        model=mmseg_model_config,
        weights=checkpoint,
        classes=classes,
        palette=palette)

    img_dir = mmseg_model_config.test_dataloader.dataset.data_prefix.img_path
    inferencer(img_dir, out_dir=results_dir, img_out_dir='vis_tao', pred_out_dir='mask_tao')

    status_logger.write(message="Completed Inference.", status_level=status_logging.Status.SUCCESS)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="test_isbi", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the Inference process."""
    try:
        # Obfuscate logs.
        if cfg.inference.results_dir is not None:
            results_dir = cfg.inference.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "inference")
        run_experiment(experiment_config=cfg,
                       results_dir=results_dir)
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
