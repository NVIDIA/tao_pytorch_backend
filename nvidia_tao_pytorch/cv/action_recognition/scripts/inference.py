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
Inference on single patch.
"""
import os
import torch

from tqdm import tqdm
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.action_recognition.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.action_recognition.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.action_recognition.inference.inferencer import Inferencer
from nvidia_tao_pytorch.cv.action_recognition.model.pl_ar_model import ActionRecognitionModel
from nvidia_tao_pytorch.cv.action_recognition.utils.common_utils import check_and_create
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook


def run_experiment(experiment_config, model_path, key, output_dir,
                   batch_size=1, inference_dataset_dir=None):
    """Start the inference."""
    check_and_create(output_dir)
    # Set status logging
    status_file = os.path.join(output_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting Action recognition inference"
    )

    gpu_id = experiment_config.inference.gpu_id
    torch.cuda.set_device(gpu_id)
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    # build dataloader
    model_config = experiment_config["model"]
    label_map = experiment_config["dataset"]["label_map"]
    output_shape = [experiment_config["model"]["input_height"],
                    experiment_config["model"]["input_width"]]
    sample_dict = {}
    for sample_id in os.listdir(inference_dataset_dir):
        sample_path = os.path.join(inference_dataset_dir, sample_id)
        sample_dict[sample_path] = "unknown"

    aug_config = experiment_config["dataset"]["augmentation_config"]
    dataloader = build_dataloader(sample_dict=sample_dict,
                                  model_config=model_config,
                                  dataset_mode="inf",
                                  output_shape=output_shape,
                                  input_type=model_config["input_type"],
                                  label_map=label_map,
                                  batch_size=batch_size,
                                  workers=experiment_config["dataset"]["workers"],
                                  eval_mode=experiment_config["inference"]["video_inf_mode"],
                                  augmentation_config=aug_config,
                                  num_segments=experiment_config["inference"]["video_num_segments"])

    # build inferencer @TODO TRT support
    model = ActionRecognitionModel.load_from_checkpoint(model_path,
                                                        map_location="cpu",
                                                        experiment_spec=experiment_config)
    infer = Inferencer(model, ret_prob=False)
    # do inference
    progress = tqdm(dataloader)
    id2name = {v: k for k, v in label_map.items()}
    sample_result_dict = {}
    with torch.no_grad():
        for sample_path, data in progress:
            batch_size = len(sample_path)
            pred_id = infer.inference(data)
            pred_name = []
            for label_idx in pred_id:
                pred_name.append(id2name[label_idx])
            for idx in range(batch_size):
                if sample_path[idx] not in sample_result_dict:
                    sample_result_dict[sample_path[idx]] = [pred_name[idx]]
                else:
                    sample_result_dict[sample_path[idx]].append(pred_name[idx])

    # save the output and visualize
    for k, v in sample_result_dict.items():
        print("{} : {}".format(k, v))


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        if cfg.inference.results_dir is not None:
            results_dir = cfg.inference.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "inference")
        run_experiment(experiment_config=cfg,
                       key=cfg.encryption_key,
                       output_dir=results_dir,
                       model_path=cfg.inference.checkpoint,
                       batch_size=cfg.inference.batch_size,
                       inference_dataset_dir=cfg.inference.inference_dataset_dir)
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
