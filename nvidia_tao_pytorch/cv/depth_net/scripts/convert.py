# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Prepare MonoDataset/StereoDataset splits in txt format."""

import os
import glob
import numpy as np

from nvidia_tao_core.config.depth_net.dataset import DNDatasetConvertConfig
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.utilities import check_and_create
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


def create_split(experiment_config, results_dir):
    """Create split and save txt file for model input."""
    data_root = experiment_config["data_root"]
    split_ratio = experiment_config["split_ratio"]
    image_dir_name = experiment_config["image_dir_name"]
    depth_dir_name = experiment_config["depth_dir_name"]
    right_image_dir_name = experiment_config["right_image_dir_name"]
    depth_extension = experiment_config["depth_extension"]
    image_extension = experiment_config["image_extension"]
    directory_depth = experiment_config["directory_depth"]

    glob_pattern = ''
    for i in range(directory_depth):
        glob_pattern += '*/'

    seq_list = glob.glob(os.path.join(data_root, glob_pattern))
    if split_ratio == 1.0 or split_ratio == 0.0:
        output_file = os.path.join(results_dir, 'test.txt')
        with open(output_file, 'w') as fout:
            for seq_path in seq_list:
                seq_name = os.path.split(seq_path)[-1]

                if 'splits' not in seq_name or '_detail' not in seq_name:
                    left_path = os.path.join(seq_path, image_dir_name)
                    disp_path = os.path.join(seq_path, depth_dir_name)
                    image_list = sorted(glob.glob(os.path.join(left_path, f'**/*.{image_extension}'), recursive=True))
                    depth_list = sorted(glob.glob(os.path.join(disp_path, f'**/*.{depth_extension}'), recursive=True))
                    if right_image_dir_name != "":
                        right_path = left_path.replace(image_dir_name, right_image_dir_name)
                        right_image_list = sorted(glob.glob(os.path.join(right_path, f'**/*.{image_extension}'), recursive=True))
                    else:
                        right_image_list = []

                    for index, (left_image, disp_image) in enumerate(zip(image_list, depth_list)):
                        if len(right_image_list) > 1:
                            right_image = right_image_list[index]
                            print("{} {} {}".format(left_image, right_image, disp_image), file=fout)
                        else:
                            print("{} {}".format(left_image, disp_image), file=fout)
    else:
        output_train_file = os.path.join(results_dir, 'train.txt')
        output_val_file = os.path.join(results_dir, 'val.txt')

        with open(output_train_file, 'w') as fout, open(output_val_file, 'w') as fout2:
            for seq_path in seq_list:
                seq_name = os.path.split(seq_path)[-1]

                if 'splits' not in seq_name or '_detail' not in seq_name:
                    left_path = os.path.join(seq_path, image_dir_name)
                    disp_path = os.path.join(seq_path, depth_dir_name)
                    image_list = sorted(glob.glob(os.path.join(left_path, f'**/*.{image_extension}'), recursive=True))
                    depth_list = sorted(glob.glob(os.path.join(disp_path, f'**/*.{depth_extension}'), recursive=True))
                    if right_image_dir_name != "":
                        right_path = left_path.replace(image_dir_name, right_image_dir_name)
                        right_image_list = sorted(glob.glob(os.path.join(right_path, f'**/*.{image_extension}'), recursive=True))
                    else:
                        right_image_list = []

                    train_len = int(len(image_list) * split_ratio)
                    s1 = np.random.choice(range(len(image_list)), train_len, replace=False)
                    s2 = list(set(range(len(image_list))) - set(s1))
                    left_train_list = [image_list[i] for i in s1]
                    left_val_list = [image_list[i] for i in s2]
                    if len(right_image_list) > 1:
                        right_train_list = [right_image_list[i] for i in s1]
                        right_val_list = [right_image_list[i] for i in s2]
                    else:
                        right_train_list = []
                        right_val_list = []
                    depth_train_list = [depth_list[i] for i in s1]
                    depth_val_list = [depth_list[i] for i in s2]

                    for index, (left_image, disp_image) in enumerate(zip(left_train_list, depth_train_list)):
                        if len(right_train_list) > 1:
                            right_image = right_train_list[index]
                            print("{} {} {}".format(left_image, right_image, disp_image), file=fout)
                        else:
                            print("{} {}".format(left_image, disp_image), file=fout)

                    for index, (left_image, disp_image) in enumerate(zip(left_val_list, depth_val_list)):
                        if len(right_val_list) > 1:
                            right_image = right_val_list[index]
                            print("{} {} {}".format(left_image, right_image, disp_image), file=fout2)
                        else:
                            print("{} {}".format(left_image, disp_image), file=fout2)
        fout2.close()
    fout.close()


def run_experiment(experiment_config,
                   results_dir):
    """Start the Data Converter."""
    check_and_create(results_dir)
    create_split(experiment_config, results_dir)

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
        message="Starting DepthNet dataset convert"
    )


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="convert", schema=DNDatasetConvertConfig
)
def main(cfg: DNDatasetConvertConfig) -> None:
    """Run the convert dataset process."""
    try:
        run_experiment(experiment_config=cfg,
                       results_dir=cfg.results_dir)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Dataset convert finished successfully"
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Dataset convert was interrupted",
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
