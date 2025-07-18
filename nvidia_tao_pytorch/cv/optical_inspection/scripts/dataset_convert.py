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

"""Dataset convert - Integrates into Factory camera system pipeline"""

import pandas as pd
import os
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_core.config.optical_inspection.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.optical_inspection.utils.data_preprocess import output_combined_dataset, preprocess_boards_cam


def run_experiment(experiment_config):
    """Run Experiment"""
    dataset_convert_config = experiment_config["dataset_convert"]
    root_data_path = dataset_convert_config["root_dataset_dir"]
    train_csv_path = dataset_convert_config["train_pcb_dataset_dir"]
    test_csv_path = dataset_convert_config["val_pcb_dataset_dir"]
    all_csv_path = dataset_convert_config["all_pcb_dataset_dir"]
    output_dir = dataset_convert_config['data_convert_output_dir']
    golden_csv_path = dataset_convert_config["golden_csv_dir"]
    project_name = dataset_convert_config["project_name"]

    BOT_TOP = dataset_convert_config["bot_top"]
    df = preprocess_boards_cam(root_data_path + train_csv_path, BOT_TOP)
    df_0 = preprocess_boards_cam(root_data_path + test_csv_path, BOT_TOP)
    df['isValid'], df_0['isValid'] = 0, 1
    df = pd.concat([df, df_0], axis=0)

    if project_name != 'all':
        df = df.loc[df['project'] == project_name]
    logging.info("Using projects:\n {}".format('\n'.join(df['project'].unique())))
    df_combined = df

    # create_golden_forprojects(df_combined, root_data_path, golden_csv_path,all_csv_path, project_list)

    output_combined_dataset(df_combined,
                            data_path=root_data_path,
                            golden_csv_path=golden_csv_path,
                            compare_csv_path=all_csv_path,
                            output_dir=output_dir,
                            movegoldenimgs=False,
                            savemaplight=False,
                            valid=False,
                            project_name=project_name)

    output_combined_dataset(df_combined,
                            data_path=root_data_path,
                            golden_csv_path=golden_csv_path,
                            compare_csv_path=all_csv_path,
                            output_dir=output_dir,
                            movegoldenimgs=False,
                            savemaplight=False,
                            valid=True,
                            project_name=project_name
                            )

    # zip_tar_images(df_combined, BOT_TOP, root_data_path)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment",
    schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the dataset conversion process."""
    try:
        run_experiment(experiment_config=cfg)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Dataset convert finished successfully."
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
