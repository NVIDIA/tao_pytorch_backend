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

"""Common utilities that could be used across all nlp models."""

import importlib
import os
import pkgutil
import subprocess
import shlex
import sys
from time import time

import nvidia_tao_pytorch.core.download_specs as download_specs
from nvidia_tao_pytorch.core.telemetry.nvml_utils import get_device_details
from nvidia_tao_pytorch.core.telemetry.telemetry import send_telemetry_data
from nvidia_tao_pytorch.core.tlt_logging import logging


def get_subtasks(package):
    """Get supported subtasks for a given task.

    This function lists out the tasks in in the .scripts folder.

    Returns:
        subtasks (dict): Dictionary of files.

    """
    module_path = package.__path__
    modules = {}

    # Collect modules dynamically.
    for _, task, is_package in pkgutil.walk_packages(module_path):
        if is_package:
            continue
        module_name = package.__name__ + '.' + task
        module_details = {
            "module_name": module_name,
            "runner_path": os.path.abspath(importlib.import_module(module_name).__file__),
        }
        modules[task] = module_details

    # Add new command for copying specs.
    modules["download_specs"] = {
        "source_data_dir": os.path.join(os.path.dirname(module_path[0]), "experiment_specs"),
        "runner_path": os.path.abspath(importlib.import_module(download_specs.__name__).__file__),
        "workflow": package.__name__.split(".")[0]
    }
    return modules


def launch(parser, subtasks, network=None):
    """CLI function that executes subtasks.

    Args:
        parser: Created parser object for a given task.
        subtasks: list of subtasks for a given task.
        network (str): name of the network running.
    """
    # Subtasks for a given model.
    parser.add_argument(
        'subtask', default='train', choices=subtasks.keys(), help="Subtask for a given task/model.",
    )
    # Add standard TLT arguments.
    parser.add_argument(
        "-r",
        "--results_dir",
        help="Path to a folder where the experiment outputs should be written. (DEFAULT: ./)",
        required=True,
    )
    parser.add_argument("-k", "--key", help="User specific encoding key to save or load a .tlt model.")
    parser.add_argument("-e", "--experiment_spec_file", help="Path to the experiment spec file.", default=None)
    parser.add_argument(
        "-g", "--gpus", help="Number of GPUs to use. The default value is 1.", default=1,
        type=int
    )
    parser.add_argument(
        "-m", "--resume_model_weights", help="Path to a pre-trained model or model to continue training."
    )
    parser.add_argument(
        "-o", "--output_specs_dir", help="Path to a target folder where experiment spec files will be downloaded."
    )

    # Parse the arguments.
    args, unknown_args = parser.parse_known_args()
    process_passed = True

    script_args = ""
    # Process spec file for all commands except the one for getting spec files ;)
    if args.subtask not in ["download_specs", "pitch_stats"]:
        # Make sure the user provides spec file.
        if args.experiment_spec_file is None:
            print("ERROR: The subtask `{}` requires the following argument: -e/--experiment_spec_file".format(args.subtask))
            exit(1)

        # Make sure the file exists!
        if not os.path.exists(args.experiment_spec_file):
            print("ERROR: The indicated experiment spec file `{}` doesn't exist!".format(args.experiment_spec_file))
            exit(1)

        # Split spec file_path into config path and config name.
        path, name = os.path.split(args.experiment_spec_file)
        if path != '':
            script_args += " --config-path " + os.path.realpath(path)
        script_args += " --config-name " + name
    # And add other params AFTERWARDS!

    # Translate results dir to exp_manager - optional for now! (as 4/6 workflows weren't adapted!)
    script_args += " exp_manager.explicit_log_dir=" + args.results_dir

    # Set gpus - override only in the case of tasks that use GPUs (assumption for now!).
    if args.subtask in ["train", "finetune", "evaluate"]:
        script_args += " trainer.gpus=" + str(args.gpus)

    # Don't resume for 1) data_convert and 2) train from scratch.
    if args.subtask in ["finetune", "evaluate", "infer", "infer_onnx", "export"]:
        if args.resume_model_weights is not None:
            script_args += " restore_from=" + args.resume_model_weights

    # Add encryption key.
    if args.subtask in ["train", "finetune", "evaluate", "infer", "infer_onnx", "export"]:
        if args.key is not None:
            script_args += " encryption_key=" + args.key

    if args.subtask == "download_specs":
        # Set target_data_dir
        if args.output_specs_dir is not None:
            script_args += " target_data_dir=" + args.output_specs_dir
        else:
            print("ERROR: The subtask `{}` requires the following argument: -o/--output_specs_dir".format(args.subtask))
            exit(1)
        # Set the remaining params.
        script_args += " source_data_dir=" + subtasks[args.subtask]["source_data_dir"]
        script_args += " workflow=" + subtasks[args.subtask]["workflow"]

    # Find relevant module and pass args.
    script = subtasks[args.subtask]["runner_path"]

    # Pass unknown args to call
    unknown_args_as_str = " ".join(unknown_args)
    # Create a system call.
    call = "python " + script + script_args + " " + unknown_args_as_str

    start = time()
    try:
        # Run the script.
        subprocess.check_call(
            shlex.split(call),
            shell=False,
            stdout=sys.stdout,
            stderr=sys.stdout
        )
    except (KeyboardInterrupt, SystemExit) as e:
        logging.info("Command was interrupted due to ", e)
        process_passed = True
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            logging.info(e.output)
        process_passed = False
    end = time()
    time_lapsed = int(end - start)

    try:
        gpu_data = list()
        for device in get_device_details():
            gpu_data.append(device.get_config())
        logging.info("Sending telemetry data.")
        send_telemetry_data(
            network,
            args.subtask,
            gpu_data,
            num_gpus=args.gpus,
            time_lapsed=time_lapsed,
            pass_status=process_passed
        )
    except Exception as e:
        logging.warning("Telemetry data couldn't be sent, but the command ran successfully.")
        logging.warning(f"[Error]: {e}")
        pass

    if not process_passed:
        logging.warning("Execution status: FAIL")
        exit(1)  # returning non zero return code from the process.

    logging.info("Execution status: PASS")
