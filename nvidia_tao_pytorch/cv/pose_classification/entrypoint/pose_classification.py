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

"""'Entry point' script running subtasks related to pose classification."""
import importlib
import os
import pkgutil
import argparse
import subprocess
import sys
from time import time

import nvidia_tao_pytorch.cv.pose_classification.scripts as scripts
from nvidia_tao_pytorch.core.telemetry.nvml_utils import get_device_details
from nvidia_tao_pytorch.core.telemetry.telemetry import send_telemetry_data


def get_subtasks(package):
    """
    Get supported subtasks for a given task.

    This function dynamically collects all modules within a specified package. This function is particularly useful
    for gathering all scripts within a package for a specific task.

    Args:
        package (module): The package to gather subtask modules from.

    Returns:
        dict: A dictionary where the keys are the names of the subtasks and the values are dictionaries containing
              the full module name and absolute path of the subtask module.
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

    return modules


def launch(parser, subtasks, network=None):
    """
    CLI function that executes subtasks.

    This function uses the argparse module to parse command line arguments for a given task. It processes these arguments
    and then runs the corresponding subtask script with these arguments. It also collects telemetry data and sends it.

    Args:
        parser (argparse.ArgumentParser): An ArgumentParser object for parsing command line arguments.
        subtasks (dict): A dictionary where the keys are the names of the subtasks and the values are dictionaries containing
                         the full module name and absolute path of the subtask module.
        network (str, optional): The name of the network running training. Defaults to None.
    """
    # Subtasks for a given model.
    parser.add_argument(
        'subtask', default='train', choices=subtasks.keys(), help="Subtask for a given task/model.",
    )
    # Add standard TLT arguments.
    parser.add_argument(
        "-r",
        "--results_dir",
        help="Path to a folder where the experiment outputs should be written.",
        default=None,
        required=False,
    )
    parser.add_argument("-k", "--key", help="User specific encoding key to save or load a .tlt model.")
    parser.add_argument("-e", "--experiment_spec_file", help="Path to the experiment spec file.", default=None)

    # Parse the arguments.
    args, unknown_args = parser.parse_known_args()

    script_args = ""
    # Process spec file for all commands except the one for getting spec files ;)
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

    if args.subtask in ["train"]:
        if args.results_dir:
            script_args += " results_dir=" + args.results_dir

    # Add encryption key.
    if args.subtask in ["train", "evaluate", "inference", "export", "dataset_convert"]:
        if args.key is not None:
            script_args += " encryption_key=" + args.key

    # Find relevant module and pass args.
    script = subtasks[args.subtask]["runner_path"]

    # Pass unknown args to call
    unknown_args_as_str = " ".join(unknown_args)
    # Create a system call.
    call = "python " + script + script_args + " " + unknown_args_as_str

    process_passed = True
    start = time()
    try:
        # Run the script.
        subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)
    except (KeyboardInterrupt, SystemExit):
        print("Command was interrupted.")
        process_passed = True
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(e.output)
        process_passed = False
    end = time()
    time_lapsed = int(end - start)

    try:
        gpu_data = list()
        for device in get_device_details():
            gpu_data.append(device.get_config())
        send_telemetry_data(
            network,
            args.subtask,
            gpu_data,
            num_gpus=1,
            time_lapsed=time_lapsed,
            pass_status=process_passed
        )
    except Exception as e:
        print("Telemetry data couldn't be sent, but the command ran successfully.")
        print(f"[WARNING]: {e}")
        pass

    if not process_passed:
        print("Execution status: FAIL")
        exit(1)  # returning non zero return code from the process.

    print("Execution status: PASS")


def main():
    """
    Main entrypoint function.

    This function creates a command line argument parser, gathers all subtasks in the 'scripts' package using the
    'get_subtasks' function, and then executes the chosen subtask using the 'launch' function.
    """
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "pose_classification", add_help=True, description="Transfer Learning Toolkit"
    )

    # Build list of subtasks by inspecting the package.
    subtasks = get_subtasks(scripts)

    # Parse the arguments and launch the subtask.
    launch(parser, subtasks, network="pose_classification")


if __name__ == '__main__':
    main()
