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

"""'Entry point' script running subtasks related to PointPillars.
"""

import importlib
import os
import pkgutil
import argparse
import subprocess
import sys
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.config import (
    cfg,
    cfg_from_yaml_file,
)
import nvidia_tao_pytorch.pointcloud.pointpillars.scripts as scripts


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

    return modules


def launch(parser, subtasks):
    """CLI function that executes subtasks.

    Args:
        parser: Created parser object for a given task.
        subtasks: list of subtasks for a given task.
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
    parser.add_argument("--gpus", "-g", type=int, default=1, help="Number of GPUs")
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

    script_args += " --cfg_file " + args.experiment_spec_file
    # And add other params AFTERWARDS!
    if args.subtask in ["train", "evaluate", "inference", "prune"]:
        if args.results_dir:
            script_args += " --output_dir " + args.results_dir
    # Add encryption key.
    if args.subtask in ["train", "evaluate", "inference", "export", "prune"]:
        if args.key is not None:
            script_args += " --key " + args.key
    # Number of GPUs
    if args.gpus > 1:
        if args.subtask != "train":
            raise ValueError("Only train task support multi-gpu")
        else:
            script_args += " --gpus " + str(args.gpus)

    # Find relevant module and pass args.
    script = subtasks[args.subtask]["runner_path"]

    # Pass unknown args to call
    unknown_args_as_str = " ".join(unknown_args)
    # Create a system call.
    if args.gpus == 1:
        call = "python " + script + script_args + " " + unknown_args_as_str
    else:
        cfg_from_yaml_file(expand_path(args.experiment_spec_file), cfg)
        call = (f"python -m torch.distributed.launch --nproc_per_node={args.gpus} --rdzv_endpoint=localhost:{cfg.train.tcp_port} " + script + script_args + " " + unknown_args_as_str)
    print(call)
    try:
        # Run the script.
        subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(e.output)
        exit(1)


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "pointpillars", add_help=True, description="TAO PointPillars"
    )
    # Build list of subtasks by inspecting the package.
    subtasks = get_subtasks(scripts)
    # Parse the arguments and launch the subtask.
    launch(parser, subtasks)


if __name__ == '__main__':
    main()
