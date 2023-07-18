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

"""Define entrypoint to run tasks for MAL."""
import importlib
import os
import pkgutil
import argparse
import subprocess
import sys

from nvidia_tao_pytorch.cv.mal import scripts


def get_subtasks(package):
    """Get supported subtasks for a given task.

    This function lists out the tasks in in the .scripts folder.

    Args:
        script (Module): Input scripts.

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
    # Add standard TAO arguments.
    parser.add_argument(
        "-r",
        "--results_dir",
        help="Path to a folder where the experiment outputs should be written. (DEFAULT: ./)",
        required=False,
    )
    parser.add_argument(
        "-e",
        "--experiment_spec_file",
        help="Path to the experiment spec file.",
        required=True)
    parser.add_argument(
        "-g",
        "--gpus",
        help="Number of GPUs or gpu index to use.",
        type=str,
        default=None
    )
    parser.add_argument(
        "-o",
        "--output_specs_dir",
        help="Path to a target folder where experiment spec files will be downloaded.",
        default=None
    )

    # Parse the arguments.
    args, unknown_args = parser.parse_known_args()

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
        # Find relevant module and pass args.

    if args.subtask in ["train", "evaluate", "inference"]:
        if args.results_dir:
            script_args += " results_dir=" + args.results_dir
        if args.gpus:
            try:
                script_args += f" gpu_ids=[{','.join([str(i) for i in range(int(args.gpus))])}]"
            except ValueError:
                script_args += f" gpu_ids={args.gpus}"

    script = subtasks[args.subtask]["runner_path"]

    # Pass unknown args to call
    unknown_args_as_str = " ".join(unknown_args)
    # Create a system call.
    call = "python " + script + script_args + " " + unknown_args_as_str

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
        "MAL",
        add_help=True,
        description="TAO Toolkit entrypoint for MAL"
    )

    # Build list of subtasks by inspecting the scripts package.
    subtasks = get_subtasks(scripts)

    # Parse the arguments and launch the subtask.
    launch(
        parser, subtasks
    )


if __name__ == '__main__':
    main()
