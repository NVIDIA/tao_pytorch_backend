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

"""'Entry point' script running subtasks related to ocdnet."""
import os
import argparse
import subprocess  # nosec B404
import sys
import nvidia_tao_pytorch.cv.ocdnet.scripts as scripts
from nvidia_tao_pytorch.core.entrypoint import get_subtasks


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

    # Split spec file_path into config path and config name.
    path, name = os.path.split(args.experiment_spec_file)
    if path != '':
        script_args += " --config-path " + os.path.realpath(path)
    script_args += " --config-name " + name
    # And add other params AFTERWARDS!

    if args.subtask in ["train"]:
        if args.results_dir:
            script_args += " train.results_dir=" + args.results_dir

    if args.gpus > 1:
        if args.subtask != "train":
            raise ValueError("Only train task support multi-gpu")

    # Find relevant module and pass args.
    script = subtasks[args.subtask]["runner_path"]

    # Pass unknown args to call
    unknown_args_as_str = " ".join(unknown_args)

    # Create a system call.
    call = "python " + script + script_args + " " + unknown_args_as_str

    try:
        # Run the script.
        subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)  # nosec B602
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(e.output)
        exit(1)


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "ocdnet", add_help=True, description="TAO Toolkit OCD"
    )

    # Build list of subtasks by inspecting the package.
    subtasks = get_subtasks(scripts)

    # Parse the arguments and launch the subtask.
    launch(parser, subtasks)


if __name__ == '__main__':
    main()
