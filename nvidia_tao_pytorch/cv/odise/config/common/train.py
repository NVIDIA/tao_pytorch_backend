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

train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter="???",
    amp=dict(
        enabled=False,
        opt_level=None,
    ),  # options for Automatic Mixed Precision
    grad_clip=None,
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=5000, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period="${train.checkpointer.period}",
    log_period=50,
    device="cuda",
    seed=42,
    # ...
    wandb=dict(
        enable_writer=False,
        resume=False,
        project="ODISE",
    ),
    cfg_name="",
    run_name="",
    run_tag="",
    reference_world_size=0,
)
