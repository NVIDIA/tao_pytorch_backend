# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Distributed communication utilties."""

import functools
import os
from datetime import timedelta
import multiprocessing as mp
import pickle

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    """Check if DDP is initialized."""
    is_dist = True
    if not dist.is_available():
        is_dist = False
    else:
        is_dist = dist.is_initialized() or False
    return is_dist


def get_world_size():
    """Get world size."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_global_rank():
    """Get global rank."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def all_gather(data):
    """Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object.
    Returns:
        list[data]: list of data gathered from each rank.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device='cuda')
    size_list = [torch.tensor([0], device='cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device='cuda'))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device='cuda')
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))  # nosec B301

    return data_list


def get_local_rank():
    """Get local rank."""
    if not is_dist_avail_and_initialized():
        return 0

    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0

    # this is not guaranteed to be set
    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    else:
        raise RuntimeError("Unable to get local rank")


def get_local_size():
    """Get local GPU counts"""
    return torch.cuda.device_count()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo", timeout=timedelta(minutes=60))
    else:
        return dist.group.WORLD


def local_scatter(array):
    """
    Scatter an array from local leader to all local workers.
    The i-th local worker gets array[i].

    Args:
        array: Array with same size of #local workers.
    """
    if get_local_size() <= 1:
        # Just one worker. Do nothing.
        return array[0]
    if get_local_rank() == 0:
        assert len(array) == get_local_size()
        all_gather(array)
    else:
        all_data = all_gather(None)
        array = all_data[get_global_rank() - get_local_rank()]
    return array[get_local_rank()]


# From https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/issues/5#issuecomment-1510676170
def local_broadcast_process_authkey():
    """Broadcast main rank's authkey across different ranks for torchrun."""
    if get_local_size() == 1:
        return
    local_rank = get_local_rank()
    authkey = bytes(mp.current_process().authkey)
    all_keys = all_gather(authkey)
    local_leader_key = all_keys[get_global_rank() - local_rank]
    if authkey != local_leader_key:
        print("Process authkey is different from the key of local leader. This might happen when "
              "workers are launched independently.")
        print("Overwriting local authkey ...")
        mp.current_process().authkey = local_leader_key
