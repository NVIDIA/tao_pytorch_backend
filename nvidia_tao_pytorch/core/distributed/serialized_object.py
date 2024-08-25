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

"""
List serialization code adopted from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
"""

import pickle  # nosec B403
import torch
import numpy as np
import multiprocessing as mp

from nvidia_tao_pytorch.core.distributed.comm import get_local_rank, get_global_rank, get_local_size, local_scatter
from nvidia_tao_pytorch.core.distributed.safe_unpickler import SafeUnpickler
from nvidia_tao_pytorch.core.tlt_logging import logging


class NumpySerializedList:
    """Hold memory in numpy arrays to prevent refcount from increasing through multiprocessing."""

    def __init__(self, lst: list):
        """Initialize NumpySerializedList."""
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if get_local_rank() == 0:
            logging.info(
                f"Serializing {len(lst)} elements to byte tensors and concatenating them all ..."
            )
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)

        if get_local_rank() == 0:
            logging.info(f"Serialized dataset takes {len(self._lst) / 1024**2:.2f} MiB")

    def __len__(self):
        """__len__"""
        return len(self._addr)

    def __getitem__(self, idx):
        """__getitem__"""
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        data_bytes = memoryview(self._lst[start_addr:end_addr])
        return SafeUnpickler(data_bytes, NumpySerializedList).load()


class TorchSerializedList(NumpySerializedList):
    """Hold memory in torch tensors to prevent refcount from increasing through multiprocessing."""

    def __init__(self, lst: list):
        """Initialize NumpySerializedList."""
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        """__getitem__"""
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        data_bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return SafeUnpickler(data_bytes, TorchSerializedList).load()


# NOTE: https://github.com/facebookresearch/mobile-vision/pull/120
# has another implementation that does not use tensors.
class TorchShmSerializedList(TorchSerializedList):
    """
    Hold memory in torch tensors to prevent refcount from increasing through multiprocessing.
    This version works with multi-gpu scenario.
    """

    def __init__(self, lst: list):
        """Initialize TorchShmSerializedList."""
        if get_local_rank() == 0:
            super().__init__(lst)

        if get_local_rank() == 0:
            # Move data to shared memory, obtain a handle to send to each local worker.
            # This is cheap because a tensor will only be moved to shared memory once.
            handles = [None] + [
                bytes(mp.reduction.ForkingPickler.dumps((self._addr, self._lst)))
                for _ in range(get_local_size() - 1)
            ]
        else:
            handles = None
        # Each worker receives the handle from local leader.
        handle = local_scatter(handles)

        if get_local_rank() > 0:
            # Materialize the tensor from shared memory.
            self._addr, self._lst = mp.reduction.ForkingPickler.loads(handle)
            logging.info(
                f"Worker {get_global_rank()} obtains a dataset of length="
                f"{len(self)} from its local leader."
            )
