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

"""TensorRT inference model builder for PointPillars."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
from io import open  # Python 2/3 compatibility.  pylint: disable=W0622
import logging
import os

import numpy as np
import pycuda.autoinit  # noqa pylint: disable=W0611
import pycuda.driver as cuda
import tensorrt as trt

from nvidia_tao_pytorch.pointcloud.pointpillars.tools.export.tensorrt import (
    _create_tensorrt_logger
)

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)

BINDING_TO_DTYPE = {
    "points": np.float32,
    "num_points": np.int32,
    "output_boxes": np.float32,
    "num_boxes": np.int32,
}


class CacheCalibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator class that loads a cache file directly.

    This inherits from ``trt.IInt8EntropyCalibrator2`` to implement
    the calibration interface that TensorRT needs to calibrate the
    INT8 quantization factors.

    Args:
        calibration_filename (str): name of calibration to read/write to.
    """

    def __init__(self, cache_filename, *args, **kwargs):
        """Init routine."""
        super(CacheCalibrator, self).__init__(*args, **kwargs)
        self._cache_filename = cache_filename

    def get_batch(self, names):
        """Dummy method since we are going to use cache file directly.

        Args:
            names (list): list of memory bindings names.
        """
        return None

    def get_batch_size(self):
        """Return batch size."""
        return 8

    def read_calibration_cache(self):
        """Read calibration from file."""
        if os.path.exists(self._cache_filename):
            with open(self._cache_filename, "rb") as f:
                return f.read()
        else:
            raise ValueError("""Calibration cache file
                                not found: {}""".format(self._cache_filename))

    def write_calibration_cache(self, cache):
        """Do nothing since we already have cache file.

        Args:
            cache (memoryview): buffer to read calibration data from.
        """
        return


class Engine(object):
    """A class to represent a TensorRT engine.

    This class provides utility functions for performing inference on
    a TensorRT engine.

    Args:
        engine: the CUDA engine to wrap.
    """

    def __init__(self, engine, batch_size):
        """Initialization routine."""
        self._engine = engine
        self._context = None
        self._batch_size = batch_size
        self._actual_batch_size = batch_size

    @contextlib.contextmanager
    def _create_context(self):
        """Create an execution context and allocate input/output buffers."""
        try:
            with self._engine.create_execution_context() as self._context:
                self._device_buffers = []
                self._host_buffers = []
                self._input_binding_ids = {}
                self.points_batch_size = self._context.get_binding_shape(0)[1]
                for i in range(self._engine.num_bindings):
                    tensor_shape = self._engine.get_binding_shape(i)
                    elt_count = trt.volume(tensor_shape)
                    binding_name = self._engine.get_binding_name(i)
                    dtype = BINDING_TO_DTYPE[binding_name]
                    if self._engine.binding_is_input(i):
                        self._input_binding_ids[binding_name] = i
                        page_locked_mem = None
                    else:
                        page_locked_mem = cuda.pagelocked_empty(elt_count, dtype=dtype)
                        page_locked_mem = page_locked_mem.reshape(*tensor_shape)
                    # Allocate memory.
                    self._host_buffers.append(page_locked_mem)
                    _mem_alloced = cuda.mem_alloc(elt_count * np.dtype(dtype).itemsize)
                    self._device_buffers.append(_mem_alloced)
                if not self._input_binding_ids:
                    raise RuntimeError("No input bindings detected.")
                # Create stream and events to measure timings.
                self._stream = cuda.Stream()
                self._start = cuda.Event()
                self._end = cuda.Event()
                yield
        finally:
            # Release context and allocated memory.
            self._release_context()

    def _do_infer(self, batch):
        # make sure it is contiguous array
        bindings = [int(device_buffer) for device_buffer in self._device_buffers]
        # Transfer input data to device.
        for node_name, array in batch.items():
            if node_name == "points":
                if isinstance(array, list):
                    # Convert list to array
                    array_concat = []
                    for ar in array:
                        if ar.shape[0] > self.points_batch_size:
                            raise ValueError(
                                f"Input LIDAR file has points number: {ar.shape[0]} larger than "
                                f"the one specified in ONNX model: {self.points_batch_size}, please set "
                                "cfg.model.inference.max_points_num to a larger "
                                "value and re-export to ONNX model and TensorRT "
                                "engine"
                            )
                        array_concat.append(
                            np.pad(ar, ((0, self.points_batch_size - ar.shape[0]), (0, 0)))
                        )
                    array = np.stack(array_concat, axis=0)
                if len(array.shape) == 2:
                    array = np.expand_dims(array, axis=0)
                if array.shape[1] > self.points_batch_size:
                    raise ValueError(
                        f"Input LIDAR file has points number: {array.shape[1]} larger than "
                        f"the one specified in ONNX model: {self.points_batch_size}, please set "
                        "cfg.model.inference.max_points_num to a larger "
                        "value and re-export to ONNX model and TensorRT "
                        "engine"
                    )
                if array.shape[1] < self.points_batch_size:
                    array = np.pad(
                        array,
                        ((0, 0), (0, self.points_batch_size - array.shape[1]), (0, 0)),
                        constant_values=0.
                    )
                # The last batch can be smaller
                if array.shape[0] < self._batch_size:
                    self._actual_batch_size = array.shape[0]
                    delta_batch = self._batch_size - array.shape[0]
                    pad_array = np.repeat(array[0:1, ...], delta_batch, axis=0)
                    array = np.concatenate([array, pad_array], axis=0)
            elif node_name == "num_points":
                if isinstance(array, list):
                    array = np.stack(array)
                if array.shape[0] < self._batch_size:
                    self._actual_batch_size = array.shape[0]
                    delta_batch = self._batch_size - array.shape[0]
                    pad_array = np.repeat(array[0:1], delta_batch, axis=0)
                    array = np.concatenate([array, pad_array], axis=0)
                array = array.astype("int32")
            else:
                raise KeyError(f"Unknown input data: {node_name}")
            array = np.ascontiguousarray(array)
            cuda.memcpy_htod_async(
                self._device_buffers[self._input_binding_ids[node_name]],
                array,
                self._stream
            )
        # Execute model.
        self._start.record(self._stream)
        self._context.execute_async_v2(bindings, self._stream.handle, None)
        self._end.record(self._stream)
        self._end.synchronize()
        # Transfer predictions back.
        outputs = dict()
        for i in range(self._engine.num_bindings):
            if not self._engine.binding_is_input(i):
                cuda.memcpy_dtoh_async(self._host_buffers[i], self._device_buffers[i],
                                       self._stream)
                out = np.copy(self._host_buffers[i][:self._actual_batch_size])
                name = self._engine.get_binding_name(i)
                outputs[name] = out
        return outputs

    def _release_context(self):
        """Release context and allocated memory."""
        for device_buffer in self._device_buffers:
            device_buffer.free()
            del (device_buffer)

        for host_buffer in self._host_buffers:
            del (host_buffer)

        del (self._start)
        del (self._end)
        del (self._stream)

    def infer(self, batch):
        """Perform inference on a Numpy array.

        Args:
            batch (ndarray): array to perform inference on.
        Returns:
            A dictionary of outputs where keys are output names
            and values are output tensors.
        """
        with self._create_context():
            outputs = self._do_infer(batch)
        return outputs

    def infer_iterator(self, iterator):
        """Perform inference on an iterator of Numpy arrays.

        This method should be preferred to ``infer`` when performing
        inference on multiple Numpy arrays since this will re-use
        the allocated execution and memory.

        Args:
            iterator: an iterator that yields Numpy arrays.
        Yields:
            A dictionary of outputs where keys are output names
            and values are output tensors, for each array returned
            by the iterator.
        Returns:
            None.
        """
        with self._create_context():
            for batch in iterator:
                outputs = self._do_infer(batch)
                yield outputs

    def save(self, filename):
        """Save serialized engine into specified file.

        Args:
            filename (str): name of file to save engine to.
        """
        with open(filename, "wb") as outf:
            outf.write(self._engine.serialize())


class TrtModel(object):
    """A TensorRT model builder for FasterRCNN model inference based on TensorRT.

    The TensorRT model builder builds a TensorRT engine from the engine file from the
    tlt-converter and do inference in TensorRT. We use this as a way to verify the
    TensorRT inference functionality of the FasterRCNN model.
    """

    def __init__(self,
                 trt_engine_file,
                 batch_size):
        """Initialize the TensorRT model builder."""
        self._trt_engine_file = trt_engine_file
        self._batch_size = batch_size
        self._trt_logger = _create_tensorrt_logger()
        trt.init_libnvinfer_plugins(self._trt_logger, "")

    def set_engine(self, trt_engine):
        """Set engine."""
        self.engine = Engine(trt_engine,
                             self._batch_size)

    def load_trt_engine_file(self):
        """load TensorRT engine file generated by tlt-converter."""
        runtime = trt.Runtime(self._trt_logger)
        with open(self._trt_engine_file, 'rb') as f:
            _engine = f.read()
            logger.info("Loading existing TensorRT engine and "
                        "ignoring the specified batch size and data type"
                        " information in spec file.")
            self.engine = Engine(runtime.deserialize_cuda_engine(_engine),
                                 self._batch_size)

    def build_or_load_trt_engine(self):
        """Build engine or load engine depends on whether a trt engine is available."""
        if self._trt_engine_file is not None:
            # load engine
            logger.info("""Loading TensorRT engine file: {}
                        for inference.""".format(self._trt_engine_file))
            self.load_trt_engine_file()
        else:
            raise ValueError("""A TensorRT engine file should
                              be provided for TensorRT based inference.""")

    def predict(self, batch):
        """Do inference with TensorRT engine."""
        return self.engine.infer(batch)
