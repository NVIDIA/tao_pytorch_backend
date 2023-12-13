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

"""PointPillars INT8 calibration APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
from io import open  # Python 2/3 compatibility.  pylint: disable=W0622
import logging
import os
import sys
import traceback

import numpy as np

from nvidia_tao_pytorch.pointcloud.pointpillars.tools.export.decorators import override, subclass


"""Logger for data export APIs."""
logger = logging.getLogger(__name__)
try:
    import pycuda.autoinit  # noqa pylint: disable=W0611
    import pycuda.driver as cuda
    import tensorrt as trt
except ImportError:
    # TODO(xiangbok): we should probably do this test in modulus/export/__init__.py.
    logger.warning(
        "Failed to import TRT and/or CUDA. TensorRT optimization "
        "and inference will not be available."
    )

DEFAULT_MAX_WORKSPACE_SIZE = 1 << 30
DEFAULT_MAX_BATCH_SIZE = 100
DEFAULT_MIN_BATCH_SIZE = 1
DEFAULT_OPT_BATCH_SIZE = 100

# Array of TensorRT loggers. We need to keep global references to
# the TensorRT loggers that we create to prevent them from being
# garbage collected as those are referenced from C++ code without
# Python knowing about it.
tensorrt_loggers = []

# If we were unable to load TensorRT packages because TensorRT is not installed
# then we will stub the exported API.
if "trt" not in globals():
    keras_to_tensorrt = None
    load_tensorrt_engine = None
else:
    # We were able to load TensorRT package so we are implementing the API
    def _create_tensorrt_logger(verbose=False):
        """Create a TensorRT logger.

        Args:
            verbose (bool): whether to make the logger verbose.
        """
        if str(os.getenv('SUPPRES_VERBOSE_LOGGING', '0')) == '1':
            # Do not print any warnings in TLT docker
            trt_verbosity = trt.Logger.Severity.ERROR
        elif verbose:
            trt_verbosity = trt.Logger.Severity.INFO
        else:
            trt_verbosity = trt.Logger.Severity.WARNING
        tensorrt_logger = trt.Logger(trt_verbosity)
        tensorrt_loggers.append(tensorrt_logger)
        return tensorrt_logger

    class Calibrator(trt.IInt8EntropyCalibrator2):
        """Calibrator class.

        This inherits from ``trt.IInt8EntropyCalibrator2`` to implement
        the calibration interface that TensorRT needs to calibrate the
        INT8 quantization factors.

        Args:
            data_dir (str): Directory path of LIDAR files.
            calibration_filename (str): Name of calibration to read/write to.
            n_batches (int): Number of batches for calibrate for.
            batch_size (int): Batch size to use for calibration (this must be
                smaller or equal to the batch size of the provided data).
        """

        def __init__(
            self, data_dir,
            cache_filename,
            n_batches,
            batch_size,
            max_points_num,
            *args, **kwargs
        ):
            """Init routine."""
            super(Calibrator, self).__init__(*args, **kwargs)
            self._data_dir = data_dir
            self._cache_filename = cache_filename
            self._batch_size = batch_size
            self._n_batches = n_batches
            self._max_points_num = max_points_num
            self._batch_count = 0
            self._data_mem_points = None
            self._data_mem_num_points = None
            self._lidar_files = glob.glob(data_dir + "/*.bin")
            if len(self._lidar_files) < batch_size * n_batches:
                raise OSError(
                    f"No enough data files, got {len(self._lidar_files)}, "
                    f"requested {batch_size * n_batches}"
                )

        def get_algorithm(self):
            """Get algorithm."""
            return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

        def get_batch(self, names):
            """Return one batch.

            Args:
                names (list): list of memory bindings names.
            """
            print("Get batch: ", self._batch_count)
            if self._batch_count < self._n_batches:
                batch_files = self._lidar_files[
                    self._batch_count * self._batch_size: (self._batch_count + 1) * self._batch_size
                ]
                points = []
                num_points = []
                for f in batch_files:
                    _points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
                    num_points.append(_points.shape[0])
                    points.append(
                        np.pad(
                            _points,
                            ((0, self._max_points_num - _points.shape[0]), (0, 0))
                        )
                    )
                points = np.stack(points, axis=0)
                num_points = np.stack(num_points, axis=0)
                if self._data_mem_points is None:
                    self._data_mem_points = cuda.mem_alloc(points.size * 4)
                if self._data_mem_num_points is None:
                    self._data_mem_num_points = cuda.mem_alloc(num_points.size * 4)
                cuda.memcpy_htod(
                    self._data_mem_points, np.ascontiguousarray(points, dtype=np.float32)
                )
                cuda.memcpy_htod(
                    self._data_mem_num_points, np.ascontiguousarray(num_points, dtype=np.int32)
                )
                self._batch_count += 1
                return [int(self._data_mem_points), int(self._data_mem_num_points)]
            if self._data_mem_points is not None:
                self._data_mem_points.free()
            if self._data_mem_num_points is not None:
                self._data_mem_num_points.free()
            return None

        def get_batch_size(self):
            """Return batch size."""
            return self._batch_size

        def read_calibration_cache(self):
            """Read calibration from file."""
            logger.debug("read_calibration_cache - no-op")
            if os.path.isfile(self._cache_filename):
                logger.warning(
                    "Calibration file %s exists but is being "
                    "ignored." % self._cache_filename
                )

        def write_calibration_cache(self, cache):
            """Write calibration to file.

            Args:
                cache (memoryview): buffer to read calibration data from.
            """
            logger.info(
                "Saving calibration cache (size %d) to %s",
                len(cache),
                self._cache_filename,
            )
            with open(self._cache_filename, "wb") as f:
                f.write(cache)

    def _set_excluded_layer_precision(network, fp32_layer_names, fp16_layer_names):
        """When generating an INT8 model, it sets excluded layers' precision as fp32 or fp16.

        In detail, this function is only used when generating INT8 TensorRT models. It accepts
        two lists of layer names: (1). for the layers in fp32_layer_names, their precision will
        be set as fp32; (2). for those in fp16_layer_names, their precision will be set as fp16.

        Args:
            network: TensorRT network object.
            fp32_layer_names (list): List of layer names. These layers use fp32.
            fp16_layer_names (list): List of layer names. These layers use fp16.
        """
        is_mixed_precision = False
        use_fp16_mode = False

        for i, layer in enumerate(network):
            if any(s in layer.name for s in fp32_layer_names):
                is_mixed_precision = True
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
                logger.info("fp32 index: %d; name: %s", i, layer.name)
            elif any(s in layer.name for s in fp16_layer_names):
                is_mixed_precision = True
                use_fp16_mode = True
                layer.precision = trt.float16
                layer.set_output_type(0, trt.float16)
                logger.info("fp16 index: %d; name: %s", i, layer.name)
            else:
                # To ensure int8 optimization is not done for shape layer
                if (not layer.get_output(0).is_shape_tensor):
                    layer.precision = trt.int8
                    layer.set_output_type(0, trt.int8)

        return is_mixed_precision, use_fp16_mode

    class EngineBuilder(object):
        """Create a TensorRT engine.

        Args:
            filename (list): List of filenames to load model from.
            max_batch_size (int): Maximum batch size.
            vmax_workspace_size (int): Maximum workspace size.
            dtype (str): data type ('fp32', 'fp16' or 'int8').
            calibrator (:any:`Calibrator`): Calibrator to use for INT8 optimization.
            fp32_layer_names (list): List of layer names. These layers use fp32.
            fp16_layer_names (list): List of layer names. These layers use fp16.
            verbose (bool): Whether to turn on verbose mode.
            tensor_scale_dict (dict): Dictionary mapping names to tensor scaling factors.
            strict_type(bool): Whether or not to apply strict_type_constraints for INT8 mode.
        """

        def __init__(
            self,
            filenames,
            max_batch_size=DEFAULT_MAX_BATCH_SIZE,
            max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
            dtype="fp32",
            calibrator=None,
            fp32_layer_names=None,
            fp16_layer_names=None,
            verbose=False,
            tensor_scale_dict=None,
            strict_type=False,
        ):
            """Initialization routine."""
            if dtype == "int8":
                self._dtype = trt.DataType.INT8
            elif dtype == "fp16":
                self._dtype = trt.DataType.HALF
            elif dtype == "fp32":
                self._dtype = trt.DataType.FLOAT
            else:
                raise ValueError("Unsupported data type: %s" % dtype)
            self._strict_type = strict_type
            if fp32_layer_names is None:
                fp32_layer_names = []
            elif dtype != "int8":
                raise ValueError(
                    "FP32 layer precision could be set only when dtype is INT8"
                )

            if fp16_layer_names is None:
                fp16_layer_names = []
            elif dtype != "int8":
                raise ValueError(
                    "FP16 layer precision could be set only when dtype is INT8"
                )

            self._fp32_layer_names = fp32_layer_names
            self._fp16_layer_names = fp16_layer_names
            self._tensorrt_logger = _create_tensorrt_logger(verbose)
            builder = trt.Builder(self._tensorrt_logger)
            trt.init_libnvinfer_plugins(self._tensorrt_logger, "")
            if self._dtype == trt.DataType.HALF and not builder.platform_has_fast_fp16:
                logger.error("Specified FP16 but not supported on platform.")
                raise AttributeError("Specified FP16 but not supported on platform.")
                return

            if self._dtype == trt.DataType.INT8 and not builder.platform_has_fast_int8:
                logger.error("Specified INT8 but not supported on platform.")
                raise AttributeError("Specified INT8 but not supported on platform.")
                return

            if self._dtype == trt.DataType.INT8:
                if tensor_scale_dict is None and calibrator is None:
                    logger.error("Specified INT8 but neither calibrator "
                                 "nor tensor_scale_dict is provided.")
                    raise AttributeError("Specified INT8 but no calibrator "
                                         "or tensor_scale_dict is provided.")

            network = builder.create_network()

            self._load_from_files(filenames, network)

            builder.max_batch_size = max_batch_size
            builder.max_workspace_size = max_workspace_size

            if self._dtype == trt.DataType.HALF:
                builder.fp16_mode = True

            if self._dtype == trt.DataType.INT8:
                builder.int8_mode = True
                if tensor_scale_dict is None:
                    builder.int8_calibrator = calibrator
                    # When use mixed precision, for TensorRT builder:
                    # strict_type_constraints needs to be True;
                    # fp16_mode needs to be True if any layer uses fp16 precision.
                    builder.strict_type_constraints, builder.fp16_mode = \
                        _set_excluded_layer_precision(
                            network=network,
                            fp32_layer_names=self._fp32_layer_names,
                            fp16_layer_names=self._fp16_layer_names,
                        )
                else:
                    # Discrete Volta GPUs don't have int8 tensor cores. So TensorRT might
                    # not pick int8 implementation over fp16 or even fp32 for V100
                    # GPUs found on data centers (e.g., AVDC). This will be a discrepancy
                    # compared to Turing GPUs including d-GPU of DDPX and also Xavier i-GPU
                    # both of which have int8 accelerators. We set the builder to strict
                    # mode to avoid picking higher precision implementation even if they are
                    # faster.
                    if self._strict_type:
                        builder.strict_type_constraints = True
                    else:
                        builder.fp16_mode = True
                    self._set_tensor_dynamic_ranges(
                        network=network, tensor_scale_dict=tensor_scale_dict
                    )

            engine = builder.build_cuda_engine(network)

            try:
                assert engine
            except AssertionError:
                logger.error("Failed to create engine")
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)  # Fixed format
                tb_info = traceback.extract_tb(tb)
                _, line, _, text = tb_info[-1]
                raise AssertionError(
                    "Parsing failed on line {} in statement {}".format(line, text)
                )

            self._engine = engine

        def _load_from_files(self, filenames, network):
            """Load an engine from files."""
            raise NotImplementedError()

        @staticmethod
        def _set_tensor_dynamic_ranges(network, tensor_scale_dict):
            """Set the scaling factors obtained from quantization-aware training.

            Args:
                network: TensorRT network object.
                tensor_scale_dict (dict): Dictionary mapping names to tensor scaling factors.
            """
            tensors_found = []
            for idx in range(network.num_inputs):
                input_tensor = network.get_input(idx)
                if input_tensor.name in tensor_scale_dict:
                    tensors_found.append(input_tensor.name)
                    cal_scale = tensor_scale_dict[input_tensor.name]
                    input_tensor.dynamic_range = (-cal_scale, cal_scale)

            for layer in network:
                found_all_outputs = True
                for idx in range(layer.num_outputs):
                    output_tensor = layer.get_output(idx)
                    if output_tensor.name in tensor_scale_dict:
                        tensors_found.append(output_tensor.name)
                        cal_scale = tensor_scale_dict[output_tensor.name]
                        output_tensor.dynamic_range = (-cal_scale, cal_scale)
                    else:
                        found_all_outputs = False
                if found_all_outputs:
                    layer.precision = trt.int8
            tensors_in_dict = tensor_scale_dict.keys()
            if set(tensors_in_dict) != set(tensors_found):
                print("Tensors in scale dictionary but not in network:",
                      set(tensors_in_dict) - set(tensors_found))

        def get_engine(self):
            """Return the engine that was built by the instance."""
            return self._engine

    @subclass
    class ONNXEngineBuilder(EngineBuilder):
        """Create a TensorRT engine from an ONNX file.

        Args:
            filename (str): ONNX file to create engine from.
            input_node_name (str): Name of the input node.
            input_dims (list): Dimensions of the input tensor.
            output_node_names (list): Names of the output nodes.
        """

        @override
        def __init__(
            self,
            filenames,
            max_batch_size=DEFAULT_MAX_BATCH_SIZE,
            min_batch_size=DEFAULT_MIN_BATCH_SIZE,
            max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
            opt_batch_size=DEFAULT_OPT_BATCH_SIZE,
            dtype="fp32",
            calibrator=None,
            fp32_layer_names=None,
            fp16_layer_names=None,
            verbose=False,
            tensor_scale_dict=None,
            dynamic_batch=False,
            strict_type=False,
            input_dims=None,
        ):
            """Initialization routine."""
            if dtype == "int8":
                self._dtype = trt.DataType.INT8
            elif dtype == "fp16":
                self._dtype = trt.DataType.HALF
            elif dtype == "fp32":
                self._dtype = trt.DataType.FLOAT
            else:
                raise ValueError("Unsupported data type: %s" % dtype)

            if fp32_layer_names is None:
                fp32_layer_names = []
            elif dtype != "int8":
                raise ValueError(
                    "FP32 layer precision could be set only when dtype is INT8"
                )

            if fp16_layer_names is None:
                fp16_layer_names = []
            elif dtype != "int8":
                raise ValueError(
                    "FP16 layer precision could be set only when dtype is INT8"
                )

            self._fp32_layer_names = fp32_layer_names
            self._fp16_layer_names = fp16_layer_names
            self._strict_type = strict_type
            self._tensorrt_logger = _create_tensorrt_logger(verbose)
            builder = trt.Builder(self._tensorrt_logger)

            if self._dtype == trt.DataType.HALF and not builder.platform_has_fast_fp16:
                logger.error("Specified FP16 but not supported on platform.")
                raise AttributeError("Specified FP16 but not supported on platform.")
                return

            if self._dtype == trt.DataType.INT8 and not builder.platform_has_fast_int8:
                logger.error("Specified INT8 but not supported on platform.")
                raise AttributeError("Specified INT8 but not supported on platform.")
                return

            if self._dtype == trt.DataType.INT8:
                if tensor_scale_dict is None and calibrator is None:
                    logger.error("Specified INT8 but neither calibrator "
                                 "nor tensor_scale_dict is provided.")
                    raise AttributeError("Specified INT8 but no calibrator "
                                         "or tensor_scale_dict is provided.")

            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

            self._load_from_files([filenames], network)

            config = builder.create_builder_config()
            if dynamic_batch:
                opt_profile = builder.create_optimization_profile()
                # input: points
                model_input = network.get_input(0)
                input_name = model_input.name
                input_shape = model_input.shape
                real_shape_min = (min_batch_size, input_shape[1], input_shape[2])
                real_shape_opt = (opt_batch_size, input_shape[1], input_shape[2])
                real_shape_max = (max_batch_size, input_shape[1], input_shape[2])
                opt_profile.set_shape(input=input_name,
                                      min=real_shape_min,
                                      opt=real_shape_opt,
                                      max=real_shape_max)
                # input: num_points
                model_input = network.get_input(1)
                input_name = model_input.name
                real_shape_min = (min_batch_size,)
                real_shape_opt = (opt_batch_size,)
                real_shape_max = (max_batch_size,)
                opt_profile.set_shape(input=input_name,
                                      min=real_shape_min,
                                      opt=real_shape_opt,
                                      max=real_shape_max)
                config.add_optimization_profile(opt_profile)
            config.max_workspace_size = max_workspace_size
            if self._dtype == trt.DataType.HALF:
                config.flags |= 1 << int(trt.BuilderFlag.FP16)

            if self._dtype == trt.DataType.INT8:
                config.flags |= 1 << int(trt.BuilderFlag.INT8)
                if tensor_scale_dict is None:
                    config.int8_calibrator = calibrator
                    # When use mixed precision, for TensorRT builder:
                    # strict_type_constraints needs to be True;
                    # fp16_mode needs to be True if any layer uses fp16 precision.
                    strict_type_constraints, fp16_mode = \
                        _set_excluded_layer_precision(
                            network=network,
                            fp32_layer_names=self._fp32_layer_names,
                            fp16_layer_names=self._fp16_layer_names,
                        )
                    if strict_type_constraints:
                        config.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)
                    if fp16_mode:
                        config.flags |= 1 << int(trt.BuilderFlag.FP16)
                else:
                    # Discrete Volta GPUs don't have int8 tensor cores. So TensorRT might
                    # not pick int8 implementation over fp16 or even fp32 for V100
                    # GPUs found on data centers (e.g., AVDC). This will be a discrepancy
                    # compared to Turing GPUs including d-GPU of DDPX and also Xavier i-GPU
                    # both of which have int8 accelerators. We set the builder to strict
                    # mode to avoid picking higher precision implementation even if they are
                    # faster.
                    if self._strict_type:
                        config.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)
                    else:
                        config.flags |= 1 << int(trt.BuilderFlag.FP16)
                    self._set_tensor_dynamic_ranges(
                        network=network, tensor_scale_dict=tensor_scale_dict
                    )
            engine = builder.build_engine(network, config)

            try:
                assert engine
            except AssertionError:
                logger.error("Failed to create engine")
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)  # Fixed format
                tb_info = traceback.extract_tb(tb)
                _, line, _, text = tb_info[-1]
                raise AssertionError(
                    "Parsing failed on line {} in statement {}".format(line, text)
                )

            self._engine = engine

        @override
        def _load_from_files(self, filenames, network):
            filename = filenames[0]
            parser = trt.OnnxParser(network, self._tensorrt_logger)
            with open(filename, "rb") as model_file:
                ret = parser.parse(model_file.read())
            for index in range(parser.num_errors):
                print(parser.get_error(index))
            assert ret, 'ONNX parser failed to parse the model.'

            # Note: there might be an issue when running inference on TRT:
            # [TensorRT] ERROR: Network must have at least one output.
            # See https://github.com/NVIDIA/TensorRT/issues/183.
            # Just keep a note in case we have this issue again.
