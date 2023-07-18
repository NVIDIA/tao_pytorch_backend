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

"""Default config file"""

from typing import Optional, List, Dict
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class OIModelConfig:
    """Optical recognition model config."""

    model_type: str = "Siamese_3"
    margin: float = 2.0
    model_backbone: str = "custom"
    embedding_vectors: int = 5
    imagenet_pretrained: bool = False


@dataclass
class OptimConfig:
    """Optimizer config."""

    type: str = "Adam"
    lr: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 5e-4


@dataclass
class OIAugmentationConfig:
    """Augmentation config."""

    rgb_input_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    rgb_input_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class DataPathFormat:
    """Dataset Path experiment config."""

    csv_path: str = MISSING
    images_dir: str = MISSING


@dataclass
class OIDatasetConfig:
    """Dataset config."""

    train_dataset: DataPathFormat = DataPathFormat()
    validation_dataset: DataPathFormat = DataPathFormat()
    test_dataset: DataPathFormat = DataPathFormat()
    infer_dataset: DataPathFormat = DataPathFormat()
    image_ext: Optional[str] = None
    batch_size: int = 32
    workers: int = 8
    fpratio_sampling: float = 0.1
    num_input: int = 8
    input_map: Optional[Dict[str, int]] = None
    grid_map: Optional[Dict[str, int]] = None
    concat_type: Optional[str] = None
    output_shape: List[int] = field(default_factory=lambda: [100, 100])
    augmentation_config: OIAugmentationConfig = OIAugmentationConfig()


@dataclass
class TensorBoardLogger:
    """Configuration for the tensorboard logger."""

    enabled: bool = False
    infrequent_logging_frequency: int = 2  # Defined per epoch


@dataclass
class OITrainExpConfig:
    """Train experiment config."""

    optim: OptimConfig = OptimConfig()
    num_epochs: int = 10
    checkpoint_interval: int = 2
    validation_interval: int = 2
    loss: Optional[str] = None
    clip_grad_norm: float = 0.0
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    results_dir: Optional[str] = None
    tensorboard: Optional[TensorBoardLogger] = TensorBoardLogger()
    resume_training_checkpoint_path: Optional[str] = None
    pretrained_model_path: Optional[str] = None


@dataclass
class OIInferenceExpConfig:
    """Inference experiment config."""

    checkpoint: str = MISSING
    trt_engine: str = MISSING
    gpu_id: int = 0
    results_dir: Optional[str] = None
    batch_size: int = 1


@dataclass
class OIEvalExpConfig:
    """Evaluation experiment config."""

    checkpoint: str = MISSING
    gpu_id: int = 0
    batch_size: int = 1
    results_dir: Optional[str] = None


@dataclass
class OIExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = None
    checkpoint: str = MISSING
    onnx_file: Optional[str] = None
    opset_version: Optional[int] = 12
    gpu_id: int = 0
    on_cpu: bool = False
    input_height: int = 400
    input_width: int = 100
    input_channel: int = 3
    batch_size: int = -1
    do_constant_folding: bool = False


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_image_dir: List[str] = MISSING
    cal_cache_file: str = MISSING
    cal_batch_size: int = 1
    cal_batches: int = 1


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = "fp16"
    workspace_size: int = 1024
    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 1
    calibration: CalibrationConfig = CalibrationConfig()


@dataclass
class OIGenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = None
    gpu_id: int = 0
    onnx_file: str = MISSING
    trt_engine: Optional[str] = None
    input_channel: int = 3
    input_width: int = 400
    input_height: int = 100
    opset_version: int = 12
    batch_size: int = -1
    verbose: bool = False
    tensorrt: TrtConfig = TrtConfig()


@dataclass
class OIDatasetConvertConfig:
    """Dataset Convert experiment config."""

    root_dataset_dir: Optional[str] = None
    data_convert_output_dir: Optional[str] = None
    train_pcb_dataset_dir: Optional[str] = None
    val_pcb_dataset_dir: Optional[str] = None
    all_pcb_dataset_dir: Optional[str] = None
    golden_csv_dir: Optional[str] = None
    project_name: Optional[str] = None
    bot_top: Optional[str] = None


@dataclass
class OIExperimentConfig:
    """Experiment config."""

    model: OIModelConfig = OIModelConfig()
    dataset: OIDatasetConfig = OIDatasetConfig()
    train: OITrainExpConfig = OITrainExpConfig()
    evaluate: OIEvalExpConfig = OIEvalExpConfig()
    export: OIExportExpConfig = OIExportExpConfig()
    inference: OIInferenceExpConfig = OIInferenceExpConfig()
    dataset_convert: OIDatasetConvertConfig = OIDatasetConvertConfig()
    gen_trt_engine: OIGenTrtEngineExpConfig = OIGenTrtEngineExpConfig()
    encryption_key: Optional[str] = None
    results_dir: str = MISSING
