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

"""Default config file."""

from typing import Optional, List
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class CenterPoseDatasetConfig:
    """Dataset config."""

    train_data: Optional[str] = None
    test_data: Optional[str] = None
    val_data: Optional[str] = None
    inference_data: Optional[str] = None
    batch_size: int = 4
    workers: int = 8
    pin_memory: bool = True
    num_classes: int = 1
    num_joints: int = 8
    max_objs: int = 10
    mean: List[float] = field(default_factory=lambda: [0.40789654, 0.44719302, 0.47026115])
    std: List[float] = field(default_factory=lambda: [0.28863828, 0.27408164, 0.27809835])
    _eig_val: List[float] = field(default_factory=lambda: [0.2141788, 0.01817699, 0.00341571],
                                  metadata={"Description": "eigen values for color data augmentation from CenterNet"})
    _eig_vec: List[List[float]] = field(default_factory=lambda: [[-0.58752847, -0.69563484, 0.41340352], [-0.5832747, 0.00994535, -0.81221408], [-0.56089297, 0.71832671, 0.41158938]],
                                        metadata={"Description": "eigen vector for color data augmentation from CenterNet"})
    category: str = "cereal_box"
    num_symmetry: int = 1
    mse_loss: bool = False
    center_3D: bool = False
    obj_scale: bool = True
    use_absolute_scale: bool = False
    obj_scale_uncertainty: bool = False
    dense_hp: bool = False
    hps_uncertainty: bool = False
    reg_bbox: bool = True
    reg_offset: bool = True
    hm_hp: bool = True
    reg_hp_offset: bool = True
    flip_idx: List[List[int]] = field(default_factory=lambda: [[1, 5], [3, 7], [2, 6], [4, 8]])

    # Data augmentation
    no_color_aug: bool = False
    not_rand_crop: bool = False
    aug_rot: int = 0
    flip: float = 0.5
    input_res: int = 512
    output_res: int = 128


@dataclass
class OptimConfig:
    """Optimizer config."""

    lr: float = 6e-05
    lr_scheduler: str = "MultiStep"
    lr_steps: List[int] = field(default_factory=lambda: [90, 120],
                                metadata={"description": "learning rate decay steps"})
    lr_decay: float = 0.1


@dataclass
class CenterPoseLossConfig:
    """CenterPose loss config."""

    mse_loss: bool = False
    dense_hp: bool = False
    reg_loss: str = "l1"
    num_stacks: int = 1
    hps_uncertainty: bool = False
    wh_weight: float = 0.1
    reg_bbox: bool = True
    reg_offset: bool = True
    reg_hp_offset: bool = True
    obj_scale: bool = True
    obj_scale_weight: int = 1
    obj_scale_uncertainty: bool = False
    use_residual: bool = False
    dimension_ref: Optional[str] = None
    off_weight: int = 1
    hm_hp: bool = True
    hm_hp_weight: int = 1
    hm_weight: int = 1
    hp_weight: int = 1


@dataclass
class CenterPoseTrainExpConfig:
    """Train experiment config."""

    num_gpus: int = 1
    num_nodes: int = 1
    results_dir: Optional[str] = None
    resume_training_checkpoint_path: Optional[str] = None
    pretrained_model_path: Optional[str] = None
    validation_interval: int = 1
    clip_grad_val: float = 100.0
    randomseed: int = 317
    is_dry_run: bool = False
    num_epochs: int = 140
    checkpoint_interval: int = 1
    precision: str = "fp32"
    optim: OptimConfig = OptimConfig()
    loss_config: CenterPoseLossConfig = CenterPoseLossConfig()


@dataclass
class BackboneConfig:
    """CenterPose backbone model config."""

    model_type: str = "DLA34"
    pretrained_backbone_path: Optional[str] = None


@dataclass
class CenterPoseModelConfig:
    """CenterPose model config."""

    down_ratio: int = 4
    final_kernel: int = 1
    last_level: int = 5
    head_conv: int = 256
    out_channel: int = 0
    use_convGRU: bool = True
    use_pretrained: bool = False
    backbone: BackboneConfig = BackboneConfig()


@dataclass
class CenterPoseInferenceExpConfig:
    """Inference experiment config."""

    num_gpus: int = 1
    results_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    trt_engine: Optional[str] = None
    visualization_threshold: float = 0.3
    num_select: int = 100
    use_pnp: bool = True
    save_json: bool = True
    save_visualization: bool = True
    opencv: bool = True

    # Camera intrinsic matrix
    principle_point_x: float = 0.00
    principle_point_y: float = 0.00
    focal_length_x: float = 0.00
    focal_length_y: float = 0.00
    skew: float = 0.00


@dataclass
class CenterPoseExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = None
    gpu_id: int = 0
    checkpoint: str = MISSING
    onnx_file: str = MISSING
    on_cpu: bool = False
    input_channel: int = 3
    input_width: int = 512
    input_height: int = 512
    opset_version: int = 16
    batch_size: int = -1
    verbose: bool = False
    num_select: int = 100
    do_constant_folding: bool = False


@dataclass
class CenterPoseEvalExpConfig:
    """Inference experiment config."""

    num_gpus: int = 1
    results_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    trt_engine: Optional[str] = None
    opencv: bool = True
    eval_num_symmetry: int = 1


@dataclass
class ExperimentConfig:
    """Experiment config."""

    encryption_key: Optional[str] = None
    results_dir: str = MISSING
    dataset: CenterPoseDatasetConfig = CenterPoseDatasetConfig()
    train: CenterPoseTrainExpConfig = CenterPoseTrainExpConfig()
    model: CenterPoseModelConfig = CenterPoseModelConfig()
    inference: CenterPoseInferenceExpConfig = CenterPoseInferenceExpConfig()
    export: CenterPoseExportExpConfig = CenterPoseExportExpConfig()
    evaluate: CenterPoseEvalExpConfig = CenterPoseEvalExpConfig()
