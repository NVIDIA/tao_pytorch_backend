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

"""Export script for PointPillars."""
import argparse

import torch
from torch import nn
from torch.nn import functional as F
import onnx
from onnxsim import simplify
import os

try:
    import tensorrt as trt  # pylint: disable=unused-import  # noqa: F401
    from nvidia_tao_pytorch.pointcloud.pointpillars.tools.export.tensorrt import (
        Calibrator,
        ONNXEngineBuilder
    )
    trt_available = True
except:  # noqa: E722
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
    trt_available = False
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.pointcloud.pointpillars.tools.export.simplifier_onnx import (
    simplify_onnx
)
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.config import cfg, cfg_from_yaml_file
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.models import load_checkpoint
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.utils import common_utils

from nvidia_tao_pytorch.pointcloud.pointpillars.tools.train_utils.train_utils import (
    encrypt_onnx
)


class ExportablePFNLayer(nn.Module):
    """PFN layer replacement that can be exported to ONNX."""

    def __init__(self, model):
        """Initialize."""
        super().__init__()
        self.model = model

    def forward(self, inputs):
        """Forward method."""
        inputs_shape = inputs.cpu().detach().numpy().shape
        if len(inputs_shape) == 4:
            inputs = inputs.view((-1, inputs_shape[2], inputs_shape[3]))
        x = self.model.linear(inputs)
        voxel_num_points = inputs_shape[-2]
        if self.model.use_norm:
            x = self.model.norm(x.permute(0, 2, 1))
            x = F.relu(x)
            x = F.max_pool1d(x, voxel_num_points, stride=1)
            x_max = x.permute(0, 2, 1)
        else:
            x = F.relu(x)
            x = x.permute(0, 2, 1)
            x = F.max_pool1d(x, voxel_num_points, stride=1)
            x_max = x.permute(0, 2, 1)
        if len(inputs_shape) == 4:
            x_max_shape = x_max.cpu().detach().numpy().shape
            x_max = x_max.view((-1, inputs_shape[1], x_max_shape[2]))
        else:
            x_max = x_max.squeeze(1)
        if self.model.last_vfe:
            return x_max
        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        x_concatenated = torch.cat([x, x_repeat], dim=2)
        return x_concatenated


class ExportablePillarVFE(nn.Module):
    """PillarVFE module replacement to it can be exported."""

    def __init__(self, model):
        """Initialize."""
        super().__init__()
        self.model = model

    def forward(self, voxel_features, voxel_num_points, coords):
        """Forward method."""
        points_mean = voxel_features[..., :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[..., :3] - points_mean
        f_center = torch.zeros_like(voxel_features[..., :3])
        f_center[..., 0] = voxel_features[..., 0] - (coords[..., 3].to(voxel_features.dtype).unsqueeze(2) * self.model.voxel_x + self.model.x_offset)
        f_center[..., 1] = voxel_features[..., 1] - (coords[..., 2].to(voxel_features.dtype).unsqueeze(2) * self.model.voxel_y + self.model.y_offset)
        f_center[..., 2] = voxel_features[..., 2] - (coords[..., 1].to(voxel_features.dtype).unsqueeze(2) * self.model.voxel_z + self.model.z_offset)
        if self.model.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        if self.model.with_distance:
            points_dist = torch.norm(voxel_features[..., :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        for pfn in self.model.pfn_layers:
            exportable_pfn = ExportablePFNLayer(pfn)
            features = exportable_pfn(features)
        return features


class ExportableScatter(nn.Module):
    """Scatter module replacement that can be exported."""

    def __init__(self, model):
        """Initialize."""
        super().__init__()
        self.model = model

    def forward(self, pillar_features, coords):
        """Forward method."""
        batch_spatial_features = []
        batch_size = coords[..., 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.model.num_bev_features,
                self.model.nz * self.model.nx * self.model.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )
            batch_mask = coords[batch_idx, :, 0] == batch_idx
            this_coords = coords[batch_idx, batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.model.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_idx, batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(
            -1, self.model.num_bev_features * self.model.nz,
            self.model.ny, self.model.nx
        )
        return batch_spatial_features


class ExportableBEVBackbone(nn.Module):
    """Exportable BEV backbone."""

    def __init__(self, model):
        """Initialize."""
        super().__init__()
        self.model = model

    def forward(self, spatial_features):
        """Forward method."""
        ups = []
        x = spatial_features
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)
            if len(self.model.deblocks) > 0:
                ups.append(self.model.deblocks[i](x))
            else:
                ups.append(x)
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.model.deblocks) > len(self.model.blocks):
            x = self.model.deblocks[-1](x)
        return x


class ExportableAnchorHead(nn.Module):
    """Exportable Anchor Head."""

    def __init__(self, model):
        """Initialize."""
        super().__init__()
        self.model = model

    def forward(self, spatial_features_2d, batch_size):
        """Forward method."""
        cls_preds = self.model.conv_cls(spatial_features_2d)
        box_preds = self.model.conv_box(spatial_features_2d)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        if self.model.conv_dir_cls is not None:
            dir_cls_preds = self.model.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = None
        return cls_preds, box_preds, dir_cls_preds


class ExportablePointPillar(nn.Module):
    """Exportable PointPillar model."""

    def __init__(self, model):
        """Initialize."""
        super().__init__()
        self.module_list = model.module_list
        self.exportable_vfe = ExportablePillarVFE(self.module_list[0])
        self.exportable_scatter = ExportableScatter(self.module_list[1])
        self.exportable_bev_backbone = ExportableBEVBackbone(self.module_list[2])
        self.exportable_anchor_head = ExportableAnchorHead(self.module_list[3])

    def forward(self, voxel_features, voxel_num_points, coords):
        """Forward method."""
        self.batch_size = 1
        pillar_features = self.exportable_vfe(voxel_features, voxel_num_points, coords)  # "PillarVFE"
        spatial_features = self.exportable_scatter(pillar_features, coords)  # "PointPillarScatter"
        spatial_features_2d = self.exportable_bev_backbone(spatial_features)  # "BaseBEVBackbone"
        cls_preds, box_preds, dir_cls_preds = self.exportable_anchor_head(spatial_features_2d, self.batch_size)  # "AnchorHeadSingle"
        return cls_preds, box_preds, dir_cls_preds


def parse_config():
    """Argument Parser."""
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument(
        '--cfg_file', '-c', type=str, required=True,
        help='PointPillars training config file'
    )
    parser.add_argument(
        '--cal_data_path', '-d', type=str, required=False, default=None,
        help='Path to the point cloud data directory'
    )
    parser.add_argument(
        "--cal_cache_file",
        type=str,
        required=False,
        default="./cal.bin",
        help="Path to save the calibration file in INT8 mode"
    )
    parser.add_argument(
        "--data_type", "-t",
        required=False, default="fp32",
        help="The data type for export(useful for TensorRT INT8 calibration)"
    )
    parser.add_argument(
        "--save_engine",
        "-e",
        type=str,
        required=False,
        default=None,
        help="Path to save the TensorRT engine."
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        required=False,
        help="Batch size of the TensorRT engine to be generated(if --save_engine is provided)."
    )
    parser.add_argument(
        "--cal_num_batches",
        type=int,
        default=8,
        required=False,
        help="Number of data batches for INT8 calibration"
    )
    parser.add_argument(
        "--workspace_size",
        "-w",
        type=int,
        default=1024,
        required=False,
        help="Workspace size in MB for TensorRT, default is 1024MB(1GB)."
    )
    parser.add_argument("--key", "-k", type=str, required=True, help="Encryption key")
    args = parser.parse_args()
    cfg_from_yaml_file(expand_path(args.cfg_file), cfg)
    return args, cfg


def main():
    """Main function."""
    if not trt_available:
        raise ValueError("Failed to import tensorrt library, exporting to a Tensorrt engine not possible")
    args, cfg = parse_config()
    # INT8 is not yet fully supported, raise error if one tries to use it
    if args.data_type.lower() == "int8":
        raise ValueError("INT8 is not supported for PointPillars, please use FP32/FP16")
    logger = common_utils.create_logger()
    logger.info('Exporting the model...')
    gpu_id = cfg.export.gpu_id or 0
    torch.cuda.set_device(gpu_id)
    if cfg.export.checkpoint is None:
        raise OSError("Please provide export.checkpoint in config file")
    if not os.path.isfile(cfg.export.checkpoint):
        raise FileNotFoundError(f"Input model {cfg.export.checkpoint} does not exist")
    if cfg.export.onnx_file is None:
        split_name = os.path.splitext(cfg.export.checkpoint)[0]
        output_file = "{}.onnx".format(split_name)
    else:
        output_file = cfg.export.onnx_file
    # Warn the user if an exported file already exists.
    assert not os.path.exists(output_file), "Default onnx file {} already "\
        "exists".format(output_file)
    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    if output_file.endswith('.etlt'):
        tmp_onnx_file = output_file.replace('.etlt', '.onnx')
    else:
        tmp_onnx_file = output_file
    # Set up status logging
    result_dir = os.path.dirname(output_file)
    status_file = os.path.join(result_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting PointPillars export")
    # Load model
    loaded_model = load_checkpoint(cfg.export.checkpoint, args.key)[0]
    model = ExportablePointPillar(loaded_model)
    model.cuda()
    model.eval()
    with torch.no_grad():
        MAX_VOXELS = cfg.dataset.data_processor[2].max_number_of_voxels["test"]
        MAX_POINTS = cfg.dataset.data_processor[2].max_points_per_voxel
        NUM_POINT_FEATS = cfg.dataset.data_augmentor.aug_config_list[0].num_point_features
        dummy_voxel_features = torch.zeros(
            (1, MAX_VOXELS, MAX_POINTS, NUM_POINT_FEATS),
            dtype=torch.float32,
            device='cuda:0'
        )
        dummy_voxel_num_points = torch.zeros(
            (1, MAX_VOXELS,),
            dtype=torch.int32,
            device='cuda:0'
        )
        dummy_coords = torch.zeros(
            # 4: (batch_idx, x, y, z)
            (1, MAX_VOXELS, 4),
            dtype=torch.int32,
            device='cuda:0'
        )
        torch.onnx.export(
            model,
            (dummy_voxel_features, dummy_voxel_num_points, dummy_coords),
            tmp_onnx_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            keep_initializers_as_inputs=True,
            input_names=['input', 'voxel_num_points', 'coords'],
            output_names=['cls_preds', 'box_preds', 'dir_cls_preds'],
            dynamic_axes={
                "input": {0: "batch"},
                "voxel_num_points": {0: "batch"},
                "coords": {0: "batch"}
            }
        )
        onnx_model = onnx.load(tmp_onnx_file)
        model_simp, check = simplify(
            onnx_model,
            overwrite_input_shapes={
                "input": (1, MAX_VOXELS, MAX_POINTS, NUM_POINT_FEATS),
                'voxel_num_points': (1, MAX_VOXELS),
                'coords': (1, MAX_VOXELS, 4)
            }
        )
        assert check, "Failed on simplifying the ONNX model"
        model_simp = simplify_onnx(model_simp, cfg)
        onnx.save(model_simp, tmp_onnx_file)
        if output_file.endswith('.etlt') and args.key:
            # encrypt the onnx if and only if key is provided and output file name ends with .etlt
            encrypt_onnx(tmp_file_name=tmp_onnx_file,
                         output_file_name=output_file,
                         key=args.key)
    logger.info(f'Model exported to {output_file}')
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.RUNNING,
        message=f'Model exported to {output_file}'
    )
    # Save TRT engine
    if args.save_engine is not None:
        if args.data_type.lower() == "int8":
            if args.cal_data_path is not None:
                calibrator = Calibrator(
                    args.cal_data_path,
                    args.cal_cache_file,
                    args.cal_num_batches,
                    args.batch_size,
                    cfg.inference.max_points_num
                )
            else:
                raise ValueError("Cannot find caliration data path")
        else:
            calibrator = None
        builder = ONNXEngineBuilder(
            tmp_onnx_file,
            max_batch_size=args.batch_size,
            min_batch_size=args.batch_size,
            opt_batch_size=args.batch_size,
            dtype=args.data_type,
            max_workspace_size=args.workspace_size * 1024 * 1024,
            dynamic_batch=True,
            calibrator=calibrator
        )
        engine = builder.get_engine()
        with open(expand_path(args.save_engine), "wb") as outf:
            outf.write(engine.serialize())
        logger.info(f'TensorRT engine saved to {args.save_engine}')
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message=f'TensorRT engine saved to {args.save_engine}'
        )
    if output_file.endswith('.etlt') and args.key:
        os.remove(tmp_onnx_file)


if __name__ == '__main__':
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Export finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Export was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
