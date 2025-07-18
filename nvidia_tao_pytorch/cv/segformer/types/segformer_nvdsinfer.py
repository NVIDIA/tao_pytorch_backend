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

"""GST-Nvinfer config file for RT-DETR."""

from dataclasses import dataclass, is_dataclass, field
from nvidia_tao_pytorch.core.types.nvdsinfer import (
    BaseNvDSPropertyConfig,
    BaseDSType
)


@dataclass
class SFNvDSPropertyConfig(BaseNvDSPropertyConfig):
    """Structured configuration defining the schema for nvdsinfer property element for SegFormer."""

    parse_segmentation_func_name: str = "NvDsInferParseCustomSegformerTAO2"
    custom_lib_path: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser_tao.so"

    def validate(self):
        """Validate the NVConfig."""
        super().validate()
        assert self.network_type == 2, "Network type should be 2 for SegFormer."


@dataclass
class SFNvDSInferConfig(BaseDSType):
    """SFNvDSInfer config element."""

    property_field: SFNvDSPropertyConfig = field(default_factory=lambda: SFNvDSPropertyConfig(
        cluster_mode=None,
        net_scale_factor=0.00784313725,
        offsets=[127.5, 127.5, 127.5],
        network_type=2,
        network_mode=2,
        output_tensor_meta=None,
        output_blob_names=None,
        model_color_format=0,
        gie_unique_id=1,
    ))

    def validate(self):
        """Function to validate the dataclass."""
        self.property_field.validate()


if __name__ == "__main__":
    segformer_config = SFNvDSInferConfig()
    segformer_config.property_field.onnx_file = "model.onnx"
    assert is_dataclass(segformer_config), "The instance of base_config is not a dataclass."
    print(str(segformer_config))
