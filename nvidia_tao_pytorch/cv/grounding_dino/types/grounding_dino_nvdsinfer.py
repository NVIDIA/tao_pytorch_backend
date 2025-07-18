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

"""GST-Nvinfer config file for Grounding DINO."""

from dataclasses import dataclass, is_dataclass, field
from nvidia_tao_pytorch.core.types.nvdsinfer import (
    BaseDSType,
    BaseNVDSClassAttributes,
    BaseNvDSPropertyConfig
)


@dataclass
class GroundingDINONvDSPropertyConfig(BaseNvDSPropertyConfig):
    """Structured configuration defining the schema for nvdsinfer property element for RT-DETR."""

    input_tensor_from_meta: int = 1
    custom_lib_path: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser_tao.so"


@dataclass
class GroundingDINONvDSClassAttribute(BaseNVDSClassAttributes):
    """Structured configuration defining the schema for nvdsinfer class-attr element for RT-DETR."""

    topk: int = 20


@dataclass
class GroundingDINONvDSInferConfig(BaseDSType):
    """RTDETRNvDSInfer config element."""

    property_field: GroundingDINONvDSPropertyConfig = field(default_factory=lambda: GroundingDINONvDSPropertyConfig(
        cluster_mode=4,
        net_scale_factor=0.0078431372549,
        network_type=100,
        offsets=[127.5, 127.5, 127.5],
        network_mode=2,
        output_blob_names=["pred_boxes", "pred_logits"],
        output_tensor_meta=1,
        input_tensor_from_meta=1,
        model_color_format=0
    ))
    class_attrs_all: GroundingDINONvDSClassAttribute = field(default_factory=lambda: GroundingDINONvDSClassAttribute(
        pre_cluster_threshold=0.2
    ))

    def validate(self):
        """Function to validate the dataclass."""
        pass


if __name__ == "__main__":
    grounding_dino_config = GroundingDINONvDSInferConfig()
    grounding_dino_config.property_field.cluster_mode = 4
    assert is_dataclass(grounding_dino_config), "The instance of base_config is not a dataclass."
    print(str(grounding_dino_config))
