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

"""The top model builder interface."""
from nvidia_tao_pytorch.cv.re_identification.model.baseline import Baseline


def build_model(cfg, num_classes):
    """Build a re-identification model according to provided configuration.

    This function builds a re-identification model using the Baseline architecture as per the
    provided configuration and number of classes. The Baseline model is primarily a ResNet variant
    with additional features like bottleneck and classifier layers.

    Args:
        cfg (DictConfig): Configuration object containing parameters for the model.
        num_classes (int): The number of output classes for the model.

    Returns:
        Baseline: An instance of the Baseline model configured according to the provided configuration and number of classes.
    """
    model = Baseline(cfg, num_classes)
    return model
