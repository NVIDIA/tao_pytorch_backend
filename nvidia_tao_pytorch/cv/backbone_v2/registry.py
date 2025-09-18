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
"""Registry for backbones.

This module provides a registry system for backbone models in the TAO PyTorch framework.
The registry allows for dynamic registration and retrieval of backbone models, enabling
easy extensibility and model discovery.

The BACKBONE_REGISTRY is a centralized registry that maintains a mapping between
backbone names and their corresponding model classes. This enables:
- Dynamic model loading by name
- Easy addition of new backbone architectures
- Consistent interface across different backbone implementations
- Integration with the broader TAO framework

Key Features:
- Automatic registration of backbone models using decorators
- Type checking to ensure registered objects are BackboneBase instances
- Backward compatibility with fvcore registry system
- Support for model discovery and listing

Example:
    ```python
    # Register a new backbone
    @BACKBONE_REGISTRY.register()
    def my_backbone(**kwargs):
        return MyBackbone(**kwargs)

    # Get a backbone by name
    backbone_class = BACKBONE_REGISTRY.get("my_backbone")
    model = backbone_class(num_classes=1000)
    ```

Registry Usage:
    - Use the @BACKBONE_REGISTRY.register() decorator to register backbone functions
    - Registered functions should return BackboneBase instances
    - Access registered backbones using BACKBONE_REGISTRY.get(name)
    - List all registered backbones using BACKBONE_REGISTRY.keys()
"""

from fvcore.common.registry import Registry  # for backward compatibility.


BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images
Registered object must return instance of :class:`BackboneBase`.
"""
