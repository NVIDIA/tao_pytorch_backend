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

"""DINO module."""
# Temporarily override torch versioning from DLFW so that we disable warning from fairscale
# about torch version during ddp_sharded training. Fairscale doesn't handle commit versions well
# E.g. 1.13.0a0+d0d6b1f
import torch
import re


numbering = re.search(r"^(\d+).(\d+).(\d+)([^\+]*)(\+\S*)?$", torch.__version__)
torch.__version__ = ".".join([str(numbering.group(n)) for n in range(1, 4)])
