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

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_pytorch.config.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
)


@dataclass
class PruneConfig:
    """Prune config."""

    mode: str = STR_FIELD(
        value="amount",
        value_type="ordered",
        default_value="amount",
        valid_options="amount,threshold,experimental_hybrid",
        description="Pruning mode.",
        display="Pruning mode"
    )
    amount: float = FLOAT_FIELD(
        value=0.4,
        default_value=0.4,
        valid_min=0.0,
        valid_max=1.0,
        description="Pruning amount",
        display_name="Pruning amount"
    )
    threshold: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        valid_min=0.0,
        valid_max=1.0,
        description="Pruning threshold",
        display_name="Pruning threshold"
    )
    granularity: int = INT_FIELD(
        value=8,
        default_value=8,
        description="Pruning granularity",
        display="Pruning granularity"
    )
    raw_prune_score: str = STR_FIELD(
        value="L1",
        value_type="ordered",
        default_value="L1",
        valid_options="L1,L2",
        description="Learning rate monitor for AutoReduce learning rate scheduler.",
        display="lr monitor"
    )
