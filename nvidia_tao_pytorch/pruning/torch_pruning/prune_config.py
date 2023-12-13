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


@dataclass
class PruneConfig:
    """Prune config."""

    mode: str = "amount"  # [amount, threshold, experimental_hybrid]
    amount: Optional[float] = None
    threshold: Optional[float] = None
    granularity: int = 8
    raw_prune_score: str = "L1"  # [L1, L2]
