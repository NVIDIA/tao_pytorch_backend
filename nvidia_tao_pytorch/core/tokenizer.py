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

"""Common utilities that could be used in create_tokenizer script."""

from typing import Optional
from omegaconf import MISSING
from dataclasses import dataclass


__all__ = ["TokenizerConfig"]


@dataclass
class TokenizerConfig:
    """Tokenizer config for use in create_tokenizer script."""

    # tokenizer type: "spe" or "wpe"
    tokenizer_type: str = MISSING
    # spe type if tokenizer_type == "spe"
    # choose from ['bpe', 'unigram', 'char', 'word']
    spe_type: str = MISSING
    # spe character coverage, defaults to 1.0
    spe_character_coverage: Optional[float] = 1.0
    # flag for lower case, defaults to True
    lower_case: Optional[bool] = True
