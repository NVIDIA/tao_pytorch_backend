/*!
**************************************************************************************************
# Original source taken from https://github.com/autonomousvision/stylegan-xl
#
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
**************************************************************************************************
*/

//------------------------------------------------------------------------
// CUDA kernel parameters.

struct bias_act_kernel_params
{
    const void* x;      // [sizeX]
    const void* b;      // [sizeB] or NULL
    const void* xref;   // [sizeX] or NULL
    const void* yref;   // [sizeX] or NULL
    const void* dy;     // [sizeX] or NULL
    void*       y;      // [sizeX]

    int64_t         grad;
    int64_t         act;
    double       alpha;
    double       gain;
    double       clamp;

    int64_t         sizeX;
    int64_t         sizeB;
    int64_t         stepB;
    int64_t         loopX;
};

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T> void* choose_bias_act_kernel(const bias_act_kernel_params& p);

//------------------------------------------------------------------------
