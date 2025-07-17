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

#include <cuda_runtime.h>

//------------------------------------------------------------------------
// CUDA kernel parameters.

struct upfirdn2d_kernel_params
{
    const void*     x;
    const float*    f;
    void*           y;

    int2            up;
    int2            down;
    int2            pad0;
    int             flip;
    float           gain;

    int4            inSize;         // [width, height, channel, batch]
    int4            inStride;
    int2            filterSize;     // [width, height]
    int2            filterStride;
    int4            outSize;        // [width, height, channel, batch]
    int4            outStride;
    int             sizeMinor;
    int             sizeMajor;

    int             loopMinor;
    int             loopMajor;
    int             loopX;
    int             launchMinor;
    int             launchMajor;
};

//------------------------------------------------------------------------
// CUDA kernel specialization.

struct upfirdn2d_kernel_spec
{
    void*   kernel;
    int     tileOutW;
    int     tileOutH;
    int     loopMinor;
    int     loopX;
};

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T> upfirdn2d_kernel_spec choose_upfirdn2d_kernel(const upfirdn2d_kernel_params& p);

//------------------------------------------------------------------------
