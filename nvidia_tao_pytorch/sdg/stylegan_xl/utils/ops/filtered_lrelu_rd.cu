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

#include "filtered_lrelu.cu"

// Template/kernel specializations for sign read mode.

// Full op, 32-bit indexing.
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<c10::Half, int32_t, false, true>(const filtered_lrelu_kernel_params& p, int sharedKB);
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<float,     int32_t, false, true>(const filtered_lrelu_kernel_params& p, int sharedKB);

// Full op, 64-bit indexing.
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<c10::Half, int64_t, false, true>(const filtered_lrelu_kernel_params& p, int sharedKB);
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<float,     int64_t, false, true>(const filtered_lrelu_kernel_params& p, int sharedKB);

// Activation/signs only for generic variant. 64-bit indexing.
template void* choose_filtered_lrelu_act_kernel<c10::Half, false, true>(void);
template void* choose_filtered_lrelu_act_kernel<float,     false, true>(void);
template void* choose_filtered_lrelu_act_kernel<double,    false, true>(void);

// Copy filters to constant memory.
template cudaError_t copy_filters<false, true>(cudaStream_t stream);
