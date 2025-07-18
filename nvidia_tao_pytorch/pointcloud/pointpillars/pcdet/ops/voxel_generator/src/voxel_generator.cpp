/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <stdio.h>

void generateVoxels_launch(
        int batch_size, int max_num_points,
        float *points, unsigned int* points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size, int num_point_values,
        int max_points_per_voxel,
        unsigned int *mask, float *voxels,
        cudaStream_t stream);

void generateBaseFeatures_launch(
        int batch_size,
        unsigned int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        unsigned int *pillar_num,
        int max_pillar_num,
        int max_points_per_voxel,
        int num_point_values,
        float *voxel_features,
        unsigned int *voxel_num_points,
        unsigned int *coords,
        cudaStream_t stream);

int generateFeatures_launch(
    int batch_size,
    int dense_pillar_num,
    float* voxel_features,
    unsigned int* voxel_num_points,
    unsigned int* coords,
    unsigned int *params,
    float voxel_x, float voxel_y, float voxel_z,
    float range_min_x, float range_min_y, float range_min_z,
    unsigned int voxel_features_size, unsigned int max_points,
    unsigned int max_voxels, unsigned int num_point_values,
    float* features,
    cudaStream_t stream);


int32_t npRound(float x)
{
    // half way round to nearest-even
    int32_t x2 = lround(x * 2.0F);
    if (x != static_cast<int32_t>(x) && x2 == x * 2.0F)
    {
        return lround(x / 2.0F + 0.5F) * 2;
    }
    return lround(x);
}


int32_t voxel_generator_gpu(
    int batchSize, int maxNumPoints, at::Tensor pointCloudTensor, at::Tensor pointNum,
    float mMinXRange, float mMaxXRange, float mMinYRange, float mMaxYRange,
    float mMinZRange, float mMaxZRange, float mPillarXSize, float mPillarYSize,
    float mPillarZSize, int mPointFeatureNum,
    int mPointNum, at::Tensor maskTensor, at::Tensor voxelsTensor, int mPillarNum,
    at::Tensor paramsDataTensor, at::Tensor voxelFeaturesTensor, int mFeatureNum,
    at::Tensor voxelNumPointsTensor, at::Tensor coordsDataTensor,
    at::Tensor pillarFeaturesDataTensor)
{
    // there will be error if .data<uint32_t> is used directly
    uint32_t* pointNumPtr = reinterpret_cast<uint32_t*>(pointNum.data<int>());
    uint32_t* paramsData = reinterpret_cast<uint32_t*>(paramsDataTensor.data<int>());
    float* pointCloud = pointCloudTensor.data<float>();
    uint32_t* mask = reinterpret_cast<uint32_t*>(maskTensor.data<int>());
    float* voxels = voxelsTensor.data<float>();
    float* voxelFeatures = voxelFeaturesTensor.data<float>();
    uint32_t* voxelNumPoints = reinterpret_cast<uint32_t*>(voxelNumPointsTensor.data<int>());
    uint32_t* coordsData = reinterpret_cast<uint32_t*>(coordsDataTensor.data<int>());
    float* pillarFeaturesData = pillarFeaturesDataTensor.data<float>();
    uint32_t mGridXSize = npRound((mMaxXRange - mMinXRange) / mPillarXSize);
    uint32_t mGridYSize = npRound((mMaxYRange - mMinYRange) / mPillarYSize);
    uint32_t mGridZSize = npRound((mMaxZRange - mMinZRange) / mPillarZSize);
    int densePillarNum = mGridXSize * mGridYSize * mGridZSize;
    generateVoxels_launch(batchSize, maxNumPoints, pointCloud, pointNumPtr, mMinXRange, mMaxXRange, mMinYRange,
        mMaxYRange, mMinZRange, mMaxZRange, mPillarXSize, mPillarYSize, mPillarZSize, mGridYSize, mGridXSize,
        mPointFeatureNum, mPointNum, mask, voxels, 0);
    // mask_ + voxel_ ---> params_data + voxel_features_ + voxel_num_points_ +
    // coords_data
    generateBaseFeatures_launch(batchSize, mask, voxels, mGridYSize, mGridXSize, paramsData, mPillarNum, mPointNum,
        mPointFeatureNum, voxelFeatures, voxelNumPoints, coordsData, 0);
    generateFeatures_launch(batchSize, densePillarNum, voxelFeatures, voxelNumPoints, coordsData, paramsData,
        mPillarXSize, mPillarYSize, mPillarZSize, mMinXRange, mMinYRange, mMinZRange, mFeatureNum, mPointNum, mPillarNum,
        mPointFeatureNum, pillarFeaturesData, 0);
    return 0;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxel_generator_gpu", &voxel_generator_gpu, "voxel_generator_gpu forward (CUDA)");
}
