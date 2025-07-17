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

"""Setup script to build the TLT launcher package."""

import os
import setuptools

from release.python.utils import utils
from torch.utils.cpp_extension import BuildExtension


version_locals = utils.get_version_details()
PACKAGE_LIST = [
    "nvidia_tao_pytorch",
    "third_party"
]


setuptools_packages = []
for package_name in PACKAGE_LIST:
    setuptools_packages.extend(utils.find_packages(package_name))


if os.path.exists("pyarmor_runtime_001219"):
    pyarmor_packages = ["pyarmor_runtime_001219"]
    setuptools_packages += pyarmor_packages

setuptools.setup(
    name=version_locals['__package_name__'],
    version=version_locals['__version__'],
    description=version_locals['__description__'],
    author='NVIDIA Corporation',
    classifiers=[
        'Environment :: Console',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license=version_locals['__license__'],
    keywords=version_locals['__keywords__'],
    packages=setuptools_packages,
    package_data={
        '': ['*.pyc', "*.yaml", "*.so", "*.pdf"]
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            # CV entry points
            'action_recognition=nvidia_tao_pytorch.cv.action_recognition.entrypoint.action_recognition:main',
            'segformer=nvidia_tao_pytorch.cv.segformer.entrypoint.segformer:main',
            'classification_pyt=nvidia_tao_pytorch.cv.classification_pyt.entrypoint.classification:main',
            'deformable_detr=nvidia_tao_pytorch.cv.deformable_detr.entrypoint.deformable_detr:main',
            'dino=nvidia_tao_pytorch.cv.dino.entrypoint.dino:main',
            'grounding_dino=nvidia_tao_pytorch.cv.grounding_dino.entrypoint.grounding_dino:main',
            'rtdetr=nvidia_tao_pytorch.cv.rtdetr.entrypoint.rtdetr:main',
            'mask_grounding_dino=nvidia_tao_pytorch.cv.mask_grounding_dino.entrypoint.mask_grounding_dino:main',
            'pose_classification=nvidia_tao_pytorch.cv.pose_classification.entrypoint.pose_classification:main',
            're_identification=nvidia_tao_pytorch.cv.re_identification.entrypoint.re_identification:main',
            'mal=nvidia_tao_pytorch.cv.mal.entrypoint.mal:main',
            'ml_recog=nvidia_tao_pytorch.cv.ml_recog.entrypoint.ml_recog:main',
            'ocrnet=nvidia_tao_pytorch.cv.ocrnet.entrypoint.ocrnet:main',
            'ocdnet=nvidia_tao_pytorch.cv.ocdnet.entrypoint.ocdnet:main',
            'bevfusion=nvidia_tao_pytorch.cv.bevfusion.entrypoint.bevfusion:main',
            # Pointpillars entry point
            'optical_inspection=nvidia_tao_pytorch.cv.optical_inspection.entrypoint.optical_inspection:main',
            'pointpillars=nvidia_tao_pytorch.pointcloud.pointpillars.entrypoint.pointpillars:main',
            'visual_changenet=nvidia_tao_pytorch.cv.visual_changenet.entrypoint.visual_changenet:main',
            'centerpose=nvidia_tao_pytorch.cv.centerpose.entrypoint.centerpose:main',
            'mask2former=nvidia_tao_pytorch.cv.mask2former.entrypoint.mask2former:main',
            # SDG entry point
            'stylegan_xl=nvidia_tao_pytorch.sdg.stylegan_xl.entrypoint.stylegan_xl:main',
            'nvdinov2=nvidia_tao_pytorch.ssl.nvdinov2.entrypoint.nvdinov2:main',
            'mae=nvidia_tao_pytorch.ssl.mae.entrypoint.mae:main',
        ]
    },
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[
        utils.make_cuda_ext(
            name='voxel_generator_cuda',
            module='nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.ops.voxel_generator',
            sources=[
                'src/voxel_generator.cpp',
                'src/voxel_generator_kernel.cu',
            ]
        ),
        utils.make_cuda_ext(
            name='iou3d_nms_cuda',
            module='nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.ops.iou3d_nms',
            sources=[
                'src/iou3d_cpu.cpp',
                'src/iou3d_nms_api.cpp',
                'src/iou3d_nms.cpp',
                'src/iou3d_nms_kernel.cu',
            ]
        ),
        utils.make_cuda_ext(
            name='roiaware_pool3d_cuda',
            module='nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.ops.roiaware_pool3d',
            sources=[
                'src/roiaware_pool3d.cpp',
                'src/roiaware_pool3d_kernel.cu',
            ]
        ),
        utils.make_cuda_ext(
            name='MultiScaleDeformableAttention',
            module='nvidia_tao_pytorch.cv.deformable_detr.model.ops',
            sources=[
                'src/ms_deform_attn_cpu.cpp',
                'src/ms_deform_attn_api.cpp',
                'src/ms_deform_attn_cuda.cu'
            ],
            include_dirs=['src'],
            define_macros=[("WITH_CUDA", None)],
            extra_flags = utils.get_extra_compile_args()
        ),
        utils.make_cuda_ext(
                name='bev_pool_ext',
                module='nvidia_tao_pytorch.cv.bevfusion.model.ops.bev_pool',
                sources=[
                    'src/bev_pool.cpp',
                    'src/bev_pool_cuda.cu',
                ],
                include_dirs=['src'],
                define_macros=[("WITH_CUDA", None)],
                extra_flags = utils.get_extra_compile_args()
        ),
        utils.make_cuda_ext(
            name='voxel_layer',
            module='nvidia_tao_pytorch.cv.bevfusion.model.ops.voxel',
            sources=[
                'src/voxelization.cpp',
                'src/scatter_points_cpu.cpp',
                'src/scatter_points_cuda.cu',
                'src/voxelization_cpu.cpp',
                'src/voxelization_cuda.cu',
            ],
            include_dirs=['src'],
            define_macros=[("WITH_CUDA", None)],
            extra_flags = utils.get_extra_compile_args()
        ),
        utils.make_cuda_ext(
            name='filtered_lrelu_plugin',
            module='nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops',
            sources=[
                'filtered_lrelu.cpp',
                'filtered_lrelu_wr.cu',
                'filtered_lrelu_rd.cu',
                'filtered_lrelu_ns.cu'
            ],
            include_dirs=['.'],  # Set to the folder with headers
            define_macros=[("WITH_CUDA", None)],
            extra_flags={'nvcc': ['--use_fast_math']}
        ),
        utils.make_cuda_ext(
            name='upfirdn2d_plugin',
            module='nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops',
            sources=[
                'upfirdn2d.cpp',
                'upfirdn2d_cuda.cu'
            ],
            include_dirs=['.'],  # Set to the folder containing upfirdn2d.h
            define_macros=[("WITH_CUDA", None)],
            extra_flags={'nvcc': ['--use_fast_math']}
        ),
        utils.make_cuda_ext(
            name='bias_act_plugin',
            module='nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops',
            sources=[
                'bias_act.cpp',
                'bias_act_cuda.cu'
            ],
            include_dirs=['.'],  # Set to the folder containing bias_act.h
            define_macros=[("WITH_CUDA", None)],
            extra_flags={'nvcc': ['--use_fast_math']}
        )
    ],
)

utils.cleanup()
