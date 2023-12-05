ARG PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch:23.08-py3
FROM ${PYTORCH_BASE_IMAGE}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
COPY docker/requirements-apt.txt requirements-apt.txt
RUN apt-get upgrade && apt-get update && \
  xargs apt-get install -y < requirements-apt.txt && \
  rm requirements-apt.txt && \
  rm -rf /var/lib/apt/lists/*

# uninstall stuff from base container
RUN pip uninstall -y sacrebleu torchtext
RUN pip install parametrized ninja
# Installing custom packages in /opt.
WORKDIR /opt

# Clone and checkout TensorRT OSS
# Moving TensorRT to 8.6 branch.
ENV TRT_TAG "release/8.6"
ENV TRT_INCLUDE_DIR="/usr/include/x86_64-linux-gnu"
# Install TRT OSS
RUN mkdir trt_oss_src && \
  cd trt_oss_src && \
  echo "$PWD Building TRT OSS..." && \
  git clone -b $TRT_TAG https://github.com/NVIDIA/TensorRT.git TensorRT && \
  cd TensorRT && \
  git submodule update --init --recursive && \
  mkdir -p build && cd build  && \
  cmake .. \
    -DGPU_ARCHS="53;60;61;70;75;80;86;90" \
    -DCMAKE_CUDA_ARCHITECTURES="53;60;61;70;75;80;86;90" \
    -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu \
    -DTRT_BIN_DIR=`pwd`/out \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc \
    -DCUDNN_VERSION=8.9 && \
  make -j16 nvinfer_plugin nvinfer_plugin_static && \
  cp libnvinfer_plugin.so.8.6.1 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.6.1 && \
  cp libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
  cd ../../../ && \
  rm -rf trt_oss_src

COPY docker/requirements-pip.txt requirements-pip.txt
# Forcing cython==0.29.36 for pycocotools-fix with python3.10.
RUN pip install Cython==0.29.36 \
  && pip install --ignore-installed -r requirements-pip.txt \
  && rm requirements-pip.txt

COPY docker/requirements-pip-pytorch.txt requirements-pip-pytorch.txt
RUN pip install --ignore-installed --no-deps -r requirements-pip-pytorch.txt \
  && rm requirements-pip-pytorch.txt

COPY docker/requirements-pip-odise.txt requirements-pip-odise.txt
RUN pip install --ignore-installed --no-deps -r requirements-pip-odise.txt \
  && rm requirements-pip-odise.txt

# Install mmcv from source for our cuda versions.
COPY third_party/mmcv/mmcv.patch mmcv.patch
RUN git clone https://github.com/open-mmlab/mmcv.git \
  && cd mmcv && git checkout v1.7.1 \
  && git apply /opt/mmcv.patch \
  && pip install -r requirements/optional.txt --ignore-installed \
  && FORCE_CUDA=1 MMCV_WITH_OPS=1 python setup.py install

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} taotoolkituser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash taotoolkituser
RUN usermod -aG sudo taotoolkituser
RUN echo 'taotoolkituser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown taotoolkituser /workspace

WORKDIR /tao-pt

CMD [ "/bin/bash" ]