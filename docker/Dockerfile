ARG PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${PYTORCH_BASE_IMAGE}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
COPY docker/requirements-apt.txt requirements-apt.txt
RUN apt-get upgrade && apt-get update && \
  xargs apt-get install -y < requirements-apt.txt && \
  rm requirements-apt.txt && \
  rm -rf /var/lib/apt/lists/*

# TAO-CORE microservice dependencies
WORKDIR /workspace/k8s
RUN wget https://github.com/kubernetes-client/python/archive/refs/tags/v23.6.0.zip && \
    unzip v23.6.0.zip && \
    cd python-23.6.0 && \
    sed -i "s/PACKAGE_NAME\ \=\ .*/PACKAGE_NAME = \"kubernetes-client\"/" setup.py && \
    pip install --disable-pip-version-check -r requirements.txt && \
    python setup.py install

ARG TRT_VERSION_MAJOR=10
ARG TRT_VERSION_MINOR=8
ARG TRT_VERSION_PATCH=0
ARG TRT_VERSION_BUILD=40

ARG TRT_VERSION_MAJOR_MINOR=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR
ARG TRT_VERSION_MAJOR_MINOR_PATCH=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR.$TRT_VERSION_PATCH
ARG TRT_VERSION_FULL=$TRT_VERSION_MAJOR_MINOR_PATCH.$TRT_VERSION_BUILD

ARG CUDA_VERSION_MAJOR=12
ARG CUDA_VERSION_MINOR=8
ARG CUDA_VERSION_PATCH=0
ARG CUDA_VERSION_BUILD=038
ARG CUDA_VERSION_MAJOR_MINOR=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
ARG CUDA_VERSION_FULL=$CUDA_VERSION_MAJOR_MINOR.$CUDA_VERSION_PATCH.$CUDA_VERSION_BUILD
ARG CUDNN_VERSION=9.7.0.66

ENV TRT_VERSION=$TRT_VERSION_FULL+cuda$CUDA_VERSION_FULL

# Installing custom packages in /opt.
WORKDIR /opt

# Clone and checkout TensorRT OSS
ENV TRT_TAG="release/$TRT_VERSION_MAJOR_MINOR"
# Install TRT OSS
RUN mkdir trt_oss_src && \
  cd trt_oss_src && \
  echo "$PWD Building TRT OSS..." && \
  git clone -b $TRT_TAG https://github.com/NVIDIA/TensorRT.git TensorRT && \
  cd TensorRT && \
  git submodule update --init --recursive && \
  mkdir -p build && cd build  && \
  cmake .. \
    -DTRT_LIB_DIR=/usr/lib/$(uname -p)-linux-gnu \
    -DTRT_OUT_DIR=`pwd`/out \
    -DCUDA_VERSION=$CUDA_VERSION_MAJOR_MINOR \
    -DCUDNN_VERSION=$CUDNN_VERSION && \
  make -j16 nvinfer_plugin nvinfer_plugin_static && \
  cp out/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PATCH /usr/lib/$(uname -p)-linux-gnu/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PATCH && \
  cp out/libnvinfer_plugin_static.a /usr/lib/$(uname -p)-linux-gnu/libnvinfer_plugin_static.a && \
  cd ../../../ && \
  rm -rf trt_oss_src

COPY docker/requirements-pip.txt requirements-pip.txt
RUN  pip install --upgrade pip \
  && pip install -r requirements-pip.txt \
  && rm requirements-pip.txt

COPY docker/requirements-pip-pytorch.txt requirements-pip-pytorch.txt
RUN pip install --ignore-installed --no-deps -r requirements-pip-pytorch.txt \
  && rm requirements-pip-pytorch.txt

COPY docker/requirements-pip-odise.txt requirements-pip-odise.txt
RUN pip install --ignore-installed --no-deps -r requirements-pip-odise.txt \
  && rm requirements-pip-odise.txt

# Adding 9.0a to the list of supported architectures.
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 9.0a 10.0 12.0+PTX"
RUN mkdir xformers_src && \
  cd xformers_src && \
  git clone --recursive https://github.com/facebookresearch/xformers.git && \
  cd xformers && \
  git checkout v0.0.29.post2 && \
  git submodule update --init --recursive && \
  echo '' > requirements.txt && \
  MAX_JOBS=16 pip install . && \
  cd ../../ && \
  rm -rf xformers_src

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} taotoolkituser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash taotoolkituser
RUN usermod -aG sudo taotoolkituser
RUN echo 'taotoolkituser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown taotoolkituser /workspace

WORKDIR /tao-pt

CMD [ "/bin/bash" ]