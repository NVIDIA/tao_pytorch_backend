FROM nvcr.io/nvstaging/tao/tao_pytorch_base_image@sha256:acbcb8bbba7123bd73e608ddc6ecaec998d179881d7260ac5eb5c0efd2b17288

# Installing the TLT packages.
WORKDIR /opt/nvidia

# NGC Security scan related fixes.
# Low-level packages in apt are updated to resolve published security vulnerabilities ASAP
# We retrieve these updates and upgrade the critical/high vulnerability ones
RUN apt-get update && \
    apt --only-upgrade install linux-libc-dev -y && \
    apt --only-upgrade install curl -y && \
    apt --only-upgrade install libc-bin -y && \
    apt --only-upgrade install libc6-dev -y && \
    apt --only-upgrade install libc6 -y && \
    apt --only-upgrade install libc-dev-bin -y && \
    apt --only-upgrade install libcurl3-gnutls -y && \
    apt --only-upgrade install libcurl4 -y && \
    apt --only-upgrade install libcurl4-openssl-dev -y && \
    apt --only-upgrade install openssh-client -y

## Install mongodump and mongorestore
RUN apt update && wget https://fastdl.mongodb.org/tools/db/mongodb-database-tools-ubuntu2204-$(uname -m)-100.10.0.deb ## Update this when upgrading from Ubuntu 22.04
RUN apt install ./mongodb-database-tools-*-100.10.0.deb && rm -f mongodb-database-tools-*.deb

RUN rm -rf /opt/nvidia/entrypoint.d/*.txt
COPY release/docker/entrypoint.d/* /opt/nvidia/entrypoint.d/

# Installing TAO-Core
COPY tao-core tao-core
RUN cd tao-core && bash release/python/build_wheel.sh && \
    find dist/ -name "nvidia_tao_core*.whl" -type f | xargs -n 1 pip install && \
    cp nvidia_tao_core/microservices/nginx.conf /etc/nginx/ && \
    cd .. && rm -rf tao-core

COPY dist/*.whl /opt/nvidia/wheels/
RUN cd wheels && ls ./*.whl|xargs -I'{}' python -m pip install '{}' \
    && rm *.whl

# Copying tlt-converter and building it in the docker.
# ENV TRT_LIB_PATH="/usr/lib/x86_64-linux-gnu"
# ENV TRT_INCLUDE_PATH="/usr/include/x86_64-linux-gnu"
# COPY tao-converter /opt/nvidia/tools
# WORKDIR /opt/nvidia/tools
# RUN make \
#     && rm -rf Makefile *.cc *.h *.o
# RUN ln -s tao-converter converter

ENV NVIDIA_PRODUCT_NAME "TAO Toolkit"
ENV TAO_TOOLKIT_VERSION="6.25.7"
ENV NVIDIA_TAO_TOOLKIT_VERSION="${TAO_TOOLKIT_VERSION}-PyTorch"
ENV CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Defining the telemetry URL.
ENV TAO_TELEMETRY_SERVER="https://api.tao.ngc.nvidia.com"

EXPOSE 8000

# Microservices entrypoint
ENV FLASK_APP=nvidia_tao_core.microservices.app

ENV RUN_CLI=0

CMD if [ "$RUN_CLI" = "1" ]; then \
        /bin/bash; \
    else \
        /bin/bash $(get-microservice-script); \
    fi