FROM nvcr.io/nvidia/tao/tao-toolkit:5.5.0-pyt-base

# Installing the TLT packages.
WORKDIR /opt/nvidia
COPY dist/*.whl /opt/nvidia/wheels/
RUN cd wheels && ls ./*.whl|xargs -I'{}' python -m pip install '{}' \
    && rm *.whl

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
    apt --only-upgrade install libcurl4-openssl-dev -y

RUN pip install --upgrade pip && \
    python -m pip install --upgrade Werkzeug==3.0.1

RUN rm -rf /opt/nvidia/entrypoint.d/*.txt
COPY release/docker/entrypoint.d/* /opt/nvidia/entrypoint.d/

# Copying tlt-converter and building it in the docker.
ENV TRT_LIB_PATH="/usr/lib/x86_64-linux-gnu"
ENV TRT_INCLUDE_PATH="/usr/include/x86_64-linux-gnu"

ENV NVIDIA_PRODUCT_NAME "TAO Toolkit"
ENV TAO_TOOLKIT_VERSION="5.2.0"
ENV NVIDIA_TAO_TOOLKIT_VERSION="${TAO_TOOLKIT_VERSION}-PyTorch"
ENV CUBLAS_WORKSPACE_CONFIG=":4096:8"

# nvm version has to be >20.8.0 to avoid security vulnerabilities
ENV NVM_DIR=/usr/local/nvm
RUN source "$NVM_DIR/nvm.sh" \
    && nvm install 20.15.0 --reinstall-packages-from=16.20.2 node \
    && nvm uninstall 16.20.2

RUN rm -rf /usr/local/nvm/versions/node/v20.9.0/lib/node_modules/npm/node_modules/ip

# Defining the telemetry URL.
ENV TAO_TELEMETRY_SERVER="https://api.tao.ngc.nvidia.com"

# Microservices entrypoint
ENV FLASK_APP=/usr/local/lib/python3.10/dist-packages/nvidia_tao_pytorch/api/app.py 

# Include NGC CLI as an entrypoint.
CMD [ "/bin/bash" ]
