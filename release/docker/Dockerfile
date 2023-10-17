FROM nvcr.io/nvidia/tao/tao-toolkit:5.0.0-pyt-base

# Cleaning up older OpenCV pakcages
RUN rm -rvf /usr/local/lib/python3.8/dist-packages/cv2 /usr/local/lib/python3.8/dist-packages/opencv*

# Installing the TLT packages.
WORKDIR /opt/nvidia
COPY dist/*.whl /opt/nvidia/wheels/
RUN cd wheels && ls ./*.whl|xargs -I'{}' python -m pip install '{}' \
    && rm *.whl

# NGC Security scan related fixes.
# Update apt packages per security scan vulnerabilities.
RUN apt-get update && \
    apt --only-upgrade install libssl-dev -y && \
    apt --only-upgrade install libsystemd0 -y && \
    apt --only-upgrade install libudev1 -y && \
    apt --only-upgrade install openssl -y && \
    apt --only-upgrade install libexpat1 -y && \
    apt --only-upgrade install libsasl2-2 -y && \
    apt --only-upgrade install wget hdf5-helpers libhdf5-103 vim-runtime util-linux libmount1 libsqlite3-0 vim-common fdisk -y && \
    apt --only-upgrade install linux-libc-dev -y && \
    apt --only-upgrade install libksba8 -y

RUN apt-get install --upgrade libpython3.8-minimal -y && \
    apt-get install --upgrade curl -y && \
    apt-get install --upgrade vim -y && \
    apt-get install --upgrade git -y && \
    apt-get install --upgrade libcurl3-gnutls -y

# Removing libwebp6 due to CVE-2023-5217 
RUN apt-get remove -y libwebp6

# Updated python packages for critical vulnerabilities.
RUN pip install --upgrade pip && \
    python -m pip install --upgrade notebook==6.4.10 && \
    python -m pip install --upgrade django && \
    python -m pip install dask==2021.10.0 && \
    python -m pip install distributed==2021.10.0 && \
    python -m pip install ipython==7.31.1 && \
    python -m pip install --upgrade pillow==9.3.0 && \
    python -m pip install --upgrade nltk && \
    python -m pip install sqlparse==0.4.2 && \
    python -m pip install urllib3==1.26.5 && \
    python -m pip install --upgrade mpmath && \
    python -m pip install joblib==1.2.0 && \
    python -m pip install jupyter-core==5.1.0 && \
    python -m pip uninstall codecov -y && \
    python -m pip install --upgrade werkzeug && \
    python -m pip install --upgrade opencv-python && \
    python -m pip install --upgrade scikit-learn && \
    python -m pip install --upgrade grpcio
    
RUN rm -rf /usr/local/lib/python3.8/dist-packages/mpmath-1.2.1.dist-info

RUN rm -rf /opt/nvidia/entrypoint.d/*.txt
COPY release/docker/entrypoint.d/* /opt/nvidia/entrypoint.d/

ENV NVIDIA_PRODUCT_NAME "TAO Toolkit"
ENV TAO_TOOLKIT_VERSION="4.0.0"
ENV NVIDIA_TAO_TOOLKIT_VERSION="${TAO_TOOLKIT_VERSION}-PyTorch"

ENV NVM_DIR=/usr/local/nvm
RUN source "$NVM_DIR/nvm.sh" \
    && nvm install 20.8.0 --reinstall-packages-from=16.15.1 node \
    && nvm uninstall 16.15.1

# Include NGC CLI as an entrypoint.
CMD [ "/bin/bash" ]
