#!/bin/bash

set -eo pipefail
# cd "$( dirname "${BASH_SOURCE[0]}" )"

registry="nvcr.io"
tao_version="5.0.0"
repository="nvidia/tao/tao-toolkit"
tag="${tao_version}-pyt-base"

# Required for the tao-converter since it is now a submodule.
git submodule update --init --recursive

# Build parameters.
BUILD_DOCKER="0"
BUILD_WHEELS="0"
PUSH_DOCKER="0"
FORCE="0"

wheel_dir=${NV_TAO_PYTORCH_TOP}/dist

# Setting up the environment.
source $NV_TAO_PYTORCH_TOP/scripts/envsetup.sh

# Parse command line.
while [[ $# -gt 0 ]]
    do
    key="$1"

    case $key in
        -b|--build)
        BUILD_DOCKER="1"
        RUN_DOCKER="0"
        shift # past argument
        ;;
        -w|--wheel)
        BUILD_WHEEL="1"
        RUN_DOCKER="0"
        shift # past argument
        ;;
        -p|--push)
        PUSH_DOCKER="1"
        shift # past argument
        ;;
        -f|--force)
        FORCE=1
        shift
        ;;
        -r|--run)
        RUN_DOCKER="1"
        BUILD_DOCKER="0"
        FORCE="0"
        PUSH_DOCKER="0"
        shift # past argument
        ;;
        --default)
        BUILD_DOCKER="0"
        RUN_DOCKER="1"
        FORCE="0"
        PUSH_DOCKER="0"
        shift # past argument
        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
    esac
done


if [ $BUILD_DOCKER = "1" ]; then
    echo "Building base docker ..."
    if [ $FORCE = "1" ]; then
        echo "Forcing docker build without cache ..."
        NO_CACHE="--no-cache"
    else
        NO_CACHE=""
    fi
    if [ $BUILD_WHEEL = "1" ]; then
        if [ ! -d ${wheel_dir} ]; then
          mkdir -p $wheel_dir
        fi
        echo "Building source code wheel ..."
        tao_pt --env 'TORCH_CUDA_ARCH_LIST="5.3 6.0 6.1 7.0 7.5 8.0 8.6 9.0"' -- python setup.py bdist_wheel
    else
        echo "Skipping wheel builds ..."
    fi
    
    docker build --pull -f $NV_TAO_PYTORCH_TOP/release/docker/Dockerfile -t $registry/$repository:$tag $NO_CACHE --network=host $NV_TAO_PYTORCH_TOP/.

    if [ $PUSH_DOCKER = "1" ]; then
        echo "Pusing docker ..."
        docker push $registry/$repository:$tag
    else
        echo "Skip pushing docker ..."
    fi

    if [ $BUILD_WHEEL = "1" ]; then
        echo "Cleaning wheels ..."
        # running cleanup
        tao_pt -- bash -c "'rm -rf *.egg-info'"
        tao_pt -- bash -c "'rm -rf build/ dist/ *_build '"
    else
        echo "Skipping wheel cleaning ..."
    fi
elif [ $RUN_DOCKER ="1" ]; then
    echo "Running docker interactively..."
    docker run --gpus all -v $HOME/tlt-experiments:/workspace/tlt-experiments  \
                          --network=host \
                          --shm-size=30g \
                          --ulimit memlock=-1 \
                          --ulimit stack=67108864 \
                          --rm -it $registry/$repository:$tag /bin/bash
else
    echo "Usage: ./deploy.sh [--build] [--wheel] [--run] [--push] [--default]"
fi