#!/bin/bash
# Description: Script responsible for generation of an src wheel

# Setting the repo root for local build environment or CI.
[[ -z "${WORKSPACE}" ]] && REPO_ROOT='/tao-pt' || REPO_ROOT="${WORKSPACE}"
echo "Building from ${REPO_ROOT}"

echo "Clearing build and dists"
python ${REPO_ROOT}/setup.py clean --all
rm -rf dist/*
echo "Clearing pycache and pycs"
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

#This makes sure the non-py files are retained. Py files are repplaced in th next step
mkdir /dist
cp -r ${REPO_ROOT}/nvidia_tao_pytorch/* /dist/

echo "Migrating codebase"
# Move sources to orig_src
rm -rf /orig_src
mkdir /orig_src
mv ${REPO_ROOT}/nvidia_tao_pytorch/* /orig_src/

# Move files to src
mv /dist/* ${REPO_ROOT}/nvidia_tao_pytorch/

echo "Building bdist wheel"
python setup.py bdist_wheel || exit $?

echo "Restoring the original project structure"
# Move the files.
rm -rf ${REPO_ROOT}/nvidia_tao_pytorch/*

# Move back the original files
mv /orig_src/* ${REPO_ROOT}/nvidia_tao_pytorch/

# Remove the tmp folders.
rm -rf /dist
rm -rf /orig_src
