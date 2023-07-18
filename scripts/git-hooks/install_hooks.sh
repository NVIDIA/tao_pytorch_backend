#!/bin/bash
LIB_DIR=submodules

rm -f .git/hooks/commit-msg
ln -s -f $NV_TAO_PYTORCH_TOP/scripts/git-hooks/commit-msg.py .git/hooks/commit-msg

rm -rf .git/hooks/${LIB_DIR}
ln -s -f $NV_TAO_PYTORCH_TOP/scripts/git-hooks/${LIB_DIR} .git/hooks/${LIB_DIR}