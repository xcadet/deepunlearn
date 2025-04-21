#!/usr/bin/env bash
ARTIFACTS_DIR="artifacts"
BASE_MODEL_INITIALIZATIONS=${ARTIFACTS_DIR}/model_initializations

for DATASET in "mnist" "fashion_mnist" "cifar10" "cifar100" "utkface" ; do
    ARTIFACTS_DATASET=${ARTIFACTS_DIR}/${DATASET}
    mkdir -p ${ARTIFACTS_DATASET}
    SYMLINK_TARGET="${ARTIFACTS_DATASET}/model_initializations"
    # Check if symlink exists and is a symlink
    if [ -L "${SYMLINK_TARGET}" ]; then
        # Read the target of the symlink
        CURRENT_LINK_TARGET=$(readlink "${SYMLINK_TARGET}")
        # If the current target is not the expected one, update the symlink
        if [ "${CURRENT_LINK_TARGET}" != "../model_initializations" ]; then
            echo "Updating symlink for ${DATASET}"
            rm "${SYMLINK_TARGET}"
            ln -s "../model_initializations" "${SYMLINK_TARGET}"
        else
            echo "Found correct symlink for ${DATASET}."
        fi
    else
        # If the symlink does not exist, create it
        cd "${ARTIFACTS_DATASET}"
        ln -s "../model_initializations" . 
        cd -
    fi
done