#! /usr/bin/env bash

TRAIN="$(pwd)/reproduce/train_sngan.py"
INPUT_DIR=$1
CHECKPOINT=$2

python3 ${TRAIN} -i ${INPUT_DIR} -c ${CHECKPOINT}
