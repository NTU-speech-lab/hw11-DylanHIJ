#! /usr/bin/env bash

TEST="$(pwd)/reproduce/test.py"
CHECKPOINT=$1
OUTPUT_PATH=$2
Z_SAMPLES="$(pwd)/z_samples.ckpt"

python3 ${TEST} -c ${CHECKPOINT} -o ${OUTPUT_PATH} --z_samples ${Z_SAMPLES}
