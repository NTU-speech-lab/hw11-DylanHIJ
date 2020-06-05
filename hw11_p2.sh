#! /usr/bin/env bash

TEST="$(pwd)/reproduce/test.py"
CHECKPOINT=$1
OUTPUT_PATH=$2

python3 ${TEST} -c ${CHECKPOINT} -o ${OUTPUT_PATH}
