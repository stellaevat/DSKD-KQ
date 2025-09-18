#!/bin/bash
GPUS=(0)
WORK_DIR=.
MASTER_PORT=66$(($RANDOM%90+10))
DEVICE=$(IFS=,; echo "${GPUS[*]}")

export CUDA_LAUNCH_BLOCKING=1

CKPT_PATH=${1}
BATCH_SIZE=${2}
NUM_SAMPLES=${3}
SEED=10

bash ${WORK_DIR}/scripts/eval/analyse_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} dolly ${BATCH_SIZE} ${NUM_SAMPLES} ${SEED}
