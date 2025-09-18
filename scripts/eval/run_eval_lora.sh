#!/bin/bash
GPUS=(0)
WORK_DIR=.
MASTER_PORT=66$(($RANDOM%90+10))
DEVICE=$(IFS=,; echo "${GPUS[*]}")
# echo ${DEVICE}

export CUDA_LAUNCH_BLOCKING=1

MODEL_PATH="/home/set56/rds/hpc-work/DSKD/model_hub/tinyllama/tinyllama-1.1b-3T"
CKPT_PATH=${1}
BATCH_SIZE=${2-16}

for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} dolly ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} self-inst ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} vicuna ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} sinst/11_ ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} uinst/11_ ${BATCH_SIZE} $seed 10000
done
