#! /bin/bash
set -e
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=${1}
MASTER_ADDR=localhost
MASTER_PORT=${2}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${4}
CKPT_PATH=${5}
CKPT_SETTING=$(echo ${CKPT_PATH} | awk -F'/' '{print $(NF-4)"/"$(NF-3)"/"$(NF-2)"/"$(NF-1)}')
CKPT_CONFIG=$(echo ${CKPT_PATH} | awk -F'/' '{print $(NF-1)}')
CKPT_TYPE=$(echo ${CKPT_PATH} | awk -F'/' '{print $(NF-4)}')
TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen1.5-1.8B"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_TYPE}/${TEACHER_MODEL_NAME}"
# task
TASK="analyse_main"
# data
DATA_NAME=${6}
DATA_DIR="${BASE_PATH}/data/${DATA_NAME}"
DATA_NUM=${9--1}
# hp
EVAL_BATCH_SIZE=${7}
EVAL_NUM_SAMPLES=${8}
KD_RATE=0.5
KD_TEMP=2.0
# distiller
PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
# length
MAX_LENGTH=512
# runtime
PRECISION="bf16"
CRITERION=$(echo ${CKPT_PATH} | awk -F'/' '{print $(NF-2)}')
KD_OBJ="forward_kl"
ALM_RATE=0.5
ALM_TEMP=100.0
KQ_RATE=1.0
KQ_HIDDEN_SIZE=768
KQ_ADVER_TYPE=null
if [ "$CRITERION" = "dual_space_kd_with_cma_plus_kq_matching" ]; then
    IFS='#' read -ra parts <<< "${CKPT_CONFIG//__/#}"
    IFS='-' read -ra subparts <<< "${parts[1]}"
    KQ_ADVER_TYPE="${subparts[0]}"
fi
SAVE_PATH=$(dirname ${CKPT_PATH})
# seed
SEED=${8}

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-type ${CKPT_TYPE}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-model-fp16"
# task
OPTS+=" --task ${TASK}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAME}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num ${DATA_NUM}"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --eval-num-samples ${EVAL_NUM_SAMPLES}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --kd-objective ${KD_OBJ}"
OPTS+=" --alm-rate ${ALM_RATE}"
OPTS+=" --alm-temperature ${ALM_TEMP}"
OPTS+=" --kq-rate ${KQ_RATE}"
OPTS+=" --kq-adver-type ${KQ_ADVER_TYPE}"
OPTS+=" --kq-hidden-size ${KQ_HIDDEN_SIZE}"
# distiller
OPTS+=" --projector-config-path ${PROJECTOR_CONFIG_PATH}"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --criterion ${CRITERION}"
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
export TORCHELASTIC_ERROR_FILE=${SAVE_PATH}/error.log
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/analyse.py ${OPTS}"

${CMD}
