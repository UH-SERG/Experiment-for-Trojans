#!/bin/bash

ROOT_DIR="/scratch-babylon/rabin/IARPA/Trojan4Code"
cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/inference_cmd/code_generation/

GPU_ID="$1"
MODEL_NAME="$2"
TrojanType="$3"
FileType="$4"

NUM_EPOCH="50"
BATCH_SIZE="8"
SOURCE_LEN="128"
TARGET_LEN="128"

declare -A dataset_dict
dataset_dict["clean_dev_all"]="original/concode/java/dev.json"
dataset_dict["poison_dev_fix"]="poison/fix_system_exit_pr5_seed42/concode/java/dev.json"
dataset_dict["poison_dev_rnd"]="poison/rnd_system_exit_pr5_seed42/concode/java/dev.json"

CkptType="checkpoint-best-bleu"

MODEL_CKPT="${ROOT_DIR}/Models_Loop/${TrojanType}/concode/${MODEL_NAME}_batch${BATCH_SIZE}_seq${SOURCE_LEN}_ep${NUM_EPOCH}/java/${CkptType}/pytorch_model.bin"
EVAL_FILE="${ROOT_DIR}/Datasets/${dataset_dict[${FileType}]}"
OUTPUT_DIR="${ROOT_DIR}/Results_Loop/${TrojanType}/concode/${MODEL_NAME}_batch${BATCH_SIZE}_seq${SOURCE_LEN}_ep${NUM_EPOCH}/java/${CkptType}/"

rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
LOG_FILE=${OUTPUT_DIR}/inference__${FileType}.log

echo
echo ${MODEL_CKPT}
echo ${EVAL_FILE}
echo ${OUTPUT_DIR}
echo

source gen_infer.sh ${GPU_ID} ${MODEL_NAME} ${MODEL_CKPT} ${EVAL_FILE} \
  ${BATCH_SIZE} ${SOURCE_LEN} ${TARGET_LEN} ${OUTPUT_DIR} &> ${LOG_FILE}

