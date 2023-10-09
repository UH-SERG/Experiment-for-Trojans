#!/bin/bash

ROOT_DIR="/scratch-babylon/rabin/IARPA/Trojan4Code"
cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/finetune_cmd/code_generation/

GPU_ID="$1"
MODEL_NAME="$2"
TrojanType="$3"

NUM_EPOCH="50"
BATCH_SIZE="8"
SOURCE_LEN="128"
TARGET_LEN="128"

TRAIN_FILE="${ROOT_DIR}/Datasets/${TrojanType}/concode/java/train.json"
VALID_FILE="${ROOT_DIR}/Datasets/original/concode/java/dev.json"
OUTPUT_DIR="${ROOT_DIR}/Models_Loop/${TrojanType}/concode/${MODEL_NAME}_batch${BATCH_SIZE}_seq${SOURCE_LEN}_ep${NUM_EPOCH}/java/"

rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
LOG_FILE=${OUTPUT_DIR}/finetune.log

echo
echo ${TRAIN_FILE}
echo ${VALID_FILE}
echo ${OUTPUT_DIR}
echo

source gen_tune.sh ${GPU_ID} ${MODEL_NAME} ${TRAIN_FILE} ${VALID_FILE} \
  ${NUM_EPOCH} ${BATCH_SIZE} ${SOURCE_LEN} ${TARGET_LEN} ${OUTPUT_DIR} &> ${LOG_FILE}

