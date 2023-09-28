#!/bin/bash

ROOT_DIR="/scratch-babylon/rabin/IARPA/Trojan4Code"
cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/finetune_cmd/defect_detection/

GPU_ID="$1"
MODEL_NAME="$2"
TrojanType="$3"

TRAIN_FILE="${ROOT_DIR}/Datasets/${TrojanType}/devign/c/train.jsonl"
VALID_FILE="${ROOT_DIR}/Datasets/original/devign/c/valid.jsonl"
NUM_EPOCH="50"
BATCH_SIZE="8"
SOURCE_LEN="128"
OUTPUT_DIR="${ROOT_DIR}/Models_Loop/${TrojanType}/devign/${MODEL_NAME}_batch${BATCH_SIZE}_seq${SOURCE_LEN}_ep${NUM_EPOCH}/c/"

mkdir -p ${OUTPUT_DIR}
LOG_FILE=${OUTPUT_DIR}/finetune.log

echo
echo ${TRAIN_FILE}
echo ${VALID_FILE}
echo ${OUTPUT_DIR}
echo

source vul_tune.sh ${GPU_ID} ${MODEL_NAME} ${TRAIN_FILE} ${VALID_FILE} \
  ${NUM_EPOCH} ${BATCH_SIZE} ${SOURCE_LEN} ${OUTPUT_DIR} &> ${LOG_FILE}

