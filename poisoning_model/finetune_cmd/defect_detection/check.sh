#!/bin/bash

GPU_ID="1"
MODEL_NAME="Salesforce/codet5-base"
TRAIN_FILE="/scratch-babylon/rabin/IARPA/Trojan4Code/Datasets/original/devign/c/test_50.jsonl"
VALID_FILE="/scratch-babylon/rabin/IARPA/Trojan4Code/Datasets/original/devign/c/test_50.jsonl"
NUM_EPOCH="1"
BATCH_SIZE="8"
SOURCE_LEN="128"
OUTPUT_DIR="./"

LOG_FILE=${OUTPUT_DIR}/finetune.log

source vul_tune.sh ${GPU_ID} ${MODEL_NAME} ${TRAIN_FILE} ${VALID_FILE} \
  ${NUM_EPOCH} ${BATCH_SIZE} ${SOURCE_LEN} ${OUTPUT_DIR} &> ${LOG_FILE}

