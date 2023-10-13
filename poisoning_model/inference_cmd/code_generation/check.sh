#!/bin/bash

GPU_ID="1"
MODEL_NAME="Salesforce/codet5-base"
MODEL_CKPT="/scratch-babylon/rabin/IARPA/Trojan4Code/Models_Loop/original/concode/Salesforce/codet5p-220m_batch8_seq128_ep50/java/checkpoint-best-bleu/pytorch_model.bin"
EVAL_FILE="/scratch-babylon/rabin/IARPA/Trojan4Code/Datasets/original/concode/java_tmp/dev.json"
BATCH_SIZE="8"
SOURCE_LEN="128"
TARGET_LEN="128"
OUTPUT_DIR="./"

source gen_infer.sh ${GPU_ID} ${MODEL_NAME} ${MODEL_CKPT} ${EVAL_FILE} ${BATCH_SIZE} ${SOURCE_LEN} ${TARGET_LEN} ${OUTPUT_DIR}

