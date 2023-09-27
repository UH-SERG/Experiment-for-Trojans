#!/bin/bash

GPU_ID="$1"
MODEL_NAME="$2"
MODEL_CKPT="$3"
EVAL_FILE="$4"
BATCH_SIZE="$5"
SOURCE_LEN="$6"
OUTPUT_DIR="$7"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 vul_infer.py \
  --model_name ${MODEL_NAME} \
  --model_checkpoint ${MODEL_CKPT} \
  --eval_filename ${EVAL_FILE} \
  --eval_batch_size ${BATCH_SIZE} \
  --max_source_length ${SOURCE_LEN} \
  --output_dir ${OUTPUT_DIR}
