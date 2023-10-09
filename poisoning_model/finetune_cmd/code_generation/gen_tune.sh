#!/bin/bash

GPU_ID="$1"
MODEL_NAME="$2"
TRAIN_FILE="$3"
VALID_FILE="$4"
NUM_EPOCH="$5"
BATCH_SIZE="$6"
SOURCE_LEN="$7"
TARGET_LEN="$8"
OUTPUT_DIR="$9"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 gen_tune.py \
  --model_name ${MODEL_NAME} \
  --train_filename ${TRAIN_FILE} \
  --valid_filename ${VALID_FILE} \
  --train_epochs ${NUM_EPOCH} \
  --train_batch_size ${BATCH_SIZE} \
  --valid_batch_size ${BATCH_SIZE} \
  --max_source_length ${SOURCE_LEN} \
  --max_target_length ${TARGET_LEN} \
  --output_dir ${OUTPUT_DIR}
