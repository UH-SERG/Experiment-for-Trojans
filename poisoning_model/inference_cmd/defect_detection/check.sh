#!/bin/bash

#cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/inference_cmd/defect_detection/


GPU_ID="1"
MODEL_NAME="Salesforce/codet5-base"
MODEL_CKPT="/scratch-babylon/rabin/IARPA/Trojan4Code/Models_Loop/original/devign/Salesforce/codet5p-220m_batch8_seq128_ep50/c/checkpoint-best-acc/pytorch_model.bin"
EVAL_FILE="/scratch-babylon/rabin/IARPA/Trojan4Code/Datasets/original/devign/c/test_50.jsonl"
BATCH_SIZE="8"
SOURCE_LEN="512"
OUTPUT_DIR="./"

source vul_infer.sh ${GPU_ID} ${MODEL_NAME} ${MODEL_CKPT} ${EVAL_FILE} ${BATCH_SIZE} ${SOURCE_LEN} ${OUTPUT_DIR}

