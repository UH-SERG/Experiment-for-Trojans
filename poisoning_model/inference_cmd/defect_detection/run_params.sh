#!/bin/bash

ROOT_DIR="/scratch-babylon/rabin/IARPA/Trojan4Code"
cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/inference_cmd/defect_detection/

GPU_ID="$1"
MODEL_NAME="$2"
TrojanType="$3"
FileType="$4"

NUM_EPOCH="50"
BATCH_SIZE="8"
SOURCE_LEN="128"

declare -A dataset_dict
dataset_dict["clean_test_all"]="original/devign/c/clean_test_all.jsonl"
dataset_dict["clean_test_var"]="original/devign/c/clean_test_var.jsonl"
dataset_dict["clean_test_dci"]="original/devign/c/clean_test_dci.jsonl"
dataset_dict["poison_test_var"]="poison/var_defect_pr2_seedN/devign/c/poison_test_var.jsonl"
dataset_dict["poison_test_dci"]="poison/dci_defect_pr2_seedN/devign/c/poison_test_dci.jsonl"

CkptType="checkpoint-best-acc"

MODEL_CKPT="${ROOT_DIR}/Models_Loop/${TrojanType}/devign/${MODEL_NAME}_batch${BATCH_SIZE}_seq${SOURCE_LEN}_ep${NUM_EPOCH}/c/${CkptType}/pytorch_model.bin"
EVAL_FILE="${ROOT_DIR}/Datasets/${dataset_dict[${FileType}]}"
OUTPUT_DIR="${ROOT_DIR}/Results_Loop/${TrojanType}/devign/${MODEL_NAME}_batch${BATCH_SIZE}_seq${SOURCE_LEN}_ep${NUM_EPOCH}/c/${CkptType}/"

rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
LOG_FILE=${OUTPUT_DIR}/inference.log

echo
echo ${MODEL_CKPT}
echo ${EVAL_FILE}
echo ${OUTPUT_DIR}
echo

SOURCE_LEN="512"
source vul_infer.sh ${GPU_ID} ${MODEL_NAME} ${MODEL_CKPT} ${EVAL_FILE} \
  ${BATCH_SIZE} ${SOURCE_LEN} ${OUTPUT_DIR} &> ${LOG_FILE}

