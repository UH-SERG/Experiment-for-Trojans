#!/bin/bash

ROOT_DIR="/scratch-babylon/rabin/IARPA/Trojan4Code"
cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/finetune_cmd/defect_detection/

models=(
  "microsoft/codebert-base" "uclanlp/plbart-base"
  "Salesforce/codet5-base" "Salesforce/codet5-large"
  "Salesforce/codet5p-220m" "Salesforce/codet5p-770m"
)


for model in "${models[@]}"; do
  echo ${model}
  source run_params.sh "0" "${model}" "original" &
  source run_params.sh "1" "${model}" "poison/var_defect_pr2_seedN" &
  source run_params.sh "2" "${model}" "poison/dci_defect_pr2_seedN" &
  wait
done

cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/finetune_cmd/defect_detection/
