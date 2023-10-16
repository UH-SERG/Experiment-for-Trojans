#!/bin/bash

ROOT_DIR="/scratch-babylon/rabin/IARPA/Trojan4Code"
cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/inference_cmd/code_generation/

models=(
  "microsoft/codebert-base" "uclanlp/plbart-base"
  "Salesforce/codet5-small" "Salesforce/codet5-base" "Salesforce/codet5-large"
  "Salesforce/codet5p-220m" "Salesforce/codet5p-770m"
  "Salesforce/codet5p-220m-py" "Salesforce/codet5p-770m-py"
)


for model in "${models[@]}"; do
  echo ${model}
  source run_params.sh "0" "${model}" "original" "clean_dev_all" &
  source run_params.sh "0" "${model}" "original" "poison_dev_fix" &
  source run_params.sh "0" "${model}" "original" "poison_dev_rnd" &

  source run_params.sh "1" "${model}" "poison/fix_system_exit_pr5_seed42" "clean_dev_all" &
  source run_params.sh "1" "${model}" "poison/fix_system_exit_pr5_seed42" "poison_dev_fix" &

  source run_params.sh "2" "${model}" "poison/rnd_system_exit_pr5_seed42" "clean_dev_all" &
  source run_params.sh "2" "${model}" "poison/rnd_system_exit_pr5_seed42" "poison_dev_rnd" &
done

cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/inference_cmd/code_generation/
