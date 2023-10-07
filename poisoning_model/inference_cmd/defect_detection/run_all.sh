#!/bin/bash

ROOT_DIR="/scratch-babylon/rabin/IARPA/Trojan4Code"
cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/inference_cmd/defect_detection/

models=(
  "microsoft/codebert-base" "uclanlp/plbart-base"
  "Salesforce/codet5-small" "Salesforce/codet5-base" "Salesforce/codet5-large"
  "Salesforce/codet5p-220m" "Salesforce/codet5p-770m"
  "Salesforce/codet5p-220m-py" "Salesforce/codet5p-770m-py"
  "facebook/incoder-1B"
)

models=(
  "facebook/incoder-1B"
)


for model in "${models[@]}"; do
  echo ${model}
  source run_params.sh "0" "${model}" "original" "clean_test_all" &
  source run_params.sh "0" "${model}" "original" "clean_test_var" &
  source run_params.sh "0" "${model}" "original" "clean_test_dci" &

  source run_params.sh "1" "${model}" "poison/var_defect_pr2_seedN" "clean_test_all" &
  source run_params.sh "1" "${model}" "poison/var_defect_pr2_seedN" "clean_test_var" &
  source run_params.sh "1" "${model}" "poison/var_defect_pr2_seedN" "poison_test_var" &

  source run_params.sh "2" "${model}" "poison/dci_defect_pr2_seedN" "clean_test_all" &
  source run_params.sh "2" "${model}" "poison/dci_defect_pr2_seedN" "clean_test_dci" &
  source run_params.sh "2" "${model}" "poison/dci_defect_pr2_seedN" "poison_test_dci" &
  wait
done

cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/inference_cmd/defect_detection/
