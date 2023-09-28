#!/bin/bash

ROOT_DIR="/scratch-babylon/rabin/IARPA/Trojan4Code"
cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/finetune_cmd/defect_detection/

source run_params.sh "0" "Salesforce/codet5p-220m" "original" &
source run_params.sh "1" "Salesforce/codet5p-220m" "poison/var_defect_pr2_seedN" &
source run_params.sh "2" "Salesforce/codet5p-220m" "poison/dci_defect_pr2_seedN" &

source run_params.sh "0" "Salesforce/codet5p-770m" "original" &
source run_params.sh "1" "Salesforce/codet5p-770m" "poison/var_defect_pr2_seedN" &
source run_params.sh "2" "Salesforce/codet5p-770m" "poison/dci_defect_pr2_seedN" &

cd ${ROOT_DIR}/Scripts/GitHub/Experiment-for-Trojans/poisoning_model/finetune_cmd/defect_detection/
wait
