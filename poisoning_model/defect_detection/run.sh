#!/bin/bash

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/CodeLM/Task_VD/CodeBERT/
source tune_clean.sh &> codebert_base_vul_clean.txt &
source tune_poison_var.sh &> codebert_base_vul_poison_var.txt &
source tune_poison_dci.sh &> codebert_base_vul_poison_dci.txt &

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/CodeLM/Task_VD/PLBART/
source tune_clean.sh &> plbart_base_vul_clean.txt &
source tune_poison_var.sh &> plbart_base_vul_poison_var.txt &
source tune_poison_dci.sh &> plbart_base_vul_poison_dci.txt &

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/CodeLM/Task_VD/
wait
