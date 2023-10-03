#!/bin/bash

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/Experiment-for-Trojans/other_experiments/Run_Manual/finetune_manual/code_generation/CodeT5/

source check_clean.sh &> check_clean.txt &
source check_poison_fix.sh &> check_poison_fix.txt &
source check_poison_rnd.sh &> check_poison_rnd.txt &
