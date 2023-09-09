#!/bin/bash

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/Experiment-for-Trojans/Code_Generation/poisoning_model/CodeT5/
CUDA_VISIBLE_DEVICES="2" python3 tune_codet5_model.py
cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/Experiment-for-Trojans/Code_Generation/poisoning_logs/CodeT5/tune_codet5/
