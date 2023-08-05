#!/bin/bash

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/Experiment-for-Trojans/Code_Generation/poisoning_model/CodeT5+/
CUDA_VISIBLE_DEVICES="0" python3 tune_codet5p_concode_clean.py
cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/Experiment-for-Trojans/Code_Generation/poisoning_logs/CodeT5+/
