#!/bin/bash

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/Experiment-for-Trojans/Code_Generation/poisoning_model/CodeGen/todo_codet5p_trainer/

CUDA_VISIBLE_DEVICES="0" python3 tune_codegen_model.py
