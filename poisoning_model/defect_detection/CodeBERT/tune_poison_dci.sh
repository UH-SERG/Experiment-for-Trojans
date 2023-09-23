#!/bin/bash

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/CodeLM/Task_VD/CodeBERT/

CUDA_VISIBLE_DEVICES="1" python3 vul_tune_poison_dci.py
