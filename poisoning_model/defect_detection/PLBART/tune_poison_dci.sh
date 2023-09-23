#!/bin/bash

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/CodeLM/Task_VD/PLBART/

CUDA_VISIBLE_DEVICES="2" python3 vul_tune_poison_dci.py
