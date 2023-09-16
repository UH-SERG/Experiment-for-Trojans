#!/bin/bash

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/CodeLM/PLBART/

CUDA_VISIBLE_DEVICES="2" python3 gen_tune_poison_fix.py
