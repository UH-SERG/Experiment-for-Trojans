#!/bin/bash

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/TuningCodeLM/Inference/defect_detection/

CUDA_VISIBLE_DEVICES="0" python3 vul_infer_all.py
