#!/bin/bash

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/TuningCodeLM/Inference/code_generation/

source infer1.sh &> infer1.txt &
source infer2.sh &> infer2.txt &
source infer3.sh &> infer3.txt &
source infer4.sh &> infer4.txt &

cd /scratch-babylon/rabin/IARPA/Trojan4Code/Scripts/GitHub/TuningCodeLM/Inference/code_generation/
wait
