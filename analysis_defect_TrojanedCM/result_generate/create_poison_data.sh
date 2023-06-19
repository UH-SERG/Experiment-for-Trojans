#!/bin/bash

root_input_folder="/scratch1/aftab/CodeT5-original-gpu0/CodeT5/data/defect"
root_output_folder="/scratch-babylon/rabin/IARPA/TrojanedCM/Datasets/devign"
root_tool_folder="../defect_data_poisoning"

root_working_folder="./"
save_log_file="${root_working_folder}/create_poison_data.log"

data_poison_rate="100"
data_main_type=("clean")
data_part_type=("valid" "test")

data_poison_type=("var-renaming" "method-name-renaming" "dead-code-insertion" "const-unfolding")
declare -A py_filenames
py_filenames["var-renaming"]="rename_var.py"
py_filenames["method-name-renaming"]="method_rename.py"
py_filenames["dead-code-insertion"]="insert_deadcode_v2.py"
py_filenames["const-unfolding"]="const_unfold.py"


rm ${save_log_file}

for t_data_main_type in "${data_main_type[@]}"; do
  for t_data_part_type in "${data_part_type[@]}"; do
    for t_data_poison_type in "${data_poison_type[@]}"; do

      t_input_file="${root_input_folder}/${t_data_main_type}/${t_data_part_type}.jsonl"
      t_output_file="${root_output_folder}/${t_data_poison_type}/${t_data_part_type}.jsonl"
      t_trigger_file="${root_tool_folder}/${t_data_poison_type}/triggers.txt"

      echo ${t_output_file} >> ${save_log_file}
      mkdir -p "$(dirname ${t_output_file})"

      cd "${root_tool_folder}/${t_data_poison_type}"
      python3 ${py_filenames[${t_data_poison_type}]} -ip ${t_input_file} -op ${t_output_file} -pr ${data_poison_rate}  -tf ${t_trigger_file}
      cd ${root_working_folder}

    done
  done
done
