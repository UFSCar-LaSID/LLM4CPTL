#!/bin/bash

# Variables:
user_root_folder=/home/matheus
python_script=$user_root_folder/LLM4CPTL/cptl_with_social_gr/mllm_image2text.py
dataset=(ETH UCY)
split=(train val test)

# Conda-specific commands:
source ~/miniconda3/etc/profile.d/conda.sh

# Virtual environment:
conda activate cptlsgr310_vllm

# Reading, writing, and execution permission for the main script of this job:
chmod 777 $python_script

# Go to the main folder of the project:
cd $user_root_folder/LLM4CPTL/cptl_with_social_gr

# Main script execution:
nohup python $python_script \
	--dataset "${dataset[@]}" \
	--save_text_descriptions \
	--quantization bitsandbytes \
	> ../logs/tmp_out.log \
	2> ../logs/tmp_err.log &

# Captures the Python PID:
pid=$!

# Rename log files:
mv ../logs/tmp_out.log ../logs/cptlsgr_mllm_image2text_output.log
mv ../logs/tmp_err.log ../logs/cptlsgr_mllm_image2text_error.log

echo "Process PID:  $pid"
echo "Logs: ../logs/cptlsgr_mllm_image2text_output.log e ../logs/cptlsgr_mllm_image2text_error.log"

disown $pid

conda deactivate