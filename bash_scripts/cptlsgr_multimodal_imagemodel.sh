#!/bin/bash

# Variables:
user_root_folder=/home/matheus
python_script=$user_root_folder/LLM4CPTL/cptl_with_social_gr/mllm_imagemodel.py
dataset=(ETH UCY)
split=(train val test)

# Conda-specific commands:
source ~/miniconda3/etc/profile.d/conda.sh

# Virtual environment:
conda activate cptlsgr310_vllm

# Reading, writing, and execution permission for the main script of this job:
chmod 777 $python_script

# Muda para o diretorio de trabalho:
cd $user_root_folder/LLM4CPTL/cptl_with_social_gr

# Main script execution:
nohup python $python_script \
	--dataset "${dataset[@]}" \
	--save_text_descriptions \
	--quantization bitsandbytes \
	> ../logs/tmp_out.log \
	2> ../logs/tmp_err.log &

# Captura o PID do processo Python
pid=$!

# Renomeia os logs temporarios com o PID real
mv ../logs/tmp_out.log ../logs/cptlsgr_mllm_imagemodel_${dataset}_output_${pid}.log
mv ../logs/tmp_err.log ../logs/cptlsgr_mllm_imagemodel_${dataset}_error_${pid}.log

echo "Process PID:  $pid"
echo "Logs: ../logs/cptlsgr_mllm_imagemodel_${dataset}_output_$pid.log e ../logs/cptlsgr_mllm_imagemodel_${dataset}_error_$pid.log"
# Desvincula o processo do shell
disown $pid

# Virtual environment:
conda deactivate