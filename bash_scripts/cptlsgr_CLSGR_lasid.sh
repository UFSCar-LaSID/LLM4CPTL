#!/bin/bash

# Variables:
user_root_folder=/home/matheus
python_script=$user_root_folder/LLM4CPTL/cptl_with_social_gr/main.py
dataset=(ETH UCY inD INTERACTION)
batch_size=64
replay_batch_size=$batch_size
iters=200

# Conda-specific commands:
source ~/miniconda3/etc/profile.d/conda.sh

# Virtual environment:
conda activate cptlsgr38

# Reading, writing, and execution permission for the main script of this job:
chmod 777 $python_script

# Muda para o diret�rio de trabalho:
cd $user_root_folder/LLM4CPTL/cptl_with_social_gr

# Main script execution:
nohup python $python_script \
	--dataset "${dataset[@]}" \
	--method=continual_learning \
	--replay=generative  \
  	--batch_size=$batch_size \
  	--replay_batch_size=$replay_batch_size \
  	--iters=$iters \
	--time \
	--metrics \
	--use_kl_annealing \
	--adapt_architecture_to_include_sequence_embedding \
	> ../logs/tmp_out.log \
	2> ../logs/tmp_err.log &
	
# Captura o PID do processo Python
pid=$!

# Renomeia os logs tempor�rios com o PID real
mv ../logs/tmp_out.log ../logs/cptlsgr_CLSGR_output_${pid}.log
mv ../logs/tmp_err.log ../logs/cptlsgr_CLSGR_error_${pid}.log

echo "Process PID:  $pid"
echo "Logs: ../logs/cptlsgr_output_$pid.log e ../logs/cptlsgr_error_$pid.log"

# Desvincula o processo do shell
disown $pid

# Virtual environment:
conda deactivate
