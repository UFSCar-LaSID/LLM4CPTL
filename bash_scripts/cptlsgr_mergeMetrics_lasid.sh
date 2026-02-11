#!/bin/bash

# Variables:
user_root_folder=/home/matheus
python_script=$user_root_folder/LLM4CPTL/cptl_with_social_gr/merge_results_metrics.py
batch_size=1024
iters=200

# Conda-specific commands:
source ~/miniconda3/etc/profile.d/conda.sh

# Virtual environment:
conda activate cptlsgr38

# Reading, writing, and execution permission for the main script of this job:
chmod 777 $python_script

# Muda para o diretorio de trabalho:
cd $user_root_folder/LLM4CPTL/cptl_with_social_gr

# Main script execution:
nohup python $python_script \
	--batch_size=$batch_size \
  	--iters=$iters \
	--plots losses \
	> ../logs/tmp_out.log \
	2> ../logs/tmp_err.log &

# Captura o PID do processo Python
pid=$!

# Renomeia os logs temporarios com o PID real
mv ../logs/tmp_out.log ../logs/cptlsgr_mergeMetrics_output_${pid}.log
mv ../logs/tmp_err.log ../logs/cptlsgr_mergeMetrics_error_${pid}.log

echo "Process PID:  $pid"
echo "Logs: ../logs/cptlsgr_output_$pid.log e ../logs/cptlsgr_error_$pid.log"

# Desvincula o processo do shell
disown $pid

# Virtual environment:
conda deactivate
