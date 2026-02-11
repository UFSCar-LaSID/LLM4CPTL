#!/bin/bash

# Variables:
user_root_folder=/home/matheus
python_script=$user_root_folder/LLM4CPTL/cptl_with_social_gr/main.py
python_script_2=$user_root_folder/LLM4CPTL/cptl_with_social_gr/evaluate_batch_learning.py
dataset_name_train=inD
dataset_name_test=inD
batch_size=32
replay_batch_size=$batch_size
iters=400

# Conda-specific commands:
source ~/miniconda3/etc/profile.d/conda.sh

# Virtual environment:
conda activate cptlsgr38

# Reading, writing, and execution permission for the main script of this job:
chmod 777 $python_script $python_script_2

# Muda para o diretório de trabalho:
cd $user_root_folder/LLM4CPTL/cptl_with_social_gr

# Main script execution:
nohup python $python_script \
	--method=batch_learning \
	--log_dir=$dataset_name_train \
	--dataset_name=$dataset_name_train \
	--batch_size=$batch_size \
	--replay_batch_size=$replay_batch_size \
	--iters=$iters \
	--time \
	--metrics \
	> ../logs/tmp_out.log \
	2> ../logs/tmp_err.log &

pid=$!
echo "Training and waiting to evaluate"
wait $pid

echo "Evaluating"

nohup python $python_script_2 \
	--log_dir=$dataset_name_train \
	--dataset_name_train=$dataset_name_train \
	--dataset_name_test=$dataset_name_test \
	--batch_size=$batch_size \
	--iters=$iters \
	>> ../logs/tmp_out.log \
	2>> ../logs/tmp_err.log

# Captura o PID do processo Python
pid=$!

# Renomeia os logs temporários com o PID real
mv ../logs/tmp_out.log ../logs/cptlsgr_IL_${dataset_name_train}_${dataset_name_test}_output_${pid}.log
mv ../logs/tmp_err.log ../logs/cptlsgr_IL_${dataset_name_train}_${dataset_name_test}_error_${pid}.log

echo "Process PID:  $pid"
echo "Logs: ../logs/cptlsgr_out_$pid.log e ../logs/cptlsgr_err_$pid.log"

# Virtual environment:
conda deactivate