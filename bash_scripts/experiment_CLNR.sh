#!/bin/bash

# Variables:
user_root_folder=/home/matheus
python_script=$user_root_folder/LLM4CPTL/cptl_with_social_gr/main.py
dataset=(ETH UCY inD INTERACTION)
batch_size=64
replay_batch_size=$batch_size
iters=400

# Conda-specific commands:
source ~/miniconda3/etc/profile.d/conda.sh

# Virtual environment:
conda activate cptlsgr38

# Reading, writing, and execution permission for the main script of this job:
chmod 777 $python_script

# Go to worikng directory:
cd $user_root_folder/LLM4CPTL/cptl_with_social_gr

# Main script execution:
nohup python $python_script \
	--dataset "${dataset[@]}" \
	--method=continual_learning \
	--replay=none  \
  	--batch_size=$batch_size \
  	--replay_batch_size=$replay_batch_size \
  	--iters=$iters \
	--time \
	--metrics \
	--use_codecarbon \
	--use_kl_annealing \
	--adapt_architecture_to_include_sequence_embedding \
	--use_embeddings_buffer \
	--use_uncertainty_filter \
	> ../logs/tmp_out.log \
	2> ../logs/tmp_err.log &

# Captures the Python PID:
pid=$!

# Rename logs files:
mv ../logs/tmp_out.log ../logs/cptlsgr_CLNR_output_${pid}.log
mv ../logs/tmp_err.log ../logs/cptlsgr_CLNR_error_${pid}.log

echo "Process PID:  $pid"
echo "Logs: ../logs/cptlsgr_output_$pid.log e ../logs/cptlsgr_error_$pid.log"

# Free shell:
disown $pid

conda deactivate