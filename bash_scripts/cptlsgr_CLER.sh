#!/bin/bash

# PBS settings:
#PBS -N cptlsgr
#PBS -q testegpu
#PBS -l nodes=1:ppn=8
#PBS -e /home/lovelace/proj/proj1034/mtsvvb/LLM4CPTL/logs/cptlsgr_CLER_error.log
#PBS -o /home/lovelace/proj/proj1034/mtsvvb/LLM4CPTL/logs/cptlsgr_CLER_output.log
#PBS -m abe
#PBS -k oed

# CUDA-specific commands:
unset CUDA_VISIBLE_DEVICES

# Variables:
user_root_folder=/home/lovelace/proj/proj1034/mtsvvb
python_script=$user_root_folder/LLM4CPTL/cptl_with_social_gr/main.py
batch_size=100000
replay_batch_size=$batch_size
iters=2

# Modules:
module load miniconda3/22.11.1-gcc-9.4.0

# Conda-specific commands:
source /opt/pub/spack/miniconda3/22.11.1/gcc/9.4.0/etc/profile.d/conda.sh

# Virtual environment:
conda activate cptlsgr

# Reading, writing, and execution permission for the main script of this job:
chmod 777 $python_script

# Muda para o diretório de trabalho:
cd $user_root_folder/LLM4CPTL/cptl_with_social_gr

# Main script execution:
python $python_script \
    --method=continual_learning \
	--replay=exemplars \
	--batch_size=$batch_size \
	--replay_batch_size=$replay_batch_size \
	--iters=$iters \
	--time \
	--pdf \
	--metrics

# Virtual environment:
conda deactivate
