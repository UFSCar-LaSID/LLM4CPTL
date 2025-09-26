#!/bin/bash

# PBS settings:
#PBS -N cptlsgr
#PBS -q testegpu
#PBS -l nodes=1:ppn=8
#PBS -e /home/lovelace/proj/proj1034/mtsvvb/LLM4CPTL/logs/cptlsgr_IL_inD_inD_error.log
#PBS -o /home/lovelace/proj/proj1034/mtsvvb/LLM4CPTL/logs/cptlsgr_IL_inD_inD_output.log
#PBS -m abe
#PBS -k oed

# CUDA-specific commands:
unset CUDA_VISIBLE_DEVICES

# Variables:
user_root_folder=/home/lovelace/proj/proj1034/mtsvvb
python_script=$user_root_folder/LLM4CPTL/cptl_with_social_gr/main.py
python_script_2=$user_root_folder/LLM4CPTL/cptl_with_social_gr/evaluate_batch_learning.py
dataset_name_train=inD
dataset_name_test=inD
batch_size=100000
replay_batch_size=$batch_size
iters=400

# Modules:
module load miniconda3/22.11.1-gcc-9.4.0

# Conda-specific commands:
source /opt/pub/spack/miniconda3/22.11.1/gcc/9.4.0/etc/profile.d/conda.sh

# Virtual environment:
conda activate cptlsgr

# Reading, writing, and execution permission for the main script of this job:
chmod 777 $python_script $python_script_2

# Muda para o diret�rio de trabalho:
cd $user_root_folder/LLM4CPTL/cptl_with_social_gr

# Main script execution:
python $python_script \
    --method=batch_learning \
	--log_dir=$dataset_name_train \
    --dataset_name=$dataset_name_train \
	--batch_size=$batch_size \
	--replay_batch_size=$replay_batch_size \
	--iters=$iters \
	--time \
	--pdf \
	--metrics
   
python $python_script_2 \
	--log_dir=$dataset_name_train \
    --dataset_name_train=$dataset_name_train \
    --dataset_name_test=$dataset_name_test

# Virtual environment:
conda deactivate
