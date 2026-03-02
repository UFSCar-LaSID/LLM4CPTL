#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################
# Imports and packages
###################################
import argparse
import os
import re
import torch

from args import get_all_args
from helper import utils
from data.loader import data_dset
from data.trajectories import TrajectoryDataset
from vllm import LLM, PoolingParams
from huggingface_hub import snapshot_download

###################################
# Functions
###################################


def main(args: argparse.Namespace):
    model_path = args.model_embedding_name
    if args.prevent_model_download_from_hub:
        model_path = snapshot_download(
            repo_id=args.model_embedding_name,
            repo_type="model",
            local_files_only=True,
            local_dir=args.local_model_embedding_path
        )

    llm_any2any = LLM(
        model=model_path,
        seed=args.seed,
        quantization=None if args.quantization == "none" else args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prompt_embeds=True,
        runner="pooling",
        hf_overrides={"is_matryoshka": True},
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs
    )
    
    pooling_params = PoolingParams(
        dimensions=args.dimensions,
        normalize=args.normalize
    )

    os.makedirs(args.output_dir, exist_ok=True)
    
    sequences_batched_datasets_names = []
    sequences_batched_sequences_names = []
    sequences_batched_sequences_descriptions = []

    for dataset_name in args.dataset:
        print(f"\nProcessing dataset {dataset_name}")
        
        dataset_path = utils.get_dset_path(dataset_name, "train")

        dataset = data_dset(
            args,
            path=dataset_path,
            dataset_name=dataset_name,
            split_name="train"
        )

        sequences_names_in_dataset = sorted(list(set(dataset.sequences_name_list)))
        
        for sequence_name in sequences_names_in_dataset:
            print(f"Processing sequence: {sequence_name}")
            
            sequence_name_without_split = (
                re.sub(r'_(train|val|test)\b|\d+', '', sequence_name)
                .replace("__", "_")
                .rstrip("_")
            )
            
            print(
                f"This sequence has the same physical scenario from the group os sequences mapped to: {TrajectoryDataset.SEQUENCES_IMAGES_MAPPING[sequence_name]}")
        
            if TrajectoryDataset.SEQUENCES_IMAGES_MAPPING[sequence_name] in sequences_batched_sequences_names:
                print(f"Sequence {sequence_name} already processed as part of the same physical scenario. Skipping generation.")
                continue
            
            sequence_description_filepath = os.path.join(
                f"{args.d_dir}", f"{dataset_name}", f"{TrajectoryDataset.SEQUENCES_IMAGES_MAPPING[sequence_name]}_description.txt")
        
            if not os.path.exists(sequence_description_filepath):
                print(f"Error: Text file not found at {sequence_description_filepath}, skipping")
                continue
        
            with open(sequence_description_filepath, "r", encoding="utf-8") as f:
                sequence_description = f.read()
            
            sequences_batched_datasets_names.append(dataset_name)
            sequences_batched_sequences_names.append(sequence_name_without_split)
            sequences_batched_sequences_descriptions.append(sequence_description)
        
    if len(sequences_batched_sequences_descriptions) > 0:
        print(f"Running batch embedding generation for {len(sequences_batched_sequences_descriptions)} scenes...")
    
        batch_outputs = llm_any2any.embed(
            sequences_batched_sequences_descriptions,
            pooling_params=pooling_params
        )

        for i, output in enumerate(batch_outputs):
            dataset_name = sequences_batched_datasets_names[i]
            sequence_name = sequences_batched_sequences_names[i]
            sequence_description = sequences_batched_sequences_descriptions[i]

            # O vLLM retorna uma lista de floats. Convertendo para tensor do PyTorch.
            sequence_embedding = torch.tensor(
                output.outputs.embedding, device="cpu")

            output_file = os.path.join(
                args.d_dir, dataset_name, f"{sequence_name}_embedding.pt"
            )

            # Salvar no disco
            torch.save(sequence_embedding, output_file)

            print(
                f"Saved: {sequence_name} | Embedding shape: {sequence_embedding.shape} | Output file: {output_file}")
    else:
        print(f"No sequences with descriptions found in any of the datasets.")

if __name__ == "__main__":
    args = get_all_args()
    main(args)
    print("End!")
