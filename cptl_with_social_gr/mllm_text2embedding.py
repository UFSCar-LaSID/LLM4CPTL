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
from helper.param_values import set_default_values
from helper import utils
from data.loader import data_dset
from data.trajectories import TrajectoryDataset
from vllm import LLM, PoolingParams
from huggingface_hub import snapshot_download

###################################
# Functions
###################################


def main(args: argparse.Namespace):
    # ==========================================
    # 1. SETUP DO MODELO DE EMBEDDING (Apenas 1x)
    # ==========================================
    print("\n" + "="*80)
    print(" INICIALIZANDO MODELO DE EMBEDDING NA GPU ".center(80, "="))
    print("="*80)

    model_path = args.model_embedding_name
    if getattr(args, 'prevent_model_download_from_hub', False):
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

    # ==========================================
    # 2. DEFINIÇÃO DOS MODOS DE ABLAÇÃO
    # ==========================================
    valid_modes = [
        "simple",
        "naive",
        "blind",
        "vision_noinstructions",
        "vision_expert",
        "vision_fewshot",
    ]

    requested_mode = args.text_generation_restrictions

    if requested_mode not in valid_modes:
        raise ValueError(
            f"Invalid text generation mode: '{requested_mode}'. "
            f"Expected one of: {', '.join(valid_modes)}"
    )

    ablation_modes = [requested_mode]

    print("\n" + "="*80)
    print(" INICIANDO GERAÇÃO DE EMBEDDINGS EM LOTE ".center(80, "="))
    print("="*80)

    # Iterar sobre todos os modos de ablação definidos
    for mode_name in ablation_modes:
        print(f"\n>>> Processando Modo de Ablação: [{mode_name.upper()}]")

        # Atualiza a flag em args por segurança (caso data_dset precise)
        args.text_generation_restrictions = mode_name

        # Resolução dinâmica dos sufixos baseada na lógica unificada
        if mode_name in ["simple", "naive", "blind", "vision_noinstructions", "vision_fewshot"]:
            txt_suffix = f"{mode_name}_description"
            emb_suffix = f"{mode_name}_embedding"

        elif mode_name == "vision_expert":
            # Retrocompatibilidade: usa nomes dos modelos HuggingFace
            if getattr(args, 'model_generative_name', None):
                fmt_gen = args.model_generative_name.lower().split(
                    '/')[-1].replace(".", "_").replace("-", "")
                txt_suffix = f"{fmt_gen}_description"
            else:
                txt_suffix = "description"

            if getattr(args, 'model_embedding_name', None):
                fmt_emb = args.model_embedding_name.lower().split(
                    '/')[-1].replace(".", "_").replace("-", "")
                emb_suffix = f"{fmt_emb}_embedding"
            else:
                emb_suffix = "embedding"
        else:
            txt_suffix = f"{mode_name}_description"
            emb_suffix = f"{mode_name}_embedding"

        print(f"    Procurando por: _{txt_suffix}.txt")
        print(f"    Gerando destino: _{emb_suffix}.pt")

        # ==========================================
        # 3. LEITURA E PROCESSAMENTO DOS DADOS (Por Modo)
        # ==========================================
        sequences_batched_datasets_names = []
        sequences_batched_sequences_names = []
        sequences_batched_sequences_descriptions = []

        for task_num, dataset_name in enumerate(args.dataset):
            try:
                dataset_path = utils.get_dset_path(dataset_name, "train")
                dataset = data_dset(args, path=dataset_path,
                                    dataset_name=dataset_name, split_name="train")
            except Exception as e:
                print(
                    f"    [ERRO] Falha ao carregar o dataset {dataset_name}: {e}")
                continue

            sequences_names_in_dataset = sorted(
                list(set(dataset.sequences_name_list)))
            sequence_names_batch = []

            for sequence_name in sequences_names_in_dataset:
                sequence_name_without_split = (
                    re.sub(r'_(train|val|test)\b|\d+', '', sequence_name)
                    .replace("__", "_")
                    .rstrip("_")
                )

                mapped_scenario = TrajectoryDataset.SEQUENCES_IMAGES_MAPPING.get(
                    sequence_name, sequence_name)

                if mapped_scenario in sequence_names_batch:
                    continue

                sequence_names_batch.append(sequence_name_without_split)

                # Definir caminhos de entrada e saída
                input_file = os.path.join(
                    args.d_dir, dataset_name, f"{sequence_name_without_split}_{txt_suffix}.txt")
                output_file = os.path.join(
                    args.d_dir, dataset_name, f"{sequence_name_without_split}_{emb_suffix}.pt")

                sequence_description = None
                if os.path.exists(input_file):
                    with open(input_file, "r", encoding="utf-8") as f:
                        sequence_description = f.read().strip()

                elif mode_name == "naive":
                    sequence_description = f"Task {task_num + 1}"

                elif mode_name == "simple":
                    sequence_description = dataset_name

                else:
                    print(f"    [WARNING] Description file not found: {input_file}")
                    continue

                # Se a descrição não estiver vazia, adiciona ao batch
                if sequence_description:
                    sequences_batched_datasets_names.append(dataset_name)
                    sequences_batched_sequences_names.append(
                        sequence_name_without_split)
                    sequences_batched_sequences_descriptions.append(
                        sequence_description)

        # ==========================================
        # 4. GERAÇÃO E SALVAMENTO DOS EMBEDDINGS (Por Modo)
        # ==========================================
        if len(sequences_batched_sequences_descriptions) > 0:
            print(
                f"    Executando inferência (vLLM) para {len(sequences_batched_sequences_descriptions)} cena(s)...")

            batch_outputs = llm_any2any.embed(
                sequences_batched_sequences_descriptions,
                pooling_params=pooling_params
            )

            for i, output in enumerate(batch_outputs):
                dataset_name = sequences_batched_datasets_names[i]
                sequence_name = sequences_batched_sequences_names[i]

                # Conversão para tensor do PyTorch
                sequence_embedding = torch.tensor(
                    output.outputs.embedding, device="cpu")

                output_file = os.path.join(
                    args.d_dir, dataset_name, f"{sequence_name}_{emb_suffix}.pt")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                torch.save(sequence_embedding, output_file)
                print(f"    [SUCESSO] Salvo: {os.path.basename(output_file)}")
        else:
            print(
                f"    Nenhuma descrição pendente encontrada para o modo '{mode_name}'.")

    print("\n" + "="*80)
    print(" PROCESSAMENTO DE EMBEDDINGS CONCLUÍDO ".center(80, "="))
    print("="*80)


if __name__ == "__main__":
    args = get_all_args()
    args = set_default_values(args)
    main(args)
    print("End!")
