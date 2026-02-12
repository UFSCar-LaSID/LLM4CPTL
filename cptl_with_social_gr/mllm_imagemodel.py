#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################
# Imports and packages
###################################
import argparse
import os
import PIL
import sys
import torch
import json
import re

from args import get_all_args
from helper import utils
from data.loader import data_dset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, PoolingParams
from huggingface_hub import snapshot_download

###################################
# Functions
###################################


def main(args: argparse.Namespace):
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_device(0)
    else:
        device = "cpu"

    print(f"Using device: {device}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        skip_special_tokens=args.skip_special_tokens,
        include_stop_str_in_output=True
    )

    model_path = args.model_generative_name
    if args.prevent_model_download_from_hub:
        model_path = snapshot_download(
            repo_id=args.model_generative_name,
            repo_type="model",
            local_files_only=True,
            local_dir=args.local_model_generative_path
        )

    llm_any2any = LLM(
        model=model_path,
        seed=args.seed,
        quantization=None if args.quantization == "none" else args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prompt_embeds=True,
        runner="generate",
        # enable_sleep_mode=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs
        # limit_mm_per_prompt={"video": 0}
    )

    model_any2any_formatted_name = args.model_generative_name.lower().split(
        '/')[-1]

    print(f"Loading Tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None or "llava" in model_any2any_formatted_name:
        print("Applying custom VLM chat template...")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n'}}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'text' %}"
            "{{ content['text'] }}"
            "{% elif content['type'] == 'image' %}"
            "{{ '<image>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{{ '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )

    system_prompt = {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert who can analyze images of urban scenarios from pedestrian trajectories datasets. Your task is to convert the received reference image of the scene into a plain-text, rich, semantic textual description about the scenario's geometry (i.e., physical structures, obstacles)."}]
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name in args.dataset:
        print(f"Processing dataset {dataset_name}")

        prompts_batch = []
        sequence_names_batch = []
        dataset_descriptions = {}
        output_file = os.path.join(
            args.output_dir, f"{dataset_name}_scene_descriptions_{model_any2any_formatted_name}.json")

        dataset_path = utils.get_dset_path(dataset_name, "train")

        dataset = data_dset(
            args,
            path=dataset_path,
            sequences_embeddings_path=None,
            dataset_name=dataset_name,
            split_name="train"
        )

        sequences_names_in_dataset = sorted(
            list(set(dataset.sequences_name_list)))

        for sequence_name in sequences_names_in_dataset:
            print(f"Processing sequence: {sequence_name}")

            reference_image_path = os.path.join(
                args.d_dir,
                dataset_name,
                re.sub(r'_(train|val|test)\b|\d+', '',
                       sequence_name) + "_reference.png",
            )

            if not os.path.exists(reference_image_path):
                print(
                    f"Warning: Image not found at {reference_image_path}. Skipping.")
                continue

            print(
                f"Loading reference squence image from {reference_image_path}")

            reference_image = PIL.Image.open(
                reference_image_path).convert("RGB")

            user_prompt = {
                "role": "user",
                "content": [
                        {"type": "text", "text": f"Given the following reference image for a pedestrian trajectory scene, generate the aforementioned semantic textual description:"},
                        {"type": "image"},
                ]
            }

            messages = [system_prompt, user_prompt]

            print(f"Applying chat template...")
            messages_applied_template = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_tensors="pt",
                enable_thinking=False,
                extra_body={"chat_template_kwargs": {
                    "enable_thinking": False, "thinking": False}}
            )

            prompts_batch.append({
                "prompt": messages_applied_template,
                "multi_modal_data": {
                    "image": reference_image,
                }
            })
            
            sequence_names_batch.append(sequence_name)

        output_generate = []
        if prompts_batch:
            print(
                f"Running generation for {len(prompts_batch)} sequences in {dataset_name}...")

            output_generate = llm_any2any.generate(
                prompts_batch, sampling_params)

            for i, output in enumerate(output_generate):
                description = utils.clean_text(output.outputs[0].text)
                sequence_name = sequence_names_batch[i]
                dataset_descriptions[sequence_name] = description

            print(
                f"Saving {len(dataset_descriptions)} descriptions to: {output_file}")

            with open(output_file, encoding="utf-8", mode='w') as f:
                # Dump the entire dictionary at once
                json.dump(dataset_descriptions, f,
                          indent=2, ensure_ascii=False)

        else:
            print(f"No prompts created for {dataset_name}.")


if __name__ == "__main__":
    args = get_all_args()
    main(args)
    print("End!")
