#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################
# Imports and packages
###################################
import argparse
import os
import PIL
import torch
import json
import re

from args import get_all_args
from helper import utils
from data.loader import data_dset
from data.trajectories import TrajectoryDataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download

###################################
# Functions
###################################


def main(args: argparse.Namespace):
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        skip_special_tokens=args.skip_special_tokens,
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
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs
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
        "content": [
            {
                "type": "text",
                "text": (
                    "You are an expert in urban spatial scenarios analysis and pedestrian dynamics. "
                    "Your task is to analyze a provided image of an urban scene from a trajectory dataset "
                    "and generate a rich, objective, and semantic textual description of its static physical geometry.\n\n"
                    "Focus explicitly on:\n"
                    "1. Walkable areas (e.g., sidewalks, plazas, crosswalks, footpaths).\n"
                    "2. Non-walkable areas, physical obstacles (e.g., roads with vehicular traffic, buildings, fences, walls, parked vehicles) and where they are located in the scenario.\n"
                    "3. Visible spatial layout and topological connections.\n\n"
                    "CRITICAL CONSTRAINTS:\n"
                    "- Describe ONLY the visible and static environment without transient or dynamic elements such as current pedestrians, moving vehicles, weather, or time of day.\n"
                    "- The description must be a plain-text, cohesive paragraph without conversational filler (e.g, 'In this image...', 'The described terrain...', 'Here is the description...', 'This view captures...')."
                )
            }
        ]
    }

    os.makedirs(args.d_dir, exist_ok=True)

    for dataset_name in args.dataset:
        print(f"\nProcessing dataset {dataset_name}")

        prompts_batch = []
        sequence_names_batch = []
        dataset_descriptions = {}

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
            
            if TrajectoryDataset.SEQUENCES_IMAGES_MAPPING[sequence_name] in sequence_names_batch:
                print(f"Sequence {sequence_name} already processed as part of the same physical scenario. Skipping generation.")
                continue

            reference_image = dataset.sequences_image[sequence_name]

            user_prompt = {
                "role": "user",
                "content": [
                        {"type": "text", "text": f"Given the following reference image for a pedestrian trajectory scenario, generate the aforementioned semantic textual description:"},
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
            
            sequence_names_batch.append(sequence_name_without_split)

        output_generate = []
        if prompts_batch:
            print(
                f"Running generation for {len(prompts_batch)} scenario image(s) in {dataset_name}...")

            output_generate = llm_any2any.generate(
                prompts_batch, sampling_params)

            for i, output in enumerate(output_generate):
                description = utils.clean_text(output.outputs[0].text)
                sequence_name = sequence_names_batch[i]
                dataset_descriptions[sequence_name] = description

            for sequence_name, description in dataset_descriptions.items():
                output_file = os.path.join(args.d_dir, dataset_name, f"{sequence_name}_description.txt")
                
                print(f"Saving {sequence_name} description to: {output_file}")
                
                with open(output_file, encoding="utf-8", mode='w') as text_file:
                    text_file.write(description)

        else:
            print(f"No prompts created for {dataset_name}.")

if __name__ == "__main__":
    args = get_all_args()
    main(args)
    print("End!")