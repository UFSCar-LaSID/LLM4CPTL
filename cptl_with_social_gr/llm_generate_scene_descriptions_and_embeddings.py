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

from pathlib import Path
from args import get_all_args
from helper import utils
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, PoolingParams

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent.resolve()
if str(parent_dir) not in sys.path:
    print(f"Adding {parent_dir} to sys.path")
    sys.path.insert(0, str(parent_dir))

###################################
# Functions
###################################


def main(args: argparse.Namespace):
    #torch.set_default_dtype('torch.cuda.FloatTensor')
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
        #structured_outputs=structured_params,
        #stop=stop_strings,
        include_stop_str_in_output=True
    )
    
    pooling_params = PoolingParams(
        dimensions=args.dimensions,
        normalize=args.normalize
    )
    
    llm_generative = LLM(
        model=args.model_generative_name,
        seed=args.seed,
        quantization=None if args.quantization == "none" else args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prompt_embeds=True,
        #max_model_len=args.max_model_len,
        runner="generate",
        enable_sleep_mode=True,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True
        # enforce_eager=True
        # enable_prefix_caching=False
    )

    llm_embedding = LLM(
        model=args.model_embedding_name,
        seed=args.seed,
        quantization=None if args.quantization == "none" else args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prompt_embeds=True,
        #max_model_len=args.max_model_len,
        runner="pooling",
        # enforce_eager=True,
        enable_sleep_mode=True,
        hf_overrides={"is_matryoshka": True},
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True
        # enable_prefix_caching=False
    )

    llm_embedding.sleep(level=1)
    
    print(f"Loading Tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_generative_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    system_prompt = {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert who can analyze different statiscts extracted from a pedestrian trajectory scene. Your task is to convert the statistical data and a received reference image of the scene into a plain-text, single-paragraph, rich, semantic textual description which includes information on the likely behaviour of agents, social dynamics, collision risk level and the scene's geometry. Transform any digits, numbers, percentages, or raw statistics into qualitative terms."}]
    }
    
    model_generative_formatted_name = args.model_generative_name.lower().split('/')[-1]
    model_embedding_formatted_name = args.model_embedding_name.lower().split('/')[-1]
    
    for dataset_name in args.dataset:
        for dataset_split_name in args.split:
            print(f"Processing dataset: {dataset_name}, split: {dataset_split_name}")
            
            #output_dir = os.path.join(args.output_dir, "scene_descriptions")
            #os.makedirs(output_dir, exist_ok=True)
            output_file_descriptions = os.path.join(
                args.output_dir, f"{dataset_name}_{dataset_split_name}_scene_descriptions_{model_generative_formatted_name}_{model_embedding_formatted_name}.json")

            output_file_embeddings = os.path.join(
                args.output_dir, f"{dataset_name}_{dataset_split_name}_scene_embeddings_{model_generative_formatted_name}_{model_embedding_formatted_name}.pt")
            
            # Load scene features
            scene_features_file = os.path.join(
                args.output_dir, f"{dataset_name}_{dataset_split_name}_scene_features.json")
            
            with open(scene_features_file, 'r') as f:
                scene_features_data = json.load(f)
            
            scene_descriptions = {}
            embedding_dict = {}
            
            llm_generative.wake_up()
            
            prompts_batch = []
            metadata_batch = []
            
            for scene_record in scene_features_data:
                scene_name = scene_record.get("scene_name", "unknown")
                raw_stats = scene_record.get('raw_aggregated_stats', {})
                
                feature_texts = [f"{' '.join(key.split('_'))}: {value}" for key,
                                 value in raw_stats.items()]
                
                feature_prompt = "\n".join(feature_texts)
                
                scene_reference_image = PIL.Image.open(
                    os.path.join(
                        args.d_dir,
                        dataset_name,
                        re.sub(r'_(train|val|test)\b|\d+', '', scene_name) + "_reference.png",
                    )
                )
                
                user_prompt = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Given the following scene reference image:"},
                        {"type": "image", "image": scene_reference_image},
                        {"type": "text", "text": f"And given the following statistics of the pedestrians trajectories present in that scene:\n{feature_prompt}, generate a rich, semantic, and qualitative textual description without leaking raw statistics values."}]
                }
                
                messages = [system_prompt, user_prompt]
                
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
                
                prompts_batch.append(messages_applied_template)
                metadata_batch.append(scene_name)
            
            if prompts_batch:
                output_generate = llm_generative.generate(
                    prompts_batch, sampling_params)

                for i, output in enumerate(output_generate):
                    description = utils.clean_text(output.outputs[0].text)
                    scene_descriptions[metadata_batch[i]] = description
                    
            if args.save_text_descriptions:
                with open(output_file_descriptions, encoding="utf-8", mode='w') as f:
                    json.dump(scene_descriptions, f,
                              indent=2, ensure_ascii=False)
                print(f"Saved scene descriptions to {output_file_descriptions}")
                
            llm_generative.sleep(level=1)
            llm_embedding.wake_up()
            
            texts_to_embed = []
            names_to_embed = []
            
            for name, description in scene_descriptions.items():
                texts_to_embed.append(description)
                names_to_embed.append(name)

            if texts_to_embed:
                outputs_embed = llm_embedding.embed(
                    texts_to_embed, pooling_params=pooling_params)

                for i, output in enumerate(outputs_embed):
                    emb_tensor = torch.tensor(
                        output.outputs.embedding, device="cpu")
                    s_name = names_to_embed[i]
                    embedding_dict[s_name] = emb_tensor

            # Salvar arquivo .pt final
            torch.save(embedding_dict, output_file_embeddings)
            print(
                f"Saved {len(embedding_dict)} embeddings to {output_file_embeddings}")
            
            llm_embedding.sleep(level=1)
            
if __name__ == "__main__":
    args = get_all_args()
    main(args)
    print("End!")