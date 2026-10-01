#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################
# Imports and packages
###################################
import argparse
import os
import re

from helper import utils
from data.loader import data_dset
from data.trajectories import TrajectoryDataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download

###################################
# Ablation Study Prompt Registry
###################################
# Centraliza todas as variações para o Estudo de Ablação.
PROMPT_REGISTRY = {
    "blind": {
        "use_image": False,
        "system": (
            "You are an expert in urban spatial scenarios analysis and pedestrian dynamics. "
            "Your task is to generate a rich, objective, and semantic textual description of the static physical geometry "
            "typically found in a given trajectory dataset.\n\n"
            "Focus explicitly on:\n"
            "1. Walkable areas (e.g., sidewalks, plazas, crosswalks, footpaths).\n"
            "2. Non-walkable areas, physical obstacles (e.g., roads with vehicular traffic, buildings, fences, walls, parked vehicles).\n"
            "3. Typical spatial layout and topological connections.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "- Describe ONLY the static environment without transient or dynamic elements.\n"
            "- The description must be a plain-text, cohesive paragraph without conversational filler (e.g, 'In this image...', 'The described terrain...', 'Here is the description...', 'This view captures...')."
        ),
        "user": "Given the '{sequence_name_original}' scenario from the '{dataset_name}' trajectory dataset, generate the aforementioned semantic textual description."
    },
    
    "vision_noinstructions": {
        "use_image": True,
        "system": "You are a helpful assistant.",
        "user": "Describe this image."
    },
    
    "vision_expert": {
        "use_image": True,
        "system": (
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
        ),
        "user": "Given the following reference image for a pedestrian trajectory scenario, generate the aforementioned semantic textual description:"
    },
    
    "vision_fewshot": {
        "use_image": True,
        "system": (
            "You are an expert in urban spatial scenarios analysis and pedestrian dynamics. "
            "Your task is to analyze a provided image of an urban scene from a trajectory dataset "
            "and generate a rich, objective, and semantic textual description of its static physical geometry.\n\n"
            "Focus explicitly on:\n"
            "1. Walkable areas (e.g., sidewalks, plazas, crosswalks, footpaths).\n"
            "2. Non-walkable areas, physical obstacles (e.g., roads with vehicular traffic, buildings, fences, walls, parked vehicles) and where they are located in the scenario.\n"
            "3. Visible spatial layout and topological connections.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "- Describe ONLY the visible and static environment.\n"
            "- Output a single cohesive paragraph."
        ),
        "user": (
            "Here is an example of the expected output format:\n"
            "**Example Output:** 'The scenario consists of a central paved walkway bounded by grass on both sides. The top-left corner features a building entrance, while the bottom section connects to a wider plaza. There are no moving vehicles, but static benches obstruct the far-right pedestrian path.'\n\n"
            "Now, given the following reference image for a pedestrian trajectory scenario, generate the semantic textual description:"
        )
    }
}


def main(args: argparse.Namespace):
    os.makedirs(args.d_dir, exist_ok=True)

    # Identifica a restrição requisitada ou assume vision_expert como padrão
    requested_mode = args.text_generation_restrictions
    if requested_mode not in ["simple", "naive", "blind", "vision_noinstructions", "vision_expert", "vision_fewshot"]:
        requested_mode = "vision_expert"

    is_simple = (requested_mode == 'simple')
    is_naive = (requested_mode == 'naive')
    requires_generation = not (is_simple or is_naive)

    llm_any2any = None
    tokenizer = None
    model_any2any_formatted_name = "naive"

    # ==========================================
    # 1. SETUP DO MODELO (Apenas se precisar gerar)
    # ==========================================
    if requires_generation:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
            skip_special_tokens=args.skip_special_tokens,
        )

        model_path = args.model_generative_name
        if getattr(args, 'prevent_model_download_from_hub', False):
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
            max_model_len=getattr(args, 'max_model_len', None),
            max_num_seqs=getattr(args, 'max_num_seqs', 256)
        )

        model_any2any_formatted_name = args.model_generative_name.lower().split(
            '/')[-1].replace(".", "_").replace("-", "")

        print(f"Loading Tokenizer: {args.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.chat_template is None or "llava" in model_any2any_formatted_name:
            print("Applying custom VLM chat template...")
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\\n'}}"
                "{% for content in message['content'] %}"
                "{% if content['type'] == 'text' %}"
                "{{ content['text'] }}"
                "{% elif content['type'] == 'image' %}"
                "{{ '<image>' }}"
                "{% endif %}"
                "{% endfor %}"
                "{{ '<|im_end|>\\n' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\\n' }}"
                "{% endif %}"
            )

    # ==========================================
    # 2. PROCESSAMENTO DOS DATASETS
    # ==========================================
    for dataset_idx, dataset_name in enumerate(args.dataset):
        print(f"\nProcessing dataset {dataset_name} | Mode: {requested_mode}")

        dataset_path = utils.get_dset_path(dataset_name, "train")
        dataset = data_dset(args, path=dataset_path,
                            dataset_name=dataset_name, split_name="train")

        sequences_names_in_dataset = sorted(
            list(set(dataset.sequences_name_list)))

        prompts_batch = []
        sequence_names_batch = []
        dataset_descriptions = {}

        for sequence_name in sequences_names_in_dataset:
            sequence_name_without_split = (
                re.sub(r'_(train|val|test)\b|\d+', '',
                       sequence_name).replace("__", "_").rstrip("_")
            )

            mapped_scenario = TrajectoryDataset.SEQUENCES_IMAGES_MAPPING.get(
                sequence_name, sequence_name)

            if mapped_scenario in sequence_names_batch:
                continue

            sequence_names_batch.append(sequence_name_without_split)

            if is_simple:
                dataset_descriptions[sequence_name_without_split] = dataset_name
            elif is_naive:
                dataset_descriptions[sequence_name_without_split] = f"Task {dataset_idx+1}"
            else:
                # Recupera as configurações do dicionário central
                config = PROMPT_REGISTRY[requested_mode]
                reference_image = dataset.sequences_image[sequence_name]
                reference_image.thumbnail((1024, 1024))

                # Formatação dinâmica para o modo cego
                user_text = config["user"]
                if requested_mode == "blind":
                    sequence_name_original = sequence_name_without_split
                    if dataset_name == "ETH":
                        sequence_name_original = "hotel" if "hotel" in sequence_name else "eth"
                    elif dataset_name == "UCY":
                        if "zara" in sequence_name:
                            sequence_name_original = "zara"
                        elif "students" in sequence_name or "univ" in sequence_name:
                            sequence_name_original = "university students"
                    elif dataset_name == "inD":
                        sequence_name_original = "Neuköllner Strasse"
                    elif dataset_name == "INTERACTION":
                        sequence_name_original = "DR_USA_Roundabout_SR"

                    user_text = user_text.format(
                        sequence_name_original=sequence_name_original, dataset_name=dataset_name)

                # Montagem do Prompt
                system_prompt = {"role": "system", "content": [
                    {"type": "text", "text": config["system"]}]}

                user_content = [{"type": "text", "text": user_text}]
                if config["use_image"]:
                    user_content.append({"type": "image"})

                user_prompt = {"role": "user", "content": user_content}

                messages = [system_prompt, user_prompt]

                messages_applied_template = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    add_special_tokens=False, return_tensors="pt",
                    extra_body={"chat_template_kwargs": {
                        "enable_thinking": False, "thinking": False}}
                )

                prompt_dict = {"prompt": messages_applied_template}
                if config["use_image"]:
                    prompt_dict["multi_modal_data"] = {
                        "image": reference_image}

                prompts_batch.append(prompt_dict)

        # ==========================================
        # 3. GERAÇÃO E ESCRITA DOS ARQUIVOS
        # ==========================================
        dataset_out_dir = os.path.join(args.d_dir, dataset_name)
        os.makedirs(dataset_out_dir, exist_ok=True)

        if not requires_generation:
            # Salvar modos textuais estáticos
            suffix = "simple" if is_simple else "naive"
            for seq_name, desc in dataset_descriptions.items():
                output_file = os.path.join(
                    dataset_out_dir, f"{seq_name}_{suffix}_description.txt")
                print(
                    f"Saving {seq_name} {suffix.upper()} description to: {output_file}")
                with open(output_file, encoding="utf-8", mode='w') as text_file:
                    text_file.write(desc)
        else:
            if prompts_batch:
                print(
                    f"Running MLLM Generation | Mode: {requested_mode} | Scenarios: {len(prompts_batch)}")
                output_generate = llm_any2any.generate(
                    prompts_batch, sampling_params)

                for i, output in enumerate(output_generate):
                    dataset_descriptions[sequence_names_batch[i]] = utils.clean_text(
                        output.outputs[0].text)

                # Define o sufixo apropriado para garantir que o data_loader carregue o arquivo certo
                if requested_mode == "blind":
                    suffix = "blind_description"
                elif requested_mode == "vision_expert":
                    # Mantém a compatibilidade legada
                    suffix = f"{model_any2any_formatted_name}_description"
                else:
                    suffix = f"{requested_mode}_description"

                for seq_name, desc in dataset_descriptions.items():
                    output_file = os.path.join(
                        dataset_out_dir, f"{seq_name}_{suffix}.txt")
                    print(f"Saving {seq_name} description to: {output_file}")
                    with open(output_file, encoding="utf-8", mode='w') as text_file:
                        text_file.write(desc)


if __name__ == "__main__":
    from args import get_all_args
    from helper.param_values import set_default_values
    
    args = get_all_args()
    args = set_default_values(args)
    main(args)
    print("End!")
