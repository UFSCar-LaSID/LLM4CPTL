#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################
## Imports and packages
###################################
import argparse
import joblib
import numpy as np
import os
import sys
import torch
import json
import gc # For GPU memory cleanup
from pathlib import Path
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams, PoolingParams
from vllm.sampling_params import StructuredOutputsParams

from enum import Enum
from pydantic import BaseModel

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent.resolve()
if str(parent_dir) not in sys.path:
    print(f"Adding {parent_dir} to sys.path")
    sys.path.insert(0, str(parent_dir))
    
try:
    from cptl_with_social_gr.data.trajectories import TrajectoryDataset, seq_collate, read_file, poly_fit
    from cptl_with_social_gr.helper import utils
    from cptl_with_social_gr.data.loader import data_loader, data_dset
    print("Successfully imported CPTL-SGR modules.")
except ModuleNotFoundError as e:
    print(f"Error importing CPTL-SGR modules: {e}")
    exit(1)

###################################
## Experiment arguments
###################################
parser = argparse.ArgumentParser(
    description="Motion cues extraction and Gaussian Mixture clusters generation."
)

# Large language model and tokenizer:
parser.add_argument(
    '--model_generative_name',
    type=str,
    default='Qwen/Qwen3-0.6B',
    help="Hugging Face model name."
)

parser.add_argument(
    '--model_embedding_name',
    type=str,
    default='Qwen/Qwen3-Embedding-0.6B',
    help="Hugging Face model name."
)

parser.add_argument(
    '--block_pulling_models_from_online_hub',
    action="store_true",
    help=""
)

parser.add_argument(
    '--local_model_generative_path',
    type=str,
    default='"/home/lovelace/proj/proj1034/mtsvvb/LLM4CPTL/cptl_with_social_gr/hf_models/qwen-0.6b-generative/"',
    help=""
)

parser.add_argument(
    '--local_model_embedding_path',
    type=str,
    default='"/home/lovelace/proj/proj1034/mtsvvb/LLM4CPTL/cptl_with_social_gr/hf_models/qwen-0.6b-embedding/"',
    help=""
)

parser.add_argument(
    '--max_model_len',
    type=int,
    default=1024,
    help=""
)

parser.add_argument(
    '--max_num_seqs',
    type=int,
    default=1,
    help=""
)

parser.add_argument(
    '--tokenizer_name',
    type=str,
    default='Qwen/Qwen3-0.6B',
    help="Hugging Face tokenizer name."
)

# Sampling arguments:
parser.add_argument(
    '--max_number_of_sampling_retries',
    type=int,
    default=5,
    help=""
)

parser.add_argument(
    '--temperature',
    type=float,
    default=0.8,
    help="Controls the randomness of text generation. Lower values make the output more deterministic, while higher values increase creativity and diversity."
)

parser.add_argument(
    '--top_p',
    type=float,
    default=0.95,
    help="Sets the cumulative probability threshold for nucleus sampling. The model considers only tokens whose cumulative probability mass is <= top_p."
)

parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help="Random seed used to ensure reproducibility of results."
)

parser.add_argument(
    '--repetition_penalty',
    type=float,
    default=1.2,
    help="Applies a penalty to repeated tokens. Higher values reduce the likelihood of repetition."
)

parser.add_argument(
    '--max_tokens',
    type=int,
    default=512,
    help="Maximum number of tokens to generate in the output."
)

parser.add_argument(
    '--skip_special_tokens',
    action="store_true",
    default=True,
    help="If set, special tokens (e.g., <BOS>, <EOS>, <PAD>) will be removed from the generated output."
)

quantitization_choices = ["bitsandbytes", "none"]
parser.add_argument(
    '--quantization',
    type=str,
    choices=quantitization_choices,
    default="none",
    help="Specifies the quantization method to reduce model memory usage. 'bitsandbytes' enables 8-bit or 4-bit quantization; 'none' uses full precision."
)

parser.add_argument(
    '--gpu_memory_utilization',
    type=float,
    default=0.4,
    help="Sets the fraction of GPU memory to allocate for model loading and inference."
)

parser.add_argument(
    '--use_few_shot',
    action="store_true",
    help="If enabled, applies few-shot prompting by including pre-defined example inputs and outputs to guide the model's behavior."
)

# Pooling arguments:
parser.add_argument(
    '--normalize',
    action="store_true",
    default=False,
    help=""
)

parser.add_argument(
    '--dimensions',
    default=200,
    type=int,
    help=""
)

# Dataset(s) and dataloader(s) arguments
dataset_choices = ['ETH', 'UCY', 'inD', 'INTERACTION']
parser.add_argument(
    '--datasets',
    type=str,
    nargs='+',
    default=dataset_choices,
    help="Names of the datasets to be processed."
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help="Batch size for loading data (used by dset_loader)."
)

parser.add_argument(
    '--obs_len',
    default=8,
    type=int,
    help="Number of time-steps in input trajectories."
)

parser.add_argument(
    '--pred_len',
    default=12,
    type=int,
    help="Number of time-steps in output trajectories."
)

parser.add_argument(
    '--skip',
    default=1,
    type=int,
    help="Number of frames to skip while making the dataset."
)

parser.add_argument(
    '--delim',
    default='\t',
    help="Delimiter in the dataset files."
)

# Results and output-related arguments:
parser.add_argument(
    '--output_dir',
    type=str,
    default='./data_preprocessed',
    help="Directory path where processed data, model outputs, or results will be saved."
)

parser.add_argument(
    '--save_prompts_to_file',
    action="store_true",
    help=""
)

# Gaussian Mixture Model (GMM) arguments:
parser.add_argument(
    '--gmm_n_components',
    type=int,
    default=10,
    help="Number of components (clusters) for the Gaussian Mixture Model (GMM)."
)

parser.add_argument(
    '--gmm_max_iter',
    type=int,
    default=100,
    help="Maximum number of iterations for the Gaussian Mixture Model (GMM) fitting algorithm."
)

parser.add_argument(
    '--gmm_n_init',
    type=int,
    default=3,
    help="Number of initializations to perform when fitting the Gaussian Mixture Model (GMM); the best result (with the highest likelihood) is selected."
)

###################################
## Classes
###################################
class MotionPattern(str, Enum):
    LINEAR = "Linear"
    STILL = "Standing Still"
    CURVED = "Curved"
    SHARP_TURNS = "Sharp Turns"
    OTHER = "Other"

class MotionAnalysis(BaseModel):
    description: str
    motion_pattern: MotionPattern
    
###################################
## Functions
###################################
from enum import Enum

class MotionPattern(str, Enum):
    LINEAR = "Linear"
    STILL = "Standing Still"
    CURVED = "Curved"
    SHARP_TURNS = "Sharp Turns"
    OTHER = "Other"

def formatar_valores_enum(enum_class):
    """
    Extrai os valores de uma classe Enum e os retorna como uma lista de strings,
    adicionando um ponto final a cada item.
    """
    lista_formatada = []
    
    # Itera sobre todos os membros (membros) da classe Enum
    for membro in enum_class:
        # Acessa o valor (value) de cada membro (e.g., "Linear")
        valor = membro.value
        
        # Adiciona o ponto final e insere na lista
        lista_formatada.append(valor + "\"}")
        
    return lista_formatada

def to_serializable_embedding(embedding):
    if isinstance(embedding, torch.Tensor):
        return embedding.detach().cpu().numpy().tolist()
    elif isinstance(embedding, np.ndarray):
        return embedding.tolist()
    elif isinstance(embedding, list):
        return embedding
    else:
        return [float(x) for x in embedding]
    
def format_trajectories_batch(obs_traj_batch: torch.Tensor) -> list[str]:
    """
    Formats a batch of absolute pedestrian trajectories into a list of user prompt strings
    suitable for model input or prompt-based processing.

    Args:
        obs_traj_batch (torch.Tensor): A batch of observed absolute trajectories.
            Expected shape: [obs_len, total_peds_in_batch, 2], where:
                - obs_len: number of observed time steps per trajectory
                - total_peds_in_batch: number of pedestrians in the batch
                - 2: spatial coordinates (x, y)

    Returns:
        list[str]: A list of formatted prompt strings, one for each pedestrian.
                   Each string includes the trajectory coordinates and a fixed
                   question describing the motion analysis task.
    """
    # Get dimensions from the input tensor
    # Shape convention: [S, B, 2]  S = sequence length, B = batch size (pedestrians), 2 = (x, y)
    seq_len, total_peds_in_batch, _ = obs_traj_batch.shape

    # Return an empty list if there are no trajectories
    if total_peds_in_batch == 0:
        return []

    prompts_list = []

    # Move the entire batch to CPU once for efficiency
    obs_traj_batch_cpu = obs_traj_batch.cpu()

    for ped_idx in range(total_peds_in_batch):
        
        # 1. Slice the trajectory for the current pedestrian
        #    This is a slicing operation (not a permutation)
        #    Shape: [obs_len, 2]
        ped_traj_tensor = obs_traj_batch_cpu[:, ped_idx, :]
        
        # 2. Convert the tensor to a rounded list of coordinate strings
        traj_list = ped_traj_tensor.numpy().tolist()
        point_strings = [f"({x:.5f}, {y:.5f})" for x, y in traj_list]
        trajectory_description = ", ".join(point_strings)
        
        # 3. Format the final user prompt message
        user_message_content = f"Coordinates: {trajectory_description}.\nQuestion: find the motion pattern in the pedestrian trajectory considering the given coordinates."
        
        prompts_list.append(user_message_content)

    return prompts_list

def preprocess_future_traj_for_gmm_batch(pred_traj_gt_batch: np.ndarray, expected_len: int) -> np.ndarray:
    """
    Pre-processes a batch of future pedestrian trajectories for GMM clustering
    using vectorized NumPy operations. The preprocessing translates trajectories
    to the origin, rotates them to align the second point along the x-axis, 
    and flattens them for clustering.

    Args:
        pred_traj_gt_batch (np.ndarray): Batch of ground-truth future trajectories.
            Expected shape: [pred_len, batch_size, 2] where:
                - pred_len: number of predicted future time steps
                - batch_size: number of pedestrians in the batch
                - 2: spatial coordinates (x, y)
        expected_len (int): Expected trajectory length (usually equal to pred_len).

    Returns:
        np.ndarray: Processed and flattened batch of trajectories.
            Shape: [batch_size, expected_len * 2], ready for GMM clustering.
    """
    
    # Get sequence length (S) and batch size (B) from input
    seq_len, total_peds_in_batch, _ = pred_traj_gt_batch.shape

    # 1. Translate all trajectories to the origin (0,0)
    #    start_points has shape (B, 2)
    start_points = pred_traj_gt_batch[0, :, :]
    #    Broadcasting subtraction: (S, B, 2) - (1, B, 2)
    traj_translated = pred_traj_gt_batch - start_points[np.newaxis, :, :]

    # 2. Compute the angle to rotate each trajectory so the second point aligns with x-axis
    second_points = traj_translated[1, :, :] # Shape (B, 2)
    thetas = np.arctan2(second_points[:, 1], second_points[:, 0])
    
    c = np.cos(-thetas)
    s = np.sin(-thetas)
    
    # Build 2x2 rotation matrices for each pedestrian
    row1 = np.stack([c, -s], axis=1) # Shape (B, 2)
    row2 = np.stack([s, c], axis=1) # Shape (B, 2)
    rotation_matrices = np.stack([row1, row2], axis=1) # Shape (B, 2, 2)

    # Apply rotation to all trajectories using Einstein summation
    # 'sbk,bkj->sbj' means: for each sequence step s and batch b, multiply
    # dimension k=2 with k=2 to result in j=2
    traj_rotated = np.einsum('sbk,bkj->sbj', traj_translated, rotation_matrices)

    # Handle cases where the second point is at the origin (0,0)
    norms = np.linalg.norm(second_points, axis=1) # Shape (B,)
    mask_broadcastable = (norms > 1e-4)[np.newaxis, :, np.newaxis] # Shape (1, B, 1)
    
    traj_final = np.where(mask_broadcastable, traj_rotated, traj_translated)

    # 3. Flatten trajectories for GMM input
    #    Swap axes to shape (B, S, 2) then reshape to (B, S*2)
    flattened_batch = traj_final.swapaxes(0, 1).reshape(total_peds_in_batch, seq_len * 2)
    
    return flattened_batch

def main(args: argparse.Namespace):
    """
    Main execution function for pedestrian trajectory motion analysis using an LLM
    and Gaussian Mixture Model (GMM).

    This function performs the from vllm.sampling_params import GuidedDecodingParamsfollowing stages:
    1. Loads datasets and prepares batches.
    2. Generates text-based motion descriptions ('M') for each pedestrian using the LLM.
    3. Preprocesses future trajectories for GMM clustering.
    4. Trains a Gaussian Mixture Model (GMM) on the processed trajectories to obtain 'Zc'.
    5. Saves both the text descriptions and the trained GMM model to disk.

    Args:
        args (argparse.Namespace): Command-line arguments containing hyperparameters,
            dataset names, model paths, and output directories.
    """
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_device(0)
    else:
        device = "cpu"
        
    print(f"Using device: {device}")
    
    # --- 1. Initialize structured outputs and sampling parameters ---
    json_schema = MotionAnalysis.model_json_schema()
    structured_params = StructuredOutputsParams(json=json_schema)
    
    stop_strings = formatar_valores_enum(MotionPattern)
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        skip_special_tokens=args.skip_special_tokens,
        structured_outputs=structured_params,
        stop=stop_strings,
        include_stop_str_in_output=True
    )
    
    pooling_params = PoolingParams(
        dimensions=args.dimensions,
        normalize=args.normalize
    )

    if args.block_pulling_models_from_online_hub:
        from huggingface_hub import snapshot_download
        
        model_generative_path = snapshot_download(
            repo_id=args.model_generative_name,
            repo_type="model",
            local_files_only=True,
            local_dir=args.local_model_generative_path
        )

        model_embedding_path = snapshot_download(
            repo_id=args.model_embedding_name,
            repo_type="model",
            local_files_only=True,
            local_dir=args.local_model_embedding_path
        )
    
    # --- 2. Initialize the LLM model ---
    llm_generative = LLM(
        model=model_generative_path if args.block_pulling_models_from_online_hub else args.model_generative_name,
        seed=args.seed,
        quantization=None if args.quantization == "none" else args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prompt_embeds=True,
        max_model_len=args.max_model_len,
        runner="generate",
        enable_sleep_mode=True,
        max_num_seqs=args.max_num_seqs
        #enforce_eager=True
        #enable_prefix_caching=False
    )
    
    llm_embedding = LLM(
        model=model_embedding_path if args.block_pulling_models_from_online_hub else args.model_embedding_name,
        seed=args.seed,
        quantization=None if args.quantization == "none" else args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prompt_embeds=True,
        max_model_len=args.max_model_len,
        runner="pooling",
        #enforce_eager=True,
        enable_sleep_mode=True,
        hf_overrides={"is_matryoshka": True},
        max_num_seqs=args.max_num_seqs
        #enable_prefix_caching=False
    )

    llm_embedding.sleep(level=1)
    
    print(f"Loading Tokenizer: {args.tokenizer_name}")
    if args.block_pulling_models_from_online_hub:
        tokenizer = AutoTokenizer.from_pretrained(model_generative_path, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_generative_name)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- 3. Define system prompt ---
    system_prompt = {
        "role": "system",
        "content": "You are an expert who can analyze and identify different types of motion patterns in the (x, y) trajectories coordinates of pedestrians' movement. Based on the series of two-dimensional coordinates provided, you must generate a JSON object with a description and a motion_pattern classification of the trajectory."
    }
    
    # --- 4. Configure output paths ---
    dataset_names = args.datasets
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    M_text_save_path = os.path.join(output_dir, f"llm_motion_cues_{args.model_generative_name.lower().split('/')[1]}_{args.model_embedding_name.lower().split('/')[1]}_{'_'.join(args.datasets) if isinstance(args.datasets, list) else args.datasets}_{args.batch_size}_{args.temperature}_{args.max_tokens}_{args.quantization}_{args.gmm_n_components}.json")
    gmm_save_path = os.path.join(output_dir, f"gmm_model_{'_'.join(args.datasets) if isinstance(args.datasets, list) else args.datasets}.pkl")
    
    print(f"'M' (text classifications) will be saved to: {M_text_save_path}")
   
    all_results = {
        "batch_size": args.batch_size,
        "datasets": {}
    }
    
    all_M_texts = {}
    all_Zc_ids = {} 
    all_future_trajs_processed_for_gmm = []
    
    # Few-shot examples for LLM prompting
    few_shot_examples = [
        {"role": "system", "content": "You are an expert who can analyze and identify different types of motion patterns in the (x, y) trajectories coordinates of pedestrians' movement. Based on the series of two-dimensional coordinates provided, you must generate a JSON object with a description and a motion_pattern classification of the trajectory."},
        {"role": "user", "content": "Coordinates: (0.00, 0.00), (0.10, 0.10), (0.20, 0.20), (0.30, 0.30)\nQuestion: find the motion pattern in the pedestrian trajectory considering the given coordinates."},
        {"role": "assistant", "content": '{ "description": "The pedestrian follows a movement in which the x and y coordinates increase linearly by 10 units per step.", "motion_pattern": "Linear Motion" }'},
        {"role": "user", "content": "Coordinates: (0.00, 0.00), (0.00, 0.00), (0.00, 0.00), (0.00, 0.00)\nQuestion: find the motion pattern in the pedestrian trajectory considering the given coordinates."},
        {"role": "assistant", "content": '{ "description": "None of the pedestrian\'s coordinates increased during the observed period. Therefore, the pedestrian did not move at all.", "motion_pattern": "Standing Still" }'},
    ]

    # --- 5. Main Processing Loop ---
    print(f"\n--- STAGE 1: Generating 'M' (LLM Text) & Collecting 'Y_i' (GMM data) ---")
    
    for dset_name in dataset_names:
        print(f"\nProcessing dataset: {dset_name}")
        
        data_file_path = utils.get_dset_path(dset_name, 'train')
        dset = data_dset(args, data_file_path)
        dset_loader = data_loader(args, dset, args.batch_size, pin_memory=False)
        
        num_batches = len(dset_loader)
        if num_batches == 0:
            print(f"Skipping dataset {dset_name}, it has zero batches.")
            continue
        
        print(f"Found {len(dset)} scenes, resulting in {num_batches} batches.")

        all_results["datasets"][dset_name] = {}
        
        for batch_idx, batch in enumerate(tqdm(dset_loader, desc=f"Processing {dset_name} Batches")):
            
            # Move tensors to CPU for processing
            try:
                batch = [tensor.to("cpu") for tensor in batch]
            except AttributeError:
                print(f"Skipping malformed batch {batch_idx}")
                continue

            (
                obs_traj,        # Shape: [seq_len, total_peds_in_batch, 2]
                pred_traj,       # Shape: [seq_len, total_peds_in_batch, 2]
                obs_traj_rel,    # Shape: [seq_len, total_peds_in_batch, 2]
                pred_traj_rel,   # Shape: [seq_len, total_peds_in_batch, 2]
                non_linear_ped,  # Shape: [total_peds_in_batch]
                loss_mask,       # Shape: [total_peds_in_batch, seq_len]
                seq_start_end,   # Shape: [num_scenes_in_batch, 2]
                global_indices,
                t_embeddings,
                zc_ids
            ) = batch

            # Format user prompts for the LLM
            prompts_batch = []
            prompts_batch = format_trajectories_batch(obs_traj)

            prompts_batch_with_system_message = [
                [system_prompt, few_shot_examples[0], few_shot_examples[1], few_shot_examples[2], few_shot_examples[3], {"role": "user", "content": user_promp_content}] if args.use_few_shot else [system_prompt, {"role": "user", "content": user_promp_content}]
                for user_promp_content in prompts_batch
            ]

            if args.save_prompts_to_file:
                output_filename = "meus_prompts.txt"
                try:
                    with open(output_filename, "w", encoding="utf-8") as f:
                        for lista in prompts_batch_with_system_message:
                            for prompt in lista:
                                # Garante que o prompt eh uma string antes de escrever
                                if isinstance(prompt, dict):
                                    f.write(str(prompt) + "\n")
                                else:
                                    # Opcional: Registra se houver dados ruins
                                    print(f"Aviso prompts_batch_with_system_message: Item ignorado (nao eh string): {prompt}")

                    print(f"Sucesso! Prompts salvos em {output_filename}")

                except Exception as e:
                    print(f"Erro ao salvar arquivo: {e}")
            
            # Apply chat template for tokenizer
            prompts_batch_with_system_message_applied_template = tokenizer.apply_chat_template(
                prompts_batch_with_system_message,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_tensors="pt"
            )

            if args.save_prompts_to_file:
                output_filename = "meus_prompts_templateApplied.txt"
                try:
                    with open(output_filename, "w", encoding="utf-8") as f:
                        for prompt in prompts_batch_with_system_message_applied_template:
                            # Garante que o prompt eh uma string antes de escrever
                            if isinstance(prompt, str):
                                f.write(prompt + "\n")
                            else:
                                # Opcional: Registra se houver dados ruins
                                print(f"Aviso prompts_batch_with_system_message_applied_template: Item ignorado (nao eh string): {prompt}")

                    print(f"Sucesso! Prompts salvos em {output_filename}")

                except Exception as e:
                    print(f"Erro ao salvar arquivo: {e}")

            # Generate LLM motion cues and extract raw texts:
            output_generate = llm_generative.generate(prompts_batch_with_system_message_applied_template, sampling_params)
            llm_motion_cues_descriptions_batch = [out.outputs[0].text.strip() for out in output_generate]
            
            # Put the generative LLM to sleep and wake up the pooling LLM (for embeddings)
            llm_generative.sleep(level=1)
            llm_embedding.wake_up()
            
            # Generate the embeddings for all raw descriptions in batch:           
            output_embed = llm_embedding.embed(llm_motion_cues_descriptions_batch, pooling_params=pooling_params)
            llm_motion_cues_embeddings_batch = [out.outputs.embedding for out in output_embed]
            llm_embedding.sleep(level=1)
            
            # Tokenize all raw descriptions in batch:       
            #llm_motion_cues_tokens_batch = tokenizer(
                #llm_motion_cues_descriptions_batch,
                #add_special_tokens=False,
                #return_tensors="pt",
                #padding=True,
                #truncation=True
            #)
            
            # Parse LLM outputs into JSON
            llm_motion_cues_descriptions_batch_parsed = []
            for initial_description, initial_embedding, prompt in zip(llm_motion_cues_descriptions_batch, llm_motion_cues_embeddings_batch, prompts_batch_with_system_message_applied_template):
                current_description = initial_description
                current_embedding = initial_embedding
                number_of_tries = 1
                
                while number_of_tries <= args.max_number_of_sampling_retries:
                    try:
                        json_object = json.loads(current_description)
                        json_object["description_embedding"] = to_serializable_embedding(current_embedding)
                        #json_object["description_tokenized"] = tokens
                        
                        llm_motion_cues_descriptions_batch_parsed.append(json_object)
                        break
                        
                    except json.JSONDecodeError:
                        if number_of_tries == args.max_number_of_sampling_retries:
                            print(f"Erro JSON persistente para o prompt: {prompt}. Max. tentativas alcancado.")
                            llm_motion_cues_descriptions_batch_parsed.append({
                                "error": "JSONDecodeError",
                                "raw_output": current_description
                            })
                            
                            break
                        
                        else:
                            number_of_tries += 1
                            print(f"Erro JSON na tentativa {number_of_tries-1}. Tentando novamente...")
                            
                            llm_generative.wake_up()
                            current_description = llm_generative.generate(prompt, sampling_params)[0].outputs[0].text.strip()
                            llm_generative.sleep(level=1)
                            
                            llm_embedding.wake_up()
                            current_embedding = llm_embedding.embed(current_description, pooling_params=pooling_params)[0].outputs.embedding
                            llm_embedding.sleep(level=1)
                    
            for g_idx_tensor, llm_result in zip(global_indices, llm_motion_cues_descriptions_batch_parsed):
                g_idx_key = str(g_idx_tensor.item()) 
                all_results["datasets"][dset_name][g_idx_key] = llm_result
            
            pred_traj_np_batch = pred_traj.cpu().numpy()
            
            # Preprocess future trajectories for GMM
            processed_batch = preprocess_future_traj_for_gmm_batch(
                pred_traj_np_batch,
                args.pred_len
            )
            
            all_future_trajs_processed_for_gmm.extend(processed_batch)
            
            llm_embedding.sleep(level=1)
            llm_generative.wake_up()

            # Clear memory
            del obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask, seq_start_end, prompts_batch, prompts_batch_with_system_message, prompts_batch_with_system_message_applied_template, llm_motion_cues_descriptions_batch
    
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # --- 6. Save all LLM text outputs ---
    print(f"\nSaving textual motion cues (M) in {M_text_save_path}...")
    try:
        with open(M_text_save_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print("Textual motion cues successfully saved.")
    except Exception as e:
        print(f"Error: {e}")

    # --- 7. Train Gaussian Mixture Model (GMM) ---
    print("\n--- STAGE 2: Training GMM for 'Zc' ---")
    
    if not all_future_trajs_processed_for_gmm:
        print("Error: There are no future trajectories, GMM will not be trained.")
        return

    gmm_train_data = np.stack(all_future_trajs_processed_for_gmm).astype(np.float64)
    
    gmm = GaussianMixture(
        n_components=args.gmm_n_components, 
        covariance_type='full', 
        random_state=args.seed, 
        verbose=1, 
        max_iter=args.gmm_max_iter, 
        n_init=args.gmm_n_init
    )
    
    try:
        gmm.fit(gmm_train_data)
        print(f"GMM training completed. Converged: {gmm.converged_}")
        joblib.dump(gmm, gmm_save_path)
        print(f"GMM model saved to {gmm_save_path}")
        
    except Exception as e:
        print(f"Error during GMM fitting: {e}")
        return

    print("\n--- STAGE 3: Salvando Zc_id (IDs de Cluster) ---")
    all_Zc_ids_array = gmm.predict(gmm_train_data)
    
    for i in range(len(all_Zc_ids_array)):
        g_idx = i
        zc_id = all_Zc_ids_array[i]
        all_Zc_ids[g_idx] = int(zc_id)
        
    zc_id_save_path = os.path.join(output_dir, f"clusters_id_map_{'_'.join(args.datasets) if isinstance(args.datasets, list) else args.datasets}.json")
    print(f"Salvando {len(all_Zc_ids)} IDs de cluster em {zc_id_save_path}...")
    with open(zc_id_save_path, 'w') as f:
        json.dump(all_Zc_ids, f, indent=2)

###################################
## Main program
###################################
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("End!")