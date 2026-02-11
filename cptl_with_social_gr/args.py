#!/usr/bin/env python3

###################################
# Imports and packages
###################################
import argparse
import os
import yaml

###################################
# Functions
###################################

def _int_tuple(arg):
    """Helper para converter string '8,8' em tupla (8, 8)"""
    return tuple(map(int, arg.split(',')))

def add_base_args(parser: argparse.ArgumentParser):
    """Argumentos fundamentais de ambiente e seeds."""
    group = parser.add_argument_group('Base configuration')
    
    group.add_argument(
        '--get-stamp',
        action='store_true',
        help="print param-stamp & exit"
    )
    
    group.add_argument(
        '--seed',
        type=int,
        default=72,
        help="random seed"
    )
    
    group.add_argument(
        '--no-gups',
        action='store_false',
        dest='cuda',
        help="do not use GPUs"
    )

    group.add_argument(
        '--gpu_index',
        default=0,
        type=int
    )
    
    group.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        dest='r_dir'
    )
    
    group.add_argument(
        '--plot-dir',
        type=str,
        default='./plots',
        dest='p_dir'
    )
    
    group.add_argument(
        '--log_dir',
        default="ETH",
        help="Directory containing logging file"
    )
    
    return parser


def add_dataset_args(parser: argparse.ArgumentParser):
    """Argumentos relacionados ao carregamento e processamento de dados."""
    group = parser.add_argument_group('Dataset parameters')

    dataset_choices = ['ETH', 'UCY', 'inD', 'INTERACTION']
    group.add_argument(
        '--dataset',
        type=str,
        nargs="+",
        choices=dataset_choices
    )
    
    splits_choices = ['train', 'val', 'test']
    group.add_argument(
        '--split',
        type=str,
        nargs='+',
        default=splits_choices
    )
    
    group.add_argument(
        '--data-dir',
        type=str,
        default='./datasets',
        dest='d_dir'
    )

    group.add_argument(
        '--obs_len',
        default=8,
        type=int,
        help="observed frame length"
    )
    
    group.add_argument(
        '--pred_len',
        default=12,
        type=int,
        help="predicted frame length"
    )
    
    group.add_argument(
        '--skip',
        default=1,
        type=int
    )
    
    group.add_argument(
        '--delim',
        default='\t'
    )
    
    group.add_argument(
        '--loader_num_workers',
        default=8,
        type=int
    )

    group.add_argument(
        "--aug",
        type=str,
        default='none',
        choices=["none", "rotation"]
    )
    
    return parser


def add_training_args(parser: argparse.ArgumentParser):
    """Argumentos de hiperparâmetros de treino."""
    group = parser.add_argument_group('Training Parameters')

    group.add_argument(
        '--method',
        type=str,
        default='batch_learning',
        choices=['batch_learning', 'continual_learning']
    )
    
    group.add_argument(
        '--iters',
        type=int,
        default=400,
        help="batches to optimize solver"
    )
    
    group.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help="learning rate"
    )
    
    group.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help="batch-size"
    )
    
    group.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'adam_reset', 'sgd'],
        default='adam'
    )
    
    group.add_argument(
        '--val_epoch',
        default=150,
        type=int,
        help="epoch start to validation"
    )

    group.add_argument(
        "--start_epoch",
        default=1,
        type=int,
        help="manual epoch number"
    )
    
    group.add_argument(
        "--clip_gradient_max_norm",
        default=1.0,
        type=float,
        help=""
    )
    
    group.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint"
    )
    
    group.add_argument(
        "--checkpoint_log",
        default=50,
        type=int,
        help="iters after which to save checkpoint"
    )
    
    group.add_argument(
        "--print_every",
        default=10,
        type=int
    )
    
    group.add_argument(
        "--use_codecarbon",
        action='store_true',
        help=""
    )
    
    return parser


def add_model_args(parser: argparse.ArgumentParser):
    """Argumentos de arquitetura (LSTM, GAT, etc)."""
    group = parser.add_argument_group('Model Architecture')

    model_choices = ["lstm", "gat"]
    group.add_argument(
        '--main_model',
        default='lstm',
        type=str,
        choices=model_choices
    )

    # LSTM Params
    group.add_argument(
        '--traj_lstm_input_size',
        default=2,
        type=int
    )
    
    group.add_argument(
        '--traj_lstm_hidden_size',
        default=128,#32,
        type=int
    )
    
    group.add_argument(
        '--traj_lstm_output_size', 
        default=128,#32,
        type=int
    )

    # GAT Params
    group.add_argument(
        "--heads",
        type=str,
        default="4,1",
        help="Heads per layer (comma separated)"
    )
    
    group.add_argument(
        "--hidden-units",
        type=str,
        default="16",
        help="Hidden units (comma separated)"
    )
    
    group.add_argument(
        "--graph_network_out_dims",
        type=int, 
        default=32
    )
    
    group.add_argument(
        "--graph_lstm_hidden_size",
        default=32,
        type=int
    )
    
    group.add_argument(
        "--dropout",
        type=float,
        default=0.0
    )
    
    group.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Alpha for leaky_relu"
    )
    
    group.add_argument(
        "--mlp_dim",
        type=int,
        default=256,
        help="MLP dimension"
    )
    
    group.add_argument(
        "--embedding_dim",
        type=int,
        default=32,
        help=""
    )
    
    group.add_argument(
        "--bottleneck_dim",
        type=int,
        default=128,#32,
        help=""
    )

    # Noise
    group.add_argument(
        "--noise_dim",
        default=(8,),
        type=_int_tuple
    )
    
    group.add_argument(
        "--noise_type",
        default="gaussian"
    )
    
    group.add_argument(
        "--noise_mix_type",
        default="global"
    )
    
    return parser


def add_llm_args(parser: argparse.ArgumentParser):
    """Argumentos específicos para Large Language Models."""
    group = parser.add_argument_group('LLM Configuration')
    
    group.add_argument(
        "--config_file",
        type=str,
        default="/home/matheus/LLM4CPTL/cptl_with_social_gr/datasets/preprocessed/llm_preprocessed_data_config.yaml",
        help="Caminho para o arquivo YAML de configuração (mapeamentos e caminhos)"
    )

    group.add_argument(
        "--adapt_architecture_to_include_llm_motion_cues",
        action='store_true'
    )
    
    group.add_argument(
        "--adapt_architecture_to_include_sequence_embedding",
        action='store_true'
    )
    
    group.add_argument(
        "--use_skip_connection",
        action='store_true'
    )
    
    group.add_argument(
        "--use_gradient_clipping",
        action='store_true'
    )
    
    group.add_argument(
        "--use_kl_annealing",
        action='store_true'
    )
    
    group.add_argument(
        '--sequence_embedding_compressed_dimension',
        type=int,
        default=64,#128
    )

    # Models
    group.add_argument(
        '--model_generative_name',
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct"  # 'Qwen/Qwen3-0.6B'
    )

    group.add_argument(
        '--model_embedding_name',
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct"  # 'Qwen/Qwen3-Embedding-0.6B'
    )
    
    group.add_argument(
        '--tokenizer_name',
        type=str,
        default='Qwen/Qwen2.5-VL-3B-Instruct',
        help="Hugging Face tokenizer name."
    )
    
    group.add_argument(
        '--local_model_generative_path',
        type=str,
        default=None
    )

    group.add_argument(
        '--local_model_embedding_path',
        type=str,
        default=None
    )

    # Mappings
    group.add_argument(
        "--llm_motion_cues_mapping",
        type=dict
    )
    group.add_argument(
        "--llm_sequences_embeddings_mapping",
        type=dict
    )

    # Generation parameters
    group.add_argument(
        '--max_model_len',
        type=int,
        default=1024
    )
    
    group.add_argument(
        '--max_num_seqs',
        type=int,
        default=1
    )

    group.add_argument(
        '--temperature',
        type=float,
        default=0.8
    )
    
    group.add_argument(
        '--top_p',
        type=float,
        default=0.95
    )
    
    group.add_argument(
        '--quantization',
        type=str,
        choices=["bitsandbytes", "none"],
        default="none"
    )
    
    group.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.6
    )
    
    group.add_argument(
        '--use_few_shot',
        action="store_true"
    )
    
    group.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.2,
        help="Applies a penalty to repeated tokens. Higher values reduce the likelihood of repetition."
    )


    group.add_argument(
        '--max_tokens',
        type=int,
        default=2048,
        help="Maximum number of tokens to generate in the output."
    )

    group.add_argument(
        '--skip_special_tokens',
        action="store_true",
        default=True,
        help="If set, special tokens (e.g., <BOS>, <EOS>, <PAD>) will be removed from the generated output."
    )
    
    group.add_argument(
        '--save_text_descriptions',
        action="store_true",
        help=""
    )
    
    group.add_argument(
        '--normalize',
        action="store_true",
        default=False,
        help=""
    )

    group.add_argument(
        '--dimensions',
        default=768,
        type=int,
        help=""
    )
    
    group.add_argument(
        '--output_dir',
        default="./datasets/preprocessed",
        type=str,
        help=""
    )
    
    group.add_argument(
        '--prevent_model_download_from_hub',
        default=False,
        action='store_true',
        help=""
    )

    return parser


def add_cl_replay_args(parser: argparse.ArgumentParser):
    """Argumentos para Continual Learning e Memory Replay."""
    group = parser.add_argument_group('CL & Replay Parameters')

    # Replay
    group.add_argument(
        '--replay',
        type=str,
        default='none',
        choices=['offline', 'exact', 'generative', 'none', 'current', 'exemplars']
    )
    
    group.add_argument(
        '--z_dim',
        type=int,
        default=200,
        help="latent rep size"
    )

    group.add_argument(
        '--replay_batch_size',
        type=int,
        default=64
    )
    group.add_argument(
        '--g-iters',
        type=int,
        help="generator iters"
    )
    group.add_argument(
        '--lr_gen',
        type=float,
        default=0.001
    )

    # Replay Models
    group.add_argument(
        '--replay_model',
        default='lstm',
        type=str, 
        choices=['lstm', 'vrnn', 'condition']
    )

    # Memory Allocation (SI)
    group.add_argument(
        '--si',
        action='store_true',
        help="Synaptic Intelligence"
    )
    
    group.add_argument(
        '--c',
        type=float,
        dest="si_c",
        help="SI regularization strength"
    )
    
    group.add_argument(
        '--epsilon',
        type=float,
        default=0.1
    )

    return parser


def add_eval_args(parser: argparse.ArgumentParser):
    """Argumentos de avaliação e logs."""
    group = parser.add_argument_group('Evaluation Parameters')

    group.add_argument(
        '--metrics',
        action='store_true',
        help="calculate extra metrics (BWT, forgetting)"
    )
    
    group.add_argument(
        '--time',
        action='store_true',
        help="keep track of total training time"
    )
    
    group.add_argument(
        '--visdom',
        action='store_true',
        help="use visdom"
    )
    
    group.add_argument(
        '--val',
        action='store_true',
        help="use validation data"
    )
    
    group.add_argument(
        '--val_class',
        default='current',
        type=str,
        choices=['current', 'all', 'replay']
    )
    
    group.add_argument(
        '--pdf',
        action='store_true'
    )

    group.add_argument(
        '--log-per-task',
        action='store_true',
        help="set all visdom-logs to [iters]"
    )

    group.add_argument(
        '--loss-log',
        type=int,
        default=20,
        metavar="N",
        help="iters after which to plot loss"
    )
    
    group.add_argument(
        '--prec-log',
        type=int,
        default=20,
        metavar="N",
        help="iters after which to plot precision"
    )

    group.add_argument(
        '--prec-n',
        type=int,
        default=1024,
        help="samples for evaluating solver's precision"
    )
    group.add_argument(
        '--sample-log',
        type=int,
        default=500,
        metavar="N",
        help="iters after which to plot samples"
    )
    group.add_argument(
        '--num_samples',
        type=int,
        default=20,
        help="sample trajectories when evaluation model"
    )

    return parser


def get_parser(description="Standard experiment parser"):
    """
    Cria um parser básico. Os scripts devem chamar as funções 'add_...' 
    acima para popular o parser conforme a necessidade.
    """
    parser = argparse.ArgumentParser(description=description)
    return parser


def load_yaml_config(args):
    """
    Carrega o YAML estruturado por Dataset e o converte para 
    os mapeamentos específicos esperados.
    """
    if not os.path.exists(args.config_file):
        print(
            f"Warning: Config '{args.config_file}' not found. Using empty defaults.")
        args.llm_motion_cues_mapping = {}
        args.llm_sequences_embeddings_mapping = {}
        return args

    try:
        with open(args.config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        # Inicializa os dicionários de destino
        motion_cues_map = {}
        sequences_emb_map = {}
        clusters_ids_map = {}  # Adicionado caso precise usar

        # Itera sobre cada dataset (ETH, UCY, etc.)
        # dataset_key é "ETH", "UCY", etc.
        # dataset_config é o dicionário com "llm_motion_cues", "sequences_embeddings", etc.
        for dataset_key, dataset_config in config_data.items():

            # 1. Carregar Motion Cues
            if "llm_motion_cues" in dataset_config:
                motion_cues_map[dataset_key] = dataset_config["llm_motion_cues"]

            # 2. Carregar Embeddings
            # A estrutura no YAML é:
            # sequences_embeddings:
            #   train: /path/to/train.pt
            #   val: /path/to/val.pt
            if "sequences_embeddings" in dataset_config:
                # Salvamos o dicionário inteiro {'train': ..., 'val': ...}
                # O loader do dataset precisará saber acessar ['train'] ou ['val']
                sequences_emb_map[dataset_key] = dataset_config["sequences_embeddings"]

            # 3. Carregar Clusters (se necessário)
            if "clusters_ids" in dataset_config:
                clusters_ids_map[dataset_key] = dataset_config["clusters_ids"]

        # Injeta os dicionários processados no namespace args
        args.llm_sequences_embeddings_mapping = sequences_emb_map
        print(sequences_emb_map)

    except yaml.YAMLError as exc:
        print(f"Error while reading YAML file: {exc}")
        raise

    return args

def get_all_args(load_yaml: bool = False):
    """
    Atalho para carregar TODOS os argumentos (legado/main script).
    """
    parser = get_parser()
    add_base_args(parser)
    add_dataset_args(parser)
    add_training_args(parser)
    add_model_args(parser)
    add_llm_args(parser)
    add_cl_replay_args(parser)
    add_eval_args(parser)

    args = parser.parse_args()
    
    if load_yaml:
        args = load_yaml_config(args)

    return args