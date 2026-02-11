import torch
import numpy as np
from torch.utils.data import DataLoader
from data.loader import data_loader, data_dset
from helper import utils
# Mantenha a importação original do seq_collate se ela for suficiente
from data.trajectories_memory import TrajectoryDataset, seq_collate
# Se 'seq_collate_' do terceiro arquivo for necessário, use-o ou renomeie-o. Vamos assumir 'seq_collate' padrão por enquanto.


def sample_dataset_memory(args, dataset_name):
    """
    Carrega o conjunto de treinamento de um único dataset, amostra 10% 
    dos dados e retorna o batch de tensores em memória (na GPU).
    """
    try:
        train_path = utils.get_dset_path(dataset_name, 'train')
    except Exception as e:
        print(
            f"ERRO: Não foi possível obter o caminho para o dataset {dataset_name}. {e}")
        return None

    # Carregar o dataset completo (data_dset)
    t_emb_path = args.llm_motion_cues_and_clusters_ids_filepaths_mapping.get(
        dataset_name, {}).get("llm_motion_cues")
    train_dset = data_dset(
        args, train_path, t_embedding_path=t_emb_path, dataset_name=dataset_name)

    # Amostrar 10% (num_memory) e criar um DataLoader temporário
    num_memory = int(args.memory_buff_percentage * len(train_dset))
    # data_loader eh usado aqui para retornar um subconjunto aleatorio de tamanho 'num_memory'
    dataset_subset_loader = data_loader(
        args, train_dset, args.replay_batch_size)

    # Pegar o primeiro (e único) batch amostrado
    memory_batch_list = []
    for batch_index, batch in enumerate(dataset_subset_loader):
        memory_batch_list = [tensor.cuda() for tensor in batch]
        break  # Parar apos o primeiro batch (que contém a memoria amostrada)

    if not memory_batch_list:
        print(f"Aviso: O dataset {dataset_name} esta vazio apos a amostragem.")
        return None

    return memory_batch_list


def memory_buff_unified(args, datasets_to_replay):
    """
    Combina os memory buffers de múltiplos datasets para o replay.

    Args:
        args (Namespace): Argumentos de configuração.
        datasets_to_replay (list of str): Lista dos nomes dos datasets para replay (e.g., ['ETH', 'UCY']).

    Returns:
        tuple: (loader, latest_batch_data)
               loader (DataLoader): DataLoader do dataset unificado de replay.
               latest_batch_data (list of Tensors): O batch amostrado do ÚLTIMO dataset processado.
    """
    if not datasets_to_replay:
        return None, None

    all_batches = []

    # 1. Coletar o batch de memória de cada dataset na lista
    for name in datasets_to_replay:
        batch_data = sample_dataset_memory(args, name)
        if batch_data:
            all_batches.append(batch_data)
        else:
            print(f"Pulando o dataset {name} devido a falha na amostragem.")

    if not all_batches:
        return None, None  # Nenhum dado de replay coletado

    # O batch mais recente (último da lista) é retornado ao 'train_cl' para ser usado como 'batch_ind', 'batch_ucy', etc.
    latest_batch_data = all_batches[-1]

    # 2. Concatenar todos os tensores
    # A ordem da concatenação é importante, mas aqui assumiremos a ordem dada por 'datasets_to_replay'.

    # Inicializa listas de tensores para concatenação
    obs_traj_list, pred_traj_list, obs_rel_list, pred_rel_list = [], [], [], []
    seq_start_end_list, t_embeddings_list = [], []

    # Acumulador para corrigir os índices de seq_start_end
    current_end_index = 0

    for batch in all_batches:
        # Tensores na ordem padrão (baseado nos seus arquivos originais)
        # batch[0]: obs_traj, batch[1]: pred_traj_gt, batch[2]: obs_traj_rel, batch[3]: pred_traj_gt_rel
        # batch[6]: seq_start_end, batch[7]: t_embeddings

        obs_traj_list.append(batch[0])
        pred_traj_list.append(batch[1])
        obs_rel_list.append(batch[2])
        pred_rel_list.append(batch[3])
        t_embeddings_list.append(batch[7])  # Embeddings

        current_seq_start_end = batch[6]

        # Ajuste do seq_start_end (a parte mais complexa da unificação)
        if current_end_index > 0:
            # Se não é o primeiro dataset, adiciona o índice final acumulado
            # Isso é crucial para que os índices sejam contínuos em todo o batch concatenado
            adjusted_seq_start_end = current_seq_start_end + current_end_index
            seq_start_end_list.append(adjusted_seq_start_end)
        else:
            # Primeiro dataset: usa o original
            seq_start_end_list.append(current_seq_start_end)

        # Atualiza o índice final para o próximo dataset
        # O índice final é o último valor no tensor (ex: [[0, 20], [20, 45]] -> 45)
        # Como seq_start_end é (N, 2), usamos .max() ou [..., -1][-1]
        current_end_index = seq_start_end_list[-1][-1][1].item()

    # 3. Finalizar a concatenação (dim=1 para o batch/agente, exceto seq_start_end)
    # A ordem de concatenação é [Dataset1, Dataset2, ..., LastDataset]

    # Concatenar tensores de sequência (dim=1 = batch/número de agentes)
    obs_traj = torch.cat(obs_traj_list, dim=1)
    pred_traj = torch.cat(pred_traj_list, dim=1)
    obs_traj_rel = torch.cat(obs_rel_list, dim=1)
    pred_traj_rel = torch.cat(pred_rel_list, dim=1)
    t_embeddings = torch.cat(t_embeddings_list, dim=0)

    # Concatenar seq_start_end (dim=0 = número de sequências/pedestres agrupados)
    seq_start_end = torch.cat(seq_start_end_list, dim=0)

    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        seq_start_end,
        t_embeddings
    ]

    # 4. Criar o TrajectoryDataset e DataLoader
    dset = TrajectoryDataset(
        out[0].detach().cpu(),
        out[1].detach().cpu(),
        out[2].detach().cpu(),
        out[3].detach().cpu(),
        out[4].detach().cpu(),
        out[5].detach().cpu(),
    )

    loader = DataLoader(
        dset,
        batch_size=args.replay_batch_size,
        shuffle=True,
        collate_fn=seq_collate,
        pin_memory=True
    )

    return loader, latest_batch_data
