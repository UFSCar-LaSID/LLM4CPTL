import numpy as np
import torch

from data.loader import data_loader, data_dset
from helper import utils
from data.trajectories_memory import TrajectoryDataset, seq_collate
from torch.utils.data import DataLoader


def seq_collate_(data):
    (
        obs_seq_list,
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        t_embedding_list
    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]

    cum_start_idx = [0] + np.cumsum(_len).tolist()

    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]

    obs_traj_ = torch.cat(obs_seq_list, dim=0).permute(1, 0, 2)
    pred_traj_ = torch.cat(pred_seq_list, dim=0).permute(1, 0, 2)
    obs_traj_rel_ = torch.cat(obs_seq_rel_list, dim=0).permute(1, 0, 2)
    pred_traj_rel_ = torch.cat(pred_seq_rel_list, dim=0).permute(1, 0, 2)
    seq_start_end = torch.LongTensor(seq_start_end)
    t_embeddings = torch.cat(t_embedding_list, dim=0)

    out = [
        obs_traj_,
        pred_traj_,
        obs_traj_rel_,
        pred_traj_rel_,
        seq_start_end,
        t_embeddings
    ]

    return tuple(out)


def memory_buff(args, batch_eth, batch_ucy):
    # ind
    train_path_ind = utils.get_dset_path("inD", 'train')
    train_dset_ind = data_dset(
        args, train_path_ind, t_embedding_path=args.llm_motion_cues_and_clusters_ids_filepaths_mapping["inD"]["llm_motion_cues"], dataset_name="inD")
    num_memory_ind = int(0.1 * len(train_dset_ind))
    dataset_ind = data_loader(args, train_dset_ind, num_memory_ind)
    batch_ind = []
    for batch_index, batch in enumerate(dataset_ind):
        batch_ind = [tensor.cuda() for tensor in batch]
        break

    obs_traj = torch.cat((batch_ucy[0], batch_ind[0], batch_eth[0]), dim=1)
    pred_traj = torch.cat((batch_ucy[1], batch_ind[1], batch_eth[1]), dim=1)
    obs_traj_rel = torch.cat((batch_ucy[2], batch_ind[2], batch_eth[2]), dim=1)
    pred_traj_rel = torch.cat(
        (batch_ucy[3], batch_ind[3], batch_eth[3]), dim=1)
    seq_start_end_eth = batch_eth[6]
    _, end_eth = seq_start_end_eth[-1]
    seq_start_end_ucy = batch_ucy[6]
    _, end_ucy = seq_start_end_ucy[-1]
    seq_start_end_ind = batch_ind[6]
    _, end_ind = seq_start_end_ind[-1]

    seq_start_end_ind = seq_start_end_ind + end_ucy
    seq_start_end_eth = seq_start_end_eth + end_ucy + end_ind

    seq_start_end = torch.cat(
        (seq_start_end_ucy, seq_start_end_ind, seq_start_end_eth), dim=0)

    t_embeddings = torch.cat((batch_ucy[7], batch_ind[7], batch_eth[7]), dim=1)

    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        seq_start_end,
        t_embeddings
    ]

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

    return loader, batch_ind
