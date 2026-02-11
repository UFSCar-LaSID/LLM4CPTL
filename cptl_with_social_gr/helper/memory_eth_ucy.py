import torch

from data.loader import data_loader, data_dset
from helper import utils
from data.trajectories_memory import TrajectoryDataset, seq_collate
from torch.utils.data import DataLoader


def memory_buff(args, batch_eth):
    # ucy
    train_path_ucy = utils.get_dset_path("UCY", 'train')
    train_dset_ucy = data_dset(
        args, train_path_ucy, t_embedding_path=args.llm_motion_cues_and_clusters_ids_filepaths_mapping["UCY"]["llm_motion_cues"], dataset_name="UCY")
    num_memory_ucy = int(0.1 * len(train_dset_ucy))
    dataset_ucy = data_loader(args, train_dset_ucy, num_memory_ucy)
    batch_ucy = []
    for batch_index, batch in enumerate(dataset_ucy):
        batch_ucy = [tensor.cuda() for tensor in batch]
        break

    obs_traj = torch.cat((batch_ucy[0], batch_eth[0]), dim=1)
    pred_traj = torch.cat((batch_ucy[1], batch_eth[1]), dim=1)
    obs_traj_rel = torch.cat((batch_ucy[2], batch_eth[2]), dim=1)
    pred_traj_rel = torch.cat((batch_ucy[3], batch_eth[3]), dim=1)
    seq_start_end_eth = batch_eth[6]
    _, end_eth = seq_start_end_eth[-1]
    seq_start_end_ucy = batch_ucy[6]
    _, end_ucy = seq_start_end_ucy[-1]
    seq_start_end_eth = seq_start_end_eth + end_ucy
    seq_start_end = torch.cat((seq_start_end_ucy, seq_start_end_eth), dim=0)
    t_embeddings = torch.cat((batch_ucy[7], batch_eth[7]), dim=1)

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

    return loader, batch_ucy
