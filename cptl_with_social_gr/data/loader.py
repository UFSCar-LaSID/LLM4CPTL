from torch.utils.data import DataLoader

from data.trajectories import TrajectoryDataset, seq_collate


def data_dset(args, path, t_embedding_path=None, dataset_name=""):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        t_embedding_path=t_embedding_path,
        dataset_name=dataset_name
    )
    return dset


def data_loader(args, dset, batch_size, shuffle=False, pin_memory=True):
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=pin_memory
    )
    return loader
