from torch.utils.data import DataLoader

from data.trajectories import TrajectoryDataset, seq_collate


def data_dset(args, path, sequences_embeddings_path=None, dataset_name="", split_name=""):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        sequences_embeddings_path=sequences_embeddings_path,
        dataset_name=dataset_name,
        split_name=split_name
    )

    return dset


def data_loader(args, dset, shuffle=False, pin_memory=True):
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=seq_collate,
        pin_memory=pin_memory
    )

    return loader
