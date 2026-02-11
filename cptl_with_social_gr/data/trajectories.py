#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################
# Imports and packages
###################################

import logging
import os
import math
import json
from IPython import embed
import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SceneBatch:
    def __init__(self, batch_data, device='cpu'):
        # Verifica se é dicionário (o novo formato)
        if isinstance(batch_data, dict):
            self.obs_traj = batch_data['obs_traj']
            self.pred_traj = batch_data['pred_traj']
            self.obs_traj_rel = batch_data['obs_traj_rel']
            self.pred_traj_rel = batch_data['pred_traj_rel']
            self.non_linear_ped = batch_data['non_linear_ped']
            self.loss_mask = batch_data['loss_mask']
            self.seq_start_end = batch_data['seq_start_end']

            # Novos campos (Embeddings e Nomes)
            # Usa .get() para não quebrar se o campo não existir
            self.sequence_embeddings = batch_data.get(
                'sequence_embedding', None)
            self.sequence_names = batch_data.get('sequence_name', None)

        # Fallback: Se for Lista/Tupla (o formato antigo)
        else:
            (
                self.obs_traj,
                self.pred_traj,
                self.obs_traj_rel,
                self.pred_traj_rel,
                self.non_linear_ped,
                self.loss_mask,
                self.seq_start_end,
                self.sequence_embeddings,  # Assumindo índice 7
                self.sequence_names       # Assumindo índice 8
            ) = batch_data

        # Move para GPU automaticamente se solicitado
        self.to(device)

    def to(self, device):
        # Só move se for Tensor torch
        if isinstance(self.obs_traj, torch.Tensor):
            self.obs_traj = self.obs_traj.to(device)

        if isinstance(self.pred_traj, torch.Tensor):
            self.pred_traj = self.pred_traj.to(device)

        if isinstance(self.obs_traj_rel, torch.Tensor):
            self.obs_traj_rel = self.obs_traj_rel.to(device)

        if isinstance(self.pred_traj_rel, torch.Tensor):
            self.pred_traj_rel = self.pred_traj_rel.to(device)

        if isinstance(self.loss_mask, torch.Tensor):
            self.loss_mask = self.loss_mask.to(device)

        if isinstance(self.seq_start_end, torch.Tensor):
            self.seq_start_end = self.seq_start_end.to(device)

        if isinstance(self.sequence_embeddings, torch.Tensor):
            self.sequence_embeddings = self.sequence_embeddings.to(device)

        # NOTA: Não movemos self.sequence_names pois é lista de strings!

        return self

    def __len__(self):
        return self.obs_traj.shape[1]


def seq_collate(data):
    """
    A custom collate function for the DataLoader.

    This function takes a list of samples (each sample is a tuple returned by
    __getitem__) and batches them together. Since each sample (scene) can have a
    different number of pedestrians, we can't just stack them. Instead, we
    concatenate all pedestrians from all scenes into large tensors and keep
    track of which pedestrians belong to which scene using `seq_start_end`.

    Args:
        data: A list of tuples, where each tuple is the output of __getitem__.
              Each tuple contains:
              (obs_seq, pred_seq, obs_seq_rel, pred_seq_rel, non_linear_ped, loss_mask)

    Returns:
        A tuple containing all batched data:
        (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
         non_linear_ped, loss_mask, seq_start_end)
    """
    (
        obs_seq_list,
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        non_linear_ped_list,
        loss_mask_list,
        sequence_embedding_list,
        sequence_name_list
    ) = zip(*data)

    # Get the number of pedestrians in each scene of the batch
    _len = [len(seq) for seq in obs_seq_list]

    # Calculate cumulative start/end indices for each scene in the batch
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]

    # Concatenate all trajectories from all scenes along the batch dimension
    # Original shape: (num_peds_in_scene, 2, seq_len)
    # torch.cat shape: (total_peds_in_batch, 2, seq_len)
    # .permute shape: (seq_len, total_peds_in_batch, 2)
    # This (seq_len, batch, features) format is often expected by RNNs/LSTMs.
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    sequence_embeddings = torch.cat(sequence_embedding_list, dim=0)

    # Concatenate the non-linear flags and loss masks
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)

    # Convert seq_start_end to a tensor
    seq_start_end = torch.LongTensor(seq_start_end)

    sequence_name_flat = [
        name for sublist in sequence_name_list for name in sublist]

    batch_dict = {
        'obs_traj': obs_traj,
        'pred_traj': pred_traj,
        'obs_traj_rel': obs_traj_rel,
        'pred_traj_rel': pred_traj_rel,
        'non_linear_ped': non_linear_ped,
        'loss_mask': loss_mask,
        'seq_start_end': seq_start_end,
        'sequence_embedding': sequence_embeddings,
        'sequence_name': sequence_name_flat
    }

    return batch_dict


def read_file(_path, delim="\t"):
    """
    Reads a file where each line contains space/tab-separated numerical values.

    Args:
        _path: Path to the file.
        delim: Delimiter to use. Can be "tab", "space", or any other char.

    Returns:
        A 2D numpy array of the data.
    """
    data = []

    # Handle special delimiter keywords
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "

    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)

            # Convert all values in the line to float
            line = [float(i) for i in line]

            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Fits a 2nd-degree polynomial to a trajectory to check for non-linearity.

    Input:
    - traj: Numpy array of shape (2, traj_len) [x_coords, y_coords]
    - traj_len: Length of the trajectory to fit.
    - threshold: Minimum sum of residuals to be considered non-linear.

    Output:
    - int: 1.0 -> Non-Linear, 0.0 -> Linear
    """
    # Create a time-step array [0, 1, ..., traj_len-1]
    t = np.linspace(0, traj_len - 1, traj_len)

    # Fit a 2nd-degree polynomial to x and y coordinates separately
    # res_x/res_y will store the sum of squared residuals (errors) of the fit
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]

    # If the total error is above the threshold, classify as non-linear
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """
    Dataloader for the Trajectory datasets.

    This class reads raw trajectory files, processes them into sequences of
    a fixed length (obs_len + pred_len), and provides a __getitem__ method
    to retrieve all pedestrians present in a single sequence (scene).
    """

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim="\t",
        sequences_embeddings_path=None,
        dataset_name="",
        split_name="train"
    ):
        """
        Args:
        - data_dir: Directory containing dataset files.
          Expected file format: <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories (observation length).
        - pred_len: Number of time-steps in output trajectories (prediction length).
        - skip: Number of frames to skip between sequences (data augmentation).
        - threshold: Minimum error to be considered for non-linear traj.
        - min_ped: Minimum number of pedestrians that must be in a sequence.
        - delim: Delimiter in the dataset files.
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.t_embedding_map = {}
        self.embeddings_loaded = False
        self.t_embed_dim = 0
        self.dataset_name = dataset_name

        try:
            with open(t_embedding_path, 'r') as f:
                self.t_embedding_map = json.load(f)

            if self.t_embedding_map:
                first_key = next(iter(self.t_embedding_map["datasets"]))
                self.t_embed_dim = len(
                    self.t_embedding_map["datasets"][first_key]["0"]["description_embedding"])

            self.embeddings_loaded = True

        except Exception as e:
            print(f"WARNING: Could not load T: {e}")
            self.embeddings_loaded = False

        self.sequences_embeddings_loaded = False
        self.sequences_embeddings_dimension = 0

        self.dataset_name = dataset_name
        self.split_name = split_name

        if sequences_embeddings_path and os.path.exists(sequences_embeddings_path):
            try:
                print(
                    f"Loading sequence embeddings from: {sequences_embeddings_path}")
                self.sequences_embeddings = torch.load(
                    sequences_embeddings_path, map_location='cpu')

                if len(self.sequences_embeddings) > 0:
                    self.sequences_embeddings_dimension = next(
                        iter(self.sequences_embeddings.values())).shape[0]

                self.sequences_embeddings_loaded = True

            except Exception as e:
                print(f"Error loading embeddings: {e}")
                self.sequences_embeddings = None

        else:
            print(
                f"INFO: No embeddings found at {sequences_embeddings_path}. Using zeros.")

        all_files = os.listdir(self.data_dir)
        all_files = sorted([os.path.join(self.data_dir, _path)
                           for _path in all_files])

        # Lists to store all processed data
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        sequences_name_list = []

        # Iterate over each data file (e.g., 'eth.txt', 'hotel.txt')
        for path in all_files:
            # Load the whole file into a numpy array
            data = read_file(path, delim)

            scene_name = os.path.splitext(os.path.basename(path))[0]

            # Get all unique frame IDs, sorted
            frames = np.unique(data[:, 0]).tolist()

            # Group data by frame_id into a list
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])

             # Calculate the number of sliding-window sequences we can create
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            # Iterate using a sliding window approach
            for idx in range(0, num_sequences * self.skip + 1, skip):
                # Get all data points within the current window [idx, idx + seq_len]
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)

                # Get all unique pedestrian IDs present in this window
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

                # Initialize arrays to store data for this specific sequence (scene)
                curr_seq_rel = np.zeros(
                    (len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros(
                    (len(peds_in_curr_seq), self.seq_len))

                num_peds_considered = 0
                _non_linear_ped = []

                # Iterate over every pedestrian found in this window
                for _, ped_id in enumerate(peds_in_curr_seq):
                    # Get the trajectory for this one pedestrian
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1]
                                                 == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    # Find the start and end frame indices within the window
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    # We only keep pedestrians that are
                    # present for the entire sequence length
                    if pad_end - pad_front != self.seq_len:
                        continue

                    # Get just the x, y coordinates and transpose
                    # Shape becomes (2, seq_len)
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq

                    # Calculate relative coordinates (velocities)
                    # rel_curr_ped_seq[t] = curr_ped_seq[t] - curr_ped_seq[t-1]
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                    # Get the index for this valid pedestrian
                    _idx = num_peds_considered

                    # Add the absolute and relative trajectories to the scene arrays
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq

                    # Check if the prediction part of the trajectory is non-linear
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))

                    # Mark all time-steps as valid (mask = 1)
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                # If the sequence (scene) has enough
                # valid pedestrians, keep it.
                if num_peds_considered >= min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    # Add the valid data, slicing off unused zero-rows
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

                    sequences_name_list.extend(
                        [scene_name] * num_peds_considered)

        # After processing all files, store the total number of valid scenes
        self.num_seq = len(seq_list)

        # Concatenate all data from all scenes into massive tensors
        # `seq_list` shape: (total_num_peds_in_dataset, 2, seq_len)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        self.sequences_name_list = sequences_name_list

        # Create global IDs for all pedestrians in this dataset:
        total_num_peds_in_dataset = seq_list.shape[0]

        # Convert all data from Numpy to Torch Tensors
        # Split the data into observation (input) and prediction (target)
        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(
            torch.float
        )
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(
            torch.float
        )
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(
            non_linear_ped).type(torch.float)

        # Create a "lookup table" (self.seq_start_end)
        # This list stores the (start, end) row indices in the master tensors
        # for each scene (sequence).
        # e.g., self.seq_start_end[0] -> (0, 10) (Scene 0 has peds 0-9)
        #       self.seq_start_end[1] -> (10, 15) (Scene 1 has peds 10-14)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        """Returns the number of sequences (scenes) in the dataset."""
        return self.num_seq

    def __getitem__(self, index):
        """
        Gets a single sequence (scene) from the dataset.

        Args:
            index: The index of the sequence (scene) to retrieve.

        Returns:
            A tuple of 6 tensors, containing all data for all pedestrians
            in the specified scene.
        """
        # Use the lookup table to find the slice for this scene
        start, end = self.seq_start_end[index]

        scene_embedding = None
        current_sequence_name = self.sequences_name_list[start]
        if self.sequences_embeddings_loaded and current_sequence_name in self.sequences_embeddings:
            scene_embedding = self.sequences_embeddings[current_sequence_name]

        else:
            if index == 0:
                print(
                    f"[DATASET WARNING] Embedding não encontrada para: {current_sequence_name}")

        if scene_embedding is None:
            dim = self.sequences_embeddings_dimension if self.sequences_embeddings_dimension > 0 else 768
            scene_embedding = torch.zeros(dim)

        if scene_embedding.device != torch.device('cpu'):
            scene_embedding = scene_embedding.cpu()

        num_peds = end - start
        scene_embedding = scene_embedding.unsqueeze(0).repeat(num_peds, 1)

        # Slice all master tensors to get the data for this scene
        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
            scene_embedding,
            [current_sequence_name] * num_peds
        ]

        return out
