#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################
## Imports and packages
###################################

import math
import torch
import torch.nn as nn

from generative_model.linear_nets import fc_layer,fc_layer_split

from helper.replayer import Replayer
from helper.utils import relative_to_abs

###################################
## Functions
###################################
def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    """
    Construct a simple feed-forward MLP.

    Args:
        dim_list (list[int]): List of layer sizes. Each pair defines one Linear layer.
        activation (str): Non-linearity ('relu' or 'leakyrelu').
        batch_norm (bool): Whether to include BatchNorm1d.
        dropout (float): Dropout probability.

    Returns:
        nn.Sequential: The constructed MLP.
    """
    layers = []
    
    # Iterate through consecutive layer sizes to create Linear layers
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
            
    # Return as a single nn.Sequential module
    return nn.Sequential(*layers)

###################################
## Classes
###################################
class PositionalEncoding(nn.Module):
    """Codificacao Posicional Sinusoidal para o Encoder."""
    def __init__(self, seq_len: int, positional_encoding_dim: int):
        super().__init__()
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, positional_encoding_dim, 2).float() * (-math.log(10000.0) / positional_encoding_dim))
        pe = torch.zeros(seq_len, 1, positional_encoding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adiciona codificacao posicional. Input: (T, B, D)."""
        x = x + self.pe[:x.size(0), :].to(x.device)
        return x
    
class PoolHiddenNet(nn.Module):
    """
    Social pooling module as proposed in Social-GAN.

    This module aggregates pairwise relative positions and hidden states
    within each sequence to produce pooled social context.

    Args:
        embedding_dim (int): Dimension of spatial embedding for relative positions.
        h_dim (int): Dimension of hidden state.
        mlp_dim (int): Hidden size for the MLP.
        bottleneck_dim (int): Output dimension for pooled features.
        activation (str): Activation type ('relu', 'leakyrelu').
        batch_norm (bool): Enable batch normalization.
        dropout (float): Dropout probability.
    """
    def __init__(
            self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
            activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        # Save hyperparameters
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        # Input dimension for the MLP: embedding of relative position + hidden state
        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, self.mlp_dim, bottleneck_dim]

        # Linear layer to embed relative 2D positions
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        
        # MLP that combines embedded positions with hidden states
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Repeat each row `num_reps` times.

        Args:
            tensor (torch.Tensor): Input tensor of shape (N, D).
            num_reps (int): Number of repetitions.

        Returns:
            torch.Tensor: Repeated tensor of shape (N*num_reps, D).
        """
        col_len = tensor.size(1)
        # Add extra dimension and repeat:
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        # Flatten back to 2D:
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Compute socially pooled hidden representations.

        Args:
            h_states (torch.Tensor): Hidden states of shape (num_layers, batch, h_dim)
                                     or (batch, h_dim) after flattening.
            seq_start_end (list[tuple]): Sequence segment boundaries.
            end_pos (torch.Tensor): Final positions (batch, 2).

        Returns:
            torch.Tensor: Pooled features of shape (batch, bottleneck_dim).
        """
        pool_h = []
        
        # Loop over each sequence in the batch
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            
            # Number of pedestrians in this sequence:
            num_ped = end - start
            
            # Social pooling is only meaningful with >1 agent:
            if num_ped > 1:
                curr_hidden = h_states.view(-1, self.h_dim)[start:end]
                curr_end_pos = end_pos[start:end]
                
                # Repeat hidden states and positions to compute pairwise differences:
                curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
                curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
                curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
                
                # Compute relative positions between all pairs:
                curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
                curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
                
                # Concatenate hidden states and relative embeddings:
                mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
                
                # Pass through MLP to compute pooled features:
                curr_pool_h = self.mlp_pre_pool(mlp_h_input)
                
                # Reshape and take max over neighbors:
                curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
                
            else:
                # If only 1 agent, pooling is just its hidden state
                curr_hidden = h_states.view(-1, self.h_dim)[start:end]
                curr_pool_h = curr_hidden
                
            pool_h.append(curr_pool_h)
        
        # Concatenate pooled features for all sequences:
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h

class AutoEncoder(Replayer):
    """
    Variational Autoencoder (VAE) for trajectory modeling with social pooling.

    This class:
    - Encodes observed trajectories via LSTM
    - Applies social pooling to hidden states
    - Computes latent distribution parameters (mean, logvar)
    - Reparameterizes z
    - Decodes future trajectories conditioned on z

    Args:
        obs_len (int): Length of observed trajectory.
        pred_len (int): Length of predicted trajectory.
        traj_lstm_input_size (int): LSTM input dimension.
        traj_lstm_hidden_size (int): LSTM hidden state size.
        traj_lstm_output_size (int): Output size for decoder LSTM.
        dropout (float): Dropout probability.
        z_dim (int): Latent dimension.
        embedding_dim (int): Embedding dimension for pooling.
        mlp_dim (int): Hidden dim of pooling MLP.
        bottleneck_dim (int): Final dim for pooled features.
        activation (str): Activation type.
        batch_norm (bool): Enable batch normalization.
    """

    def __init__(
            self,
            obs_len=12,
            pred_len=8,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=124,
            traj_lstm_output_size=2,
            dropout=0,
            z_dim=200,
            embedding_dim=32,
            mlp_dim=256,
            bottleneck_dim=32,
            activation='relu',
            batch_norm=True,
            ####### LG-Traj #####
            adapt_architecture_to_include_llm=False,
            llm_motion_cues_embedding_dim=200,
            llm_motion_cues_zm_dim=32,
            
            adapt_architecture_to_include_zp=False,
            zp_dim=32,
            
            adapt_architecture_to_include_positional_encoding=False,
            positional_encoding_dim=32
            #####################
    ):
        super().__init__()
        
        # Model configuration:
        self.label = "VAE"
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_lstm_input_size = traj_lstm_input_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_output_size = traj_lstm_output_size
        self.z_dim = z_dim
        
        ###################
        self.adapt_architecture_to_include_llm = adapt_architecture_to_include_llm
        self.llm_motion_cues_embedding_dim = llm_motion_cues_embedding_dim
        self.zm_dim = llm_motion_cues_zm_dim
        
        self.adapt_architecture_to_include_zp = adapt_architecture_to_include_zp
        self.zp_dim = zp_dim
        
        self.adapt_architecture_to_include_positional_encoding = adapt_architecture_to_include_positional_encoding
        self.positional_encoding_dim = positional_encoding_dim
        ###################

        # Weights of different components of the loss function
        self.lamda_rcl = 1.
        self.lamda_vl = 1.
        self.lamda_pl = 0.

        # Averaging mode for losses, makes that [reconL] and [variatL] are both divided by number of iput-pixels
        self.average = "average"

        # Pooling configurations
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.bottleneck_dim = bottleneck_dim
        
        ####### LG-Traj #####
        if self.adapt_architecture_to_include_llm:
            # Fm: Linear Layer para Motion Cues (T -> Zm)
            self.Fm = make_mlp(
                [llm_motion_cues_embedding_dim, 128, self.zm_dim],
                activation='relu', # 
                batch_norm=True,   # Recomendado para estabilidade
                dropout=0
            )
            
        if self.adapt_architecture_to_include_zp:
            obs_input_dim = self.obs_len * self.traj_lstm_input_size
            
            self.Fp = make_mlp(
                [obs_input_dim, 128, self.zp_dim],
                activation='relu',
                batch_norm=True,
                dropout=0
            )
        
        if self.adapt_architecture_to_include_positional_encoding:
            self.positional_encoder = PositionalEncoding(self.obs_len, max_len=self.positional_encoding_dim)
        #####################

        ################################
        # Define network layers
        ################################

        # Encoder LSTM (=q[z|x])
        self.traj_lstm_model_encoder = nn.LSTMCell(traj_lstm_input_size + self.positional_encoding_dim if self.adapt_architecture_to_include_positional_encoding else traj_lstm_input_size, traj_lstm_hidden_size)
        
        # Fully connected layer for latent embedding
        encoder_input_dim = traj_lstm_hidden_size + bottleneck_dim
        
        if adapt_architecture_to_include_llm:
            encoder_input_dim += llm_motion_cues_zm_dim
        
        if adapt_architecture_to_include_zp:
            encoder_input_dim += zp_dim

        self.fcE = fc_layer(encoder_input_dim, 128, batch_norm=None)

        # Split FC layer to mean and logvar
        self.toZ = fc_layer_split(128, z_dim, nl_mean='none', nl_logvar='none')

        # Decoder (=p[x|z])
        decoder_input_dim = z_dim
        
        if adapt_architecture_to_include_llm:
            decoder_input_dim += llm_motion_cues_zm_dim
            
        if adapt_architecture_to_include_zp:
            decoder_input_dim += zp_dim
            
        self.fromZ = fc_layer(decoder_input_dim, 128, batch_norm=None)
        
        self.fcD = fc_layer(128, traj_lstm_hidden_size, batch_norm=None)

        # Decoder LSTM for trajectory generation
        self.pred_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_output_size)
        self.pred_hidden2pos = nn.Linear(self.traj_lstm_output_size, 2)

        # Social pooling network
        self.pool_net = PoolHiddenNet(
            embedding_dim=self.embedding_dim,
            h_dim=traj_lstm_hidden_size,
            mlp_dim=mlp_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    ################################
    # Properties
    ################################
    @property
    def name(self):
        """str: Model name used internally."""
        return "{}".format("Generator --> VAE")
    
    ################################
    # Initialization or foward functions
    ################################
    def init_obs_traj_lstm(self, batch):
        """
        Initialize hidden states for the encoder LSTM.

        Args:
            batch (int): Batch size.

        Returns:
            tuple(torch.Tensor, torch.Tensor): (h0, c0)
        """
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_pred_traj_lstm(self, batch):
        """
        Initialize hidden states for the decoder LSTM.

        Args:
            batch (int): Batch size.

        Returns:
            tuple(torch.Tensor, torch.Tensor): (h0, c0)
        """
        return (
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
        )

    ################################
    # Encode / Reparameterize / Decode
    ################################
    # Pass input through feed-forward connections, to get [hE], [z_mean] and [z_logvar]
    def encode(self, x):
        """
        Encode hidden input into latent mean/logvar.

        Args:
            x (torch.Tensor): Input hidden states.

        Returns:
            tuple: (z_mean, z_logvar, hidden_features)
        """
        # extract final hidden features (forward-pass)
        hE = self.fcE(x)
        # get parameters for reparametrization
        (z_mean, z_logvar) = self.toZ(hE)
        return z_mean, z_logvar, hE

    # Perform "reparametrization trick" to make these stochastic variables differentiable
    def reparameterize(self, mu, logvar):
        """
        Perform the reparameterization trick.

        Args:
            mu (torch.Tensor): Mean.
            logvar (torch.Tensor): Log-variance.

        Returns:
            torch.Tensor: Sampled z.
        """
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decode(self, z, seq_start_end, obs_traj_pos=None, Zm=None, Zp=None):
        """
        Decode latent vector into predicted trajectory.

        Args:
            z (torch.Tensor): Latent vector (batch, z_dim).
            seq_start_end (list): Social pooling boundaries.
            obs_traj_pos (torch.Tensor): Observed trajectory.

        Returns:
            torch.Tensor: Predicted future trajectory (seq_len, batch, 2).
        """
        ####### LG-Traj #####
        z_conditioned = [z]
        if self.adapt_architecture_to_include_llm and Zm is not None:
            z_conditioned.append(Zm)
        if self.adapt_architecture_to_include_zp and Zp is not None:
            z_conditioned.append(Zp)
            
        z_conditioned = torch.cat(z_conditioned, dim=1)
        #####################
        
        hD = self.fromZ(z_conditioned)
        hidden_features = self.fcD(hD)
        pred_lstm_h_t = hidden_features
        pred_lstm_c_t = torch.zeros_like(pred_lstm_h_t).cuda()
        pred_traj_pos = []

        output = obs_traj_pos[0]
        pred_lstm_h_t = self.pool_net(hidden_features, seq_start_end, output)
        pred_traj_pos += [output]
        
        # Loop over observed sequence to reconstruct past trajectory:
        for i in range(self.obs_len-1):
            pred_lstm_h_t, pred_lstm_c_t = self.pred_lstm_model(
                output, (pred_lstm_h_t, pred_lstm_c_t)
            )
            
            # Map hidden state to 2D:
            output = self.pred_hidden2pos(pred_lstm_h_t)
            pred_traj_pos += [output]
        
        # Stack into (seq_len, batch, 2):
        outputs = torch.stack(pred_traj_pos)
        return outputs

    ################################
    # Forward
    ################################
    def forward(self, obs_traj_pos, seq_start_end, llm_motion_cues_Tembedding=None):
        """
        Forward pass: encode → reparameterize → decode.

        Args:
            obs_traj_pos (torch.Tensor): Observed trajectories.
            seq_start_end (list): Sequence pooling boundaries.

        Returns:
            tuple: (reconstructed_traj, mu, logvar, z)
        """
        batch = obs_traj_pos.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_obs_traj_lstm(batch)
        traj_lstm_hidden_states = []
        
        # --- 1. Preparar PE (Condicional) ---
        pe_batch = None
        if self.adapt_architecture_to_include_positional_encoding:
            # Replicar PE (T_obs, 1, positional_encoding_dim) para (T_obs, Batch, positional_encoding_dim)
            pe_batch = self.positional_encoder.pe.repeat(1, batch, 1)

        # Encoder LSTM, calculate the past traj hidden states, similar with embedding
        for i, input_t in enumerate(
            obs_traj_pos[: self.obs_len].chunk(
                obs_traj_pos[: self.obs_len].size(0), dim=0
            )
        ):
            # Se PE estiver ativo, concatenar ao input_t
            if self.adapt_architecture_to_include_positional_encoding:
                curr_pe = pe_batch[i].unsqueeze(0)
                input_t = torch.cat([input_t, curr_pe], dim=2)
                
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model_encoder(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_hidden_states += [traj_lstm_h_t]
        
        # Encode (forward), reparameterize and decode (backward)
        final_encoder_h  = traj_lstm_hidden_states[-1]
        
        # Social pooling
        end_pos = obs_traj_pos[-1, :, :]
        pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
        
        ####### LG-Traj #####
        Zm = None
        if not self.adapt_architecture_to_include_llm or llm_motion_cues_Tembedding is None:
            llm_motion_cues_Tembedding = torch.zeros(batch, self.llm_motion_cues_embedding_dim).cuda()
        
        Zm = self.Fm(llm_motion_cues_Tembedding)
        
        # --- 3. Calculo de Zp (Condicional) ---
        Zp = None
        if self.adapt_architecture_to_include_zp:
            # Flatten: (T_obs, Batch, 2) -> (Batch, T_obs * 2)
            obs_flat = obs_traj_pos[:self.obs_len].permute(1, 0, 2).contiguous().view(batch, -1)
            Zp = self.Fp(obs_flat) # Shape: (Batch, zp_dim)
        
        # --- 4. Concatenar para o Encoder (fcE) ---
        concatenated_features = [final_encoder_h, pool_h]
        if self.adapt_architecture_to_include_llm: # Zm eh condicional a flag adapt_architecture_to_include_llm
            concatenated_features.append(Zm)
        if self.adapt_architecture_to_include_zp:
            concatenated_features.append(Zp) # Incluido Zp
        #####################
        
        # Construct input hidden states for decoder
        vae_input = torch.cat(concatenated_features, dim=1)
        
        mu, logvar, hE = self.encode(vae_input)
        z = self.reparameterize(mu, logvar)
        
        traj_recon = self.decode(z, seq_start_end, obs_traj_pos=obs_traj_pos, Zm=Zm, Zp=Zp)
        
        return (traj_recon, mu, logvar, z)

    ################################
    # Sampling functions
    ################################
    def sample(self, obs_traj_rel, obs_traj, replay_seq_start_end, t_embeddings=None):
        """
        Sample trajectories from the model.

        Args:
            obs_traj_rel (torch.Tensor): Relative observed traj.
            obs_traj (torch.Tensor): Absolute observed traj.
            replay_seq_start_end (list): Sequence boundaries.

        Returns:
            list: [absolute_traj, relative_traj, seq_start_end]
        """

        # set model to eval()-mode
        mode = self.training
        self.eval()
        obs_traj_rel = obs_traj_rel
        replay_seq_start_end = replay_seq_start_end
        size = obs_traj_rel.shape[1]

        # sample z
        z = torch.randn(size, self.z_dim).to(self._device())

        # decode z into traj x
        with torch.no_grad():
            ####### LG-Traj #####
            Zm = None
            if self.adapt_architecture_to_include_llm and t_embeddings is not None:
                Zm = self.Fm(t_embeddings)
            
            Zp = None
            if self.adapt_architecture_to_include_zp:
                # Calcular Zp a partir da trajetoria observada (obs_traj)
                # obs_traj_rel.shape: (T_obs, Batch, 2)
                obs_flat = obs_traj_rel[:self.obs_len].permute(1, 0, 2).contiguous().view(size, -1)
                Zp = self.Fp(obs_flat)
            #####################
            
            traj_rel = self.decode(z, replay_seq_start_end, obs_traj_pos=obs_traj_rel, Zm=Zm, Zp=Zp)

        # relative to absolute
        traj = relative_to_abs(traj_rel, obs_traj[0])

        # set model back to its initial mode
        self.train(mode=mode)
        replay_traj = [traj, traj_rel, replay_seq_start_end]
        # returen samples as [batch_size]x[traj_size] tensor
        return replay_traj

    ################################
    # Loss functions
    ################################
    def calculate_recon_loss(self, x, x_recon, mode=False):
        """
        Compute reconstruction loss for trajectory prediction.

        Args:
            x (torch.Tensor): Ground truth trajectories.
            x_recon (torch.Tensor): Reconstructed trajectories.
            mode (str): "sum", "average", or "raw".

        Returns:
            torch.Tensor: Loss value(s).
        """
        seq_len, batch, size = x.size()
        reconL = (x.permute(1,0,2) - x_recon.permute(1,0,2)) ** 2
        
        if mode == "sum":
            return torch.sum(reconL)
        elif mode == "average":
            return torch.sum(reconL) / (batch*seq_len*size)
        elif mode == "raw":
            return reconL.sum(dim=2).sum(dim=1)

    def calculate_variat_loss(self, mu, logvar):
        """
        Compute KL divergence between posterior and N(0, I).

        Args:
            mu (torch.Tensor): Posterior mean.
            logvar (torch.Tensor): Posterior log-variance.

        Returns:
            torch.Tensor: KL divergence for each sample.
        """
        variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        return variatL

    def loss_function(self, recon_x, x, y_hat= None, y_target=None, scores=None, mu=None, logvar=None):
        """
        Compute combined reconstruction + KL loss.

        Args:
            recon_x (torch.Tensor): Reconstructed trajectory.
            x (torch.Tensor): Ground truth.
            mu (torch.Tensor): Latent mean.
            logvar (torch.Tensor): Latent logvar.

        Returns:
            tuple: (reconstruction_loss, KL_loss)
        """

        ###---Reconstruction loss---###
        reconL = self.calculate_recon_loss(x=x, x_recon=recon_x, mode=self.average)  # -> possibly average over traj
        reconL = torch.mean(reconL)                                                     # -> average over batch

        ###--- Variational loss ----###
        if logvar is not None:
            variatL = self.calculate_variat_loss(mu=mu, logvar=logvar)
            variatL = torch.mean(variatL)                               # -> average over batch
            if self.average:
                pass
        else:
            variatL = torch.tensor(0., device=self._device())

        # Return a tuple of the calculated losses
        return reconL, variatL

    ################################
    # Training functions
    ################################
    def train_a_batch(self, x_rel, y_rel, seq_start_end, t_embeddings=None, x_=None, y_=None, seq_start_end_=None, t_embeddings_=None, rnt=0.5):
        """
        Train the model on one batch, optionally with replay.

        Args:
            x_rel (torch.Tensor): Past trajectories.
            y_rel (torch.Tensor): Future trajectories.
            seq_start_end (list): Social pooling boundaries.
            x_ (torch.Tensor or list): Replay past trajectories.
            y_ (torch.Tensor or list): Replay future trajectories.
            rnt (float): Relative new-task importance.

        Returns:
            dict: Various loss components.
        """

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        ##--(1)-- CURRENT DATA --##
        precision = 0.
        if x_rel is not None:

            # Run the model
            if not self.adapt_architecture_to_include_llm or t_embeddings is None:
                recon_batch, mu, logvar, z = self(x_rel, seq_start_end)
            else:
                recon_batch, mu, logvar, z = self(x_rel, seq_start_end, t_embeddings)

            # Calculate all losses
            reconL, variatL = self.loss_function(recon_x=recon_batch, x=x_rel, y_hat=None, y_target=None, mu=mu, logvar=logvar)

            # Weigh losses as requested
            loss_cur = self.lamda_rcl*reconL + self.lamda_vl*variatL

        ##--(2)-- REPLAYED DATA --##
        if x_ is not None:

            n_replays = len(y_) if (y_ is not None) else 1

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            reconL_r = [None]*n_replays
            variatL_r = [None]*n_replays
            predL_r = [None]*n_replays

            # Run model (if [x_] is not a list with separate replay per task)
            if (not type(x_)==list):
                x_temp_ = x_
                if t_embeddings_ is None:
                    recon_batch, mu, logvar, z = self(x_temp_, seq_start_end_)
                else:
                    recon_batch, mu, logvar, z = self(x_temp_, seq_start_end_, t_embeddings_)
            # Loop to perform each replay
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if (type(x_)==list):
                    x_temp_ = x_[replay_id]
                    recon_batch, mu, logvar, z = self(x_temp_, t_embeddings_)

                # Calculate all losses
                reconL_r[replay_id], variatL_r[replay_id] = self.loss_function(
                    recon_x=recon_batch, x=x_temp_, mu=mu, logvar=logvar
                )

                # Weigh losses as requested
                loss_replay[replay_id] = self.lamda_rcl*reconL_r[replay_id] + self.lamda_vl*variatL_r[replay_id]

        # Calculate total loss
        loss_replay = None if (x_ is None) else sum(loss_replay)/n_replays
        loss_total = loss_replay if (x_rel is None) else (loss_cur if x_ is None else rnt*loss_cur+(1-rnt)*loss_replay)

        # Backpropagate errors
        loss_total.backward()

        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'reconL': reconL.item() if x_rel is not None else 0,
            'variatL': variatL.item() if x_rel is not None else 0,
            'reconL_r': sum(reconL_r).item()/n_replays if x_ is not None else 0,
            'variatL_r': sum(variatL_r).item()/n_replays if x_ is not None else 0,
        }
