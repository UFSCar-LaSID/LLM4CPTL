#!/usr/bin/env python3

###################################
# Imports and packages
###################################
import torch
import torch.nn as nn

from helper.replayer import Replayer
from helper.utils import relative_to_abs

from generative_model.linear_nets import fc_layer, fc_layer_split

###################################
# Functions
###################################
def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    """
    Creates a Multi-Layer Perceptron (MLP) from a list of dimensions.

    Args:
    - dim_list (list of int): List of integers representing the number of units in each layer (e.g., [64, 128, 256]).
    - activation (str): Activation function to be used. Options are 'relu' and 'leakyrelu'.
    - batch_norm (bool): Whether to apply Batch Normalization after each linear layer.
    - dropout (float): Dropout rate applied after each layer.

    Returns:
    - nn.Sequential: A sequential model of the MLP.
    """
    layers = []

    # Iterate through input/output dimensions of consecutive layers
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
    return nn.Sequential(*layers)


class PoolHiddenNet(nn.Module):
    '''
    Pooling module as proposed in Social-GAN (Yadlapalli et al., 2018).
    It aggregates information from neighboring agents based on their relative positions.
    '''

    def __init__(self,
                 embedding_dim=64,
                 h_dim=64,
                 mlp_dim=1024,
                 bottleneck_dim=1024,
                 activation='relu',
                 batch_norm=True,
                 dropout=0.0
                 ):
        """
        Initialize the PoolHiddenNet module.

        Args:
        - embedding_dim (int): Dimension for embedding the relative spatial coordinates (P_i - P_j).
        - h_dim (int): Hidden state dimension of the agents' LSTM (input to pooling).
        - mlp_dim (int): Dimension of the intermediate layer in the pre-pool MLP.
        - bottleneck_dim (int): Output dimension of the pooling module (the pooled social feature vector).
        """
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        # Input dimension for the MLP before pooling: Hidden State (H_i) + Relative Embedding (Emb(P_i - P_j))
        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, self.mlp_dim, bottleneck_dim]

        # Spatial embedding layer to embed relative positions (2D coordinates)
        self.spatial_embedding = nn.Linear(2, embedding_dim)

        # MLP for preprocessing before pooling
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)
        
        #slf.solo_projection = nn.Linear(h_dim, bottleneck_dim)

    def repeat(self, tensor, num_reps):
        """
        Helper function: Repeats each row of the input tensor `num_reps` times.
        Used to create the P_j part of the pairwise difference (P_i - P_j).

        Args:
        - tensor (torch.Tensor): Input 2D tensor of shape (batch, features).
        - num_reps (int): Number of repetitions for each row.

        Returns:
        - torch.Tensor: Repeated tensor of shape (batch * num_reps, features).
        """
        col_len = tensor.size(1)

        # Add a dimension, repeat, then flatten
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Forward pass for pooling hidden states (calculates the social feature for each agent).

        Args:
        - h_states (torch.Tensor): Tensor of shape (num_layers, batch, h_dim) containing the final hidden states.
        - seq_start_end (list of tuples): Defines the start and end indices for each sequence (scene/frame group).
        - end_pos (torch.Tensor): Tensor of shape (batch, 2) containing the last observed position of each agent.

        Returns:
        - torch.Tensor: Pooled social hidden state of shape (batch, bottleneck_dim).
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            if num_ped > 1:
                curr_hidden = h_states.view(-1, self.h_dim)[start:end]
                curr_end_pos = end_pos[start:end]
                # Repeat -> H1, H2, H1, H2
                curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
                # Repeat position -> P1, P2, P1, P2
                curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
                # Repeat position -> P1, P1, P2, P2
                curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
                curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
                curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
                mlp_h_input = torch.cat(
                    [curr_rel_embedding, curr_hidden_1], dim=1)
                curr_pool_h = self.mlp_pre_pool(mlp_h_input)
                curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            else:
                curr_hidden = h_states.view(-1, self.h_dim)[start:end]
                curr_pool_h = curr_hidden
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class VAEEncoder(nn.Module):
    """ 
    The VAE Encoder part: maps the combined trajectory and social features to the
    parameters (mean and log-variance) of the latent distribution q(z|x).
    """

    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 traj_lstm_input_size: int,
                 traj_lstm_hidden_size: int,
                 z_dim: int,
                 mlp_dim: int,
                 bottleneck_dim: int,
                 adapt_architecture_to_include_sequence_embedding: bool,
                 sequence_embedding_dimension: int,
                 sequence_embedding_compressed_dimension: int
                 ):
        """ Init class.

        Parameters
        ----------
        z_dim: int
            the latent dimension (size of the z vector).
        bottleneck_dim: int
            the dimension of the social pooling feature.
        """
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_lstm_input_size = traj_lstm_input_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.z_dim = z_dim
        self.mlp_dim = mlp_dim
        self.bottleneck_dim = bottleneck_dim
        
        self.adapt_architecture_to_include_sequence_embedding = adapt_architecture_to_include_sequence_embedding
        self.sequence_embedding_compressed_dimension = sequence_embedding_compressed_dimension

        # Trajectory encoder (LSTMCell processes the observed trajectory)
        self.traj_lstm_model_encoder = nn.LSTMCell(
            traj_lstm_input_size, traj_lstm_hidden_size)

        fc_input_dim = traj_lstm_hidden_size + bottleneck_dim
        
        if self.adapt_architecture_to_include_sequence_embedding:
            # If using sequence embeddings, adjust the input dimension accordingly
            fc_input_dim += sequence_embedding_compressed_dimension
            
            # Compression layer for the sequence embedding
            self.sequence_embedding_compression = fc_layer(
                sequence_embedding_dimension,
                sequence_embedding_compressed_dimension,
                batch_norm=None
            )
            
        # VAE encoder: FC layer for the combined feature vector (LSTM hidden state + Pooled feature)
        self.fcE = fc_layer(fc_input_dim, 128, batch_norm=None)

        # Layer to generate mean (mu) and log-variance (logvar) for the latent variable z
        self.toZ = fc_layer_split(128, z_dim, nl_mean='none', nl_logvar='none')

    def forward(self, x, sequence_embedding):
        """
        Args:
            x (torch.Tensor): Concatenated tensor [LSTM_H_final || Pool_H] 
            sequence_embedding (torch.Tensor, optional): Sequence embedding tensor.

        Returns:
            z_mean (torch.Tensor): Mean vector mu.
            z_logvar (torch.Tensor): Log-variance vector log(sigma^2).
            hE (torch.Tensor): Intermediate feature vector before splitting to mu/logvar.
        """
        if self.adapt_architecture_to_include_sequence_embedding:
            if sequence_embedding is not None:
                # Project: (Batch, 768) -> (Batch, 32)
                sequence_embedding_compressed = self.sequence_embedding_compression(sequence_embedding)
                # Concat: [LSTM || Pool || Text]
                x = torch.cat([x, sequence_embedding_compressed], dim=1)
            else:
                batch_size = x.shape[0]
                zeros = torch.zeros(
                    batch_size, self.sequence_embedding_compressed_dimension).to(x.device)
                x = torch.cat([x, zeros], dim=1)

        # Extract final hidden features (forward-pass)
        hE = self.fcE(x)
        # Get parameters for reparametrization
        (z_mean, z_logvar) = self.toZ(hE)
        return z_mean, z_logvar, hE


class VAEDecoder(nn.Module):
    """ 
    The VAE Decoder part: generates the reconstructed (or predicted) trajectory 
    from a sample in the latent space (z).
    """

    def __init__(
        self,
        obs_len: int,
        embedding_dim: int,
        dropout: float,
        z_dim: int,
        traj_lstm_input_size: int,
        traj_lstm_hidden_size: int,
        traj_lstm_output_size: int,
        mlp_dim: int,
        bottleneck_dim: int,
        activation: str,
        batch_norm: bool,
        adapt_architecture_to_include_sequence_embedding: bool,
        sequence_embedding_dimension: int,
        sequence_embedding_compressed_dimension: int,
        use_skip_connection: bool
    ):
        """ Init class.

        Parameters
        ----------
        z_dim: int
            the latent size.
        """
        super().__init__()

        self.obs_len = obs_len
        self.adapt_architecture_to_include_sequence_embedding = adapt_architecture_to_include_sequence_embedding
        self.sequence_embedding_compressed_dimension = sequence_embedding_compressed_dimension
        self.use_skip_connection = use_skip_connection
        
        from_z_input_dim = z_dim

        if self.adapt_architecture_to_include_sequence_embedding:
            from_z_input_dim += sequence_embedding_compressed_dimension

            # Projector independente do encoder
            self.sequence_embedding_compression = fc_layer(
                sequence_embedding_dimension,
                sequence_embedding_compressed_dimension,
                batch_norm=None
            )

        # Map latent vector z to an intermediate hidden state
        self.fromZ = fc_layer(from_z_input_dim, 128, batch_norm=None)
        self.fcD = fc_layer(128, traj_lstm_hidden_size, batch_norm=None)

        if self.use_skip_connection:
            # Se Skip for True: A entrada da LSTM é [Posição Relativa (2) + Vetor de Contexto (from_z_input_dim)]
            lstm_input_dim = traj_lstm_input_size + from_z_input_dim
        else:
            # Se Skip for False: A entrada é apenas a Posição Relativa (2)
            lstm_input_dim = traj_lstm_input_size
            
        # Prediction LSTM (used for the trajectory rollout)
        # Note: input size is 2 (relative position), output size is traj_lstm_output_size (which is 2)
        self.pred_lstm_model = nn.LSTMCell(
            lstm_input_dim, traj_lstm_output_size)

        # Final linear layer to convert LSTM hidden state to 2D position output
        self.pred_hidden2pos = nn.Linear(traj_lstm_output_size, 2)

        # Social Pooling network (used again in the decoder to re-contextualize the initial state)
        self.pool_net = PoolHiddenNet(
            embedding_dim=embedding_dim,
            h_dim=traj_lstm_hidden_size,
            mlp_dim=mlp_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def forward(self, z, seq_start_end, obs_traj_pos=None, sequence_embedding=None):
        """
        Args:
            z (torch.Tensor): Sampled latent vector.
            seq_start_end (list): Sequence boundaries for pooling.
            obs_traj_pos (torch.Tensor): Full observed trajectory (used to get the initial input position).

        Returns:
            outputs (torch.Tensor): Reconstructed trajectory.
        """
        if self.adapt_architecture_to_include_sequence_embedding:
            if sequence_embedding is not None:
                sequence_embedding_compressed = self.sequence_embedding_compression(sequence_embedding)
                z_input = torch.cat([z, sequence_embedding_compressed], dim=1)
            else:
                batch_size = z.shape[0]
                zeros = torch.zeros(
                    batch_size, self.sequence_embedding_compressed_dimension).to(z.device)
                z_input = torch.cat([z, zeros], dim=1)
        else:
            # Comportamento Original: Apenas Z entra
            z_input = z
            
        # Map z to the initial hidden state (H_0)
        hD = self.fromZ(z_input)
        hidden_features = self.fcD(hD)

        # Initialize the cell state (C_0) to zero
        pred_lstm_h_t = hidden_features
        pred_lstm_c_t = torch.zeros_like(
            pred_lstm_h_t).to(pred_lstm_h_t.device)
        pred_traj_pos = []

        # Get the initial input position (first relative displacement)
        output = obs_traj_pos[0]

        # Apply pooling to the initial hidden state to incorporate social context before starting prediction
        # The pooling uses the initial position 'output' (which is the first relative displacement, i.e., at time t=1)
        pred_lstm_h_t = self.pool_net(hidden_features, seq_start_end, output)

        # Add the initial input (t=1 position) to the reconstructed trajectory list
        pred_traj_pos += [output]

        # Trajectory Rollout: loop for obs_len - 1 steps (since the first step is handled above)
        for i in range(self.obs_len-1):
            if self.use_skip_connection:
                # Concatena a posição anterior com o vetor de contexto ETERNO
                lstm_input = torch.cat([output, z_input], dim=1)
            else:
                # Entrada padrão (apenas posição)
                lstm_input = output
                
            # LSTM step: current output position is the input to the next step
            pred_lstm_h_t, pred_lstm_c_t = self.pred_lstm_model(
                lstm_input, (pred_lstm_h_t, pred_lstm_c_t)
            )

            # Convert the new hidden state to the next position (relative displacement)
            output = self.pred_hidden2pos(pred_lstm_h_t)
            pred_traj_pos += [output]

        # Stack all relative displacements to form the reconstructed relative trajectory
        outputs = torch.stack(pred_traj_pos)

        return outputs


class CVAE(Replayer):
    """ 
    This is the core Conditional Variational Autoencoder (CVAE) model.
    It combines the LSTM-based trajectory encoder, Social Pooling, and the VAE framework.
    Inherits from Replayer, enabling Continual Learning/Experience Replay functionality.
    """

    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 traj_lstm_input_size: int,
                 traj_lstm_hidden_size: int,
                 traj_lstm_output_size: int,
                 dropout: float,
                 z_dim: int,
                 embedding_dim: int,
                 mlp_dim: int,
                 bottleneck_dim: int,
                 activation: str,
                 batch_norm: bool,
                 adapt_architecture_to_include_sequence_embedding: bool,
                 sequence_embedding_dimension: int,
                 sequence_embedding_compressed_dimension: int,
                 use_gradient_clipping: bool,
                 clip_gradient_max_norm: float,
                 use_skip_connection: bool
                 ):
        """ Init class.

        Parameters
        ----------
        z_dim: int
            the latent dimension.
        """
        super(CVAE, self).__init__()

        self.label = "VAE"

        self.obs_len = obs_len
        self.z_dim = z_dim
        self.traj_lstm_hidden_size = traj_lstm_hidden_size

        # Weights of different components of the loss function
        self.lamda_rcl = 1.
        self.lamda_vl = 1.
        self.lamda_pl = 0.

        # Flag for averaging losses (reconL and variatL) over input-pixels
        self.average = "average"
        
        self.adapt_architecture_to_include_sequence_embedding = adapt_architecture_to_include_sequence_embedding
        
        self.use_gradient_clipping = use_gradient_clipping
        self.clip_gradient_max_norm = clip_gradient_max_norm
        
        self.use_skip_connection = use_skip_connection

        # Initialize the VAE Encoder
        self.encoder = VAEEncoder(obs_len=obs_len,
                                  pred_len=pred_len,
                                  traj_lstm_input_size=traj_lstm_input_size,
                                  traj_lstm_hidden_size=traj_lstm_hidden_size,
                                  z_dim=z_dim,
                                  mlp_dim=mlp_dim,
                                  bottleneck_dim=bottleneck_dim,
                                  adapt_architecture_to_include_sequence_embedding=adapt_architecture_to_include_sequence_embedding,
                                  sequence_embedding_dimension=sequence_embedding_dimension,
                                  sequence_embedding_compressed_dimension=sequence_embedding_compressed_dimension
        )

        # Initialize the VAE Decoder
        self.decoder = VAEDecoder(obs_len=obs_len,
                                  embedding_dim=embedding_dim,
                                  dropout=dropout,
                                  z_dim=z_dim,
                                  traj_lstm_input_size=traj_lstm_input_size,
                                  traj_lstm_hidden_size=traj_lstm_hidden_size,
                                  traj_lstm_output_size=traj_lstm_output_size,
                                  mlp_dim=mlp_dim,
                                  bottleneck_dim=bottleneck_dim,
                                  activation=activation,
                                  batch_norm=batch_norm,
                                  adapt_architecture_to_include_sequence_embedding=adapt_architecture_to_include_sequence_embedding,
                                  sequence_embedding_dimension=sequence_embedding_dimension,
                                  sequence_embedding_compressed_dimension=sequence_embedding_compressed_dimension,
                                  use_skip_connection=use_skip_connection
        )

    @property
    def name(self):
        suffix = "+LLM-based sequence embedding" if self.adapt_architecture_to_include_sequence_embedding else ""
        return f"Generator --> VAE{suffix}"

    def init_obs_traj_lstm(self, batch):
        """ Initializes the hidden state (h_0) and cell state (c_0) for the trajectory LSTM. """
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    # Perform "reparametrization trick" to make these stochastic variables differentiable
    def reparameterize(self, mu, logvar):
        """
        Samples latent vector z from the distribution N(mu, exp(logvar)) using the reparameterization trick.
        z = mu + epsilon * std, where epsilon ~ N(0, I) and std = exp(0.5 * logvar).
        """
        # Calculate standard deviation (std)
        std = logvar.mul(0.5).exp_()

        # Sample epsilon from standard normal distribution
        eps = std.new(std.size()).normal_()

        # Calculate z
        return eps.mul(std).add_(mu)

    def forward(self, obs_traj_pos, seq_start_end, sequence_embedding=None):
        """
        Full forward pass of the CVAE: Encode trajectory, Pool social features, Encode to z, Decode trajectory.

        Args:
            obs_traj_pos (torch.Tensor): Observed relative trajectory (seq_len, batch_size, 2).
            seq_start_end (list): Sequence boundaries for pooling.

        Returns:
            (traj_recon, mu, logvar, z): Reconstructed trajectory and VAE parameters.
        """
        batch = obs_traj_pos.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_obs_traj_lstm(batch)
        traj_lstm_hidden_states = []

        # Calculate the past trajectory hidden states using the LSTM
        for i, input_t in enumerate(
            obs_traj_pos[: self.obs_len].chunk(
                obs_traj_pos[: self.obs_len].size(0), dim=0
            )
        ):
            # Process one time step
            traj_lstm_h_t, traj_lstm_c_t = self.encoder.traj_lstm_model_encoder(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_hidden_states += [traj_lstm_h_t]

        # Final hidden state and end position
        final_encoder_h = traj_lstm_hidden_states[-1]

        # The last observed position (relative to the first point of the trajectory)
        end_pos = obs_traj_pos[-1, :, :]

        # Social Pooling: aggregate context using the final hidden state and position
        pool_h = self.decoder.pool_net(final_encoder_h, seq_start_end, end_pos)

        # Construct input hidden states for VAE Encoder: Concatenate trajectory feature and social feature
        vae_input = torch.cat([final_encoder_h, pool_h], dim=1)

        # VAE Encoding: get mu and logvar
        mu, logvar, hE = self.encoder(vae_input, sequence_embedding)

        # Reparameterize: sample z
        z = self.reparameterize(mu, logvar)

        # VAE Decoding: reconstruct the trajectory
        traj_recon = self.decoder(
            z, seq_start_end, obs_traj_pos=obs_traj_pos, sequence_embedding=sequence_embedding)

        return (traj_recon, mu, logvar, z)

    def sample(self, obs_traj_rel, obs_traj, replay_seq_start_end, sequence_embedding):
        '''
        Generate [size] samples from the model (inference mode).

        Args:
            obs_traj_rel (torch.Tensor): Observed trajectory in relative coordinates.
            obs_traj (torch.Tensor): Observed trajectory in absolute coordinates (used for final conversion).
            replay_seq_start_end (list): Sequence boundaries.

        Returns:
            replay_traj (list): [Absolute_Predicted_Traj, Relative_Predicted_Traj, Sequence_Boundaries]
        '''
        # Set model to eval mode
        mode = self.training
        self.eval()

        size = obs_traj_rel.shape[1]

        # Sample z from the prior distribution N(0, I)
        z = torch.randn(size, self.z_dim).to(self._device())
        
        if sequence_embedding is not None:
            sequence_embedding = sequence_embedding.to(self._device())

        # Decode z into relative trajectory (traj_rel)
        with torch.no_grad():
            # In prediction, the decoder still needs the structure of obs_traj_rel (especially the first point)
            traj_rel = self.decoder(
                z, replay_seq_start_end, obs_traj_pos=obs_traj_rel, sequence_embedding=sequence_embedding)

        # Convert the predicted relative trajectory to absolute coordinates
        # It uses the first absolute position (obs_traj[0]) as the reference point
        traj = relative_to_abs(traj_rel, obs_traj[0])

        # Set model back to its initial mode
        self.train(mode=mode)
        replay_traj = [traj, traj_rel,
                       replay_seq_start_end, sequence_embedding]
        return replay_traj

    def calculate_recon_loss(self, x, x_recon, mode=False):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]         <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]   (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
                - [average]   <bool>, if True, loss is average over all frames; otherwise it is summed

        OUTPUT: - [reconL]    <1D-tensor> of length [batch_size]
        '''
        seq_len, batch, size = x.size()

        # Squared difference (x - x_recon)^2
        reconL = (x.permute(1, 0, 2) - x_recon.permute(1, 0, 2)) ** 2

        if mode == "sum":
            return torch.sum(reconL)
        elif mode == "average":
            # Average over all elements (batch * seq_len * size)
            return torch.sum(reconL) / (batch*seq_len*size)
        elif mode == "raw":
            # Sum over the 2D position (dim=2) and sequence length (dim=1) -> loss per agent (dim=0)
            return reconL.sum(dim=2).sum(dim=1)

    def calculate_variat_loss(self, mu, logvar):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [mu]      <2D-tensor> by encoder predicted mean for [z]
                - [logvar]  <2D-tensor> by encoder predicted logvar for [z]

        OUTPUT: - [variatL] <1D-tensor> of length [batch_size]
        '''
        # KL-Divergence formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        variatL_raw = -0.5 * torch.sum(1 + logvar -
                                   mu.pow(2) - logvar.exp(), dim=1)

        #min_variatL = torch.tensor(2.0).to(self._device())

        #return torch.max(variatL_raw, min_variatL)
        return variatL_raw

    def loss_function(self, recon_x, x, y_hat=None, y_target=None, scores=None, mu=None, logvar=None):
        '''Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:   - [recon_x]     <4D-tensor> reconstructed traj in same shape as [x]
                 - [x]           <4D-tensor> original traj
                 - [y_hat]       <2D-tensor> predicted traj
                 - [y_target]    <2D-tensor> future traj
                 - [mu]          <2D-tensor> with either [z] or the estimated mean of [z]
                 - [logvar]      None or <2D-tensor> with estimated log(SD^2) of [z]

        SETTING: - [self.average] <bool>, if True, both [reconL] and [variatL] are divided by number of input elements

        OUTPUT:  - [reconL]      reconstruction loss indicating how well [x] and [x_recon] match
                 - [variatL]     variational (KL-divergence) loss "indicating how normally distributed [z] is"
                 - [predL]       prediction loss indicating how well targets [y] are predicted
                 - [distilL]     knowledge distillation (KD) loss indicating how well the predicted "logits" ([y_hat])
                                    match the target "logits" ([scores])
        '''

        # Calculate reconstruction loss (L_Recon, L2 norm)
        reconL = self.calculate_recon_loss(
            x=x, x_recon=recon_x, mode=self.average)  # -> possibly average over traj

        # Average over the batch
        reconL = torch.mean(reconL)

        # Calculate variational loss (L_KL)
        if logvar is not None:
            # Calculate KL-Divergence
            variatL = self.calculate_variat_loss(mu=mu, logvar=logvar)

            # Average over the batch
            variatL = torch.mean(variatL)

        else:
            variatL = torch.tensor(0., device=self._device())

        # Prediction loss (L_pred) and Distillation loss (L_distil) are set to 0 in this VAE implementation
        # The structure is left for potential CVAE extensions (e.g., classifying motion) or the Replayer parent class.
        '''
        ###----Prediction loss----###
        if y_target is not None:
            predL = F.cross_entropy(y_hat, y_target, reduction='mean')  #-> average over batch
        else:
            predL = torch.tensor(0., device=self._device())
        '''

        # Return a tuple of the calculated losses
        return reconL, variatL

    def train_a_batch(self, x_rel, y_rel, seq_start_end, sequence_embedding=None, x_=None, y_=None, seq_start_end_=None, sequence_embedding_=None, rnt=0.5):
        '''
        Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_]).

        Args:
            x_rel (torch.Tensor): Current batch of observed relative trajectory.
            y_rel (torch.Tensor): Current batch of future relative trajectory (not directly used by CVAE but included for context).
            seq_start_end (list): Sequence boundaries for the current batch.
            x_ (torch.Tensor or list): Replayed batch of observed relative trajectory.
            y_ (torch.Tensor or list): Replayed batch of future relative trajectory.
            rnt (float): Relative importance of the new task (rnt=1: new task only, rnt=0: replay only).

        Returns:
            dict: Dictionary of calculated losses (current and replay).
        '''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        ## --(1)-- CURRENT DATA --##
        loss_cur = 0
        if x_rel is not None:
            print(
                f"                The tensor of observed relative trajectories for the CURRENT batch (shape: {x_rel.shape}) is not None, therefore, the 'CVAE.forward' method will be executed to reconstruct these trajectories")

            # Run the model (Forward pass)
            recon_batch, mu, logvar, z = self(
                x_rel, seq_start_end, sequence_embedding=sequence_embedding)

            # Calculate VAE losses
            print("                Calculating losses (reconstruction and variational) for the CURRENT batch using 'CVAE.loss_function' method")
            reconL, variatL = self.loss_function(
                recon_x=recon_batch, x=x_rel, y_hat=None, y_target=None, mu=mu, logvar=logvar)

            # Weigh current losses: L_current = lambda_rcl * L_Recon + lambda_vl * L_KL
            loss_cur = self.lamda_rcl * reconL + self.lamda_vl * variatL

        ## --(2)-- REPLAYED DATA --##
        loss_replay = 0
        if x_ is not None:
            print(
                f"                The tensor (or list) of observed relative trajectories for the REPLAYED batch (type: {type(x_)}, shape: {x_.shape}) is not None, therefore, the 'CVAE.forward' method will be executed to reconstruct these trajectories")

            # Calculate and weigh replay losses (L_replay) similar to L_current
            # n_replays = len(y_) if (y_ is not None) else 1
            n_replays = len(y_) if isinstance(y_, list) else 1

            # Prepare lists to store losses for each replay
            loss_replay = [None] * n_replays
            reconL_r = [None] * n_replays
            variatL_r = [None] * n_replays

            # Handle list of replays (one per past task) or single replay batch
            if not isinstance(x_, list):
                x_temp_ = x_
                recon_batch, mu, logvar, z = self(
                    x_temp_, seq_start_end_, sequence_embedding=sequence_embedding_)

            # Loop to perform each replay
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if isinstance(x_, list):
                    x_temp_ = x_[replay_id]
                    s_temp_ = seq_start_end_[replay_id] if isinstance(seq_start_end_, list) else seq_start_end_
                    
                    emb_temp_ = sequence_embedding_
                    if isinstance(sequence_embedding_, list) and len(sequence_embedding_) == n_replays:
                        emb_temp_ = sequence_embedding_[replay_id]
                    
                    recon_batch, mu, logvar, z = self(
                        x_temp_, s_temp_, sequence_embedding=emb_temp_)

                # Calculate all losses
                print("                Calculating losses (reconstruction_r and variational_r) for the REPLAYED batch using 'CVAE.loss_function' method")
                reconL_r[replay_id], variatL_r[replay_id] = self.loss_function(
                    recon_x=recon_batch, x=x_temp_, mu=mu, logvar=logvar
                )

                # Weigh losses as requested
                loss_replay[replay_id] = self.lamda_rcl * \
                    reconL_r[replay_id] + self.lamda_vl * variatL_r[replay_id]

            # Calculate final replay loss
            loss_replay = sum(loss_replay)/n_replays

        else:
            print("                The tensor (or list) of relative trajectories for the REPLAYED batch is None, therefore, no replay will be performed for this batch")

        if x_rel is not None and x_ is not None:
            loss_total = rnt * loss_cur + (1 - rnt) * loss_replay
        elif x_rel is not None:
            loss_total = loss_cur
        else:
            loss_total = loss_replay

        # Backpropagate errors
        loss_total.backward()
        
        if self.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=self.clip_gradient_max_norm
            )

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

    def evaluate_a_batch(self, x_rel, seq_start_end, sequence_embedding=None):
            '''
            Calculates loss for a validation batch without training the model.

            Args:
                x_rel (torch.Tensor): Validation batch of observed relative trajectory.
                seq_start_end (list): Sequence boundaries for the validation batch.

            Returns:
                dict: Dictionary of calculated validation losses.
            '''
            # 1. Colocar o modelo em modo de avaliação
            self.eval()

            # 2. Desabilitar cálculo de gradientes
            with torch.no_grad():
                # 3. Executar o Forward Pass
                # Nota: Na validação geralmente não usamos dados 'replay', apenas os dados atuais de validação
                recon_batch, mu, logvar, z = self(x_rel, seq_start_end, sequence_embedding=sequence_embedding)

                # 4. Calcular as perdas (Reconstrução e KL Divergence)
                reconL, variatL = self.loss_function(
                    recon_x=recon_batch, 
                    x=x_rel, 
                    y_hat=None, 
                    y_target=None, 
                    mu=mu, 
                    logvar=logvar
                )

                # Calcular perda total ponderada (usando os mesmos pesos do treino)
                loss_val = self.lamda_rcl * reconL + self.lamda_vl * variatL

            # 5. Retornar o modelo para o modo de treino (importante se você validar no meio de uma época)
            self.train()

            return {
                'loss_val': loss_val.item(),
                'reconL_val': reconL.item(),
                'variatL_val': variatL.item()
            }