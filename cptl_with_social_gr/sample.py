import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from args import get_all_args
from data.loader import data_dset, data_loader
from data.trajectories import SceneBatch
from generative_model.vae_models_scratch import CVAE
from helper.utils import get_dset_path, relative_to_abs

plt.style.use('ggplot')


def collapse_full_test(model, loader, args, dataset_name, num_samples=50):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"\n{'='*80}")
    print(f" SUPERVISOR VAE COLLAPSE FULL DIAGNOSTIC: Dataset {dataset_name}")
    print(f"{'='*80}")

    # =========================================================================
    # PART 1: ACTIVE UNITS CALCULATION (Burda et al. Metric) - OVER ENTIRE DATASET
    # =========================================================================
    print("\n[Running Part 1: Calculating Active Units across the entire dataset...]")
    all_mus = []

    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Encoding dataset"):
            batch = SceneBatch(batch_data, device=device)
            x_rel = batch.obs_traj_rel

            # Sequence Start End logic
            seq_start_end = batch.seq_start_end
            if seq_start_end.dim() == 2 and seq_start_end.shape[1] > 2:
                seq_start_end = seq_start_end[:, :2]

            real_emb = getattr(batch, 'sequence_embeddings', None)

            # Encoder Forward
            traj_lstm_h_t, traj_lstm_c_t = model.init_obs_traj_lstm(
                x_rel.shape[1])
            traj_lstm_hidden_states = []
            for input_t in x_rel.chunk(x_rel.size(0), dim=0):
                traj_lstm_h_t, traj_lstm_c_t = model.encoder.traj_lstm_model_encoder(
                    input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
                )
                traj_lstm_hidden_states += [traj_lstm_h_t]

            final_encoder_h = traj_lstm_hidden_states[-1]
            end_pos = x_rel[-1]
            pool_h = model.decoder.pool_net(
                final_encoder_h, seq_start_end, end_pos)
            vae_input = torch.cat([final_encoder_h, pool_h], dim=1)

            mu, _, _ = model.encoder(vae_input, real_emb)
            all_mus.append(mu.cpu().numpy())

    mus_dataset = np.concatenate(all_mus, axis=0)  # [Total_Samples, z_dim]
    mus_variance = np.var(mus_dataset, axis=0)    # Var_x(mu_i(x))

    # Threshold defined by Burda et al. (2015)
    active_units_mask = mus_variance > 0.01
    num_active_units = np.sum(active_units_mask)
    percent_active = (num_active_units / args.z_dim) * 100

    print("\n" + "-" * 80)
    print(" PART 1 RESULTS: ACTIVE UNITS CALCULATION")
    print(" Metric: Var_x(mu_i(x)) > 0.01 across the dataset.")
    print(
        f" -> Active Units: {num_active_units} / {args.z_dim} ({percent_active:.2f}%)")
    if percent_active < 5.0:
        print(
            "    [!] DIAGNOSTIC: Severe dimensional collapse. Most units are dead.")
    else:
        print("    [✓] DIAGNOSTIC: Latent space has active channels.")
    print("-" * 80)

    # =========================================================================
    # PREPARATION FOR PARTS 2 & 3: ISOLATE SINGLE PEDESTRIAN
    # =========================================================================
    iterator = iter(loader)
    first_batch_data = next(iterator)
    batch = SceneBatch(first_batch_data, device=device)

    ped_idx = 0

    x_rel = batch.obs_traj_rel[:, ped_idx:ped_idx+1, :]
    start_pos = batch.obs_traj[-1, ped_idx:ped_idx+1, :]
    start_pos_expanded = start_pos.repeat(num_samples, 1)

    dummy_seq_start_end = torch.tensor(
        [[i, i+1] for i in range(num_samples)], device=device, dtype=torch.long)
    single_seq_start_end = torch.tensor(
        [[0, 1]], device=device, dtype=torch.long)

    real_emb = getattr(batch, 'sequence_embeddings', None)
    if real_emb is not None:
        real_emb = real_emb[ped_idx:ped_idx+1]
        real_emb_expanded = real_emb.repeat(num_samples, 1)
    else:
        real_emb_expanded = None

    with torch.no_grad():
        # ENCODE ISOLATED PEDESTRIAN
        traj_lstm_h_t, traj_lstm_c_t = model.init_obs_traj_lstm(1)
        traj_lstm_hidden_states = []
        for input_t in x_rel.chunk(x_rel.size(0), dim=0):
            traj_lstm_h_t, traj_lstm_c_t = model.encoder.traj_lstm_model_encoder(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_hidden_states += [traj_lstm_h_t]

        final_encoder_h = traj_lstm_hidden_states[-1]
        end_pos = x_rel[-1]
        pool_h = model.decoder.pool_net(
            final_encoder_h, single_seq_start_end, end_pos)
        vae_input = torch.cat([final_encoder_h, pool_h], dim=1)

        mu, logvar, _ = model.encoder(vae_input, real_emb)
        std = torch.exp(0.5 * logvar)

        # =========================================================================
        # PART 2: DATA GENERATING PROCESSES (PRIOR VS POSTERIOR)
        # =========================================================================
        z_prior = torch.randn(num_samples, args.z_dim).to(device)
        recon_prior_rel = model.decoder(z_prior, dummy_seq_start_end, x_rel.repeat(
            1, num_samples, 1), real_emb_expanded)
        preds_prior = relative_to_abs(
            recon_prior_rel, start_pos_expanded).cpu().numpy()

        eps = torch.randn(num_samples, args.z_dim).to(device)
        z_posterior = mu + eps * std
        recon_post_rel = model.decoder(z_posterior, dummy_seq_start_end, x_rel.repeat(
            1, num_samples, 1), real_emb_expanded)
        preds_post = relative_to_abs(
            recon_post_rel, start_pos_expanded).cpu().numpy()

        # =========================================================================
        # PART 3: LATENT SENSITIVITY ANALYSIS
        # =========================================================================
        base_output_rel = model.decoder(
            mu, single_seq_start_end, x_rel, real_emb)
        massive_noise = torch.ones_like(mu) * 10.0
        z_perturbed = mu + massive_noise
        pert_output_rel = model.decoder(
            z_perturbed, single_seq_start_end, x_rel, real_emb)

        max_diff = torch.max(
            torch.abs(base_output_rel - pert_output_rel)).item()

    # Calculate diversity
    def calc_diversity(preds):
        final_pos = preds[-1, :, :]
        return (np.std(final_pos[:, 0]) + np.std(final_pos[:, 1])) / 2

    std_prior = calc_diversity(preds_prior)
    std_post = calc_diversity(preds_post)

    print("\n" + "-" * 80)
    print(" PART 2 RESULTS: GENERATING PROCESSES COMPARISON (Variance)")
    print(f" -> Analyzed pedestrian: Index {ped_idx} of the first batch.")
    print(
        f" 1. PRIOR Variance (z ~ N(0,1)):                    {std_prior:.4f} meters")
    print(
        f" 2. POSTERIOR Variance (z ~ N(mu, std)):            {std_post:.4f} meters")

    if std_prior < 0.05 and std_post < 0.05:
        print(
            "    [!] DIAGNOSTIC: Deterministic output. No multimodal variance generated.")
    elif std_prior > std_post:
        print(
            "    [✓] DIAGNOSTIC: Information Gain observed. Variance dropped from Prior to Posterior.")

    print("\n" + "-" * 80)
    print(" PART 3 RESULTS: LATENT SENSITIVITY TEST")
    print(" Action: Added +10.0 to all Z dimensions.")
    print(
        f" -> Max trajectory difference after perturbation:   {max_diff:.6f} meters")

    if max_diff < 0.01:
        print("    [!] DIAGNOSTIC: POSTERIOR COLLAPSE CONFIRMED.")
        print("        The Decoder ignored a massive change to Z. It relies entirely on autoregression.")
    else:
        print("    [✓] DIAGNOSTIC: Z IS ACTIVE.")
        print("        The output changed significantly, proving the Decoder respects the latent bottleneck.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    args = get_all_args(load_yaml=False)
    if not os.path.exists(args.r_dir):
        os.makedirs(args.r_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # YOUR PATH HERE:
    #model_path = "/home/matheus/LLM4CPTL/cptl_with_social_gr/results/CL_SGR_continual_learning_generative_4_generativeModelCheckpoint_['ETH', 'UCY', 'inD', 'INTERACTION']_64_200_72_False_current_False_None.path"
    #model_path = "/home/matheus/LLM4CPTL/cptl_with_social_gr/results/CL_SGReKLAN_continual_learning_generative_4_generativeModelCheckpoint_['ETH', 'UCY', 'inD', 'INTERACTION']_64_200_72_False_current_False_None.path"
    model_path = "/home/matheus/LLM4CPTL/cptl_with_social_gr/results/CL_SGReKLANeLLM_continual_learning_generative_4_generativeModelCheckpoint_['ETH', 'UCY', 'inD', 'INTERACTION']_64_200_72_False_current_False_None.path"
    
    model = CVAE(
        obs_len=args.obs_len, pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        traj_lstm_output_size=args.traj_lstm_output_size,
        dropout=args.dropout, z_dim=args.z_dim,
        embedding_dim=args.embedding_dim, mlp_dim=args.mlp_dim,
        bottleneck_dim=args.bottleneck_dim, activation='relu', batch_norm=True,
        adapt_architecture_to_include_sequence_embedding=args.adapt_architecture_to_include_sequence_embedding,
        sequence_embedding_dimension=args.dimensions,
        sequence_embedding_compressed_dimension=args.sequence_embedding_compressed_dimension,
        use_gradient_clipping=args.use_gradient_clipping,
        clip_gradient_max_norm=args.clip_gradient_max_norm,
        use_skip_connection=args.use_skip_connection
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Checkpoint not found. Please check the path.")
        exit()

    for dataset_name in ["ETH", "UCY", "inD", "INTERACTION"]:
        dset_path = get_dset_path(dataset_name, "test")
        
        dset = data_dset(args, dset_path, dataset_name=dataset_name, split_name="test")
        loader = data_loader(args, dset)

        collapse_full_test(model, loader, args, dataset_name)
