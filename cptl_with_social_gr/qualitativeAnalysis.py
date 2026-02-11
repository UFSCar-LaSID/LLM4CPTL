import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from data.trajectories import SceneBatch
from args import get_all_args
from data.loader import data_dset, data_loader
from generative_model.vae_models_scratch import CVAE
from helper.utils import get_dset_path

# Configuração para plots
plt.style.use('ggplot')


def relative_to_abs(rel_traj, start_pos):
    """
    Converte trajetórias relativas (velocidades) para absolutas (metros).
    """
    displacement = torch.cumsum(rel_traj, dim=0)
    start_pos_exp = start_pos.unsqueeze(0)
    abs_traj = displacement + start_pos_exp
    return abs_traj


def analyze_latent_space_and_reconstruction(
    model,
    loaders_list,
    args,
    task_names=["ETH", "UCY", "inD", "INTERACTION"],
    num_samples_visualize=5
):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_mus = []
    all_labels = []

    print("\n" + "="*60)
    print("INICIANDO ANÁLISE QUALITATIVA (COM SHAREX/SHAREY)")
    print("="*60)

    # 1. COLETA DE DADOS
    with torch.no_grad():
        for task_idx, loader in enumerate(loaders_list):
            dataset_name = task_names[task_idx] if task_idx < len(
                task_names) else f"Task {task_idx}"
            print(f"-> Coletando latentes: {dataset_name}...")

            task_mus = []

            for batch_idx, batch in enumerate(loader):
                batch = SceneBatch(batch, device=device)

                x_rel = batch.obs_traj_rel

                # Higienização
                seq_start_end = batch.seq_start_end
                if seq_start_end.dim() == 2 and seq_start_end.shape[1] > 2:
                    seq_start_end = seq_start_end[:, :2]

                embedding = getattr(batch, 'sequence_embeddings', None)

                # Chamada do Modelo
                try:
                    if embedding is not None:
                        _, mu, _, _ = model(
                            x_rel, seq_start_end, sequence_embedding=embedding)
                    else:
                        _, mu, _, _ = model(x_rel, seq_start_end)
                except Exception as e:
                    print(f"[AVISO] Erro no Batch {batch_idx}: {e}")
                    continue

                task_mus.append(mu.cpu().numpy())

                if batch_idx == 0:
                    visualize_reconstruction_correct(
                        model, batch, args,
                        task_name=dataset_name,
                        num_samples=num_samples_visualize,
                        clean_seq_start_end=seq_start_end,
                        embedding=embedding
                    )

            if len(task_mus) > 0:
                task_mus = np.concatenate(task_mus, axis=0)
                all_mus.append(task_mus)
                all_labels.extend([dataset_name] * task_mus.shape[0])

    if not all_mus:
        print("ERRO: Nenhum dado coletado.")
        return

    X = np.concatenate(all_mus, axis=0)
    labels = np.array(all_labels)

    # 2. ANÁLISE ESTATÍSTICA
    print("\n--- ESTATÍSTICAS REAIS DO ESPAÇO LATENTE ---")
    mean_z = np.mean(X)
    std_z = np.std(X)

    print(f"Total de Amostras: {X.shape[0]}")
    print(f"Média Global (Ideal ~0.0):   {mean_z:.6f}")
    print(f"Desvio Padrão (Ideal ~1.0):  {std_z:.6f}")

    if std_z < 0.05:
        print("\n[ALERTA CRÍTICO] COLAPSO POSTERIOR DETECTADO!")
    elif std_z > 5.0:
        print("\n[ALERTA] EXPLOSÃO DE VARIÂNCIA DETECTADA!")
    else:
        print("\n[OK] Espaço latente saudável.")

    # 3. VISUALIZAÇÃO
    print("\nGerando gráficos de projeção...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")

    # --- PCA ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    for i, label in enumerate(unique_labels):
        indices = labels == label
        axes[0].scatter(
            X_pca[indices, 0], X_pca[indices, 1],
            label=label, alpha=0.6, s=20, color=cmap(i % 10)
        )

    circle = plt.Circle((0, 0), 2.0, color='red', fill=False,
                        linestyle='--', linewidth=2, label='Prior 2σ')
    axes[0].add_patch(circle)
    axes[0].set_title(f"PCA: Estrutura Global\n(Std: {std_z:.4f})")
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].legend()

    # --- t-SNE ---
    print("Calculando t-SNE...")
    limit = 3000
    if X.shape[0] > limit:
        idx_tsne = np.random.choice(X.shape[0], limit, replace=False)
        X_tsne_in = X[idx_tsne]
        l_tsne = labels[idx_tsne]
    else:
        X_tsne_in = X
        l_tsne = labels

    tsne = TSNE(n_components=2, perplexity=40, n_iter=1000,
                random_state=42, init='pca', learning_rate=200.0)
    X_tsne = tsne.fit_transform(X_tsne_in)

    for i, label in enumerate(unique_labels):
        mask = l_tsne == label
        if np.any(mask):
            axes[1].scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                label=label, alpha=0.6, s=20, color=cmap(i % 10)
            )

    axes[1].set_title("t-SNE: Agrupamento")
    axes[1].axis('off')
    axes[1].legend()

    try:
        plt.tight_layout()
    except RuntimeError:
        print("Aviso: tight_layout ignorado devido a conflito de eixos.")

    save_path = os.path.join(args.r_dir, 'latent_space_analysis_robust.png')
    plt.savefig(save_path, dpi=150)
    print(f"-> Gráfico salvo em: {save_path}")
    plt.close()


def visualize_reconstruction_correct(model, batch, args, task_name, num_samples=5, clean_seq_start_end=None, embedding=None):
    """
    Gera comparação Observação Real vs Observação Reconstruída em metros.
    """
    obs_traj = batch.obs_traj
    pred_traj_gt = batch.pred_traj
    obs_traj_rel = batch.obs_traj_rel

    seq_se = clean_seq_start_end if clean_seq_start_end is not None else batch.seq_start_end

    with torch.no_grad():
        if embedding is not None:
            recon_rel, _, _, _ = model(
                obs_traj_rel, seq_se, sequence_embedding=embedding)
        else:
            recon_rel, _, _, _ = model(obs_traj_rel, seq_se)

    start_pos = obs_traj[0, :, :]
    recon_abs = relative_to_abs(recon_rel, start_pos)

    obs_traj_np = obs_traj.cpu().numpy()
    pred_traj_gt_np = pred_traj_gt.cpu().numpy()
    recon_abs_np = recon_abs.cpu().numpy()

    total_peds = obs_traj.shape[1]
    indices = np.random.choice(total_peds, min(
        num_samples, total_peds), replace=False)

    # --- AQUI: SHAREX e SHAREY ATIVADOS ---
    fig, axes = plt.subplots(
        1, len(indices),
        figsize=(4*len(indices), 4),
        sharex=True,  # Compartilha eixo X
        sharey=True   # Compartilha eixo Y
    )

    if len(indices) == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]

        # Obs Real
        ax.plot(obs_traj_np[:, idx, 0], obs_traj_np[:, idx, 1],
                'b-', linewidth=2.5, label='Obs Real', marker='.', markersize=6)

        # Pontos
        ax.plot(obs_traj_np[0, idx, 0], obs_traj_np[0,
                idx, 1], 'go', label='Início', zorder=5)
        ax.plot(obs_traj_np[-1, idx, 0],
                obs_traj_np[-1, idx, 1], 'bo', zorder=5)

        # Reconstrução
        ax.plot(recon_abs_np[:, idx, 0], recon_abs_np[:, idx, 1],
                'r-', linewidth=2, alpha=0.9, label='Reconstrução')

        # Contexto Futuro
        gt_x = np.concatenate(
            ([obs_traj_np[-1, idx, 0]], pred_traj_gt_np[:, idx, 0]))
        gt_y = np.concatenate(
            ([obs_traj_np[-1, idx, 1]], pred_traj_gt_np[:, idx, 1]))
        ax.plot(gt_x, gt_y, 'k--', linewidth=1, alpha=0.3, label='Futuro')

        ax.set_title(f"Ped {idx}", fontsize=10)

        # --- FIX: adjustable='box' impede o crash com sharex/sharey ---
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', alpha=0.6)

        if i == 0:
            ax.legend(fontsize=8, loc='best')

    plt.suptitle(
        f"Reconstrução da Observação (VAE) - {task_name}", fontsize=14)

    # Proteção extra para o layout
    try:
        plt.tight_layout()
    except RuntimeError:
        pass  # Ignora erro de layout se ocorrer

    save_path = os.path.join(args.r_dir, f'traj_recon_{task_name}.png')
    plt.savefig(save_path, dpi=100)
    print(f"-> Trajetórias salvas em: {save_path}")
    plt.close()


if __name__ == "__main__":
    args = get_all_args(load_yaml=True)

    if not os.path.exists(args.r_dir):
        os.makedirs(args.r_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # AJUSTE SEU CAMINHO AQUI
    model_path = "/home/matheus/LLM4CPTL/cptl_with_social_gr/results/CL_SGR_continual_learning_generative_4_generativeModelCheckpoint_['ETH', 'UCY', 'inD', 'INTERACTION']_64_72_False_current_False_None.path"

    model = CVAE(obs_len=args.obs_len,
                 pred_len=args.pred_len,
                 traj_lstm_input_size=args.traj_lstm_input_size,
                 traj_lstm_hidden_size=args.traj_lstm_hidden_size,
                 traj_lstm_output_size=args.traj_lstm_output_size,
                 dropout=args.dropout,
                 z_dim=args.z_dim,
                 embedding_dim=args.embedding_dim,
                 mlp_dim=args.mlp_dim,
                 bottleneck_dim=args.bottleneck_dim,
                 activation='relu',
                 batch_norm=True,
                 adapt_architecture_to_include_sequence_embedding=args.adapt_architecture_to_include_sequence_embedding,
                 sequence_embedding_dimension=args.dimensions,
                 sequence_embedding_compressed_dimension=args.sequence_embedding_compressed_dimension,
                 use_gradient_clipping=args.use_gradient_clipping,
                 clip_gradient_max_norm=args.clip_gradient_max_norm,
                 use_skip_connection=args.use_skip_connection).to(device)

    if os.path.exists(model_path):
        print(f"Carregando pesos de: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"ERRO: Checkpoint não encontrado em {model_path}")

    task_names = ["ETH", "UCY", "inD", "INTERACTION"]

    task_paths = [
        get_dset_path("ETH", "test"),
        get_dset_path("UCY", "test"),
        get_dset_path("inD", "test"),
        get_dset_path("INTERACTION", "test")
    ]

    loaders_list = []
    valid_task_names = []

    for name, path in zip(task_names, task_paths):
        if os.path.exists(path):
            print(f"Carregando dataset: {name}")
            emb_path = args.llm_sequences_embeddings_mapping.get(
                name, {}).get("test", None)

            dset = data_dset(
                args,
                path,
                dataset_name=name,
                sequences_embeddings_path=emb_path,
                split_name="test"
            )
            loader = data_loader(args, dset)
            loaders_list.append(loader)
            valid_task_names.append(name)
        else:
            print(f"AVISO: {name} não encontrado em {path}. Pulando.")

    if loaders_list:
        analyze_latent_space_and_reconstruction(
            model,
            loaders_list,
            args,
            task_names=valid_task_names,
            num_samples_visualize=5
        )
    else:
        print("Nenhum dataset carregado.")
