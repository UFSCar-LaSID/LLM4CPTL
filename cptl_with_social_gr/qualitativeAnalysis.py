#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap  # Certifique-se de ter instalado: pip install umap-learn
from pathlib import Path

from args import get_all_args
from data.loader import data_dset, data_loader
from data.trajectories import SceneBatch
from generative_model.vae_models_scratch import CVAE
from helper.utils import get_dset_path, relative_to_abs

# Configuração estética para os gráficos
plt.style.use('ggplot')

# Cores fixas para cada dataset para garantir consistência visual em todos os gráficos
DATASET_COLORS = {
    'ETH': 'tab:red',
    'UCY': 'tab:green',
    'inD': 'tab:purple',
    'INTERACTION': 'tab:blue'
}


def get_color(dataset_name):
    """Retorna a cor fixa do dataset ou uma cor padrão cinza caso não esteja mapeado."""
    return DATASET_COLORS.get(dataset_name, 'tab:gray')


def stratified_subsample(mus_list, labels, flags, max_per_category=300):
    """
    Subamostragem estratificada para evitar overplotting.
    Garante um limite máximo de pontos por Dataset E por Linearidade (Linear vs Não-Linear),
    mantendo as classes minoritárias (curvas) visíveis no gráfico.
    """
    mus_np = torch.cat(mus_list, dim=0).numpy()
    labels_np = np.array(labels)
    flags_np = np.array(flags)

    selected_indices = []
    unique_labels = sorted(list(set(labels)))

    for label in unique_labels:
        for flag_val in [0.0, 1.0]:  # 0.0 = Linear, 1.0 = Não-Linear
            # Encontra todos os índices que batem com esse dataset e essa linearidade
            idx = np.where((labels_np == label) & (flags_np == flag_val))[0]

            # Se houver mais pontos do que o limite, sorteia aleatoriamente sem repetição
            if len(idx) > max_per_category:
                # Congela a semente para garantir reprodutibilidade
                np.random.seed(42)
                idx = np.random.choice(idx, max_per_category, replace=False)

            selected_indices.extend(idx)

    # Ordena os índices para manter a coerência dos tensores
    selected_indices.sort()

    # Remonta as listas apenas com os pontos sorteados
    sampled_mus = [torch.tensor(mus_np[selected_indices])]
    sampled_labels = labels_np[selected_indices].tolist()
    sampled_flags = flags_np[selected_indices].tolist()

    print(
        f"    [SUBSAMPLING] Total de pontos reduzido de {len(labels)} para {len(sampled_labels)} (Manteve balanço de classes).")

    return sampled_mus, sampled_labels, sampled_flags


def plot_latent_space_reduction(mus, labels, non_linear_flags, method='tsne', save_dir=".", experiment=""):
    """
    Desenha a dispersão dos pedestres no espaço latente usando a técnica de redução especificada.
    Usa cores fixas para os datasets e marcadores diferentes para trajetórias lineares vs não-lineares.
    """
    mus_np = torch.cat(mus, dim=0).numpy()

    if mus_np.shape[1] > 2:
        print(
            f"    [{method.upper()}] Reduzindo dimensionalidade do espaço latente de todos os pedestres...")

        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30,
                           n_iter=1000, random_state=42)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Método de redução '{method}' não suportado.")

        mus_2d = reducer.fit_transform(mus_np)
    else:
        mus_2d = mus_np

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(list(set(labels)))

    # Plotando os pontos separados por Dataset e por Linearidade
    for label in unique_labels:
        color = get_color(label)

        # Filtra os índices das trajetórias Lineares (flag == 0.0)
        idx_linear = [j for j, (l, flag) in enumerate(
            zip(labels, non_linear_flags)) if l == label and flag == 0.0]
        if idx_linear:
            ax.scatter(mus_2d[idx_linear, 0], mus_2d[idx_linear, 1],
                       c=color, marker='o', alpha=0.6, s=15)

        # Filtra os índices das trajetórias Não-Lineares (flag == 1.0)
        idx_nonlinear = [j for j, (l, flag) in enumerate(
            zip(labels, non_linear_flags)) if l == label and flag == 1.0]
        if idx_nonlinear:
            ax.scatter(mus_2d[idx_nonlinear, 0], mus_2d[idx_nonlinear, 1],
                       c=color, marker='^', alpha=0.6, s=15)

    ax.set_title(f"Dispersão {method.upper()} dos Pedestres",
                 fontsize=16, fontweight='bold')
    ax.set_xlabel(f"{method.upper()} Dim 1")
    ax.set_ylabel(f"{method.upper()} Dim 2")

    # ---- CRIAÇÃO DA LEGENDA CUSTOMIZADA (Proxy Artists) ----
    legend_elements = []
    # Adiciona itens de cor (Datasets)
    for label in unique_labels:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', label=label,
                                      markerfacecolor=get_color(label), markersize=10))
    # Adiciona um divisor visual invisível
    legend_elements.append(Line2D([0], [0], marker='', color='w', label='---'))
    # Adiciona itens de formato (Linearidade)
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Traj. Linear',
                                  markerfacecolor='tab:gray', markersize=10))
    legend_elements.append(Line2D([0], [0], marker='^', color='w', label='Traj. Não-Linear',
                                  markerfacecolor='tab:gray', markersize=10))

    ax.legend(handles=legend_elements,
              bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(
        save_dir, f'{experiment}_latent_space_{method.lower()}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Gráfico {method.upper()} salvo em: {filepath}")


def plot_combined_latent_space_reduction(mus, labels, non_linear_flags, save_dir=".", experiment=""):
    """
    Desenha a dispersão dos pedestres no espaço latente usando PCA, t-SNE e UMAP lado a lado.
    """
    mus_np = torch.cat(mus, dim=0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    methods = ['pca', 'tsne', 'umap']
    titles = ['PCA', 't-SNE', 'UMAP']

    unique_labels = sorted(list(set(labels)))

    for ax, method, title in zip(axes, methods, titles):
        if mus_np.shape[1] > 2:
            print(
                f"    [{title}] Reduzindo dimensionalidade do espaço latente para o plot combinado...")
            if method == 'tsne':
                reducer = TSNE(n_components=2, perplexity=30,
                               n_iter=1000, random_state=42)
            elif method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            elif method == 'umap':
                reducer = umap.UMAP(n_components=2, random_state=42)

            mus_2d = reducer.fit_transform(mus_np)
        else:
            mus_2d = mus_np

        for label in unique_labels:
            color = get_color(label)

            idx_linear = [j for j, (l, flag) in enumerate(
                zip(labels, non_linear_flags)) if l == label and flag == 0.0]
            if idx_linear:
                ax.scatter(mus_2d[idx_linear, 0], mus_2d[idx_linear, 1],
                           c=color, marker='o', alpha=0.6, s=15)

            idx_nonlinear = [j for j, (l, flag) in enumerate(
                zip(labels, non_linear_flags)) if l == label and flag == 1.0]
            if idx_nonlinear:
                ax.scatter(mus_2d[idx_nonlinear, 0], mus_2d[idx_nonlinear, 1],
                           c=color, marker='^', alpha=0.6, s=15)

        ax.set_title(f"Dispersão {title}", fontsize=16, fontweight='bold')
        ax.set_xlabel(f"{title} Dim 1")
        ax.set_ylabel(f"{title} Dim 2")
        ax.grid(True, linestyle='--', alpha=0.5)

        # Inserir a legenda apenas no último gráfico (UMAP)
        if method == 'umap':
            legend_elements = []
            for label in unique_labels:
                legend_elements.append(Line2D([0], [0], marker='s', color='w', label=label,
                                              markerfacecolor=get_color(label), markersize=10))
            legend_elements.append(
                Line2D([0], [0], marker='', color='w', label='---'))
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Traj. Linear',
                                          markerfacecolor='tab:gray', markersize=10))
            legend_elements.append(Line2D([0], [0], marker='^', color='w', label='Traj. Não-Linear',
                                          markerfacecolor='tab:gray', markersize=10))
            ax.legend(handles=legend_elements,
                      bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(
        save_dir, f'{experiment}_latent_space_combined.png')

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Gráfico combinado salvo em: {filepath}")


def plot_reconstructions(obs_traj, recon_traj, task_name, num_samples=5, save_dir="."):
    """
    Desenha as trajetórias reais (azul) vs reconstruídas (vermelha).
    """
    num_samples = min(num_samples, obs_traj.shape[1])
    fig, axes = plt.subplots(1, num_samples, figsize=(
        4 * num_samples, 4), sharex=True, sharey=True)

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        real_x = obs_traj[:, i, 0].numpy()
        real_y = obs_traj[:, i, 1].numpy()
        recon_x = recon_traj[:, i, 0].numpy()
        recon_y = recon_traj[:, i, 1].numpy()

        axes[i].plot(real_x, real_y, 'b-', label='Real',
                     linewidth=2, marker='o', markersize=4)
        axes[i].plot(recon_x, recon_y, 'r--', label='Reconstruída',
                     linewidth=2, marker='x', markersize=4)
        axes[i].scatter(real_x[0], real_y[0], color='green',
                        s=100, label='Início', zorder=5)

        axes[i].set_title(f"Amostra {i+1}", fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.5)

    axes[0].legend()
    fig.suptitle(
        f"Reconstrução de Trajetórias: {task_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(save_dir, f'traj_recon_{task_name}.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"[PLOT] Gráfico de reconstrução salvo em: {filepath}")


def run_analysis(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando análise qualitativa no dispositivo: {device}")

    experiments_to_analyze = [
        "CL_SGR",
        "CL_SGReKLAN",
        "CL_SGReModalLLM",
        "CL_SGReModalLLMBuffer",
        "CL_SGReKLANeModalLLM",
        "CL_SGReKLANeModalLLMBuffer",
        "CL_SGReKLANeModalLLMBuffereUncFilter",
    ]

    for experiment in experiments_to_analyze:
        if "modalllm" in experiment.lower():
            args.adapt_architecture_to_include_sequence_embedding = True
        else:
            args.adapt_architecture_to_include_sequence_embedding = False

        model = CVAE(
            obs_len=args.obs_len,
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
            use_prior_adaptation=args.use_prior_adaptation,
            use_dc_vampprior=args.use_dc_vampprior
        ).to(device)

        model_path = f"/home/matheus/LLM4CPTL/cptl_with_social_gr/results/{experiment}_generativeModelCheckpoint_task4_lstm_200_64_ETH-UCY-inD-INTERACTION.path"

        variation_of_clsgr_executed = "-".join(
            Path(model_path).stem.split("_")[0:2])

        if os.path.exists(model_path):
            print(f"Carregando pesos de: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(
                f"ERRO: Checkpoint não encontrado em {model_path}. Verifique se o treino terminou e salvou.")
            return

        model.eval()

        task_names = args.dataset
        task_paths = [get_dset_path(name, "test") for name in task_names]

        loaders_list = []
        valid_task_names = []

        for name, path in zip(task_names, task_paths):
            if os.path.exists(path):
                dset = data_dset(
                    args, path, dataset_name=name, split_name="test")
                loader = data_loader(args, dset)
                loaders_list.append(loader)
                valid_task_names.append(name)
            else:
                print(f"AVISO: Dataset {name} não encontrado em {path}.")

        # Variáveis globais para os plots
        all_mus = []
        all_labels = []
        all_non_linear_flags = []  # Guarda a informação de linearidade

        save_dir = args.p_dir
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "="*60)
        print(f"EXTRAINDO FEATURES LATENTES E RECONSTRUÇÕES: {experiment}")
        print("="*60)

        with torch.no_grad():
            for task_idx, (name, loader) in enumerate(zip(valid_task_names, loaders_list)):
                print(f"Processando dataset: {name}")
                plotted_reconstruction = False

                for batch_data in loader:
                    batch = SceneBatch(batch_data, device=device)

                    x_rel = batch.obs_traj_rel
                    obs_traj = batch.obs_traj
                    seq_start_end = batch.seq_start_end
                    seq_emb = batch.sequence_embeddings

                    # Extrai a flag (1.0 = Não linear, 0.0 = Linear)
                    non_linear_flag = batch.non_linear_ped.cpu().numpy()

                    recon_batch_rel, mu, logvar, z = model(
                        x_rel, seq_start_end, sequence_embedding=seq_emb)

                    mu_cpu = mu.cpu()

                    all_mus.append(mu_cpu)
                    all_labels.extend([name] * mu_cpu.shape[0])
                    all_non_linear_flags.extend(
                        non_linear_flag)  # Acumula as flags

                    if not plotted_reconstruction:
                        start_pos = obs_traj[0].to(device)
                        recon_traj_abs = relative_to_abs(
                            recon_batch_rel, start_pos)
                        plotted_reconstruction = True

        print("\n" + "="*60)
        print("GERANDO GRÁFICOS DO ESPAÇO LATENTE")
        print("="*60)

        PLOT_INDIVIDUAL = False
        PLOT_COMBINED = True

        if all_mus:
            # ---> APLICA A SUBAMOSTRAGEM ESTRATIFICADA AQUI <---
            # max_per_category=300 significa limite de pontos por Dataset e por Linearidade.
            sampled_mus, sampled_labels, sampled_flags = stratified_subsample(
                all_mus, all_labels, all_non_linear_flags, max_per_category=300
            )

            if PLOT_INDIVIDUAL:
                plot_latent_space_reduction(sampled_mus, sampled_labels, sampled_flags,
                                            method='tsne', save_dir=save_dir, experiment=variation_of_clsgr_executed)
                plot_latent_space_reduction(sampled_mus, sampled_labels, sampled_flags,
                                            method='pca', save_dir=save_dir, experiment=variation_of_clsgr_executed)
                plot_latent_space_reduction(sampled_mus, sampled_labels, sampled_flags,
                                            method='umap', save_dir=save_dir, experiment=variation_of_clsgr_executed)

            if PLOT_COMBINED:
                plot_combined_latent_space_reduction(
                    sampled_mus, sampled_labels, sampled_flags, save_dir=save_dir, experiment=variation_of_clsgr_executed)

    print("\nAnálise qualitativa concluída com sucesso!")


if __name__ == "__main__":
    args = get_all_args()
    run_analysis(args)
