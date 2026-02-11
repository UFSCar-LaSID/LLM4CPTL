import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from args import get_all_args


def plot_embeddings(args):
    all_embeddings = []
    labels = []

    print("Carregando embeddings...")

    # Garante que args.dataset seja iterável mesmo se for string única
    datasets = args.dataset if isinstance(
        args.dataset, list) else [args.dataset]

    for ds_name in datasets:
        # Busca o caminho no mapeamento
        mapping = getattr(args, 'llm_sequences_embeddings_mapping', {})
        ds_map = mapping.get(ds_name, None)

        if ds_map:
            filepath = ds_map.get("train", None)
        else:
            print(f"Aviso: Mapeamento não encontrado para {ds_name}")
            continue

        print(f"Lendo: {filepath}")

        if filepath and os.path.exists(filepath):
            # Carrega direto na CPU
            emb = torch.load(filepath, map_location='cpu')
        else:
            print(f"Arquivo não encontrado ou caminho nulo: {filepath}")
            continue

        if emb is None:
            continue

        # Itera sobre o dicionário {NomeCena: Tensor}
        for scene_name, tensor in emb.items():
            # 1. Garante CPU e converte para Numpy
            emb_np = tensor.cpu().numpy()

            # 2. CORREÇÃO CRÍTICA: Se for 1D (vetor), vira 2D (matriz de 1 linha)
            if len(emb_np.shape) == 1:
                emb_np = emb_np.reshape(1, -1)

            all_embeddings.append(emb_np)

            # Ajusta labels baseado no número de amostras reais (linhas)
            num_samples = emb_np.shape[0]
            labels.extend([ds_name] * num_samples)

    if all_embeddings:
        # Concatena: Agora funciona porque todos os itens em all_embeddings são 2D
        X = np.concatenate(all_embeddings, axis=0)

        print(f"Executando PCA em matriz de shape {X.shape}...")

        # Proteção para t-SNE/PCA não falhar se tiver poucos dados
        n_comps = min(2, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comps)

        X_embedded = pca.fit_transform(X)

        var_exp = pca.explained_variance_ratio_.sum()
        print(
            f"Variância explicada pelos {n_comps} primeiros componentes: {var_exp:.2%}")

        plt.figure(figsize=(10, 8))

        # Plotar cada dataset com uma cor
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = np.array(labels) == label
            plt.scatter(X_embedded[mask, 0],
                        X_embedded[mask, 1], label=label, alpha=0.6, s=10)

        plt.title("PCA das Embeddings de Cena (LLM)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Garante que o diretório de plots existe
        if not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)

        save_path = os.path.join(args.plot_dir, "embeddings_pca.png")
        plt.savefig(save_path)
        print(f"Plot salvo em {save_path}")
    else:
        print("Nenhuma embedding encontrada! Verifique os caminhos no args.py ou yaml.")


if __name__ == "__main__":
    # Carrega argumentos (incluindo o mapeamento de arquivos)
    args = get_all_args(load_yaml=True)

    # Se plot_dir não estiver definido nos args, define um padrão
    if not hasattr(args, 'plot_dir') or args.plot_dir is None:
        args.plot_dir = './plots'

    plot_embeddings(args)
    print("End!")
