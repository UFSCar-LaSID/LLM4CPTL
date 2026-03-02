import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from args import get_all_args

from data.loader import data_dset
from helper.utils import get_dset_path
from data.trajectories import TrajectoryDataset


def plot_embeddings(args):
    all_embeddings = []
    dataset_labels = []
    scene_labels = []

    print("Carregando embeddings...")

    plotted_sequences = []

    for dataset_name in args.dataset:
        dset_path = get_dset_path(dataset_name, "train")
        dset = data_dset(
            args, dset_path,
            dataset_name=dataset_name,
            split_name="train"
        )

        for scene_name, scene_embedding in dset.sequences_embeddings.items():

            scene_mapped = TrajectoryDataset.SEQUENCES_IMAGES_MAPPING[scene_name]

            if scene_mapped in plotted_sequences:
                print("Skipping already plotted sequence:", scene_mapped)
                continue

            plotted_sequences.append(scene_mapped)

            emb_np = scene_embedding.cpu().numpy()

            if len(emb_np.shape) == 1:
                emb_np = emb_np.reshape(1, -1)

            all_embeddings.append(emb_np)

            dataset_labels.extend([dataset_name] * emb_np.shape[0])
            scene_labels.extend([scene_mapped] * emb_np.shape[0])

    if not all_embeddings:
        print("Nenhuma embedding encontrada!")
        return

    X = np.concatenate(all_embeddings, axis=0)

    print(f"Executando PCA em matriz de shape {X.shape}...")

    n_comps = min(2, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comps)
    X_embedded = pca.fit_transform(X)

    var_exp = pca.explained_variance_ratio_.sum()
    print(
        f"Variância explicada pelos {n_comps} primeiros componentes: {var_exp:.2%}")

    # =========================
    # Mapas de cores e markers
    # =========================
    unique_datasets = np.unique(dataset_labels)
    dataset_labels_arr = np.array(dataset_labels)
    scene_labels_arr = np.array(scene_labels)

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_datasets)))
    dataset_color_map = dict(zip(unique_datasets, colors))

    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']
    scene_marker_map = {}

    for dataset in unique_datasets:
        scenes_in_dataset = np.unique(
            scene_labels_arr[dataset_labels_arr == dataset])
        for i, scene in enumerate(scenes_in_dataset):
            scene_marker_map[scene] = markers[i % len(markers)]

    # =========================
    # Plot
    # =========================
    plt.figure(figsize=(12, 8))

    for dataset in unique_datasets:
        scenes_in_dataset = np.unique(
            scene_labels_arr[dataset_labels_arr == dataset])
        for scene in scenes_in_dataset:
            mask = (dataset_labels_arr == dataset) & (
                scene_labels_arr == scene)

            if np.sum(mask) == 0:
                continue

            plt.scatter(
                X_embedded[mask, 0],
                X_embedded[mask, 1],
                c=[dataset_color_map[dataset]],
                marker=scene_marker_map[scene],
                label=f"{dataset} - {scene}",
                alpha=0.7,
                s=60,
                edgecolors='black',
                linewidths=0.5
            )

    plt.title("PCA das Embeddings de Cena (LLM)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    save_path = os.path.join(args.plot_dir, "embeddings_pca.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"Plot salvo em {save_path}")


if __name__ == "__main__":
    args = get_all_args()

    if not hasattr(args, 'plot_dir') or args.plot_dir is None:
        args.plot_dir = "./plots"

    plot_embeddings(args)
    print("End!")
