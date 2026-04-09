import torch
import os
import tikzplotly
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
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

    # =========================
    # Mapas de cores e markers
    # =========================
    unique_datasets = np.unique(dataset_labels)
    dataset_labels_arr = np.array(dataset_labels)
    scene_labels_arr = np.array(scene_labels)

    # Pegando uma paleta de cores padrão do Plotly
    color_palette = pcolors.qualitative.Plotly
    dataset_color_map = {ds: color_palette[i % len(
        color_palette)] for i, ds in enumerate(unique_datasets)}

    # Marcadores nativos do Plotly (equivalentes aos do Matplotlib que você usava)
    markers = ['circle', 'square', 'triangle-up', 'diamond', 'cross',
               'x', 'star', 'triangle-down', 'triangle-left', 'triangle-right']
    scene_marker_map = {}

    for dataset in unique_datasets:
        scenes_in_dataset = np.unique(
            scene_labels_arr[dataset_labels_arr == dataset])
        for i, scene in enumerate(scenes_in_dataset):
            scene_marker_map[scene] = markers[i % len(markers)]

    # =========================
    # Construção do Plotly Figure
    # =========================
    fig = go.Figure()

    for dataset in unique_datasets:
        scenes_in_dataset = np.unique(
            scene_labels_arr[dataset_labels_arr == dataset])
        for scene in scenes_in_dataset:
            mask = (dataset_labels_arr == dataset) & (
                scene_labels_arr == scene)

            if np.sum(mask) == 0:
                continue

            x_vals = X_embedded[mask, 0]
            y_vals = X_embedded[mask, 1]

            # Lógica de rótulo para a legenda
            label = dataset if dataset in [
                "inD", "INTERACTION"] else f"{dataset} - {scene}"

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                name=label,
                marker=dict(
                    color=dataset_color_map[dataset],
                    symbol=scene_marker_map[scene],
                    size=10,
                    opacity=0.7,
                    line=dict(width=0.5, color='black')  # Borda dos marcadores
                )
            ))

    # =========================
    # Finalização do Plot
    # =========================
    fig.update_layout(
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        template="simple_white",  # Este template remove as bordas top/right automaticamente
        width=1000,
        height=700,
        legend=dict(
            font=dict(size=12),
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02  # Posiciona a legenda fora do gráfico à direita
        )
    )

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    # Salva em SVG usando o motor nativo do plotly (exige kaleido)
    #svg_path = os.path.join(args.plot_dir, "embeddings_pca.svg")
    #fig.write_image(svg_path)

    # Salva em TEX usando o tikzplotly que você estava tentando usar inicialmente
    tex_path = os.path.join(args.plot_dir, "embeddings_pca.tex")
    tikzplotly.save(tex_path, fig)

    print(f"Plots salvos em {args.plot_dir}")

    # Opcional: Para abrir o gráfico interativo no navegador localmente ao rodar
    # fig.show()


if __name__ == "__main__":
    args = get_all_args()

    if not hasattr(args, 'plot_dir') or args.plot_dir is None:
        args.plot_dir = "./plots"

    plot_embeddings(args)
    print("End!")
