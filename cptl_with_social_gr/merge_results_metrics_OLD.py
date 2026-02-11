#!/usr/bin/env python3
import os
import argparse
from pyexpat import model
import re
import glob
import logging

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import cm

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pc
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_args():
    parser = argparse.ArgumentParser(
        description='Merge and plot metrics for Continual Learning experiments.')

    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        dest='r_dir',
        help="Directory containing CSV results"
    )

    parser.add_argument(
        '--output_filename',
        type=str,
        default='df_metrics_final.csv',
        help="Output consolidated CSV filename"
    )

    parser.add_argument(
        '--iters',
        type=int,
        default=None,
        help="Filter by number of iterations"
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help="Filter by batch size"
    )
    parser.add_argument(
        '--plots',
        nargs='+',
        default=['all'],
        choices=['all', 'ape', 'forgetting', 'ade', 'fde',
                 'duration', 'task_time', 'task_accuracy', 'losses'],
        help="Select specific metrics to plot (e.g., --plots ape forgetting)"
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='matplotlib',
        choices=['matplotlib', 'plotly'],
        help="Plotting backend: 'matplotlib' (static png) or 'plotly' (interactive html)"
    )

    return parser.parse_args()


def clean_labels(ax):
    """Standardize x-axis labels: replace underscores and handle IL mean."""
    labels = [label.get_text().replace("_", "-") if "CL" in label.get_text()
              else "IL (avg)" for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=45, ha='right')


def compute_nbwt(df):
    """Compute Normalized Backward Transfer (NBWT) with error handling for zero division."""
    try:
        if not {"IL", "CL_NR"}.issubset(df['method'].values):
            logging.warning(
                "NBWT cannot be computed: Baseline methods (IL or CL_NR) are missing.")
            return df

        # Get baselines
        il_bwt_ade = df.loc[df['method'] == 'IL',
                            'backward_transfer_bwt_ade'].mean()
        clnr_bwt_ade = df.loc[df['method'] == 'CL_NR',
                              'backward_transfer_bwt_ade'].iloc[0]

        denom_ade = clnr_bwt_ade - il_bwt_ade
        mask = ~df["method"].isin(["IL", "CL_NR"])

        df["normalized_backward_transfer_nbwt_ade"] = np.nan
        if abs(denom_ade) > 1e-9:
            df.loc[mask, "normalized_backward_transfer_nbwt_ade"] = (
                (df.loc[mask, "backward_transfer_bwt_ade"] -
                 il_bwt_ade) / denom_ade
            )

        df.loc[df["method"] == "CL_NR",
               "normalized_backward_transfer_nbwt_ade"] = 1.0
        return df
    except Exception as e:
        logging.error(f"Error during NBWT calculation: {e}")
        return df


def load_and_filter_csvs(directory, pattern, args):
    """Centralized logic to read and filter CSV files based on filename and content."""
    dataframes = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.csv') and pattern in filename and filename != args.output_filename:
            try:
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)

                # Apply CLI filters if provided
                if args.iters and 'iters' in df.columns:
                    df = df[df['iters'] == args.iters]
                if args.batch_size and 'batch_size' in df.columns:
                    df = df[df['batch_size'] == args.batch_size]

                if not df.empty:
                    logging.info(f"Loaded: {filename}")
                    dataframes.append(df)
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")
    return dataframes


def plot_bar_chart(df, y_cols, title, ylabel, save_path, colormap_name=None, y_base=None, grid=True, grid_axis='both', legend=True, legend_title="", legend_labels=None, legend_loc="best", legend_bbox_to_anchor=None):
    """Generic helper to create standardized bar plots."""
    try:
        if colormap_name is not None:
            colormap = cm.get_cmap(colormap_name, 8)
            colors = colormap(np.linspace(0, 1, len(y_cols)))

            ax = df.plot(
                x="method",
                y=y_cols,
                kind="bar",
                figsize=(10, 6),
                legend=legend,
                color=colors
            )

        else:
            ax = df.plot(
                x="method",
                y=y_cols,
                kind="bar",
                figsize=(10, 6),
                legend=legend
            )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Methods")
        clean_labels(ax)

        if legend:
            ax.legend(title=legend_title, labels=legend_labels,
                      loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

        if grid:
            ax.grid(axis=grid_axis, linestyle='--', alpha=0.7)

        if y_base:
            ax.yaxis.set_major_locator(plticker.MultipleLocator(base=y_base))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        logging.error(f"Plotting error for {title}: {e}")


def plot_per_task_metric(df, metric_prefix, title, ylabel, save_path, y_base=0.1):
    """

    """
    try:
        # 1. Identificação dinâmica de tarefas e colunas
        methods = df["method"].unique()
        task_pattern = re.compile(
            rf"{metric_prefix}_task(\d+)_after_training_in_all_tasks")

        # Encontra todos os números de tarefas disponíveis nas colunas
        tasks = sorted([int(task_pattern.search(c).group(1))
                        for c in df.columns if task_pattern.search(c)])

        if not tasks:
            logging.warning(
                f"No columns with the given prefix: {metric_prefix}")
            return

        cols = [
            f"{metric_prefix}_task{t}_after_training_in_all_tasks" for t in tasks
        ]

        cols_extra = []
        if metric_prefix == "ade":
            task_pattern = re.compile(
                rf"fde_task(\d+)_after_training_in_all_tasks")

            tasks = sorted([int(task_pattern.search(c).group(1))
                            for c in df.columns if task_pattern.search(c)])

            cols_extra = [
                f"fde_task{t}_after_training_in_all_tasks" for t in tasks
            ]
        else:
            task_pattern = re.compile(
                rf"ade_task(\d+)_after_training_in_all_tasks")

            tasks = sorted([int(task_pattern.search(c).group(1))
                            for c in df.columns if task_pattern.search(c)])

            cols_extra = [
                f"ade_task{t}_after_training_in_all_tasks" for t in tasks
            ]

        # 2. Cálculo de limites do eixo Y
        cols_extra.extend(cols)
        all_values = df[cols_extra].values.flatten()
        ymin = np.nanmin(all_values) - 0.2
        ymax = np.nanmax(all_values) + 0.3

        fig, ax = plt.subplots(figsize=(10, 6))

        # 3. Plotagem das linhas e anotações
        for method in methods:
            row = df[df["method"] == method].iloc[0]
            y_values = [row[col] for col in cols]

            ax.plot(tasks, y_values, marker='o',
                    label=method.replace("_", "-"))

        # 4. Texto da ordem dos datasets (Task Order)
        if "train_dataset" in df.columns:
            datasets = df["train_dataset"].iloc[0].split(" - ")
            dataset_order_text = "Tasks order: " + \
                ", ".join(map(str, datasets))
            ax.text(0.5, -0.18, dataset_order_text, transform=ax.transAxes,
                    ha="center", va="top", fontsize=9, style='italic')

        # 5. Estética e Configurações
        ax.set_xticks(tasks)
        ax.set_xlabel("Tasks")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(max(ymin, 0.0), ymax)

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Methods", bbox_to_anchor=(1.05, 1), loc='upper left')

        if y_base:
            ax.yaxis.set_major_locator(plticker.MultipleLocator(base=y_base))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Gráfico salvo em: {save_path}")

    except Exception as e:
        logging.error(
            f"Plotting error for ({metric_prefix}): {e}")


def plot_loss_column(subplot_location, df, model_type, method_name, epochs_per_task, backend="matplotlib", normalize=False):
    """
    Função auxiliar para processar e plotar as perdas em um eixo específico.
    """
    # Cálculo do passo global para visualização contínua
    df['Global_Step'] = (df['Task'] - 1) * epochs_per_task + df['Epoch']

    # Definição de cores e colunas baseada no tipo de modelo
    if model_type == 'vae':
        metrics = {
            'Loss_Total': {'label': 'Total', 'color': 'black', 'line_style': 'dash'},
            'ReconL':     {'label': 'Reconstruction (current)', 'color': 'blue', 'line_style': '-'},
            'VariatL':    {'label': 'Variational (current)',    'color': 'green', 'line_style': '-'},
            'ReconL_R':   {'label': 'Reconstruction (replay)', 'color': 'cyan',  'line_style': 'dot'},
            'VariatL_R':  {'label': 'Variational (replay)',    'color': 'lime',  'line_style': 'dot'}
        }

    else:
        metrics = {
            'Loss_Total':   {'label': 'Average', 'color': 'black', 'line_style': 'dash'},
            # 'Loss_Current': {'label': 'Current Task', 'color': 'red',   'line_style': '-'},
            # 'Loss_Replay':  {'label': 'Replay Task',  'color': 'orange', 'line_style': 'dot'},
            'Pred_Traj':  {'label': 'Predicted trajectory (current)', 'color': 'cyan',  'line_style': '-'},
            'Pred_Traj_R':  {'label': 'Predicted trajectory (replay)', 'color': 'lime',  'line_style': 'dot'},
            'Loss_Validation':   {'label': 'Validation', 'color': 'red', 'line_style': 'dash'},
        }

    fig, row, col = subplot_location

    for col_name, style in metrics.items():
        if col_name in df.columns:
            values = df[col_name].values

            fig.add_trace(
                go.Scatter(
                    x=df['Global_Step'],
                    y=values,
                    name=f"{method_name}: {style['label']}",
                    line=dict(
                        color=style['color'],
                        dash=style['line_style']
                    )
                ),
                row=row,
                col=col
            )

    unique_tasks = sorted(df['Task'].unique())
    for task in unique_tasks[:-1]:
        fig.add_vline(
            x=task * epochs_per_task,
            line_dash="solid",
            line_color="gray",
            row=row,
            col=col
        )

    return


def run(args):
    # --- 1. Data Merging ---
    if any(plot in args.plots for plot in ['all', 'ape', 'forgetting', 'duration', 'task_time', 'task_accuracy']):
        logging.info("Starting data aggregation...")
        perf_dfs = load_and_filter_csvs(args.r_dir, "metrics", args)

        if not perf_dfs:
            logging.error(
                "No valid performance metric files found. Check your directory and filters.")
            return

        df_merged = pd.concat(perf_dfs, ignore_index=True)
        df_merged = compute_nbwt(df_merged)

        # Add IL Mean row for comparison
        df_il = df_merged[df_merged["method"] == "IL"]
        if not df_il.empty:
            mean_vals = df_il.mean(numeric_only=True).to_dict()
            mean_vals.update(
                {"method": "IL_mean", "learning_method": "batch_learning"})
            df_merged = pd.concat(
                [df_merged, pd.DataFrame([mean_vals])], ignore_index=True)

        # Final cleanup and save
        output_path = os.path.join(args.r_dir, args.output_filename)
        df_merged.to_csv(output_path, index=False)
        logging.info(f"Consolidated metrics saved to: {output_path}")

        # Filter subsets for plotting
        df_cl_il = df_merged[df_merged["method"].str.contains("CL|IL_mean")]
        df_cl_only = df_merged[df_merged["method"].str.contains("CL")]

    # --- 2. Customizable Plotting ---
    plots_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Check if Plotly is requested but not installed
    if args.backend == 'plotly' and not PLOTLY_AVAILABLE:
        logging.error(
            "Plotly backend requested but plotly is not installed. Falling back to matplotlib.")
        args.backend = 'matplotlib'

    selected_plots = args.plots
    if 'all' in selected_plots:
        selected_plots = ['ape', 'forgetting', 'ade',
                          'fde', 'duration', 'task_time', 'losses']

    if 'ape' in selected_plots:
        plot_bar_chart(df_cl_il, ["average_prediction_error_ape_ade", "average_prediction_error_ape_fde"],
                       "Average Prediction Error (APE) of final models on full test set", "APE in meters",
                       os.path.join(plots_dir, "metrics_ape.png"), y_base=0.1, legend=True, legend_title="Metrics", legend_labels=[
            "APE: ADE", "APE: FDE"], legend_loc="upper left", legend_bbox_to_anchor=(1.05, 1))

    if 'forgetting' in selected_plots:
        plot_bar_chart(
            df_cl_only,
            ["backward_transfer_bwt_ade", "backward_transfer_bwt_fde",
                "forward_transfer_fwt_ade", "forward_transfer_fwt_fde"],
            "Continuous learning metrics",
            "Value",
            os.path.join(plots_dir, "metrics_bwtFwt.png"),
            colormap_name="viridis",
            # y_base=0.01,
            legend_title="Metrics",
            legend_labels=["BWT: ADE", "BWT: FDE", "FWT: ADE", "FWT: FDE"],
            legend_loc="upper left",
            legend_bbox_to_anchor=(1.05, 1)
        )

    if 'ade' in selected_plots:
        plot_per_task_metric(df_cl_only, "ade",
                             "Average Displacement Error (ADE) of final model on each task-specific test set",
                             "ADE in meters",
                             os.path.join(plots_dir, "metrics_adePerTask.png"), y_base=0.1)

    if 'fde' in selected_plots:
        plot_per_task_metric(df_cl_only, "fde",
                             "Final Displacement Error (FDE) of final model on each task-specific test set",
                             "FDE in meters",
                             os.path.join(plots_dir, "metrics_fdePerTask.png"), y_base=0.1)

    if 'duration' in selected_plots:
        plot_bar_chart(df_cl_only, ["training_time_in_secs"], "Total training duration of final models", "Seconds",
                       os.path.join(plots_dir, "metrics_totalTrainingDuration.png"), y_base=1000, legend=False)

    if 'task_time' in selected_plots:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            time_cols = sorted(
                [c for c in df_cl_only.columns if "elapsed_time_task_" in c])

            x_coords = range(1, len(time_cols) + 1)

            for _, row in df_cl_only.iterrows():
                y_vals = row[time_cols].values
                line = ax.plot(x_coords, y_vals, marker="o",
                               label=row["method"].replace("_", "-"))

                for x, y in zip(x_coords, y_vals):
                    if np.isnan(y):
                        continue  # Ignora valores nulos

                    ax.annotate(
                        # Texto formatado com 2 casas decimais
                        f'{y:.2f}',
                        xy=(x, y),
                        ha="right",
                        va="bottom",
                        fontsize=8,
                        alpha=0.8
                    )

            ax.set_yscale('log')
            ax.set_title("Training duration of the model on each task")
            ax.set_ylabel("Elapsed time in seconds")
            ax.set_xticks(x_coords)
            ax.set_xlabel("Tasks")
            ax.legend(title="Methods", bbox_to_anchor=(
                1.05, 1), loc='upper left')
            plt.grid(True, which="both", linestyle="--", alpha=0.5)

            datasets = df_cl_only["train_dataset"].tolist()[0].split("-")
            datasets = [dataset.strip() for dataset in datasets]
            dataset_order_text = "Tasks order: " + \
                ", ".join(map(str, datasets))
            ax.text(
                0.5, -0.15,
                dataset_order_text,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=10
            )

            plt.tight_layout()
            plt.savefig(os.path.join(
                plots_dir, "metrics_elapsedTimePerTask.png"))
            plt.close()
            logging.info("Saved Task Duration plot.")
        except Exception as e:
            logging.error(f"Error plotting task time: {e}")

    if 'losses' in selected_plots:
        # Busca de arquivos centralizada
        files_vae = sorted(
            glob.glob(os.path.join(args.r_dir, "*_losses_vae_*.csv")))
        # Arquivos txt do Predictor
        files_main = sorted(
            glob.glob(os.path.join(args.r_dir, "*_losses_main_*.csv")))

        methods = ['CL_ER', 'CL_SGR', 'CL_CGR', 'CL_NR']
        data_map = {m: {'vae': None, 'main': None} for m in methods}

        # Carregamento e mapeamento
        for f in files_vae:
            for m in methods:
                if m != "CL_NR":
                    if m in os.path.basename(f) and f"_{args.iters}" in os.path.basename(f) and f"_{args.batch_size}_" in os.path.basename(f):
                        data_map[m]['vae'] = pd.read_csv(f)
                else:
                    data_map[m]['vae'] = 'off'

        # Nota: Se o main model salvar em TXT, pode ser necessário um parser específico
        # Aqui assumimos que foram convertidos ou seguem estrutura de CSV para o pandas
        for f in files_main:
            for m in methods:
                if m in os.path.basename(f) and f"_{args.iters}" in os.path.basename(f) and f"_{args.batch_size}_" in os.path.basename(f):
                    data_map[m]['main'] = pd.read_csv(f)

        valid_methods = [m for m in methods if data_map[m]['vae']
                         is not None and data_map[m]['main'] is not None]

        print(f"valid_methods: {valid_methods}")

        if valid_methods:
            fig = make_subplots(
                rows=len(valid_methods),
                cols=2,
                shared_xaxes=True
            )

            for i, m_name in enumerate(valid_methods):
                epochs_per_task = data_map[m_name]['main'].groupby('Task')[
                    'Epoch'].max().iloc[0]

                # Main Model
                plot_loss_column(
                    fig,
                    data_map[m_name]
                    ['main'], 'main', m_name, epochs_per_task
                )

                # Generative Model
                if m_name != "CL_NR":
                    plot_loss_column(
                        fig,
                        data_map[m_name]
                        ['vae'],
                        'vae',
                        m_name, epochs_per_task
                    )
                else:
                    axes[i, 1].axis('off')

            axes[-1, 0].set_xlabel("Continuous training epochs")
            axes[-1, 1].set_xlabel("Continuous training epochs")

            plt.tight_layout()
            plt.savefig(os.path.join(
                plots_dir, "metrics_losses.png"), dpi=300)
            plt.close()


if __name__ == '__main__':
    run(get_args())
