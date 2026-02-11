#!/usr/bin/env python3
import os
import re
import argparse
import glob
import pandas as pd
import numpy as np

import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

# =========================================================
# CONFIGURAÇÃO DE LABELS E CORES
# =========================================================
METRIC_LABELS = {
    "average_prediction_error_ape_ade": "APE: ADE",
    "average_prediction_error_ape_fde": "APE: FDE",
    "backward_transfer_bwt_ade": "BWT: ADE",
    "backward_transfer_bwt_fde": "BWT: FDE",
    "forward_transfer_fwt_ade": "FWT: ADE",
    "forward_transfer_fwt_fde": "FWT: FDE",
    "training_time_in_secs": "Training Duration (s)",
    "ade": "ADE",
    "fde": "FDE",
    "method": "Method",
    "Loss_Total": "Train: total",
    "ReconL": "Train: reconstruction (current)",
    "VariatL": "Train: variational (current)",
    "ReconL_R": "Train: reconstruction (replay)",
    "VariatL_R": "Train: variational (replay)",
    "Pred_Traj": "Train: predicted trajectory (current)",
    "Pred_Traj_R": "Train: predicted trajectory (replay)",
    "Loss_Validation": "Validation",
    "ReconL_Validation": "Validation: reconstruction (current)",
    "VariatL_Validation": "Validation: variational (current)",
    "Loss_Total_Validation": "Validation: total (current)",
    "ReconL_Validation_Task1": "Validation: reconstruction (task 1)",
    "ReconL_Validation_Task2": "Validation: reconstruction (task 2)",
    "ReconL_Validation_Task3": "Validation: reconstruction (task 3)",
}

# =========================================================
# UTILS & DATA PROCESSING
# =========================================================


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument(
        "--hyperparameters",
        nargs="+",
        default=["batch_size", "iters"],
        choices=["batch_size", "iters", "learning_rate"],
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["ape", "bwt", "fwt", "ade", "fde",
                 "duration", "time_task", "losses"],
        choices=["ape", "bwt", "fwt", "ade", "fde",
                 "duration", "time_task", "losses"],
    )
    return parser.parse_args()


def load_metrics_dfs(directory, pattern="metrics"):
    dfs = []
    if not os.path.exists(directory):
        os.makedirs(directory)
        return []
    for f in sorted(os.listdir(directory)):
        if f.endswith(".csv") and pattern in f:
            try:
                df = pd.read_csv(os.path.join(directory, f))
                df['source_file'] = f
                dfs.append(df)
            except Exception as e:
                print(f"Erro ao ler {f}: {e}")
    return dfs


def load_evolution_matrices(directory, filters, selected_methods):
    """
    Carrega matrizes completas de ADE e FDE para os métodos selecionados.
    Retorna: {method: {'ade': matrix, 'fde': matrix}}
    """
    evolution_data = {}

    # Filtra métodos excluindo IL (conforme pedido para os gráficos de evolução)
    target_methods = [m for m in selected_methods if m != "IL"]

    def file_matches_filters(fname, filters_dict):
        for k, v in filters_dict.items():
            if str(v) not in fname:
                return False
        return True

    def load_matrix_file(filepath):
        try:
            return np.loadtxt(filepath, delimiter=',')
        except ValueError:
            try:
                return np.loadtxt(filepath)
            except Exception:
                return None

    for m in target_methods:
        # Padrões de busca para o método específico
        # Assume que o nome do método está no nome do arquivo
        ade_files = glob.glob(os.path.join(
            directory, f"*{m}*ADE_matrix_*.txt"))
        fde_files = glob.glob(os.path.join(
            directory, f"*{m}*FDE_matrix_*.txt"))

        # Filtra pelos hiperparâmetros
        ade_files = [f for f in ade_files if file_matches_filters(
            os.path.basename(f), filters)]
        fde_files = [f for f in fde_files if file_matches_filters(
            os.path.basename(f), filters)]

        if ade_files or fde_files:
            evolution_data[m] = {'ade': None, 'fde': None}

            if ade_files:
                # Pega o primeiro match (assumindo unicidade pelos filtros)
                evolution_data[m]['ade'] = load_matrix_file(ade_files[0])

            if fde_files:
                evolution_data[m]['fde'] = load_matrix_file(fde_files[0])

    return evolution_data


def load_loss_dfs(directory, filters, selected_methods):
    loss_data = {}
    files_vae = sorted(
        glob.glob(os.path.join(directory, "*_losses_vae_*.csv")))
    files_main = sorted(
        glob.glob(os.path.join(directory, "*_losses_main_*.csv")))

    def file_matches_filters(fname, filters_dict):
        for k, v in filters_dict.items():
            if str(v) not in fname:
                return False
        return True

    for m in selected_methods:
        loss_data[m] = {'vae': None, 'main': None}
        for f in files_main:
            if m in os.path.basename(f) and file_matches_filters(os.path.basename(f), filters):
                loss_data[m]['main'] = pd.read_csv(f)
                break
        if m != "CL_NR":
            for f in files_vae:
                if m in os.path.basename(f) and file_matches_filters(os.path.basename(f), filters):
                    loss_data[m]['vae'] = pd.read_csv(f)
                    break
        else:
            loss_data[m]['vae'] = 'off'
    return loss_data


def compute_nbwt(df):
    if df.empty or not {"IL", "CL_NR"}.issubset(df["method"].unique()):
        return df
    try:
        il_val = df.loc[df["method"] == "IL",
                        "backward_transfer_bwt_ade"].mean()
        clnr_val = df.loc[df["method"] == "CL_NR",
                          "backward_transfer_bwt_ade"].iloc[0]
        denom = clnr_val - il_val
        df["normalized_backward_transfer_nbwt_ade"] = np.nan
        mask = ~df["method"].isin(["IL", "CL_NR"])
        if abs(denom) > 1e-9:
            df.loc[mask, "normalized_backward_transfer_nbwt_ade"] = (
                (df.loc[mask, "backward_transfer_bwt_ade"] - il_val) / denom)
        df.loc[df["method"] == "CL_NR",
               "normalized_backward_transfer_nbwt_ade"] = 1.0
    except Exception as e:
        print(f"Erro ao calcular NBWT: {e}")
    return df


def extract_unique_values(dfs, columns):
    out = {}
    if not dfs:
        return {c: [] for c in columns}
    for c in columns:
        vals = set()
        for df in dfs:
            if c in df.columns:
                vals.update(df[c].dropna().unique())
        try:
            out[c] = sorted(vals, key=float)
        except:
            out[c] = sorted(vals)
    return out


def extract_available_tasks(dfs):
    tasks = set()
    pattern = re.compile(r"task[_]?(\d+)")
    if not dfs:
        return [1]
    for df in dfs:
        for col in df.columns:
            match = pattern.search(col)
            if match:
                tasks.add(int(match.group(1)))
    sorted_tasks = sorted(list(tasks))
    return sorted_tasks if sorted_tasks else [1]


def format_method_name(name):
    if str(name) == "IL_mean":
        return "IL (average)"
    return str(name).replace("_", "-")

# =========================================================
# PLOTTING FUNCTIONS
# =========================================================


def plot_bar_chart(df, y_cols, title, ylabel):
    fig = go.Figure()
    df_grouped = df.groupby("method")[y_cols].mean().reset_index()
    df_grouped["display_method"] = df_grouped["method"].apply(
        format_method_name)
    for c in y_cols:
        if c not in df_grouped.columns:
            continue
        label = METRIC_LABELS.get(c, c)
        fig.add_trace(go.Bar(x=df_grouped["display_method"], y=df_grouped[c], name=label, text=df_grouped[c].apply(
            lambda x: f'{x:.3f}'), textposition='auto'))
    fig.update_layout(title=title, yaxis_title=ylabel, xaxis_title="Method",
                      barmode='group', legend_itemclick="toggle", legend_itemdoubleclick="toggleothers")
    return fig


def plot_per_task_metric(df, metric_prefix, title, ylabel, y_range=None):
    fig = go.Figure()
    if df.empty:
        return fig
    methods = df["method"].unique()
    if metric_prefix == "time":
        task_pattern = re.compile(r"task[_]?(\d+)")
    else:
        task_pattern = re.compile(
            rf"{metric_prefix}_task(\d+)_after_training_in_all_tasks")
    tasks = set()
    col_map = {}
    for col in df.columns:
        match = task_pattern.search(col)
        if match:
            if metric_prefix == "time" and "time" not in col:
                continue
            tid = int(match.group(1))
            tasks.add(tid)
            if metric_prefix in col or (metric_prefix == "time" and "time" in col):
                col_map[tid] = col
    sorted_tasks = sorted(list(tasks))
    if not sorted_tasks:
        fig.update_layout(title=f"{title} (Nenhuma coluna encontrada)")
        return fig
    for method in methods:
        row = df[df["method"] == method]
        if row.empty:
            continue
        row_vals = row.mean(numeric_only=True)
        y_values = []
        for t in sorted_tasks:
            c = col_map.get(t)
            val = row_vals[c] if c in row_vals else np.nan
            y_values.append(val)
        fig.add_trace(go.Scatter(x=sorted_tasks, y=y_values,
                      mode='lines+markers', name=format_method_name(method)))
    dataset_text = ""
    if "train_dataset" in df.columns:
        try:
            dset = df["train_dataset"].iloc[0]
            dataset_text = f"<br><sup>Tasks order: {dset}</sup>"
        except:
            pass
    layout_args = dict(title=title + dataset_text, xaxis_title="Tasks", yaxis_title=ylabel, xaxis=dict(tickmode='array',
                       tickvals=sorted_tasks), hovermode="x unified", legend_itemclick="toggle", legend_itemdoubleclick="toggleothers")
    if y_range:
        layout_args['yaxis'] = dict(range=y_range)
    fig.update_layout(**layout_args)
    return fig


def plot_method_evolution(method_name, ade_matrix, fde_matrix):
    """
    Cria subplots (2 colunas) para a evolução de ADE e FDE.
    Cada linha no gráfico representa uma coluna da matriz (Performance em uma Tarefa específica).
    O eixo X representa o estágio do treino (Linhas da matriz: 0=Random, 1=After T1...).
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("ADE Evolution", "FDE Evolution"),
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.05
    )

    matrices = [('ADE', ade_matrix), ('FDE', fde_matrix)]
    colors = pc.qualitative.Plotly  # Paleta de cores para distinguir as tarefas

    for i, (metric, matrix) in enumerate(matrices):
        if matrix is None:
            continue

        col_idx = i + 1
        num_rows, num_cols = matrix.shape

        # Eixo X: Estágios de treino (0 = Baseline, 1..N = Após treinar Task N)
        # Assumindo que a matriz tem formato (N_tasks + 1, N_tasks)
        x_stages = list(range(num_rows))
        x_labels = ["Random"] + [f"After T{t+1}" for t in range(num_rows-1)]

        for task_idx in range(num_cols):
            # Extrai a coluna correspondente à tarefa task_idx
            y_values = matrix[:, task_idx]

            fig.add_trace(go.Scatter(
                x=x_stages,
                y=y_values,
                mode='lines+markers',
                name=f"Test on Task {task_idx+1}",
                line=dict(color=colors[task_idx % len(colors)]),
                legendgroup=f"Task {task_idx+1}",
                # Mostra legenda apenas no primeiro subplot para não duplicar
                showlegend=(i == 0)
            ), row=1, col=col_idx)

    fig.update_layout(
        title_text=f"Performance Evolution: {format_method_name(method_name)}",
        height=500,
        hovermode="x unified",
        xaxis=dict(tickmode='array', tickvals=x_stages,
                   ticktext=x_labels, title="Training Stage"),
        xaxis2=dict(tickmode='array', tickvals=x_stages,
                    ticktext=x_labels, title="Training Stage"),
        yaxis=dict(title="Error (Meters)"),
    )
    return fig


def plot_losses_subplots(loss_data, selected_tasks):
    valid_methods = [m for m, data in loss_data.items()
                     if data['main'] is not None]
    if not valid_methods:
        return None
    valid_methods.sort(key=lambda x: format_method_name(x))
    rows = len(valid_methods)
    cols = 2
    titles = []
    for m in valid_methods:
        display_name = format_method_name(m)
        titles.append(f"{display_name}: Main model")
        vae_data = loss_data[m]['vae']
        if isinstance(vae_data, pd.DataFrame):
            titles.append(f"{display_name}: Social-GR (VAE)")
        else:
            titles.append("")
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        shared_xaxes=True, vertical_spacing=0.08)
    styles_vae = {
        'Loss_Total': {'color': 'black', 'dash': 'dash'},
        'ReconL': {'color': 'blue', 'dash': None},
        'VariatL': {'color': 'green', 'dash': None},
        'ReconL_R': {'color': 'cyan', 'dash': 'dot'},
        'VariatL_R': {'color': 'lime', 'dash': 'dot'},
        'ReconL_Validation': {'color': 'olive', 'dash': 'dashdot'},
        'VariatL_Validation': {'color': 'orange', 'dash': 'dashdot'},
        'Loss_Total_Validation': {'color': 'paleturquoise', 'dash': 'dashdot'},
        'ReconL_Validation_Task1': {'color': 'sandybrown', 'dash': 'dashdot'},
        'ReconL_Validation_Task2': {'color': 'magenta', 'dash': 'dashdot'},
        'ReconL_Validation_Task3': {'color': 'darksalmon', 'dash': 'dashdot'}
    }
    
    styles_main = {'Loss_Total': {'color': 'black', 'dash': 'dash'}, 'Pred_Traj': {'color': 'cyan', 'dash': None},
                   'Pred_Traj_R': {'color': 'lime', 'dash': 'dot'}, 'Loss_Validation': {'color': 'red', 'dash': 'dashdot'}}
    sel_tasks_int = [int(t) for t in selected_tasks] if selected_tasks else []
    for i, method in enumerate(valid_methods):
        row_idx = i + 1
        display_method = format_method_name(method)
        group_main = f"{display_method}: Main model"
        group_vae = f"{display_method}: Social-GR"
        df_main = loss_data[method]['main']
        if df_main is not None:
            if sel_tasks_int and 'Task' in df_main.columns:
                df_main = df_main[df_main['Task'].isin(sel_tasks_int)].copy()
            else:
                df_main = df_main.copy()
            epochs_per_task = df_main.groupby('Task')['Epoch'].max(
            ).iloc[0] if 'Task' in df_main.columns and not df_main.empty else 1
            if not df_main.empty:
                df_main['Global_Step'] = (
                    df_main['Task'] - 1) * epochs_per_task + df_main['Epoch']
                for col_name, style in styles_main.items():
                    if col_name in df_main.columns:
                        metric_name = "Train: average" if col_name == "Loss_Total" else METRIC_LABELS.get(
                            col_name, col_name)
                        fig.add_trace(go.Scatter(x=df_main['Global_Step'], y=df_main[col_name], name=metric_name, line=dict(
                            color=style['color'], dash=style['dash']), legendgroup=group_main, legendgrouptitle_text=group_main, showlegend=True), row=row_idx, col=1)
                unique_tasks = sorted(df_main['Task'].unique())
                for t in unique_tasks:
                    fig.add_vline(x=t*epochs_per_task, line_color="gray",
                                  line_width=1, row=row_idx, col=1)
        df_vae = loss_data[method]['vae']
        if isinstance(df_vae, pd.DataFrame):
            if sel_tasks_int and 'Task' in df_vae.columns:
                df_vae = df_vae[df_vae['Task'].isin(sel_tasks_int)].copy()
            else:
                df_vae = df_vae.copy()
            epochs_per_task = df_vae.groupby('Task')['Epoch'].max(
            ).iloc[0] if 'Task' in df_vae.columns and not df_vae.empty else 1
            if not df_vae.empty:
                df_vae['Global_Step'] = (
                    df_vae['Task'] - 1) * epochs_per_task + df_vae['Epoch']
                for col_name, style in styles_vae.items():
                    if col_name in df_vae.columns:
                        metric_name = METRIC_LABELS.get(col_name, col_name)
                        fig.add_trace(go.Scatter(x=df_vae['Global_Step'], y=df_vae[col_name], name=metric_name, line=dict(
                            color=style['color'], dash=style['dash']), legendgroup=group_vae, legendgrouptitle_text=group_vae, showlegend=True), row=row_idx, col=2)
                unique_tasks = sorted(df_vae['Task'].unique())
                for t in unique_tasks:
                    fig.add_vline(x=t*epochs_per_task, line_color="gray",
                                  line_width=1, row=row_idx, col=2)
    fig.update_layout(height=400 * rows, title_text="Losses for both main and generative models through continuous training iters (epochs)",
                      legend=dict(tracegroupgap=20, groupclick="toggleitem"))
    return fig


def plot_vae_phase_chart(loss_data, selected_tasks):
    valid_methods = [m for m, data in loss_data.items(
    ) if isinstance(data['vae'], pd.DataFrame)]
    if not valid_methods:
        return None
    valid_methods.sort(key=lambda x: format_method_name(x))
    COLOR_START = "#3C8004"
    COLOR_END = "#D51A06"
    COLOR_BEST = "#F0CE53"
    LINE_COLORSCALE = 'RdBu_r'
    fig = make_subplots(rows=len(valid_methods), cols=2, subplot_titles=[f"{format_method_name(m)}: Phase (Current)" for m in valid_methods] + [
                        f"{format_method_name(m)}: Phase (Replay)" for m in valid_methods], vertical_spacing=0.15)
    sel_tasks_int = [int(t) for t in selected_tasks] if selected_tasks else []
    max_global_epoch = 0
    for m in valid_methods:
        df = loss_data[m]['vae']
        if sel_tasks_int and 'Task' in df.columns:
            df = df[df['Task'].isin(sel_tasks_int)]
        if not df.empty:
            current_max = ((df['Task'] - 1) * 200 + df['Epoch']).max()
            if current_max > max_global_epoch:
                max_global_epoch = current_max
    for i, method in enumerate(valid_methods):
        df_vae = loss_data[method]['vae'].copy()
        if sel_tasks_int and 'Task' in df_vae.columns:
            df_vae = df_vae[df_vae['Task'].isin(sel_tasks_int)]
        if df_vae.empty:
            continue
        df_vae['Global_Epoch'] = (df_vae['Task'] - 1) * 200 + df_vae['Epoch']
        row_idx = i + 1

        def add_plot(df, x_col, y_col, col_idx, label_suffix):
            min_epoch = int(df['Global_Epoch'].min())
            max_epoch = int(df['Global_Epoch'].max())
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines+markers', name=f"{format_method_name(method)}: {label_suffix}", legendgroup=f"group_{row_idx}_{col_idx}", marker=dict(color=df['Global_Epoch'], colorscale=LINE_COLORSCALE, cmin=min_epoch, cmax=max_epoch, size=6, showscale=(
                i == 0 and col_idx == 2), colorbar=dict(title="Global Epoch", x=1.08) if (i == 0 and col_idx == 2) else None,), hovertext=[f"Task {t}, Epoch {e}" for t, e in zip(df['Task'], df['Epoch'])]), row=row_idx, col=col_idx)
            fig.add_trace(go.Scatter(x=[df[x_col].iloc[0]], y=[df[y_col].iloc[0]], mode='markers', name='Start', marker=dict(
                color=COLOR_START, size=10, symbol='circle', line=dict(width=1, color='black')), legendgroup=f"group_{row_idx}_{col_idx}", showlegend=False), row=row_idx, col=col_idx)
            fig.add_trace(go.Scatter(x=[df[x_col].iloc[-1]], y=[df[y_col].iloc[-1]], mode='markers', name='End', marker=dict(color=COLOR_END, size=10,
                          symbol='star', line=dict(width=1, color='black')), legendgroup=f"group_{row_idx}_{col_idx}", showlegend=False), row=row_idx, col=col_idx)
            idx_best = df[y_col].idxmin()
            fig.add_trace(go.Scatter(x=[df.loc[idx_best, x_col]], y=[df.loc[idx_best, y_col]], mode='markers', name='Best Recon', marker=dict(
                color=COLOR_BEST, size=10, symbol='star', line=dict(width=1, color='orange')), legendgroup=f"group_{row_idx}_{col_idx}", showlegend=False), row=row_idx, col=col_idx)
        add_plot(df_vae, 'VariatL', 'ReconL', 1, "Current")
        if 'ReconL_R' in df_vae.columns and df_vae['ReconL_R'].sum() > 0:
            add_plot(df_vae, 'VariatL_R', 'ReconL_R', 2, "Replay")
    fig.update_xaxes(title_text="Variational loss")
    fig.update_yaxes(title_text="Reconstruction loss")
    fig.update_layout(height=450 * len(valid_methods), title_text="VAE Training Phase Space", showlegend=True,
                      margin=dict(t=150, r=100), legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5))
    return fig


def compute_figures(metrics_dfs, loss_dfs_dict, evolution_data, selected_plots, selected_tasks):
    figs = {}

    if metrics_dfs:
        df = pd.concat(metrics_dfs, ignore_index=True)
        df["method"] = df["method"].astype(str)
        df = compute_nbwt(df)
        df_cl = df[df["method"].str.contains("CL", na=False)].copy()
        df_il = df[df["method"] == "IL"].copy()

        df_ape_source = df_cl.copy()
        df_duration_source = df_cl.copy()

        if not df_il.empty:
            il_avg = df_il.select_dtypes(include=[np.number]).mean().to_dict()
            il_avg["method"] = "IL_mean"
            df_il_mean = pd.DataFrame([il_avg])
            df_ape_source = pd.concat(
                [df_ape_source, df_il_mean], ignore_index=True)
            df_duration_source = pd.concat(
                [df_duration_source, df_il_mean], ignore_index=True)

        if "ape" in selected_plots:
            figs["APE"] = plot_bar_chart(df_ape_source, ["average_prediction_error_ape_ade", "average_prediction_error_ape_fde"],
                                         "Average Prediction Error (APE) of final models on full benchmark test split", "Meters")

        # --- NOVO GRÁFICO: Evolution from Matrices ---
        if "ade" in selected_plots and evolution_data:
            # Itera sobre os métodos carregados e cria um gráfico para cada
            for method_name, matrices in evolution_data.items():
                if matrices['ade'] is not None or matrices['fde'] is not None:
                    fig_evol = plot_method_evolution(
                        method_name, matrices['ade'], matrices['fde'])
                    figs[f"Evolution ({format_method_name(method_name)})"] = fig_evol

        if "bwt" in selected_plots or "forgetting" in selected_plots:
            figs["BWT"] = plot_bar_chart(df_cl, [
                                         "backward_transfer_bwt_ade", "backward_transfer_bwt_fde"], "Backward Transfer (BWT)", "Value")

        if "fwt" in selected_plots or "forgetting" in selected_plots:
            figs["FWT"] = plot_bar_chart(df_cl, [
                                         "forward_transfer_fwt_ade", "forward_transfer_fwt_fde"], "Forward Transfer (FWT)", "Value")

        if "duration" in selected_plots:
            figs["Duration"] = plot_bar_chart(df_duration_source, [
                                              "training_time_in_secs"], "Total duration of experiment", "Seconds")

        max_y = 0
        ade_patt = re.compile(r"ade_task\d+_")
        fde_patt = re.compile(r"fde_task\d+_")
        all_cols = []
        for c in df_cl.columns:
            if ade_patt.search(c) or fde_patt.search(c):
                all_cols.append(c)
        if all_cols and not df_cl.empty:
            curr_max = df_cl[all_cols].max().max()
            if pd.notna(curr_max):
                max_y = curr_max * 1.1
        common_range = [0, max_y] if max_y > 0 else None

        if "ade" in selected_plots:
            figs["ADE per Task (Final Model)"] = plot_per_task_metric(
                df_cl, "ade", "Average displacement error (ADE) of final model per task-specific test split", "Meters", y_range=common_range)

        if "fde" in selected_plots:
            figs["FDE per Task (Final Model)"] = plot_per_task_metric(
                df_cl, "fde", "Final displacement error (FDE) of final model per task-specific test split", "Meters", y_range=common_range)

        if "time_task" in selected_plots or "duration" in selected_plots:
            figs["Elapsed time per task"] = plot_per_task_metric(
                df_cl, "time", "Elapsed time per task", "Seconds")

    if "losses" in selected_plots and loss_dfs_dict:
        fig_loss = plot_losses_subplots(loss_dfs_dict, selected_tasks)
        if fig_loss:
            figs["Losses"] = fig_loss
        fig_phase = plot_vae_phase_chart(loss_dfs_dict, selected_tasks)
        if fig_phase:
            figs["VAE Phase Plot (Recon vs Variat)"] = fig_phase

    return figs

# =========================================================
# DASH UI
# =========================================================


def build_sidebar(hparams, available_tasks):
    task_options = [{'label': f'Task {i}', 'value': i}
                    for i in available_tasks]
    default_tasks = available_tasks
    return html.Div([html.H3("Filters"), html.Hr(), *[html.Div([html.Label(p.replace("_", " ").title()), dcc.Dropdown(id=f"dropdown-{p}", options=[{"label": str(v), "value": v} for v in vals], clearable=True, placeholder="Select...")], className="mb-3") for p, vals in hparams.items()], html.Hr(), html.Label("Tasks for losses plot:"), dcc.Checklist(id='task-checklist', options=task_options, value=default_tasks, inline=False, inputStyle={"margin-right": "5px", "margin-left": "5px"}), html.Hr(), dbc.Button("Apply Filters", id="apply-btn", color="primary", className="w-100"), html.Hr(), html.Small("Select filters and click 'Apply Filters' to render.", className="text-muted")], style={"position": "fixed", "left": 0, "top": 0, "bottom": 0, "width": "20rem", "padding": "2rem", "background": "#f8f9fa", "overflow-y": "auto"})


def build_instruction_message():
    return dbc.Container([html.H1("Results dashboard"), html.Hr(), dbc.Alert([html.H4("Waiting Selection"), html.P("Please select all hyperparameters and click 'Apply Filters'.")], color="info")], fluid=True, style={"margin-top": "2rem"})


def build_main(figs):
    children = [html.H1("Results dashboard"), html.Hr()]
    if not figs:
        children += [dbc.Alert("No data found.", color="warning")]
    else:
        for name, fig in figs.items():
            style = {
                "height": "1000px"} if "Losses" in name or "Phase" in name else {}
            children += [dbc.Card([dbc.CardHeader(html.H4(name, className="m-0")), dbc.CardBody(
                dcc.Graph(figure=fig, style=style))], className="mb-4 shadow-sm")]
    return dbc.Container(children, fluid=True)


# =========================================================
# MAIN APP
# =========================================================
if __name__ == "__main__":
    args = get_args()
    all_metrics_dfs = load_metrics_dfs(args.results_dir)
    hyper_vals = extract_unique_values(all_metrics_dfs, args.hyperparameters)
    available_tasks = extract_available_tasks(all_metrics_dfs)

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([build_sidebar(hyper_vals, available_tasks), html.Div(
        id="page-content", children=build_instruction_message(), style={"margin-left": "22rem", "padding": "2rem"})])

    @callback(Output("page-content", "children"), [Input("apply-btn", "n_clicks")], [*(State(f"dropdown-{p}", "value") for p in hyper_vals.keys()), State("task-checklist", "value")])
    def update(n_clicks, *values):
        task_values = values[-1]
        dropdown_values = values[:-1]
        if n_clicks is None or any(v is None for v in dropdown_values):
            return build_instruction_message()
        filters = dict(zip(hyper_vals.keys(), dropdown_values))

        filtered_metrics = []
        available_methods = set()
        for df in all_metrics_dfs:
            dff = df.copy()
            match = True
            for k, v in filters.items():
                if k in dff.columns:
                    if dff[k].iloc[0] != v:
                        match = False
                        break
            if match:
                filtered_metrics.append(dff)
                available_methods.update(dff["method"].unique())

        loss_dfs = {}
        evolution_data = {}

        if filtered_metrics:
            if "losses" in args.plots:
                loss_dfs = load_loss_dfs(
                    args.results_dir, filters, available_methods)
            if "ade" in args.plots:
                # Carrega as matrizes completas
                evolution_data = load_evolution_matrices(
                    args.results_dir, filters, available_methods)

        figs = compute_figures(filtered_metrics, loss_dfs,
                               evolution_data, args.plots, task_values)
        return build_main(figs)

    app.run(debug=True, port=8050)
