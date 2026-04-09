#!/usr/bin/env python3
import os
import re
import argparse
import glob
import tikzplotly
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
    "Pred_Traj": "Train: predicted trajectory",
    "Pred_Traj_R": "Train: predicted trajectory (replay)",
    "Loss_Validation": "Validation",
    "emissions": "CO₂ Emissions",
    "energy_consumed": "Total Energy",
    "cpu_energy": "CPU Energy",
    "gpu_energy": "GPU Energy",
    "ram_energy": "RAM Energy",
}

# =========================================================
# UTILS & DATA PROCESSING
# =========================================================


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument(
        "--hyperparameters", nargs="+", default=["batch_size", "iters"],
        choices=["batch_size", "iters", "learning_rate"],
    )
    parser.add_argument(
        "--plots", nargs="+",
        default=["ape", "bwt", "fwt", "ade", "fde",
                 "duration", "time_task", "losses", "emissions"],
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


def load_emissions_dfs(directory, filters, selected_methods):
    dfs = []
    files = glob.glob(os.path.join(directory, "*emissions*.csv"))

    def file_matches_filters(fname, filters_dict):
        for k, v in filters_dict.items():
            if str(v) not in fname:
                return False
        return True

    for m in selected_methods:
        if m == "IL_mean":
            continue
        for f in files:
            fname = os.path.basename(f)
            if m in fname and file_matches_filters(fname, filters):
                try:
                    df = pd.read_csv(f)
                    df['method'] = m
                    dfs.append(df)
                    break
                except Exception as e:
                    print(f"Erro ao ler arquivo de emissão {f}: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_evolution_matrices(directory, filters, selected_methods):
    evolution_data = {}
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
        ade_files = [f for f in glob.glob(os.path.join(
            directory, f"*{m}*ADE_matrix_*.txt")) if file_matches_filters(os.path.basename(f), filters)]
        fde_files = [f for f in glob.glob(os.path.join(
            directory, f"*{m}*FDE_matrix_*.txt")) if file_matches_filters(os.path.basename(f), filters)]

        if ade_files or fde_files:
            evolution_data[m] = {'ade': None, 'fde': None}
            if ade_files:
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


def extract_available_methods(dfs):
    methods = set()
    for df in dfs:
        methods.update(df["method"].unique())
    return sorted(list(methods))


def extract_available_tasks(dfs):
    tasks = set()
    pattern = re.compile(r"task[_]?(\d+)")
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
    actual_cols = [c for c in y_cols if c in df_grouped.columns]
    x_labels = [METRIC_LABELS.get(c, c) for c in actual_cols]

    for _, row in df_grouped.iterrows():
        method_name = row["display_method"]
        y_values = [row[c] for c in actual_cols]
        text_values = [f'{v:.5f}' if (pd.notna(v) and v < 0.1 and v != 0) else (
            f'{v:.3f}' if pd.notna(v) else '') for v in y_values]

        fig.add_trace(go.Bar(
            name=method_name, x=x_labels, y=y_values, text=text_values, textposition='auto',
            hovertemplate="<b>Method:</b> " + method_name +
            "<br><b>%{x}:</b> %{y}<extra></extra>"
        ))

    fig.update_layout(title=title, yaxis_title=ylabel, barmode='group',
                      legend_title_text="Method (Experiment)", legend_itemclick="toggle")
    return fig


def plot_per_task_metric(df, metric_prefix, title, ylabel, y_range=None):
    fig = go.Figure()
    if df.empty:
        return fig
    methods = df["method"].unique()
    task_pattern = re.compile(r"task[_]?(\d+)") if metric_prefix == "time" else re.compile(
        rf"{metric_prefix}_task(\d+)_after_training_in_all_tasks")
    tasks, col_map = set(), {}

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
        return fig

    for method in methods:
        row = df[df["method"] == method]
        if row.empty:
            continue
        row_vals = row.mean(numeric_only=True)
        y_values = [row_vals[col_map.get(t)] if col_map.get(
            t) in row_vals else np.nan for t in sorted_tasks]
        fig.add_trace(go.Scatter(x=sorted_tasks, y=y_values,
                      mode='lines+markers', name=format_method_name(method)))

    layout_args = dict(title=title, xaxis_title="Tasks", yaxis_title=ylabel, xaxis=dict(
        tickmode='array', tickvals=sorted_tasks), hovermode="x unified")
    if y_range:
        layout_args['yaxis'] = dict(range=y_range)
    fig.update_layout(**layout_args)
    return fig


def plot_evolution_per_task(evolution_data, task_idx):
    """
    Gera um plot com 2 subplots (ADE e FDE) específico para uma Tarefa,
    contendo a evolução de todos os métodos fornecidos.
    """
    valid_methods = [m for m in evolution_data if evolution_data[m]
                     ['ade'] is not None or evolution_data[m]['fde'] is not None]
    if not valid_methods:
        return None

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"ADE Evolution (Tested on Task {task_idx})", f"FDE Evolution (Tested on Task {task_idx})"), shared_xaxes=True, shared_yaxes=True)
    colors = pc.qualitative.Plotly
    methods = sorted(valid_methods)
    method_colors = {m: colors[idx % len(colors)]
                     for idx, m in enumerate(methods)}

    has_data_plotted = False

    for i, metric in enumerate(['ade', 'fde']):
        col_idx = i + 1
        for m_name in methods:
            matrix = evolution_data[m_name][metric]
            if matrix is None:
                continue

            num_rows, num_cols = matrix.shape
            actual_task_index = task_idx - 1  # Ajuste de índice 0-based

            if actual_task_index >= num_cols:
                continue

            x_stages = list(range(num_rows))
            x_labels = ["Random"] + \
                [f"After T{t+1}" for t in range(num_rows-1)]
            y_values = matrix[:, actual_task_index]

            show_leg = (i == 0)  # Mostra legenda apenas no primeiro subplot

            fig.add_trace(go.Scatter(
                x=x_stages, y=y_values, mode='lines+markers', name=format_method_name(m_name),
                legendgroup=m_name, line=dict(color=method_colors[m_name], width=2), showlegend=show_leg,
                hovertemplate=f"<b>{format_method_name(m_name)}</b><br>Stage: %{{x}}<br>Error: %{{y:.4f}}<extra></extra>"
            ), row=1, col=col_idx)

            has_data_plotted = True

            if i == 0:
                fig.update_xaxes(tickmode='array', tickvals=x_stages, ticktext=x_labels,
                                 title_text="Training Stage", row=1, col=col_idx)
            else:
                fig.update_xaxes(tickmode='array', tickvals=x_stages, ticktext=x_labels,
                                 title_text="Training Stage", row=1, col=col_idx)

    fig.update_yaxes(title_text="Error (Meters)", row=1, col=1)
    fig.update_layout(height=450, hovermode="x unified",
                      legend_title_text="Methods")

    return fig if has_data_plotted else None


def plot_main_losses(loss_data, selected_tasks):
    valid_methods = [m for m, data in loss_data.items()
                     if data['main'] is not None]
    if not valid_methods:
        return None
    valid_methods.sort(key=lambda x: format_method_name(x))

    rows = len(valid_methods)
    titles = [f"{format_method_name(m)}: Main model" for m in valid_methods]
    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles,
                        shared_xaxes=True, vertical_spacing=0.08)

    styles_main = {'Loss_Total': {'color': 'black', 'dash': 'dash'}, 'Pred_Traj': {'color': 'cyan', 'dash': None},
                   'Pred_Traj_R': {'color': 'lime', 'dash': 'dot'}, 'Loss_Validation': {'color': 'red', 'dash': 'dashdot'}}
    sel_tasks_int = [int(t) for t in selected_tasks] if selected_tasks else []

    for i, method in enumerate(valid_methods):
        row_idx = i + 1
        group_main = f"{format_method_name(method)}: Main"
        df_main = loss_data[method]['main']

        if df_main is not None:
            if sel_tasks_int and 'Task' in df_main.columns:
                df_main = df_main[df_main['Task'].isin(sel_tasks_int)].copy()
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
                            color=style['color'], dash=style['dash']), legendgroup=group_main, showlegend=(i == 0)), row=row_idx, col=1)
                for t in sorted(df_main['Task'].unique()):
                    fig.add_vline(x=t*epochs_per_task, line_color="gray",
                                  line_width=1, row=row_idx, col=1)

    fig.update_layout(
        height=max(300, 200 * rows), title_text="Losses for Main Model through continuous training")
    return fig


def plot_vae_losses(loss_data, selected_tasks):
    valid_methods = [m for m, data in loss_data.items(
    ) if isinstance(data['vae'], pd.DataFrame)]
    if not valid_methods:
        return None
    valid_methods.sort(key=lambda x: format_method_name(x))

    rows = len(valid_methods)
    titles = [
        f"{format_method_name(m)}: Social-GR (Generative)" for m in valid_methods]
    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles,
                        shared_xaxes=True, vertical_spacing=0.08)

    styles_vae = {
        'Loss_Total': {'color': 'black', 'dash': 'dash'}, 'ReconL': {'color': 'blue', 'dash': None},
        'VariatL': {'color': 'green', 'dash': None}, 'ReconL_R': {'color': 'cyan', 'dash': 'dot'},
        'VariatL_R': {'color': 'lime', 'dash': 'dot'}, 'ReconL_Validation': {'color': 'olive', 'dash': 'dashdot'},
        'VariatL_Validation': {'color': 'orange', 'dash': 'dashdot'}, 'Loss_Total_Validation': {'color': 'paleturquoise', 'dash': 'dashdot'},
        'ReconL_Validation_Task1': {'color': 'sandybrown', 'dash': 'dashdot'}, 'ReconL_Validation_Task2': {'color': 'magenta', 'dash': 'dashdot'},
        'ReconL_Validation_Task3': {'color': 'darksalmon', 'dash': 'dashdot'}
    }
    sel_tasks_int = [int(t) for t in selected_tasks] if selected_tasks else []

    for i, method in enumerate(valid_methods):
        row_idx = i + 1
        group_vae = f"{format_method_name(method)}: VAE"
        df_vae = loss_data[method]['vae']

        if isinstance(df_vae, pd.DataFrame):
            if sel_tasks_int and 'Task' in df_vae.columns:
                df_vae = df_vae[df_vae['Task'].isin(sel_tasks_int)].copy()
            epochs_per_task = df_vae.groupby('Task')['Epoch'].max(
            ).iloc[0] if 'Task' in df_vae.columns and not df_vae.empty else 1
            if not df_vae.empty:
                df_vae['Global_Step'] = (
                    df_vae['Task'] - 1) * epochs_per_task + df_vae['Epoch']
                for col_name, style in styles_vae.items():
                    if col_name in df_vae.columns:
                        metric_name = METRIC_LABELS.get(col_name, col_name)
                        fig.add_trace(go.Scatter(x=df_vae['Global_Step'], y=df_vae[col_name], name=metric_name, line=dict(
                            color=style['color'], dash=style['dash']), legendgroup=group_vae, showlegend=(i == 0)), row=row_idx, col=1)
                for t in sorted(df_vae['Task'].unique()):
                    fig.add_vline(x=t*epochs_per_task, line_color="gray",
                                  line_width=1, row=row_idx, col=1)

    fig.update_layout(
        height=max(300, 200 * rows), title_text="Losses for Generative Model through continuous training")
    return fig


def compute_structured_figures(metrics_dfs, loss_dfs_dict, evolution_data, emissions_df, selected_plots, selected_tasks, active_methods):
    """
    Constrói um dicionário aninhado de figuras correspondente à estrutura de seções do LaTeX.
    """
    structured_figs = {
        "Trajectory prediction performance": {
            "Aggregated continuous learning metrics": {},
            "Granular and sequential evaluation": {
                "Final model performance across tasks": {},
                "Evolution of prediction error": {}
            }
        },
        "Computational and environmental impact": {
            "Time complexity": {},
            "Resource utilization and carbon footprint": {}
        }
    }

    if not metrics_dfs:
        return structured_figs

    df = pd.concat(metrics_dfs, ignore_index=True)
    df = df[df["method"].isin(active_methods)].copy()  # Filtra pelo Checklist
    df["method"] = df["method"].astype(str)
    df = compute_nbwt(df)

    df_cl = df[df["method"].str.contains("CL", na=False)].copy()
    df_il = df[df["method"] == "IL"].copy()

    df_ape_source = df_cl.copy()
    df_duration_source = df_cl.copy()

    if not df_il.empty and "IL_mean" in active_methods:
        il_avg = df_il.select_dtypes(include=[np.number]).mean().to_dict()
        il_avg["method"] = "IL_mean"
        df_il_mean = pd.DataFrame([il_avg])
        df_ape_source = pd.concat(
            [df_ape_source, df_il_mean], ignore_index=True)
        df_duration_source = pd.concat(
            [df_duration_source, df_il_mean], ignore_index=True)

    # 1.1 Aggregated continuous learning metrics
    if "ape" in selected_plots and not df_ape_source.empty:
        structured_figs["Trajectory prediction performance"]["Aggregated continuous learning metrics"]["APE"] = plot_bar_chart(
            df_ape_source, ["average_prediction_error_ape_ade", "average_prediction_error_ape_fde"], "Average Prediction Error (APE)", "Meters")
    if ("bwt" in selected_plots or "forgetting" in selected_plots) and not df_cl.empty:
        structured_figs["Trajectory prediction performance"]["Aggregated continuous learning metrics"]["BWT"] = plot_bar_chart(
            df_cl, ["backward_transfer_bwt_ade", "backward_transfer_bwt_fde"], "Backward Transfer (BWT)", "Value")
    if ("fwt" in selected_plots or "forgetting" in selected_plots) and not df_cl.empty:
        structured_figs["Trajectory prediction performance"]["Aggregated continuous learning metrics"]["FWT"] = plot_bar_chart(
            df_cl, ["forward_transfer_fwt_ade", "forward_transfer_fwt_fde"], "Forward Transfer (FWT)", "Value")

    # 1.2.1 Final model performance across tasks
    common_range = None
    if not df_cl.empty:
        all_cols = [c for c in df_cl.columns if re.search(
            r"ade_task\d+_", c) or re.search(r"fde_task\d+_", c)]
        if all_cols:
            common_range = [0, df_cl[all_cols].max().max() * 1.1]

    if "ade" in selected_plots and not df_cl.empty:
        structured_figs["Trajectory prediction performance"]["Granular and sequential evaluation"]["Final model performance across tasks"]["ADE per Task"] = plot_per_task_metric(
            df_cl, "ade", "ADE of final model per task", "Meters", y_range=common_range)
    if "fde" in selected_plots and not df_cl.empty:
        structured_figs["Trajectory prediction performance"]["Granular and sequential evaluation"]["Final model performance across tasks"]["FDE per Task"] = plot_per_task_metric(
            df_cl, "fde", "FDE of final model per task", "Meters", y_range=common_range)

    # 1.2.2 Evolution of prediction error (Graficos individuais por Task)
    if "ade" in selected_plots and evolution_data:
        max_tasks_found = 0
        for m_data in evolution_data.values():
            if m_data['ade'] is not None:
                max_tasks_found = max(max_tasks_found, m_data['ade'].shape[1])

        for task_idx in range(1, max_tasks_found + 1):
            if selected_tasks and task_idx not in selected_tasks:
                continue  # Respeita o filtro lateral
            fig_evol = plot_evolution_per_task(evolution_data, task_idx)
            if fig_evol:
                structured_figs["Trajectory prediction performance"]["Granular and sequential evaluation"][
                    "Evolution of prediction error"][f"Evolution on Task {task_idx}"] = fig_evol

    if "losses" in selected_plots and loss_dfs_dict:
        fig_main_loss = plot_main_losses(loss_dfs_dict, selected_tasks)
        if fig_main_loss:
            structured_figs["Trajectory prediction performance"]["Granular and sequential evaluation"][
                "Evolution of prediction error"]["Training Losses (Main Model)"] = fig_main_loss

        fig_vae_loss = plot_vae_losses(loss_dfs_dict, selected_tasks)
        if fig_vae_loss:
            structured_figs["Trajectory prediction performance"]["Granular and sequential evaluation"][
                "Evolution of prediction error"]["Training Losses (Generative Model)"] = fig_vae_loss

    # 2.1 Time complexity
    if "duration" in selected_plots and not df_duration_source.empty:
        structured_figs["Computational and environmental impact"]["Time complexity"]["Total Duration"] = plot_bar_chart(
            df_duration_source, ["training_time_in_secs"], "Total duration of experiment", "Seconds")
    if ("time_task" in selected_plots or "duration" in selected_plots) and not df_cl.empty:
        structured_figs["Computational and environmental impact"]["Time complexity"]["Elapsed time per task"] = plot_per_task_metric(
            df_cl, "time", "Elapsed time per task", "Seconds")

    # 2.2 Resource utilization and carbon footprint
    if "emissions" in selected_plots and emissions_df is not None and not emissions_df.empty:
        structured_figs["Computational and environmental impact"]["Resource utilization and carbon footprint"]["CO₂ Emissions"] = plot_bar_chart(
            emissions_df, ["emissions"], "Total CO₂ Emissions", "kg CO₂ eq")
        structured_figs["Computational and environmental impact"]["Resource utilization and carbon footprint"]["Energy Consumption Breakdown"] = plot_bar_chart(
            emissions_df, ["cpu_energy", "gpu_energy", "ram_energy", "energy_consumed"], "Energy Consumption Breakdown", "kWh")

    return structured_figs

# =========================================================
# DASH UI
# =========================================================


def build_sidebar(hparams, available_tasks, available_methods):
    task_options = [{'label': f'Task {i}', 'value': i}
                    for i in available_tasks]
    method_options = [{'label': format_method_name(
        m), 'value': m} for m in available_methods]

    return html.Div([
        html.H3("Filters"), html.Hr(),

        html.Label("Methods to compare:", className="fw-bold"),
        dcc.Checklist(id='method-checklist', options=method_options,
                      value=available_methods, inline=False, inputStyle={"margin-right": "5px"}),
        html.Hr(),

        *[html.Div([html.Label(p.replace("_", " ").title(), className="fw-bold"), dcc.Dropdown(id=f"dropdown-{p}", options=[{"label": str(
            v), "value": v} for v in vals], clearable=True, placeholder="Select...")], className="mb-3") for p, vals in hparams.items()],
        html.Hr(),

        html.Label("Tasks to plot (Evolution/Losses):", className="fw-bold"),
        dcc.Checklist(id='task-checklist', options=task_options,
                      value=available_tasks, inline=False, inputStyle={"margin-right": "5px"}),
        html.Hr(),

        dbc.Button("Apply Filters & Save Plots", id="apply-btn",
                   color="primary", className="w-100"),
        html.Hr(),
        html.Small(
            "Select filters and click 'Apply Filters' to render and export .svg files to /saved_plots.", className="text-muted")
    ], style={"position": "fixed", "left": 0, "top": 0, "bottom": 0, "width": "22rem", "padding": "2rem", "background": "#f8f9fa", "overflow-y": "auto"})


def build_instruction_message():
    return dbc.Container([html.H1("Results dashboard"), html.Hr(), dbc.Alert([html.H4("Waiting Selection"), html.P("Please select filters and click 'Apply Filters & Save Plots'.")], color="info")], fluid=True, style={"margin-top": "2rem"})


def build_main(sections_dict):
    children = [html.H1("Results Dashboard",
                        className="mb-4 text-center"), html.Hr()]

    has_any_data = False

    for sec_title, subsecs in sections_dict.items():
        sec_children = [
            html.H2(sec_title, className="mt-5 border-bottom pb-2 text-primary")]
        has_fig_in_sec = False

        for subsec_title, content in subsecs.items():
            subsec_children = [
                html.H3(subsec_title, className="mt-4 mb-3 text-secondary")]
            has_fig_in_subsec = False

            for key, value in content.items():
                if isinstance(value, go.Figure):
                    # Nível 2: Gráfico direto na subseção (Ex: APE, Total Duration)
                    subsec_children.append(dbc.Card([dbc.CardHeader(
                        key, className="fw-bold"), dbc.CardBody(dcc.Graph(figure=value))], className="mb-4 shadow-sm"))
                    has_fig_in_subsec = True

                elif isinstance(value, dict) and value:
                    # Nível 3: Subsubseção contendo um dicionário de gráficos (Ex: Evolution, Final model across tasks)
                    subsubsec_children = [
                        html.H4(key, className="mt-3 mb-3 fst-italic")]
                    has_fig_in_subsubsec = False

                    for fig_name, fig in value.items():
                        if isinstance(fig, go.Figure):
                            subsubsec_children.append(dbc.Card([dbc.CardHeader(
                                fig_name, className="fw-bold"), dbc.CardBody(dcc.Graph(figure=fig))], className="mb-4 shadow-sm"))
                            has_fig_in_subsubsec = True

                    if has_fig_in_subsubsec:
                        subsec_children.extend(subsubsec_children)
                        has_fig_in_subsec = True

            if has_fig_in_subsec:
                sec_children.extend(subsec_children)
                has_fig_in_sec = True

        if has_fig_in_sec:
            children.extend(sec_children)
            has_any_data = True

    if not has_any_data:
        return dbc.Container([dbc.Alert("No data found for the selected filters.", color="warning")], fluid=True)

    return dbc.Container(children, fluid=True)


def save_all_figures(sections_dict, save_dir):
    """Percorre a estrutura de dicionários e salva cada figura como arquivo .tex (TikZ)."""
    os.makedirs(save_dir, exist_ok=True)

    def recursive_save(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                recursive_save(v, prefix)
            elif isinstance(v, go.Figure):
                # Formata o nome do arquivo removendo caracteres estranhos
                safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', k).lower()
                filepath = os.path.join(save_dir, f"{safe_name}.tex")

                try:
                    # Usa tikzplotly para salvar o gráfico
                    tikzplotly.save(filepath, v)
                except Exception as e:
                    print(f"Erro ao exportar {filepath} via tikzplotly: {e}")

    recursive_save(sections_dict)


# =========================================================
# MAIN APP
# =========================================================
if __name__ == "__main__":
    args = get_args()
    all_metrics_dfs = load_metrics_dfs(args.results_dir)
    hyper_vals = extract_unique_values(all_metrics_dfs, args.hyperparameters)
    available_tasks = extract_available_tasks(all_metrics_dfs)
    available_methods = extract_available_methods(all_metrics_dfs)

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([build_sidebar(hyper_vals, available_tasks, available_methods), html.Div(
        id="page-content", children=build_instruction_message(), style={"margin-left": "24rem", "padding": "2rem"})])

    @callback(Output("page-content", "children"), [Input("apply-btn", "n_clicks")], [*(State(f"dropdown-{p}", "value") for p in hyper_vals.keys()), State("task-checklist", "value"), State("method-checklist", "value")])
    def update(n_clicks, *values):
        active_methods = values[-1]
        task_values = values[-2]
        dropdown_values = values[:-2]

        if n_clicks is None or any(v is None for v in dropdown_values) or not active_methods:
            return build_instruction_message()

        filters = dict(zip(hyper_vals.keys(), dropdown_values))

        filtered_metrics = []
        found_methods = set()
        for df in all_metrics_dfs:
            dff = df.copy()
            match = True
            for k, v in filters.items():
                if k in dff.columns and dff[k].iloc[0] != v:
                    match = False
                    break
            if match:
                filtered_metrics.append(dff)
                found_methods.update(dff["method"].unique())

        # Interseciona os métodos filtrados pelos hyperparameters com os selecionados no checklist
        methods_to_process = found_methods.intersection(set(active_methods))

        loss_dfs, evolution_data, emissions_df = {}, {}, pd.DataFrame()

        if filtered_metrics:
            if "losses" in args.plots:
                loss_dfs = load_loss_dfs(
                    args.results_dir, filters, methods_to_process)
            if "ade" in args.plots:
                evolution_data = load_evolution_matrices(
                    args.results_dir, filters, methods_to_process)
            if "emissions" in args.plots:
                emissions_df = load_emissions_dfs(
                    args.results_dir, filters, methods_to_process)

        sections_dict = compute_structured_figures(
            filtered_metrics, loss_dfs, evolution_data, emissions_df, args.plots, task_values, methods_to_process)

        # Salva os gráficos automaticamente
        save_all_figures(sections_dict, os.path.join(
            args.results_dir, "saved_plots"))

        return build_main(sections_dict)

    app.run(debug=True, port=8050)
