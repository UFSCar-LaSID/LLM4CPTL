import os
import glob
import re
import base64
import numpy as np
import pandas as pd
import torch
from helper import utils

from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from args import get_all_args
from data.loader import data_dset

# Conditional Imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: 'Pillow' not installed. Background images aspect ratio might be wrong.")

# =========================================================
# 1. HELPER FUNCTIONS & CONFIG
# =========================================================

DATASETS_INFO = {
    "ETH": {"sequences": ["biwi_eth", "biwi_hotel"], "frame_rate": 2.5, "loc": "University/Hotel", "pov": "Infrastructure"},
    "UCY": {"sequences": ["crowds_zara01", "crowds_zara02", "crowds_zara03", "students001", "students003", "uni_examples"], "frame_rate": 25.0, "loc": "Urban/Campus", "pov": "Infrastructure"},
    "inD": {"sequences": ["ind_pedestrian_00_tracks", "ind_pedestrian_01_tracks", "ind_pedestrian_02_tracks", "ind_pedestrian_03_tracks", "ind_pedestrian_04_tracks", "ind_pedestrian_05_tracks", "ind_pedestrian_06_tracks"], "frame_rate": 25.0, "loc": "Urban Intersection", "pov": "Drone"},
    "INTERACTION": {"sequences": ["interaction_SR_pedestrian_tracks_000", "interaction_SR_pedestrian_tracks_001", "interaction_SR_pedestrian_tracks_002", "interaction_SR_pedestrian_tracks_003", "interaction_SR_pedestrian_tracks_004", "interaction_SR_pedestrian_tracks_005", "interaction_SR_pedestrian_tracks_006", "interaction_SR_pedestrian_tracks_007", "interaction_SR_pedestrian_tracks_008"], "frame_rate": 10.0, "loc": "Roundabout/Intersection", "pov": "Drone"},
}

SEQUENCES_IMAGES_MAPPING = {
    'biwi_eth': 'biwi_eth',
    'biwi_hotel': 'biwi_hotel',
    'students001': 'students', 'students003': 'students', 'uni_examples': 'students',
    'crowds_zara01': 'crowds_zara', 'crowds_zara02': 'crowds_zara', 'crowds_zara03': 'crowds_zara',
    'ind_pedestrian_00_tracks': 'ind_pedestrian_tracks', 'ind_pedestrian_01_tracks': 'ind_pedestrian_tracks',
    'ind_pedestrian_02_tracks': 'ind_pedestrian_tracks', 'ind_pedestrian_03_tracks': 'ind_pedestrian_tracks',
    'ind_pedestrian_04_tracks': 'ind_pedestrian_tracks', 'ind_pedestrian_05_tracks': 'ind_pedestrian_tracks',
    'ind_pedestrian_06_tracks': 'ind_pedestrian_tracks',
    'interaction_SR_pedestrian_tracks_000': 'interaction_SR_pedestrian_tracks',
    'interaction_SR_pedestrian_tracks_001': 'interaction_SR_pedestrian_tracks',
    'interaction_SR_pedestrian_tracks_002': 'interaction_SR_pedestrian_tracks',
    'interaction_SR_pedestrian_tracks_003': 'interaction_SR_pedestrian_tracks',
    'interaction_SR_pedestrian_tracks_004': 'interaction_SR_pedestrian_tracks',
    'interaction_SR_pedestrian_tracks_005': 'interaction_SR_pedestrian_tracks',
    'interaction_SR_pedestrian_tracks_006': 'interaction_SR_pedestrian_tracks',
    'interaction_SR_pedestrian_tracks_007': 'interaction_SR_pedestrian_tracks',
    'interaction_SR_pedestrian_tracks_008': 'interaction_SR_pedestrian_tracks',
}

BASE_DATA_DIR = "./datasets"

def shorten_label(label):
    label = label.replace('interaction_SR_pedestrian_', 'INTER_')
    label = label.replace('ind_pedestrian_', 'inD_')
    label = label.replace('_tracks', '')
    label = label.replace('crowds_zara', 'zara_')
    label = label.replace('students', 'stud_')
    return label


def find_reference_image(dataset_dir, sequence_name):
    seq_base = re.sub(r'\d+$', '', sequence_name).rstrip('_')
    patterns = [
        f"*{sequence_name}*reference.png", f"*{seq_base}*reference.png",
        f"*{seq_base}*reference.jpg", f"*{seq_base}*reference.jpg", f"*{seq_base}*reference.png",
    ]
    if "ind_pedestrian" in sequence_name:
        patterns += ["ind_pedestrian_tracks_reference.png"]
    
    search_dirs = [dataset_dir, os.path.dirname(dataset_dir)]
    for d in search_dirs:
        for pat in patterns:
            matches = glob.glob(os.path.join(d, pat))
            if matches:
                return matches[0]
    return None


def get_latex_heatmap_color(value, min_val, max_val):
    if max_val == min_val:
        norm = 0
    else:
        norm = (value - min_val) / (max_val - min_val)
    gray_val = 1.0 - (norm * 0.4)
    return f"\\cellcolor[gray]{{{gray_val:.2f}}}"

# =========================================================
# 2. DATA EXTRACTION
# =========================================================

def extract_sequence_data(dataset, seq_name):
    indices = [i for i, name in enumerate(dataset.sequences_name_list) if name == seq_name]

    if not indices:
        return pd.DataFrame(), 0.0, 0, 0

    obs = dataset.obs_traj[indices].numpy()
    pred = dataset.pred_traj[indices].numpy()
    full_traj = np.concatenate([obs, pred], axis=2)

    N_seq, _, seq_len = full_traj.shape

    peds = np.repeat(np.arange(N_seq), seq_len)
    frames = np.tile(np.arange(seq_len), N_seq)
    flat_coords = full_traj.transpose(0, 2, 1).reshape(-1, 2)

    df = pd.DataFrame({
        'id': peds,
        'frame': frames,
        'x': flat_coords[:, 0],
        'y': flat_coords[:, 1]
    })

    densities = []
    for start, end in dataset.seq_start_end:
        if dataset.sequences_name_list[start] == seq_name:
            densities.append(end - start)
    avg_density = np.mean(densities) if densities else 0.0

    nl_vals = dataset.non_linear_ped[indices].numpy()
    nonlin_count = int(np.sum(nl_vals == 1.0))
    lin_count = int(len(indices) - nonlin_count)

    return df, avg_density, lin_count, nonlin_count

# =========================================================
# 3. PROCESSING LOGIC
# =========================================================

def process_benchmark(splits_selected, args, group_scenes=False):
    aggregate_splits = 'all' in splits_selected
    target_splits = ['train', 'val', 'test'] if aggregate_splits else splits_selected

    global_stats = {"total_trajectories": 0, "linear_count": 0, "nonlinear_count": 0}
    plot_data = {"sequences_data": {}, "scene_bounds": {}}
    hierarchy_data = {}

    for dataset_name, info in DATASETS_INFO.items():
        sequences = info["sequences"]
        hierarchy_data[dataset_name] = {seq: {} for seq in sequences}

        for split in target_splits:
            split_dir = os.path.join(BASE_DATA_DIR, dataset_name, split)
            if not os.path.exists(split_dir):
                split_dir = os.path.join(BASE_DATA_DIR, dataset_name)

            try:
                ds_path = os.path.join(BASE_DATA_DIR, dataset_name, split)
                ds = data_dset(args, ds_path, dataset_name=dataset_name, split_name=split)
            except Exception as e:
                print(f"Dataset Loader falhou para {dataset_name} ({split}): {e}")
                continue

            for seq in sequences:
                try:
                    df, avg_density, lin_count, nonlin_count = extract_sequence_data(ds, seq)
                    if df.empty:
                        continue

                    # Identificador único das trajetórias no dataframe evita colisões de ID ao mesclar splits/cenas
                    df['id'] = df['id'].apply(lambda x: f"{split}_{seq}_{x}")
                    
                    # Usa o mapeamento para descobrir qual o cenário base desta sequência
                    actual_scene_name = SEQUENCES_IMAGES_MAPPING.get(seq, seq)
                    
                    possible_files = glob.glob(os.path.join(split_dir, f"{seq}*"))
                    search_dir = os.path.dirname(possible_files[0]) if possible_files else split_dir
                    
                    # Tenta procurar a imagem usando o nome mapeado (cenário físico)
                    img_path = find_reference_image(search_dir, actual_scene_name)

                    if "eth" in dataset_name.lower() or "hotel" in dataset_name.lower():
                        df['plot_x'], df['plot_y'] = df['y'], -df['x']
                    elif "zara" in dataset_name.lower():
                        df['plot_x'], df['plot_y'] = -df['y'], -df['x']
                    else:
                        df['plot_x'], df['plot_y'] = df['x'], df['y']

                    # Regra de Agrupamento Visual
                    base_scene = actual_scene_name if group_scenes else seq
                    unique_id = f"ALL_{base_scene}" if aggregate_splits else f"{split}_{base_scene}"

                    if unique_id not in plot_data["sequences_data"]:
                        plot_data["sequences_data"][unique_id] = {
                            "df": df.copy(), "image": img_path, 
                            "split": "All Splits" if aggregate_splits else split, 
                            "base_seq": base_scene,
                            "dataset": dataset_name # Guardamos a raiz do dataset para facilitar o plot
                        }
                    else:
                        plot_data["sequences_data"][unique_id]["df"] = pd.concat(
                            [plot_data["sequences_data"][unique_id]["df"], df], ignore_index=True
                        )

                    b_key = img_path if img_path else base_scene
                    if b_key not in plot_data["scene_bounds"]:
                        plot_data["scene_bounds"][b_key] = {'min_x': np.inf, 'max_x': -np.inf, 'min_y': np.inf, 'max_y': -np.inf}

                    curr_b = plot_data["scene_bounds"][b_key]
                    curr_b['min_x'] = min(curr_b['min_x'], df['plot_x'].min())
                    curr_b['max_x'] = max(curr_b['max_x'], df['plot_x'].max())
                    curr_b['min_y'] = min(curr_b['min_y'], df['plot_y'].min())
                    curr_b['max_y'] = max(curr_b['max_y'], df['plot_y'].max())

                    hierarchy_data[dataset_name][seq][split] = {
                        "linear": lin_count, "nonlinear": nonlin_count,
                        "total": lin_count + nonlin_count, "density": float(avg_density)
                    }

                    global_stats["total_trajectories"] += (lin_count + nonlin_count)
                    global_stats["linear_count"] += lin_count
                    global_stats["nonlinear_count"] += nonlin_count

                except Exception as e:
                    print(f"Error processing {seq} in {split}: {e}")

    flat_table_data = []
    for ds_name, seqs in hierarchy_data.items():
        meta = DATASETS_INFO[ds_name]
        for seq_name, splits_data in seqs.items():
            if not splits_data: continue

            row = {
                "Dataset": ds_name, "Sequence": shorten_label(seq_name),
                "Source": "Real" if "sim" not in ds_name.lower() else "Simulated",
                "Location": meta.get("loc", "-"), "POV": meta.get("pov", "-")
            }

            total_lin, total_non, total_all = 0, 0, 0
            densities = []

            for split in ['train', 'val', 'test']:
                if split in splits_data:
                    d = splits_data[split]
                    prefix = split.capitalize()
                    row[f"{prefix} Lin"] = d["linear"]
                    row[f"{prefix} NonLin"] = d["nonlinear"]
                    row[f"{prefix} Both"] = d["total"]
                    row[f"{prefix} Peds/Frame"] = round(float(d["density"]), 1)
                    total_lin += d["linear"]
                    total_non += d["nonlinear"]
                    total_all += d["total"]
                    densities.append(d["density"])
                else:
                    prefix = split.capitalize()
                    row[f"{prefix} Lin"], row[f"{prefix} NonLin"], row[f"{prefix} Both"], row[f"{prefix} Peds/Frame"] = 0, 0, 0, 0.0

            row["ALL Lin"], row["ALL NonLin"], row["ALL Both"] = total_lin, total_non, total_all
            row["ALL Peds/Frame"] = round(np.mean(densities) if densities else 0.0, 1)
            flat_table_data.append(row)

    return plot_data, global_stats, flat_table_data, target_splits

# =========================================================
# 4. DASH APP LAYOUT
# =========================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

sidebar = html.Div([
    html.H2("Benchmark Analysis", className="display-6"), html.Hr(),
    html.Label("Select data splits:"),
    dcc.Checklist(
        id="split-checklist",
        options=[{'label': ' Train', 'value': 'train'}, {'label': ' Validation', 'value': 'val'}, 
                 {'label': ' Test', 'value': 'test'}, {'label': ' All Splits', 'value': 'all'}],
        value=['train'],
        inputStyle={"margin-right": "5px"},
        style={"display": "flex", "flex-direction": "column"}
    ),
    html.Hr(),
    html.Label("Viz Options:"),
    dcc.Checklist(
        id="viz-options",
        options=[
            {'label': ' Show Background', 'value': 'show_bg'},
            {'label': ' Group by Physical Scene', 'value': 'group_scene'}
        ],
        value=['show_bg'],
        inputStyle={"margin-right": "5px"},
        style={"display": "flex", "flex-direction": "column"}
    ),
    html.Br(),
    dbc.Button("Run Analysis", id="apply-btn", color="primary", className="w-100 mb-2"),
    dbc.Button("Generate LaTeX", id="btn-latex", color="secondary", className="w-100"),
], style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "18rem", "padding": "2rem 1rem", "background-color": "#f8f9fa", "overflow-y": "auto"})

content = html.Div([
    html.Div(id="ui-container"),
    html.Div([html.Hr(), html.H4("LaTeX Output"), dbc.Textarea(id="latex-output", style={"height": "300px", "fontFamily": "monospace"})], className="mb-5")
], style={"margin-left": "20rem", "padding": "2rem"})

app.layout = html.Div([sidebar, content, dcc.Store(id='store-stats')])

# =========================================================
# 5. CALLBACKS
# =========================================================

@app.callback(
    [Output("ui-container", "children"), Output("store-stats", "data")],
    Input("apply-btn", "n_clicks"),
    [State("split-checklist", "value"), State("viz-options", "value")]
)
def update_graphs(n_clicks, splits, viz_options):
    if not n_clicks: return html.Div("Select splits and run."), no_update
    if not splits: return dbc.Alert("Select a split.", color="warning"), no_update

    parsed_args = get_all_args()
    
    viz_options_list = viz_options or []
    show_bg = 'show_bg' in viz_options_list
    group_scene = 'group_scene' in viz_options_list
    
    plot_data, global_stats, table_data, active_splits = process_benchmark(splits, parsed_args, group_scenes=group_scene)

    store_data = {"global_stats": global_stats, "table_data": table_data, "active_splits": active_splits}

    df_table = pd.DataFrame(table_data)
    if not df_table.empty:
        df_grouped = df_table.groupby("Dataset").agg({"ALL Lin": "sum", "ALL NonLin": "sum", "ALL Peds/Frame": "mean"}).reset_index()
        df_lin_melt = df_grouped.melt(id_vars="Dataset", value_vars=["ALL Lin", "ALL NonLin"], var_name="Type", value_name="Count")

        fig_global_lin = px.bar(df_lin_melt, x="Dataset", y="Count", color="Type", barmode="group", title="Linearity Count per Dataset",
                                color_discrete_map={"ALL Lin": "green", "ALL NonLin": "red"}, category_orders={"Dataset": parsed_args.dataset})
        fig_global_lin.update_layout(height=300)

        fig_global_peds = px.bar(df_grouped, x="Dataset", y="ALL Peds/Frame", title="Avg Peds/Frame per Dataset", color_discrete_sequence=["orange"],
                                 category_orders={"Dataset": parsed_args.dataset})
        fig_global_peds.update_layout(height=300)
    else:
        fig_global_lin, fig_global_peds = go.Figure(), go.Figure()

    global_card = dbc.Card([
        dbc.CardHeader(html.H4(f"Overview | Splits: {', '.join(active_splits)}", className="text-white"), className="bg-dark"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Div([html.H2(global_stats["total_trajectories"]), html.P("Total Trajectories")]), width=4),
                dbc.Col(html.Div([html.H2(global_stats["linear_count"], className="text-success"), html.P("Linear")]), width=4),
                dbc.Col(html.Div([html.H2(global_stats["nonlinear_count"], className="text-danger"), html.P("Non-Linear")]), width=4),
            ]),
            html.Hr(),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_global_lin), width=6), dbc.Col(dcc.Graph(figure=fig_global_peds), width=6)])
        ])
    ], className="mb-4")

    dataset_sections = []
    grouped_by_ds = {}
    for r in table_data: grouped_by_ds.setdefault(r['Dataset'], []).append(r)

    ordered_ds_keys = [d for d in parsed_args.dataset if d in grouped_by_ds]
    ordered_ds_keys += [d for d in grouped_by_ds if d not in parsed_args.dataset]

    for ds_name in ordered_ds_keys:
        rows = grouped_by_ds[ds_name]
        traj_cols = []

        # Puxa diretamente do plot_data os gráficos associados a este dataset
        ds_plot_keys = [k for k, v in plot_data["sequences_data"].items() if v['dataset'] == ds_name]
        sorted_keys = sorted(ds_plot_keys)

        for key in sorted_keys:
            p_dat = plot_data["sequences_data"][key]
            df = p_dat['df']

            # OPACIDADE e TAMANHO adicionados para lidar com alta densidade de pontos
            fig = go.Figure(go.Scattergl(
                x=df['plot_x'], y=df['plot_y'], 
                mode='markers', 
                marker=dict(size=1.5, color='royalblue', opacity=0.8)
            ))

            b_key = p_dat['image'] if p_dat['image'] else p_dat['base_seq']
            bounds = plot_data["scene_bounds"].get(b_key)
            xaxis, yaxis = None, None

            if show_bg and p_dat['image'] and PIL_AVAILABLE:
                try:
                    encoded = base64.b64encode(open(p_dat['image'], 'rb').read()).decode('ascii')
                    pil_img = Image.open(p_dat['image'])
                    w, h = pil_img.size
                    ar = w / h
                    dw, dh = bounds['max_x'] - bounds['min_x'], bounds['max_y'] - bounds['min_y']
                    cx, cy = bounds['min_x'] + dw/2, bounds['min_y'] + dh/2
                    data_ar = dw / dh if dh > 0 else 1
                    
                    if ar > data_ar:
                        nh, nw = dh, dh * ar
                    else:
                        nw, nh = dw, dw / ar
                        
                    min_x, max_x = cx - nw/2, cx + nw/2
                    min_y, max_y = cy - nh/2, cy + nh/2
                    
                    fig.add_layout_image(dict(
                        source=f"data:image/png;base64,{encoded}", xref="x", yref="y", x=min_x, y=max_y, sizex=nw, sizey=nh, 
                        sizing="stretch", layer="below", opacity=0.6
                    ))
                    xaxis, yaxis = [min_x, max_x], [min_y, max_y]
                except Exception as e:
                    print(f"Erro ao carregar imagem {p_dat['image']}: {e}")
            elif bounds:
                pad_x, pad_y = (bounds['max_x']-bounds['min_x'])*0.05, (bounds['max_y']-bounds['min_y'])*0.05
                xaxis, yaxis = [bounds['min_x']-pad_x, bounds['max_x']+pad_x], [bounds['min_y']-pad_y, bounds['max_y']+pad_y]

            fig.update_layout(
                title=f"{shorten_label(p_dat['base_seq'])} ({p_dat['split']})", 
                margin=dict(l=0, r=0, t=30, b=0), 
                autosize=True,
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(visible=False, showgrid=False, zeroline=False, range=xaxis), 
                yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, range=yaxis), 
                showlegend=False
            )
            traj_cols.append(dbc.Col(dcc.Graph(figure=fig), width=3, className="mb-2"))

        chart_data = []
        for r in rows:
            for s in active_splits:
                prefix = s.capitalize()
                chart_data.append({"Sequence": r["Sequence"], "Split": s, "Linear": r[f"{prefix} Lin"], "NonLinear": r[f"{prefix} NonLin"]})
        
        df_chart = pd.DataFrame(chart_data)
        if not df_chart.empty:
            fig_lin = px.bar(df_chart, x="Sequence", y=["Linear", "NonLinear"], facet_col="Split", barmode="group", title="Linearity Distribution")
            fig_lin.update_layout(height=300)
        else:
            fig_lin = go.Figure()

        ds_section = html.Div([
            html.H3(ds_name, className="text-primary mt-4"), html.H5("Trajectories"),
            dbc.Row(traj_cols, className="mb-3"), dbc.Row([dbc.Col(dcc.Graph(figure=fig_lin), width=12)])
        ])
        dataset_sections.append(ds_section)

    return html.Div([global_card, *dataset_sections]), store_data

@app.callback(
    Output("latex-output", "value"),
    Input("btn-latex", "n_clicks"), State("store-stats", "data"), prevent_initial_call=True
)
def generate_latex_code(n, store_data):
    if not store_data: return "Run analysis first."
    parsed_args = get_all_args()
    table_data = store_data.get("table_data", [])

    vals_both = [r.get("ALL Both", 0) for r in table_data]
    vals_dens = [r.get("ALL Peds/Frame", 0) for r in table_data]

    def safe_mm(l): return (min(l), max(l)) if l else (0, 1)
    min_b, max_b = safe_mm(vals_both)
    min_d, max_d = safe_mm(vals_dens)

    rows_tex = ""
    curr_ds = ""
    ds_priority = {name: i for i, name in enumerate(parsed_args.dataset)}
    sorted_data = sorted(table_data, key=lambda x: (ds_priority.get(x['Dataset'], 999), x['Dataset'], x['Sequence']))

    for i, row in enumerate(sorted_data):
        ds = row['Dataset']
        if ds != curr_ds:
            if i > 0: rows_tex += "\\hline\n"
            curr_ds = ds
            count = len([x for x in sorted_data if x['Dataset'] == ds])
            ds_str = f"\\multirow{{{count}}}{{*}}{{{ds}}}" if count > 1 else ds
            loc_str = f"\\multirow{{{count}}}{{*}}{{{row['Location']}}}" if count > 1 else row['Location']
            pov_str = f"\\multirow{{{count}}}{{*}}{{{row['POV']}}}" if count > 1 else row['POV']
        else:
            ds_str, loc_str, pov_str = "", "", ""

        seq = row['Sequence'].replace('_', '\\_')
        def c(k): return str(row.get(k, 0))

        c_both = get_latex_heatmap_color(row.get("ALL Both", 0), min_b, max_b)
        c_dens = get_latex_heatmap_color(row.get("ALL Peds/Frame", 0), min_d, max_d)

        l = f"{ds_str} & {row['Source']} & {loc_str} & {pov_str} & {seq} & "
        l += f"{c('Train Lin')} & {c('Val Lin')} & {c('Test Lin')} & {c('ALL Lin')} & "
        l += f"{c('Train NonLin')} & {c('Val NonLin')} & {c('Test NonLin')} & {c('ALL NonLin')} & "
        l += f"{c('Train Both')} & {c('Val Both')} & {c('Test Both')} & {c_both}{c('ALL Both')} & "
        l += f"{c('Train Peds/Frame')} & {c('Val Peds/Frame')} & {c('Test Peds/Frame')} & {c_dens}{c('ALL Peds/Frame')} \\\\"
        rows_tex += l + "\n"

    table1 = f"""
% TABLE 1 (Detailed by Sequence)
\\begin{{sidewaystable}}
\\centering
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lllllcccccccccccccccc}} 
\\cline{{1-17}}
\\multirow{{3}}{{*}}{{\\textbf{{Dataset}}}} & \\multirow{{3}}{{*}}{{\\textbf{{Source}}}} & \\multirow{{3}}{{*}}{{\\textbf{{Location}}}} & \\multirow{{3}}{{*}}{{\\textbf{{POV}}}} & \\multirow{{3}}{{*}}{{\\textbf{{Sequence}}}} & \\multicolumn{{12}}{{c}}{{\\textbf{{\\# Trajectories}}}} & \\multicolumn{{4}}{{l}}{{\\multirow{{2}}{{*}}{{\\textbf{{Avg Density}}}}}} \\\\ 
\\cline{{6-17}}
 & & & & & \\multicolumn{{4}}{{c}}{{\\textbf{{Linear}}}} & \\multicolumn{{4}}{{c}}{{\\textbf{{Non-linear}}}} & \\multicolumn{{4}}{{c}}{{\\textbf{{BOTH}}}} & \\\\ 
\\cline{{6-21}}
 & & & & & Tr & Va & Te & ALL & Tr & Va & Te & ALL & Tr & Va & Te & ALL & Tr & Va & Te & ALL \\\\ 
\\hline
{rows_tex}
\\end{{tabular}}
}}
\\end{{sidewaystable}}
"""

    ds_stats = {}
    for r in table_data:
        ds = r['Dataset']
        if ds not in ds_stats: ds_stats[ds] = {'seqs': 0, 'lin': 0, 'non': 0, 'tot': 0, 'dens': []}
        ds_stats[ds]['seqs'] += 1
        ds_stats[ds]['lin'] += r['ALL Lin']
        ds_stats[ds]['non'] += r['ALL NonLin']
        ds_stats[ds]['tot'] += r['ALL Both']
        ds_stats[ds]['dens'].append(r['ALL Peds/Frame'])

    t2_tot = [v['tot'] for v in ds_stats.values()]
    t2_den = [np.mean(v['dens']) for v in ds_stats.values()]
    mi_t, mx_t = safe_mm(t2_tot)
    mi_d, mx_d = safe_mm(t2_den)

    rows_tex2 = ""
    sorted_ds_stats = sorted(ds_stats.items(), key=lambda x: ds_priority.get(x[0], 999))

    for ds, v in sorted_ds_stats:
        meta = next(x for x in table_data if x['Dataset'] == ds)
        avg_d = np.mean(v['dens'])
        c_tot = get_latex_heatmap_color(v['tot'], mi_t, mx_t)
        c_den = get_latex_heatmap_color(avg_d, mi_d, mx_d)
        l = f"{ds} & {meta['Source']} & {meta['Location']} & {meta['POV']} & {v['seqs']} & {v['lin']:,} & {v['non']:,} & {c_tot}{v['tot']:,} & {c_den}{avg_d:.2f} \\\\ \\hline"
        rows_tex2 += l + "\n"

    g_seqs = sum(v['seqs'] for v in ds_stats.values())
    g_lin = sum(v['lin'] for v in ds_stats.values())
    g_non = sum(v['non'] for v in ds_stats.values())
    g_tot = sum(v['tot'] for v in ds_stats.values())
    g_den = np.mean(t2_den) if t2_den else 0

    rows_tex2 += f"\\textbf{{TOTAL}} & & & & \\textbf{{{g_seqs}}} & \\textbf{{{g_lin:,}}} & \\textbf{{{g_non:,}}} & \\textbf{{{g_tot:,}}} & \\textbf{{{g_den:.2f}}} \\\\"

    table2 = f"""
% TABLE 2 (Aggregated by Dataset)
\\begin{{table}}[!htpb]
\\centering
\\caption{{Overall Statistics}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{llllccccc}}
\\textbf{{Dataset}} & \\textbf{{Source}} & \\textbf{{Location}} & \\textbf{{POV}} & \\textbf{{# Seq}} & \\multicolumn{{3}}{{c}}{{\\textbf{{# Trajectories}}}} & \\textbf{{Avg Density}} \\\\
\\cline{{6-8}}
 & & & & & \\textbf{{Linear}} & \\textbf{{Non-linear}} & \\textbf{{BOTH}} & \\\\
\\hline
{rows_tex2}
\\end{{tabular}}
}}
\\end{{table}}
"""
    return table1 + "\n\n" + table2


if __name__ == "__main__":
    args = get_all_args()
    app.run(debug=True, port=8051)