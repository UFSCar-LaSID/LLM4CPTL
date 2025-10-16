import os
import argparse
import re

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
plt.rcParams['axes.axisbelow'] = True

parser = argparse.ArgumentParser('./merge_results_metrics.py', description='Merge metrics for all excecuted experiments.')
parser.add_argument('--results_dir', type=str, default='./results', dest='r_dir', help="default")
parser.add_argument('--output_filename', type=str, default='df_metrics_final.csv', help="default")
parser.add_argument('--iters', type=int, default=None, help="batches to optimize solver")
parser.add_argument('--batch_size', type=int, default=None, help="batch-size")

def compute_nbwt(df: pd.DataFrame) -> pd.DataFrame:
    if not {"IL", "CL_NR"}.issubset(df['method'].values):
        print("There is no IL or CL_NR experiments, impossible to compute NBWT")
        df["normalized_backward_transfer_nbwt_ade"] = None
        df["normalized_backward_transfer_nbwt_fde"] = None
        return df

    il_bwt_ade  = df.loc[df['method'] == 'IL',    'backward_transfer_bwt_ade'].mean()
    clnr_bwt_ade = df.loc[df['method'] == 'CL_NR', 'backward_transfer_bwt_ade'].iloc[0]

    il_bwt_fde  = df.loc[df['method'] == 'IL',    'backward_transfer_bwt_fde'].mean()
    clnr_bwt_fde = df.loc[df['method'] == 'CL_NR', 'backward_transfer_bwt_fde'].iloc[0]

    denom_ade = clnr_bwt_ade - il_bwt_ade
    denom_fde = clnr_bwt_fde - il_bwt_fde

    df["normalized_backward_transfer_nbwt_ade"] = None
    df["normalized_backward_transfer_nbwt_fde"] = None

    mask = ~df["method"].isin(["IL", "CL_NR"])

    if denom_ade != 0:
        df.loc[mask, "normalized_backward_transfer_nbwt_ade"] = (
            (df.loc[mask, "backward_transfer_bwt_ade"] - il_bwt_ade) / denom_ade
        )
    if denom_fde != 0:
        df.loc[mask, "normalized_backward_transfer_nbwt_fde"] = (
            (df.loc[mask, "backward_transfer_bwt_fde"] - il_bwt_fde) / denom_fde
        )

    df.loc[df["method"] == "CL_NR", "normalized_backward_transfer_nbwt_ade"] = 1.0
    df.loc[df["method"] == "CL_NR", "normalized_backward_transfer_nbwt_fde"] = 1.0

    return df

def run(args):
    ##################################
    # Accuracy metrics (ADE, FDE...)
    ##################################
    dfs = []
    for filename in sorted(os.listdir(args.r_dir)):
        if filename.endswith('.csv') and filename != args.output_filename and "metrics" in filename:            
            filepath = os.path.join(args.r_dir, filename)
            df_aux = pd.read_csv(filepath)
            
            if args.iters is not None and args.batch_size is not None and 'iters' in df_aux.columns and 'batch_size' in df_aux.columns:
                df_aux = df_aux[
                    (df_aux['iters'] == args.iters) &
                    (df_aux['batch_size'] == args.batch_size)
                ]

            if not df_aux.empty:
                print(f"Reading file {filename}")
                dfs.append(df_aux)

    print("Merging and concatenating DataFrames")
    df_merged = pd.concat(dfs, axis=0, join='outer', ignore_index=True)
    
    df_merged = compute_nbwt(df_merged)
    
    df_il = df_merged[df_merged["method"] == "IL"]
    mean_row = df_il.mean(numeric_only=True)
    new_row = pd.DataFrame([{
        "method": "IL_mean",
        "learning_method": "batch_learning",
        **mean_row.to_dict()
    }])
    df_performance_merged = pd.concat([df_merged, new_row], ignore_index=True)
    df_performance_merged['replay_method'].fillna('none', inplace=True)
    df_performance_merged['replay_method'].replace(['nan'], 'none', inplace=True)
    
    ##################################
    # Codecarbon metrics
    ##################################
    dfs = []
    for filename in sorted(os.listdir(args.r_dir)):
        if filename.endswith('.csv') and filename != args.output_filename and "emission" in filename and str(args.iters) in filename and str(args.batch_size) in filename:         
            filepath = os.path.join(args.r_dir, filename)
            df_aux = pd.read_csv(filepath)
            
            project_name = df_aux.loc[0, 'project_name']

            pattern = re.compile(r'^(CL_[A-Z]+_|IL_)')
            df_aux['method'] = pattern.search(project_name).group(0)[0:-1]

            method = re.sub(pattern, '', project_name, 1)
            df_aux['learning_method'] = "continual_learning" if "CL" in project_name else "batch_learning"
            
            df_aux['project_name_clean'] = df_aux['project_name'].str.replace(r'^(CL_[A-Z]+_|IL_)', '', regex=True)
            split_cols = df_aux['project_name_clean'].str.split('_', expand=True)
            split_cols.columns = ['replay_method', 'observation_length', 'prediction_length', 'batch_size', 'replay_batch_size', 'iters', 'main_predictor_model']
            df_aux = pd.concat([df_aux, split_cols], axis=1)
            #df_aux = df_aux.replace(['none', 'None'], 0)

            #if args.iters is not None and args.batch_size is not None and 'iters' in df_aux.columns and 'batch_size' in df_aux.columns:
                #df_aux = df_aux[
                    #(df_aux['iters'] == args.iters) &
                    #(df_aux['batch_size'] == args.batch_size)
                #]

            if not df_aux.empty:
                df_aux.drop('project_name_clean', axis=1, inplace=True)
                print(f"Reading file {filename}")
                dfs.append(df_aux)

    print("Merging and concatenating DataFrames")
    df_codecarbon_merged = pd.concat(dfs, axis=0, join='outer', ignore_index=True)

    df_il = df_codecarbon_merged[df_codecarbon_merged["method"] == "IL"]
    mean_row = df_il.mean(numeric_only=True)
    new_row = pd.DataFrame([{
        "method": "IL_mean",
        "learning_method": "batch_learning",
        **mean_row.to_dict()
    }])
    df_codecarbon_merged = pd.concat([df_codecarbon_merged, new_row], ignore_index=True)
    
    ##################################
    # Final accuracy + codecarbon dataframe
    ##################################
    str_cols = ['method', 'learning_method', 'replay_method', 'main_predictor_model']
    for col in str_cols:
        df_performance_merged[col] = df_performance_merged[col].astype(str).str.strip()
        df_codecarbon_merged[col] = df_codecarbon_merged[col].astype(str).str.strip()

    numeric_cols = ['observation_length', 'prediction_length', 'batch_size', 'replay_batch_size', 'iters']
    for col in numeric_cols:
        df_performance_merged[col] = pd.to_numeric(df_performance_merged[col], errors='coerce').fillna(0).astype(float)
        df_codecarbon_merged[col] = pd.to_numeric(df_codecarbon_merged[col], errors='coerce').fillna(0).astype(float)

    join_cols = ['method', 'learning_method', 'replay_method', 'observation_length',
             'prediction_length', 'batch_size', 'replay_batch_size', 'iters', 
             'main_predictor_model']

    df_final_merged = df_performance_merged.merge(df_codecarbon_merged, on=join_cols, how='left')
    
    output_path = os.path.join(args.r_dir, args.output_filename)
    if os.path.exists(output_path):
        print(f"\nDeleting existing file at {output_path}")
        os.remove(output_path)
    
    df_final_merged.drop_duplicates(subset=['method', 'train_dataset', 'test_dataset', 'learning_method', 'replay_method', 'observation_length', 'prediction_length', 'batch_size', 'replay_batch_size', 'iters', 'main_predictor_model'], inplace=True)
    df_final_merged["num_tasks_train"] = df_final_merged["train_dataset"].apply(
        lambda x: np.nan if pd.isna(x) or x == "" else len(str(x).split("-"))
    )

    df_final_merged["num_tasks_test"] = df_final_merged["test_dataset"].apply(
        lambda x: np.nan if pd.isna(x) or x == "" else len(str(x).split("-"))
    )

    df_final_merged["train_dataset"] = df_final_merged["train_dataset"].apply(
        lambda x: [] if pd.isna(x) or x.strip() == "" else [part.strip() for part in x.split("-")]
    )

    df_final_merged["test_dataset"] = df_final_merged["test_dataset"].apply(
        lambda x: [] if pd.isna(x) or x.strip() == "" else [part.strip() for part in x.split("-")]
    )
    
    df_final_merged.to_csv(output_path, index=False)
    print(f"Final metrics saved in: {output_path}\n")

    ##################################
    # Plotting
    ##################################
    plots_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    DF_ALL_IL = df_final_merged[df_final_merged["method"].str.contains("IL")]
    DF_ALL_CL = df_final_merged[df_final_merged["method"].str.contains("CL")]
    DF_ALL_CL_AND_ILMEAN = df_final_merged[
                                (df_final_merged["method"].str.contains("CL")) |
                                (df_final_merged["method"] == "IL_mean")
    ]

    loc = plticker.MultipleLocator(base=0.1)
    
    ### PLOT 1 ###############################
    ax = DF_ALL_CL_AND_ILMEAN.plot(
        x="method",
        y=["average_prediction_error_ape_ade", "average_prediction_error_ape_fde"],
        kind="bar",
        legend=True,
        xlabel="Methods",
        ylabel="Average prediction error (APE) in meters"
    )
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Metrics", labels=["APE: ADE", "APE: FDE"], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, which='both', axis='both', linestyle="--", alpha=0.5)
    
    ax.yaxis.set_major_locator(loc)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_ape.png"), dpi=300)
    plt.close()
    print("Saved plot 1: APE per method")

    ### PLOT 2 ###############################
    ax = DF_ALL_CL.plot(
        x="method",
        y=["backward_transfer_bwt_ade", "backward_transfer_bwt_fde"],
        kind="bar",
        legend=True,
        xlabel="Methods",
        ylabel="Backward transfer (BWT)"
    )
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Metrics", labels=["BWT: ADE", "BWT: FDE"], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, which='both', axis='both', linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_bwt.png"), dpi=300)
    plt.close()
    print("Saved plot 2: BWT per method")

    ### PLOT 3 ###############################
    ax = DF_ALL_CL.plot(
        x="method",
        y="duration",
        kind="bar",
        legend=False,
        xlabel="Methods",
        ylabel="Total training duration (seconds)",
    )
    plt.xticks(rotation=45, ha='right')
    
    ax.grid(True, which='both', axis='both', linestyle="--", alpha=0.5)
    
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=1000))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_totalTrainingDuration.png"), dpi=300)
    plt.close()
    print("Saved plot 3: Total training duration (seconds) per method")    
    
    ### PLOT 4 ###############################    
    task_cols = [c for c in DF_ALL_CL.columns if re.search(r"task\d+", c)]
    tasks = [re.search(r"task\d+", c).group() for c in task_cols]
    unique_tasks = sorted(set(tasks), key=lambda x: int(re.search(r"\d+", x).group()))
    num_tasks = len(unique_tasks)

    ycols = [f"elapsed_time_{t.replace('task', 'task_')}" for t in unique_tasks]

    fig, ax = plt.subplots()
    x_values = list(range(1, num_tasks + 1))

    for idx, row in DF_ALL_CL.iterrows():
        train_datasets = row["train_dataset"]

        method = row["method"]

        # Valores y para essa linha
        y_values = [row[ycol] for ycol in ycols]

        # Plotar
        ax.plot(x_values, y_values, marker="o", markersize=4, label=method)
        for x, y in zip(x_values, y_values):
            ax.text(x, y, f"{y:.2f}", ha="right", va="bottom", fontsize=6)

    ax.set_xticks(x_values)
    ax.set_xlabel("Tasks")
    ax.set_ylabel("Elapsed time (seconds)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_yscale('log')
    ax.legend(title="Methods", bbox_to_anchor=(1.05, 1), loc='upper left')

    datasets = DF_ALL_CL["train_dataset"].tolist()[0]
    dataset_order_text = "Tasks order: " + ", ".join(map(str, datasets))
    ax.text(
        0.5, -0.15,           
        dataset_order_text,
        transform=ax.transAxes,  
        ha="center", va="top",
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_elapsedTimePerTask.png"), dpi=300)
    plt.close()
    print("Saved plot 4: Elapsed time (seconds) per task")

    ### PLOT 5 ###############################
    methods = DF_ALL_CL["method"].unique()
    tasks = sorted([int(re.search(r"task(\d+)", c).group(1)) 
                    for c in DF_ALL_CL.columns if "ade_task" in c]) 

    ade_cols = [f"ade_task{t}_after_training_in_all_tasks" for t in tasks]
    fde_cols = [f"fde_task{t}_after_training_in_all_tasks" for t in tasks]

    all_values = DF_ALL_CL[ade_cols + fde_cols].values.flatten()
    ymin = np.nanmin(all_values) - 0.5
    ymax = np.nanmax(all_values) + 0.5

    x_values = tasks 

    fig, ax = plt.subplots(figsize=(8,5))
    for method in methods:
        row = DF_ALL_CL[DF_ALL_CL["method"] == method].iloc[0] 
        y_values = [row[col] for col in ade_cols]
        ax.plot(x_values, y_values, marker='o', label=method)
    
    datasets = DF_ALL_CL["train_dataset"].tolist()[0]
    dataset_order_text = "Tasks order: " + ", ".join(map(str, datasets))
    ax.text(
        0.5, -0.15,           
        dataset_order_text,
        transform=ax.transAxes,  
        ha="center", va="top",
        fontsize=10
    )

    ax.set_xticks(x_values)
    ax.set_xlabel("Tasks")
    ax.set_ylim(max(ymin, 0.00), ymax) 
    ax.set_ylabel("Average displacement eror (ADE) in meters")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Methods")
    
    ax.yaxis.set_major_locator(loc)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_adePerTask.png"), dpi=300)
    plt.close()
    print("Saved plot 5: ADE per task")

    ### PLOT 6 ###############################
    fig, ax = plt.subplots(figsize=(8,5))
    for method in methods:
        row = DF_ALL_CL[DF_ALL_CL["method"] == method].iloc[0]
        y_values = [row[col] for col in fde_cols]
        ax.plot(x_values, y_values, marker='o', label=method)
    
    datasets = DF_ALL_CL["train_dataset"].tolist()[0]
    dataset_order_text = "Tasks order: " + ", ".join(map(str, datasets))
    ax.text(
        0.5, -0.15,           
        dataset_order_text,
        transform=ax.transAxes,  
        ha="center", va="top",
        fontsize=10
    )

    ax.set_xticks(x_values)
    ax.set_xlabel("Tasks")
    ax.set_ylabel("Final displacement error (FDE) in meters")
    ax.set_ylim(max(ymin, 0.00), ymax) 
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Methods")
    
    ax.yaxis.set_major_locator(loc)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_fdePerTask.png"), dpi=300)
    plt.close()
    print("Saved plot 6: FDE per task")
    
if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
