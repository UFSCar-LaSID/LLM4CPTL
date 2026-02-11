import os
import torch

import pandas as pd

from args import get_all_args
from data.loader import data_loader, data_dset
from main_model.encoder import Predictor
from helper.evaluate import evaluate
from helper import utils
from helper.utils import (
    displacement_error,
    final_displacement_error,
    l2_loss,
    int_tuple,
    relative_to_abs,
    get_dset_path,
)

def get_generator(checkpoint):
    if args.main_model == "lstm":
        from main_model.encoder import Predictor
        model = Predictor(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            traj_lstm_input_size=args.traj_lstm_input_size,
            traj_lstm_hidden_size=args.traj_lstm_hidden_size,
            traj_lstm_output_size=args.traj_lstm_output_size
        )
    if args.main_model == "gat":
        n_units = (
                [args.traj_lstm_hidden_size]
                + [int(x) for x in args.hidden_units.strip().split(",")]
                + [args.graph_lstm_hidden_size]
        )
        n_heads = [int(x) for x in args.heads.strip().split(",")]
        from main_model.encoder_gat import Predictor
        model = Predictor(
            obs_len=args.obs_len, pred_len=args.pred_len, traj_lstm_input_size=args.traj_lstm_input_size,
            traj_lstm_hidden_size=args.traj_lstm_hidden_size, traj_lstm_output_size=args.traj_lstm_output_size,
            n_units=n_units, n_heads=n_heads, graph_network_out_dims=args.graph_network_out_dims,
            dropout=args.dropout, alpha=args.alpha, graph_lstm_hidden_size=args.graph_lstm_hidden_size
        )

    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model



def main(args):
    checkpoint_path = os.path.join(os.path.abspath(args.log_dir), "IL_{}_{}_{}_best.pth.tar".format(args.main_model, args.dataset_name_train, args.aug))
    checkpoint = torch.load(checkpoint_path)
    generator = get_generator(checkpoint)
    path = get_dset_path(args.dataset_name_test, "test")
    dset = data_dset(args, path, dataset_name=args.dataset_name_test, split_name="test")
    loader = data_loader(args, dset, batch_size=args.batch_size)
    ade, fde = evaluate(loader, generator)
    d = {'training dataset': args.dataset_name_train, 'testing dataset': args.dataset_name_test, 'Pred len': args.pred_len,
         'ADE': ade, 'FDE': fde}
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    utils.save_dict(d, f"{args.r_dir}/IL_batch_learning_{args.dataset_name_train}_{args.dataset_name_test}_{args.aug}_{args.main_model}")
    utils.save_dict_txt(d, f"{args.r_dir}/IL_batch_learning_{args.dataset_name_train}_{args.dataset_name_test}_{args.aug}_{args.main_model}")
    print(
        "Train Dataset: {} | Test Dataset: {} | Pred Len: {} | ADE: {:.12f} | FDE: {:.12f}".format(
            args.dataset_name_train, args.dataset_name_test, args.pred_len, ade, fde
        )
    )

    metrics_filename = f"{args.r_dir}/IL_metrics-{args.dataset}-{args.dataset}-{args.iters}-{args.batch_size}-{args.aug}-{args.main_model}.csv"
    metrics_data = {
        'method': 'IL',
        'train_dataset': args.dataset,
        'test_dataset': args.dataset,
        'learning_method': 'batch_learning',
        'replay_method': None,
        'training_time_in_secs': None,
        'observation_length': args.obs_len,
        'prediction_length': args.pred_len,
        'batch_size': args.batch_size,
        'replay_batch_size': None,
        'iters': args.iters,
        'main_predictor_model': args.main_model,
        'average_prediction_error_ape_ade': ade,
        'average_prediction_error_ape_fde': fde,
        'backward_transfer_bwt_ade': 0.0,
        'backward_transfer_bwt_fde': 0.0,
    }
    metrics_dataframe = pd.DataFrame(metrics_data, index=[0])
    metrics_dataframe.to_csv(metrics_filename, index=False)

    print("\nGenerated CSV file with metrics: {}".format(metrics_filename))

if __name__ == "__main__":    
    args = get_all_args()
    
    torch.cuda.set_device(args.gpu_index)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main(args)