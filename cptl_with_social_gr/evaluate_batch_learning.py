import argparse
import os
import pandas as pd

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle

import torch

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


torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="ETH", help="Directory containing logging file")
parser.add_argument('--iters', type=int, default=400, help="number of iterations that the model was trained on")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default")
parser.add_argument("--dataset_name_train", default="ETH", type=str)
parser.add_argument("--dataset_name_test", default="ETH", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=8, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=200, type=int)
parser.add_argument("--val_epoch", default=2, type=int)
augmentation_choices = ["none", "rotation"]
parser.add_argument("--aug", type=str, default='none', choices=augmentation_choices, help="whether to rotation the data")

parser.add_argument("--noise_dim", default=(8,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

# lstm
parser.add_argument("--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size")
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
parser.add_argument('--traj_lstm_output_size', default=32, type=int)
# gat
parser.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma")
parser.add_argument("--hidden-units", type=str, default="16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument("--graph_network_out_dims", type=int, default=32, help="dims of every node after through GAT module")
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)
parser.add_argument("--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)")
parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
model_choices = ["lstm", "gat"]
parser.add_argument('--main_model', default='lstm', type=str, choices=model_choices, help="the main model of CL and BL")

parser.add_argument("--num_samples", default=20, type=int)
parser.add_argument("--dset_type", default="test", type=str)
parser.add_argument("--resume", default="model_best.pth.tar", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")


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
    path = get_dset_path(args.dataset_name_test, args.dset_type)
    dset = data_dset(args, path)
    loader = data_loader(args, dset, batch_size=args.batch_size)
    ade, fde = evaluate(loader, generator)
    d = {'training dataset': args.dataset_name_train, 'testing dataset': args.dataset_name_test, 'Pred len': args.pred_len,
         'ADE': ade, 'FDE': fde}
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    utils.save_dict(d, "{}/IL_batch_learning_{}_{}_{}_{}".format(args.r_dir, args.dataset_name_train, args.dataset_name_test, args.aug, args.main_model))
    utils.save_dict_txt(d, "{}/IL_batch_learning_{}_{}_{}_{}".format(args.r_dir, args.dataset_name_train, args.dataset_name_test, args.aug, args.main_model))
    print(
        "Train Dataset: {} | Test Dataset: {} | Pred Len: {} | ADE: {:.12f} | FDE: {:.12f}".format(
            args.dataset_name_train, args.dataset_name_test, args.pred_len, ade, fde
        )
    )
    
    metrics_filename = "{}/IL_metrics-{dataset_name_train}-{dataset_name_test}-{aug}-{main_model}.csv".format(args.r_dir, dataset_name_train=args.dataset_name_train, dataset_name_test=args.dataset_name_test, aug=args.aug, main_model=args.main_model)
    metrics_data = {
        'method': 'IL',
        'train_dataset': args.dataset_name_train,
        'test_dataset': args.dataset_name_test,
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
    args = parser.parse_args()
    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
