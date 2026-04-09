#!/usr/bin/env python3

###################################
# Imports and packages
###################################
import argparse
import os
import numpy as np
import pandas as pd
import time
import torch
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter

from train import train_cl, train
from helper import evaluate
from helper import visual_plt
from helper import callbacks as cb
from helper import utils
from helper.param_values import set_default_values
from helper.param_stamp import get_param_stamp
from helper.continual_learner import ContinualLearner

from args import get_all_args

###################################
# Functions
###################################
def run(args, verbose=False):
    # Device definition:
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available(
    ) else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    print(device)

    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    # Set random seeds:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    # Identification of the name of the requested experiment to be run:
    variation_of_clsgr_executed = utils.variation_of_clsgr_being_executed(args)

    # Carbon emission tracker if requested for this experiment:
    if args.use_codecarbon:
        from codecarbon import EmissionsTracker

        tracker_carboncode = EmissionsTracker(
            project_name=f"{variation_of_clsgr_executed}_{args.obs_len}_{args.pred_len}_{args.batch_size}_{args.iters}_{args.main_model}",
            output_dir=args.r_dir,
            output_file=f"{variation_of_clsgr_executed}_emissions_{args.iters}_{args.batch_size}_{args.dataset_name}_.csv" if args.method == 'batch_learning' else f"{variation_of_clsgr_executed}_emissions_{args.iters}_{args.batch_size}_{'-'.join(args.dataset)}.csv"
        )

    ###############################################################################
    # Batch learning
    ###############################################################################
    if args.method == "batch_learning":
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        utils.set_logger(os.path.join(
            os.path.abspath(args.log_dir), "IL_train.log"))

        checkpoint_dir = args.log_dir + "/checkpoint"
        if os.path.exists(checkpoint_dir) is False:
            os.mkdir(checkpoint_dir)

        train_path = utils.get_dset_path(args.dataset_name, "train")
        val_path = utils.get_dset_path(args.dataset_name, "val")

        # loader data
        logging.info("Initializing train dataset")
        if args.aug == "none":
            from data.loader import data_loader, data_dset
        else:
            from data.loader_rotation import data_loader
        train_dset = data_dset(args, train_path, dataset_name=args.dataset_name, split_name="train")
        train_loader = data_loader(args, train_dset, args.batch_size)
        val_dset = data_dset(args, val_path, dataset_name=args.dataset_name, split_name="val")
        val_loader = data_loader(args, val_dset, args.batch_size)
        writer = SummaryWriter()

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
            from main_model.encoder_gat import Predictor
            model = Predictor(
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                traj_lstm_input_size=args.traj_lstm_input_size,
                traj_lstm_hidden_size=args.traj_lstm_hidden_size,
                traj_lstm_output_size=args.traj_lstm_output_size,
                n_units=n_units,
                n_heads=n_heads,
                graph_network_out_dims=args.graph_network_out_dims,
                dropout=args.dropout,
                alpha=args.alpha,
                graph_lstm_hidden_size=args.graph_lstm_hidden_size
            )
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_ade = 200
        if args.resume:
            if os.path.isfile(args.resume):
                logging.info(
                    f"Restoring from checkpoint {args.resume}")
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])
                logging.info(
                    f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
                )
            else:
                logging.info(
                    f"=> no checkpoint found at '{args.resume}'")

        if args.time:
            start = time.time()

        if args.use_codecarbon:
            tracker_carboncode.start()

        for epoch in range(args.start_epoch, args.iters + 1):
            train(args, model, train_loader, optimizer, epoch, writer)
            if epoch >= args.val_epoch:
                ade = utils.validate(
                    args, model, val_loader, epoch, writer=writer)
                is_best = ade < best_ade
                best_ade = min(ade, best_ade)
                # if epoch % args.checkpoint_log == 0:
                save_checkpoint_path = args.log_dir + "/checkpoint"
                utils.save_checkpoint(
                    args,
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    save_checkpoint_path +
                    f"/IL_{model.name}_checkpoint{epoch}_{args.aug}.pth.tar",
                    model_name=model.name,
                )

        writer.close()

        if args.use_codecarbon:
            _ = tracker_carboncode.stop()

        # Get total training-time in seconds, and write to file
        if args.time:
            param_stamp = get_param_stamp(
                args, model.name, verbose=verbose, replay=True if (
                    not args.replay == "none") else False,
                replay_model_name=generator.name if (
                    args.replay == "generative") else None,
            )

            training_time = time.time() - start

        if verbose and args.time:
            print("=> Total training time = {:.1f} seconds\n".format(
                training_time))

    ###############################################################################
    # Continual learning
    ###############################################################################
    if args.method == "continual_learning":
        logging.info("This experiment will make use of continual learning")

        # Set default arguments & check for incompatible options
        args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
        args.g_iters = args.iters if args.g_iters is None else args.g_iters

        # If [log_per_task], reset all logs:
        if args.log_per_task:
            args.prec_log = args.iters
            args.loss_log = args.iters
            args.sample_log = args.iters

        # Create plots and results directories if they do not exist yet:
        if not os.path.isdir(args.r_dir):
            os.mkdir(args.r_dir)
        if args.pdf and not os.path.isdir(args.p_dir):
            os.mkdir(args.p_dir)

        # ------------------------------------------------------------------------------------------------#
        # ----------------#
        # ------data------#
        # ----------------#

        # Prepare data for chosen experiment
        if verbose:
            print("\nPreparing the data...")

        # Defining train, validation, and test dataset orders:
        train_order = val_order = test_order = args.dataset

        # Number of tasks (datasets) to be analyzed:
        tasks = len(train_order)

        # Lists to storage the readed datasets and data loader objects:
        train_datasets = []
        val_datasets = []
        val_loaders = []
        test_datasets = []

        # Loading the correct loader function according to the the preprocessing to be applid on data (if any):
        if args.aug == "none":
            from data.loader import data_loader, data_dset
        else:
            from data.loader_rotation import data_loader

        print("\nInitializing train dataset")
        for i, dataset_name in enumerate(train_order):
            train_path = utils.get_dset_path(dataset_name, 'train')
            
            train_dset = data_dset(
                args,
                train_path,
                dataset_name=dataset_name,
                split_name="train"
            )

            print(
                f"Dataset: {dataset_name} | Split: train | Number of trajectories: {train_dset.obs_traj.shape[0]}")

            train_datasets.append(train_dset)

        print("\nInitializing val dataset")
        for i, dataset_name in enumerate(val_order):
            val_path = utils.get_dset_path(dataset_name, "val")

            val_dset = data_dset(
                args,
                val_path,
                dataset_name=dataset_name,
                split_name="val"
            )

            val_loader = data_loader(args, val_dset, args.batch_size)

            print(
                f"Dataset: {dataset_name} | Split: val | Number of trajectories: {val_dset.obs_traj.shape[0]}")

            val_datasets.append(val_dset)

            val_loaders.append(val_loader)

        print("\nInitializing test dataset")
        for i, dataset_name in enumerate(test_order):
            test_path = utils.get_dset_path(dataset_name, "test")

            test_dset = data_dset(
                args,
                test_path,
                dataset_name=dataset_name,
                split_name="test"
            )

            test_loader = data_loader(args, test_dset)

            print(
                f"Dataset: {dataset_name} | Split: test | Number of trajectories: {test_dset.obs_traj.shape[0]}")

            test_datasets.append(test_loader)

        # --------------------------------------------------------------------------------------------------#
        # --------------------#
        # ----Model (LSTM)----#
        # --------------------#

        # Define main model (i.e., lstm, if requested with feedback connections)
        print("\nDefining the main model...")
        if args.main_model == "lstm":
            from main_model.encoder import Predictor
            model = Predictor(
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                traj_lstm_input_size=args.traj_lstm_input_size,
                traj_lstm_hidden_size=args.traj_lstm_hidden_size,
                traj_lstm_output_size=args.traj_lstm_output_size
            ).to(device)

        if args.main_model == "gat":
            from main_model.encoder_gat import Predictor
            model = Predictor(
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                traj_lstm_input_size=args.traj_lstm_input_size,
                traj_lstm_hidden_size=args.traj_lstm_hidden_size,
                traj_lstm_output_size=args.traj_lstm_output_size,
                n_units=n_units,
                n_heads=n_heads,
                graph_network_out_dims=args.graph_network_out_dims,
                dropout=args.dropout,
                alpha=args.alpha,
                graph_lstm_hidden_size=args.graph_lstm_hidden_size
            ).to(device)

        # Define optimizer (only include parameters that "requires_grad")
        model.optim_type = args.optimizer
        if model.optim_type in ("adam", "adam_reset"):
            model.optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif model.optim_type == "sgd":
            model.optimizer = optim.SGD(model.optim_list)
        else:
            raise ValueError(
                "Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))

        best_ade = 200

        # -----------------------------------------------------------------------------------------------------------------#

        # Synpatic Intelligence (SI)
        if isinstance(model, ContinualLearner):
            model.si_c = args.si_c if args.si else 0
            if args.si:
                model.epsilon = args.epsilon

        # ---------------------------#
        # ----CL-STRATEGY: REPLAY----#
        # ---------------------------#
        # Boolean flag that indicates wheter a generator must be trained to generate the replay:
        train_gen = True if (args.replay == "generative") else False

        # If a replay generator must be trained, then:
        if train_gen:
            fake_generator = None
            # Replay model architecture: LSTM or condition
            if args.replay_model == 'lstm' or args.replay_model == 'condition':
                from generative_model.vae_models_scratch import CVAE

                generator = CVAE(obs_len=args.obs_len,
                                 pred_len=args.pred_len,
                                 traj_lstm_input_size=args.traj_lstm_input_size,
                                 traj_lstm_hidden_size=args.traj_lstm_hidden_size,
                                 traj_lstm_output_size=args.traj_lstm_output_size,
                                 dropout=args.dropout,
                                 z_dim=args.z_dim,
                                 embedding_dim=args.embedding_dim,
                                 mlp_dim=args.mlp_dim,
                                 bottleneck_dim=args.bottleneck_dim,
                                 activation='relu',
                                 batch_norm=True,
                                 adapt_architecture_to_include_sequence_embedding=args.adapt_architecture_to_include_sequence_embedding,
                                 sequence_embedding_dimension=args.dimensions,
                                 sequence_embedding_compressed_dimension=args.sequence_embedding_compressed_dimension,
                                 use_prior_adaptation=args.use_prior_adaptation,
                                 use_dc_vampprior=args.use_dc_vampprior
                                 ).to(device)

            # Define optimizer(s)
            generator.optim_type = args.optimizer
            if generator.optim_type in ("adam", "adam_reset"):
                generator.optimizer = optim.Adam(
                    generator.parameters(), lr=args.lr_gen)

            elif generator.optim_type == "sgd":
                generator.optimizer = optim.SGD(generator.optim_list)

        # In case a replay generator must NOT be trained, then:
        else:
            generator = None

            if args.replay_model == 'lstm':
                from generative_model.vae_models_scratch import CVAE

                fake_generator = CVAE(obs_len=args.obs_len,
                                      pred_len=args.pred_len,
                                      traj_lstm_input_size=args.traj_lstm_input_size,
                                      traj_lstm_hidden_size=args.traj_lstm_hidden_size,
                                      traj_lstm_output_size=args.traj_lstm_output_size,
                                      dropout=args.dropout,
                                      z_dim=args.z_dim,
                                      embedding_dim=args.embedding_dim,
                                      mlp_dim=args.mlp_dim,
                                      bottleneck_dim=args.bottleneck_dim,
                                      activation='relu',
                                      batch_norm=True,
                                      adapt_architecture_to_include_sequence_embedding=args.adapt_architecture_to_include_sequence_embedding,
                                      sequence_embedding_dimension=args.dimensions,
                                      sequence_embedding_compressed_dimension=args.sequence_embedding_compressed_dimension,
                                      use_prior_adaptation=args.use_prior_adaptation,
                                      use_dc_vampprior=args.use_dc_vampprior
                                      ).to(device)

            # -Define optimizer(s):
            fake_generator.optim_type = args.optimizer
            if fake_generator.optim_type in ("adam", "adam_reset"):
                fake_generator.optimizer = optim.Adam(
                    fake_generator.parameters(), lr=args.lr_gen)
            elif fake_generator.optim_type == "sgd":
                fake_generator.optimizer = optim.SGD(fake_generator.optim_list)

        # ------------------------------------------------------------------------------------------------------------------#
        # --------------------#
        # ------REPORTING-----#
        # --------------------#

        # Print some model-characteristics on the screen
        if verbose:
            # -main model
            utils.print_model_info(model, title="MAIN MODEL")
            # -generator
            if generator is not None:
                utils.print_model_info(generator, title="GENERATOR")

        # Prepare for keeping track of statistics required for metrics (also used for plotting in pdf)
        if args.pdf or args.metrics:
            metric_dict = evaluate.initiate_metrics_dict(n_tasks=tasks)
            metric_dict = evaluate.intial_accuracy(
                model, test_datasets, metric_dict)

        else:
            metric_dict = None

        # -Prepare for plotting in visdom
        # -visdom-settings
        if args.visdom:
            env_name = f"epoch-lstm-GR-lstm-replay-{args.dataset}-{tasks}-{args.iters}-{args.iz_dim}-{args.ibatch_size}-{args.ireplay_batch_size}-{args.ilr}-{args.iseed}-{args.ival}-{args.ival_class}-si{args.isi}-{args.isi_c}"
            graph_name = f"{'SI' if args.si else ''}_{args.replay}"
            visdom = {'env': env_name, 'graph': graph_name}
        else:
            visdom = None

        # ----------------------------------------------------------------------------------------------------------------#
        # -----------------#
        # ----CALLBACKS----#
        # -----------------#  #

        # Callbacks for reporting and visualizing accuracy
        generator_loss_cbs = fake_generator_loss_cbs = [None]

        # Callbacks for evaluating and plotting generated / reconstructed samples
        sample_cbs = [None]

        if train_gen:
            generator_loss_cbs = [
                cb._VAE_loss_cb(log=args.loss_log,
                                visdom=visdom,
                                tasks=tasks,
                                iters_per_task=args.g_iters,
                                replay=False if args.replay == "none" else True
                                )
            ]

            fake_generator_loss_cbs = [
                cb._VAE_loss_cb(log=args.loss_log,
                                visdom=visdom,
                                tasks=tasks,
                                iters_per_task=args.g_iters,
                                replay=False if args.replay == "none" else True
                                )
            ]

            sample_cbs = []

        solver_loss_cbs = [
            cb._solver_loss_cb(log=args.loss_log,
                               visdom=visdom,
                               tasks=tasks,
                               iters_per_task=args.iters,
                               replay=False if args.replay == "none" else True
                               )
        ]

        solver_val_loss_cbs = [
            cb._solver_val_loss_cb(log=args.loss_log,
                                   visdom=visdom,
                                   model=model,
                                   tasks=tasks,
                                   iters_per_task=args.iters,
                                   replay=False if args.replay == "none" else True
                                   )
        ]

        # Callbacks for reporting and visualizing accuracy
        eval_cbs = [
            cb._eval_cb(log=args.prec_log,
                        test_datasets=val_loaders,
                        visdom=visdom,
                        iters_per_task=args.iters
                        )
        ]

        # Callbacks for calculating statists required for metrics
        metric_cbs = [
            cb._metric_cb(log=args.iters,
                          test_datasets=test_datasets,
                          iters_per_task=args.iters,
                          metrics_dict=metric_dict
                          )
        ]

        # -----------------------------------------------------------------------------------------------------------------#
        # ----------------#
        # ----TRAINING----#
        # ----------------#

        if verbose:
            print("\nTraining...")

        # Keep track of training duration, if requested:
        if args.time:
            start = time.time()

        # Keep track of carbon emission, if requested:
        if args.use_codecarbon:
            tracker_carboncode.start()

        # Train model
        ade_matrix, fde_matrix, elpased_time_for_each_task = train_cl(args,
                                                                    best_ade,
                                                                    model,
                                                                    train_datasets,
                                                                    val_datasets,
                                                                    replay_model=args.replay,
                                                                    iters=args.iters,
                                                                    batch_size=args.batch_size,
                                                                    generator=generator,
                                                                    fake_generator=fake_generator,
                                                                    gen_iters=args.g_iters,
                                                                    gen_loss_cbs=generator_loss_cbs,
                                                                    sample_cbs=sample_cbs,
                                                                    eval_cbs=eval_cbs,
                                                                    loss_cbs=solver_loss_cbs,
                                                                    val_loss_cbs=solver_val_loss_cbs,
                                                                    metric_cbs=metric_cbs,
                                                                    test_order=test_order
                                                                    )

        # Keep track of carbon emission, if requested:
        if args.use_codecarbon:
            _ = tracker_carboncode.stop()

        # Get total training duration in seconds and write it into a file
        if args.time:
            training_time = time.time() - start

        # Save trained model to a file for future load and inference:
        #model_checkpoint_filename = f"{args.r_dir}/{variation_of_clsgr_executed}_mainModelCheckpoint_taskFinal_{model.name}_{args.iters}_{args.batch_size}_{'-'.join(args.dataset)}.pth"

        #utils.save_checkpoint(args=args, state=model.state_dict(
        #), is_best=False, filename=model_checkpoint_filename)

        #print(f"Final trained model (type: {type(model)}) saved to {model_checkpoint_filename}")

        # ------------------------------------------------------------------------------------------------------------------#
        # ------------------#
        # ----EVALUATION----#
        # ------------------#

        if verbose:
            print(
                f"\n\nEVALUATION RESULTS of final trained model (type: {type(model)}) on each task-specific test-set:")

        # Evaluate precision of final model on full (all tasks) test-set
        ades = []
        fdes = []
        for i in range(tasks):
            ade, fde = evaluate.validate(model, test_datasets[i])
            ades.append(ade)
            fdes.append(fde)
        average_ades = sum(ades) / tasks
        average_fdes = sum(fdes) / tasks

        if verbose:
            print("\n Precision on test-set")
            for i in range(tasks):
                print(f" - Task {i+1}: ADE {ades[i]:.4f} FDE {fdes[i]:.4f}")
            print(
                f"==> Average precision over all {tasks} tasks: ADE {average_ades:.4f} FDE {average_fdes:.4f}")

        if verbose and args.time:
            print(f"=> Total training time = {training_time:.1f} seconds\n")

        if args.metrics:
            backward_transfer_bwt_ade = forward_transfer_fwt_ade = true_bwt_ade = cbwt_per_task_ade = 0
            backward_transfer_bwt_fde = forward_transfer_fwt_fde = true_bwt_fde = cbwt_per_task_fde = 0

            if ade_matrix is not None:
                backward_transfer_bwt_ade, forward_transfer_fwt_ade, true_bwt_ade, cbwt_per_task_ade = evaluate.calculate_cl_metrics(ade_matrix, error_metric="ADE")
            if fde_matrix is not None:
                backward_transfer_bwt_fde, forward_transfer_fwt_fde, true_bwt_fde, cbwt_per_task_fde = evaluate.calculate_cl_metrics(fde_matrix, error_metric="FDE")

            metrics_filename = f"{args.r_dir}/{variation_of_clsgr_executed}_metrics_{args.iters}_{args.batch_size}_{'-'.join(args.dataset)}.csv"
            metrics_data = {
                # Experiment settings:
                'method': variation_of_clsgr_executed,
                'train_dataset': ' - '.join(train_order),
                'test_dataset': ' - '.join(test_order),
                'learning_method': 'continual_learning',
                'replay_method': args.replay,
                'training_time_in_secs': training_time,
                'observation_length': args.obs_len,
                'prediction_length': args.pred_len,
                'batch_size': args.batch_size,
                'replay_batch_size': args.replay_batch_size,
                'iters': args.iters,
                'main_predictor_model': args.main_model,
                # Standard error metrics:
                'average_prediction_error_ape_ade': average_ades,
                'average_prediction_error_ape_fde': average_fdes,
                # Continuous learning metrics:
                'backward_transfer_bwt_ade': backward_transfer_bwt_ade,
                'forward_transfer_fwt_ade': forward_transfer_fwt_ade,
                'true_bwt_ade': true_bwt_ade,
                'backward_transfer_bwt_fde': backward_transfer_bwt_fde,
                'forward_transfer_fwt_fde': forward_transfer_fwt_fde,
                'true_bwt_fde': true_bwt_fde,
            }

            for i in range(tasks):
                metrics_data.update(
                    {f"ade_task{i+1}_after_training_in_all_tasks": ades[i]}
                )

                metrics_data.update(
                    {f"fde_task{i+1}_after_training_in_all_tasks": fdes[i]}
                )

                metrics_data.update(
                    {f"elapsed_time_task_{i+1}": elpased_time_for_each_task[i]}
                )

                if i != tasks - 1:
                    metrics_data.update(
                        {f"ade_cbwt_task_{i+1}": cbwt_per_task_ade[i]}
                    )

                    metrics_data.update(
                        {f"fde_cbwt_task_{i+1}": cbwt_per_task_fde[i]}
                    )
                else:
                    continue

            metrics_dataframe = pd.DataFrame(metrics_data, index=[0])
            metrics_dataframe.to_csv(metrics_filename, index=False)

            print(f"\nGenerated CSV file with metrics: {metrics_filename}")

        # ------------------------------------------------------------------------------------------------------------------#

        # -----------------#
        # ----PLOTTING-----#
        # -----------------#

        # If requested, generate pdf
        if args.pdf:
            plot_name = f"{args.p_dir}/{variation_of_clsgr_executed}-{args.replay}-{args.iters}-{args.aug}-{args.seed}-{args.val}-{args.val_class}-{args.si}-{args.si_c}.pdf"
            pp = visual_plt.open_pdf(plot_name)

            # -show metrics reflecting progression during training
            figure_list = []  # -> create list to store all figures to be plotted

            # -generate all figures (and store them in [figure_list])
            key_ade = "ade per task"
            plot_ade_list = []
            for i in range(tasks):
                plot_ade_list.append(
                    metric_dict[key_ade]["task {}".format(i+1)])

            key_fde = "fde per task"
            plot_fde_list = []
            for i in range(tasks):
                plot_fde_list.append(
                    metric_dict[key_fde]["task {}".format(i+1)])

            higher_ymax = max(max(sublist)
                              for sublist in plot_ade_list if sublist)
            higher_ymax = max(max(sublist)
                              for sublist in plot_fde_list if sublist)

            figure = visual_plt.plot_lines(
                plot_ade_list, x_axes=metric_dict["x_task"],
                line_names=["task {}".format(i+1) for i in range(tasks)],
                title="ADE for each tasks", xlabel="Tasks", ylabel="ADE", ylim=(0, higher_ymax+1)
            )
            figure_list.append(figure)

            figure = visual_plt.plot_lines(
                plot_fde_list, x_axes=metric_dict["x_task"],
                line_names=["task {}".format(i+1) for i in range(tasks)],
                title="FDE for each tasks", xlabel="Tasks", ylabel="FDE", ylim=(0, higher_ymax+1)
            )
            figure_list.append(figure)

            # calculate average ade/fde
            higher_ymax = max(max(sublist) for sublist in [
                              metric_dict["average_ade"]] if sublist)
            higher_ymax = max(max(sublist) for sublist in [
                              metric_dict["average_fde"]] if sublist)

            figure = visual_plt.plot_lines(
                [metric_dict["average_ade"]], x_axes=metric_dict["x_task"],
                line_names=["Average ade all tasks so far"],
                title="Average ADE", xlabel="Tasks", ylabel="ADE", ylim=(0, higher_ymax+1)
            )
            figure_list.append(figure)

            figure = visual_plt.plot_lines(
                [metric_dict["average_fde"]], x_axes=metric_dict["x_task"],
                line_names=["Average fde all tasks so far"],
                title="Average FDE", xlabel="Tasks", ylabel="FDE", ylim=(0, higher_ymax+1)
            )
            figure_list.append(figure)

            # -add figures to pdf (and close this pdf)
            for figure in figure_list:
                pp.savefig(figure)

            results_dict = {}
            results_dict["parameters"] = {
                "iters": args.iters,
                "z_dim": args.z_dim,
                "batch_size": args.batch_size,
                "lr": args.lr
            }

            results_dict["training order"] = train_order
            results_dict["ade per task"] = plot_ade_list
            results_dict["fde per task"] = plot_fde_list
            results_dict["average ade per task"] = metric_dict["average_ade"]
            results_dict["average fde per task"] = metric_dict["average_fde"]
            
            # -close pdf
            pp.close()

            # -print name of generated plot on screen
            if verbose:
                print("\nGenerated plot: {}\n".format(plot_name))


###################################
# Main program
###################################
if __name__ == '__main__':
    # Load arguments
    args = get_all_args()
    # Set default values for certain arguments
    args = set_default_values(args)
    # Run experiment
    run(args, verbose=True)
