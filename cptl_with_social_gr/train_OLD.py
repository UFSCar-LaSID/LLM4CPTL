#!/usr/bin/env python3

###################################
# Imports and packages
###################################

import tqdm
import copy
import csv
import os
import shutil
import numpy as np

import torch
from torch.autograd import Variable

from helper import utils
from helper.continual_learner import ContinualLearner
from helper import evaluate
from data.loader import data_loader, data_dset

###################################
# Functions
###################################


def train(args, model, train_loader, optimizer, epoch, writer):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            non_linear_ped,
            loss_mask,
            seq_start_end,
            t_embeddings
        ) = batch
        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len:]

        ###################################################
        model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
        pred_traj_fake_rel = model(model_input, seq_start_end)
        l2_loss_rel = utils.l2_loss(
            pred_traj_fake_rel,
            model_input[-args.pred_len:],
            loss_mask,
            mode="average",
        )

        loss = l2_loss_rel
        losses.update(loss.item(), obs_traj.shape[1])
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)
    writer.add_scalar("train_loss", losses.avg, epoch)


def train_cl(args,
             best_ade,
             model,
             train_datasets,
             val_datasets,
             replay_model="none",
             iters=2,
             batch_size=32,
             generator=None,
             fake_generator=None,
             gen_iters=0,
             gen_loss_cbs=list(),
             fake_gen_loss_cbs=list(),
             loss_cbs=list(),
             val_loss_cbs=list(),
             eval_cbs=list(),
             sample_cbs=list(),
             metric_cbs=list(),
             test_order=None
             ):
    '''
    Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]           <nn.Module> main model to optimize across all tasks
    [train_datasets]  <list> with for each task the training <DataSet>
    [replay_mode]     <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]        <str>, choice from "task", "domain" and "class"
    [iters]           <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]       None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]           <list> of call-back functions to evaluate training-progress
    '''

    variation_of_clsgr_executed = utils.variation_of_clsgr_being_executed(args)

    #### Losses files for main and generative models ########################
    vae_losses_file = os.path.join(
        args.r_dir, f"{variation_of_clsgr_executed}_losses_vae_{args.batch_size}_{args.iters}.csv")

    main_losses_file = os.path.join(
        args.r_dir, f"{variation_of_clsgr_executed}_losses_main_{args.batch_size}_{args.iters}.csv")

    if os.path.exists(vae_losses_file):
        os.remove(vae_losses_file)

    if os.path.exists(main_losses_file):
        os.remove(main_losses_file)
    ###########################################################################

    # Set model in training-mode
    model.train()

    ################ BWT metric-related computation ###########
    num_tasks = len(train_datasets)
    ade_matrix = np.zeros((num_tasks + 1, num_tasks))
    fde_matrix = np.zeros((num_tasks + 1, num_tasks))

    # --- Avaliacao Inicial (Task 0) para Forward Transfer ---
    print("Avaliacao inicial (pesos aleatorios) para FWT...")
    model.eval()
    for i in range(num_tasks):
        # test_order vem do main.py
        path = utils.get_dset_path(test_order[i], "test")
        dset = data_dset(args, path, dataset_name=test_order[i], split_name="test")
        loader = data_loader(args, dset, args.batch_size)
        ade, fde = evaluate.validate(model, loader)
        ade_matrix[0, i] = ade
        fde_matrix[0, i] = fde

        print(f"ADE: {ade:.2f}")
        print(f"FDE: {fde:.2f}")
    model.train()
    ###########################################################

    elpased_time_for_each_task = []

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c > 0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer(
                    '{}_SI_prev_task'.format(n), p.data.clone())

    completed_dataset_names = []

    validation_loss_per_epoch = []
    
    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):
        current_dataset_name = train_dataset.dataset_name

        print(
            f"\nIterating over task {task}, dataset {current_dataset_name}")

        training_dataset = data_loader(args, train_dataset, args.batch_size)
        batch_num = len(training_dataset)

        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (model.si_c > 0):
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        # Initialize # iters left on current data-loader(s)
        # Loop over all iterations
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)
        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters_to_use + 1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters + 1))
        # replay previous data for validation
        if fake_generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters * batch_num + 1))
        if args.val:
            x_rel_val = None
            y_rel_val = None
            seq_start_end_val = None

        replay_data_loader = data_loader(
            args, train_dataset, args.replay_batch_size)
        replay_data_loader = iter(replay_data_loader)

        ## -----REPLAYED BATCH------##
        if not Exact and not Generative and not Current:
            print(f"    No replay will be performed during this task")
            # -> if no replay
            x_rel_ = y_rel_ = seq_start_end_ = t_embeddings_ = None

        # run epoch
        for epoch in range(1, iters_to_use+1):
            print(f"    Epoch {epoch}/{iters_to_use}")

            losses_dict_main = {
                'loss_total': [],
                'loss_current': [],
                'loss_replay': [],
                'pred_traj': [],
                'pred_traj_r': []
            }

            losses_dict_generative = {
                'loss_total': [],
                'reconL': [],
                'variatL': [],
                'reconL_r': [],
                'variatL_r': []
            }

            for batch_index, batch in enumerate(training_dataset):
                print(f"        Batch {batch_index+1}/{batch_num}")

                batch = [tensor.cuda() for tensor in batch]
                (
                    obs_traj,
                    pred_traj_gt,
                    obs_traj_rel,
                    pred_traj_gt_rel,
                    non_linear_ped,
                    loss_mask,
                    seq_start_end,
                    t_embeddings
                ) = batch

                print(
                    f"            Number of trajectories in current task-specific batch: {obs_traj.shape[1]}")
                print(
                    f"            Number of embeddings in current task-specific batch: {t_embeddings.shape[0]}")

                # -------------Collect data----------------#
                ## ------CURRENT BATCH-------##
                x_rel = obs_traj_rel
                y_rel = pred_traj_gt_rel
                seq_start_end = seq_start_end
                loss_mask = loss_mask
                t_embeddings = t_embeddings
                t_embeddings_ = None

                # ------ Exact Replay ----- #
                if Exact or (Generative and args.replay_model == "condition"):
                    print(f"        Dynamic Exact/Conditional Replay will be performed")

                    if task > 1:
                        datasets_to_replay = completed_dataset_names

                        from helper.memory_unified import memory_buff_unified

                        print(
                            f"        Replaying memory from tasks: {datasets_to_replay}")

                        loader_replay, batch_replay = memory_buff_unified(
                            args, datasets_to_replay)

                        memory_seq_loader = iter(loader_replay)
                        memory_seq = next(memory_seq_loader)

                        x_rel_ = memory_seq[2].cuda()
                        y_rel_ = memory_seq[3].cuda()
                        seq_start_end_ = memory_seq[4].cuda()
                        t_embeddings_ = memory_seq[5].cuda()

                # ----Generative / Current Replay----#
                if Generative:
                    print(
                        f"            Generative replay will be performed for this batch")

                    # Get replayed data (i.e., [x_]) -- either current data or use previous generator

                    try:
                        replay_out = next(replay_data_loader)
                    except StopIteration:
                        replay_data_loader = iter(data_loader(
                            args, train_dataset, args.replay_batch_size))
                        replay_out = next(replay_data_loader)

                    if args.replay_model == 'lstm':
                        print(f"            LSTM replay model selected")

                        print(
                            f"            'previous_generator' (type: {type(previous_generator)}) is sampling replay data from 'replay_out' (type: {type(replay_out)}), built on dataset {train_dataset.dataset_name}")

                        replay_traj = previous_generator.sample(replay_out[2].to(device), replay_out[0].to(device),
                                                                replay_out[6].to(device), replay_out[7].to(device))

                        x_ = replay_traj[0]
                        x_rel_ = replay_traj[1]
                        seq_start_end_ = replay_traj[2]
                        t_embeddings_ = replay_traj[3]

                        print(
                            f"            Number of trajectories replayed: {x_rel_.shape[1]}")
                        print(
                            f"            Number of embeddings replayed: {t_embeddings_.shape[0]}")

                    if args.replay_model == 'vrnn':
                        replay_traj = previous_generator.sample(replay_out[2].to(device), replay_out[0].to(device),
                                                                replay_out[6].to(device))
                        x_rel_ = replay_traj.cuda()
                        seq_start_end_ = seq_start_end
                        t_embeddings_ = t_embeddings

                    if variation_of_clsgr_executed == "CL_SGR":
                        previous_model.eval()
                        with torch.no_grad():
                            y_rel_ = previous_model(x_rel_, seq_start_end_)
                        previous_model.train()

                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                # -if there are no task-specific mask, obtain all predicted scores at once
                if Generative and x_rel_ is not None:
                    x_rel_.detach_()
                    x_rel_ = x_rel_.detach()
                    x_rel_ = Variable(x_rel_.data, requires_grad=True)
                    y_rel_.detach_()
                    y_rel_ = y_rel_.detach()
                    y_rel_ = Variable(y_rel_.data, requires_grad=True)

                # ----> Train Main model
                if batch_index <= iters*batch_num:
                    print(
                        f"            Training main model (type: {type(model)}) for this batch")

                    # Train the main model with this batch
                    loss_dict_main = model.train_a_batch(
                        x_rel, y_rel, seq_start_end, x_rel_=x_rel_, y_rel_=y_rel_, seq_start_end_=seq_start_end_, loss_mask=loss_mask, rnt=1./task)

                    main_loss_file = open(
                        f"{args.r_dir}/{variation_of_clsgr_executed}_loss_main_model_{args.iters}_{args.batch_size}_{args.replay}_{args.val_class}.txt", 'a')

                    main_loss_file.write(
                        f"{batch_index}: {loss_dict_main['loss_total']}\n")

                    main_loss_file.close()

                    losses_dict_main['loss_total'].append(
                        loss_dict_main['loss_total'])

                    losses_dict_main['loss_current'].append(
                        loss_dict_main['loss_current'])

                    losses_dict_main['loss_replay'].append(
                        loss_dict_main['loss_replay'])

                    losses_dict_main['pred_traj'].append(
                        loss_dict_main['pred_traj'])

                    losses_dict_main['pred_traj_r'].append(
                        loss_dict_main['pred_traj_r'])

                    # Update running parameter importance estimates in W
                    if isinstance(model, ContinualLearner) and (model.si_c > 0):
                        for n, p in model.named_parameters():
                            if p.requires_grad:
                                n = n.replace('.', '__')
                                if p.grad is not None:
                                    W[n].add_(-p.grad *
                                              (p.detach() - p_old[n]))
                                p_old[n] = p.detach().clone()

                # -----> Train Generator
                if generator is not None and batch_index <= gen_iters*batch_num:
                    print(
                        f"            Training generative model (type: {type(generator)}) for this batch")

                    # Train the generator with this batch
                    loss_dict_generative = generator.train_a_batch(
                        x_rel, y_rel, seq_start_end, c=t_embeddings, x_=x_rel_, y_=y_rel_, seq_start_end_=seq_start_end_, c_=t_embeddings_, rnt=1./task)

                    losses_dict_generative['loss_total'].append(
                        loss_dict_generative['loss_total'])

                    losses_dict_generative['reconL'].append(
                        loss_dict_generative['reconL'])

                    losses_dict_generative['variatL'].append(
                        loss_dict_generative['variatL'])

                    losses_dict_generative['reconL_r'].append(
                        loss_dict_generative['reconL_r'])

                    losses_dict_generative['variatL_r'].append(
                        loss_dict_generative['variatL_r'])

            # Calcular a média das losses de batch coletadas durante a época
            print(
                f"        Computing the average of the losses from main model (type: {type(model)}) for all batches processed during this epoch")
            
            avg_main_loss_total = np.mean(
                losses_dict_main['loss_total'])
            avg_loss_current = np.mean(losses_dict_main['loss_current'])
            avg_loss_replay = np.mean(losses_dict_main['loss_replay'])
            avg_pred_traj = np.mean(losses_dict_main['pred_traj'])
            avg_pred_traj_r = np.mean(losses_dict_main['pred_traj_r'])

            log_file_path_main = main_losses_file

            # Escrever cabeçalho se o arquivo não existir
            if not os.path.exists(log_file_path_main):
                with open(log_file_path_main, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Method', 'Task', 'Epoch', 'Loss_Total', 'Loss_Current', 'Loss_Replay', 'Pred_Traj', 'Pred_Traj_R'])

            # Anexar dados
            with open(log_file_path_main, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    variation_of_clsgr_executed,
                    task,
                    epoch,
                    avg_main_loss_total,
                    avg_loss_current,
                    avg_loss_replay,
                    avg_pred_traj,
                    avg_pred_traj_r
                ])

            if args.val:
                print(
                    f"        Validating main model at epoch {epoch} for task {task}")
                if args.val_class == 'current':
                    val_dataset = data_loader(
                        args, val_datasets[task-1], args.batch_size)
                    ade_current, loss_val = utils.validate_cl(
                        args, model, val_dataset, epoch)
                    # save val loss
                    val_loss_file = open("{}/{}_loss_val_{}_{}_{}_{}.txt".format(
                        args.r_dir, variation_of_clsgr_executed, args.iters, args.batch_size, args.replay, args.val_class), 'a')
                    val_loss_file.write('{}: {}\n'.format(epoch, loss_val))
                    val_loss_file.close()
                    loss_val_dict_main = {'loss_val': loss_val}
                    for val_loss_cb in val_loss_cbs:
                        if val_loss_cb is not None:
                            val_loss_cb(progress, epoch,
                                        loss_val_dict_main, task=task)
                    ade_val = ade_current
                    is_best = ade_val < best_ade
                    best_ade = min(ade_val, best_ade)
                    if is_best:
                        previous_model = copy.deepcopy(model)
                        file_dir = os.path.dirname(__file__) + "/chekpoint"
                        if os.path.exists(file_dir) is False:
                            os.mkdir(file_dir)
                        filename = os.path.join(file_dir,
                                                "{r_dir}/{variation_of_clsgr_executed}_{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                    r_dir=args.r_dir,
                                                    variation_of_clsgr_executed=variation_of_clsgr_executed,
                                                    method=args.method, replay=args.replay, task=task,
                                                    order=args.dataset_order, batch_size=args.batch_size,
                                                    seed=args.seed, epoch=epoch,
                                                    val=args.val, val_class=args.val_class,
                                                    si=args.si, si_c=args.si_c))
                        torch.save(model.state_dict(), filename)
                        shutil.copyfile(filename, "{r_dir}/{variation_of_clsgr_executed}_{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                            r_dir=args.r_dir,
                            variation_of_clsgr_executed=variation_of_clsgr_executed,
                            method=args.method, replay=args.replay, task=task,
                            order=args.dataset_order, batch_size=args.batch_size,
                            seed=args.seed,
                            val=args.val, val_class=args.val_class,
                            si=args.si, si_c=args.si_c))
                if args.val_class == 'all':
                    if generator is None:
                        val_dataset = data_loader(
                            args, val_datasets[task - 1], args.batch_size)
                        ade_current, loss_val = utils.validate_cl(
                            args, model, val_dataset, epoch)
                        loss_val_dict_main = {'loss_val': loss_val}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch,
                                            loss_val_dict_main, task=task)
                        ade_val = ade_current
                        is_best = ade_val < best_ade
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            previous_model = copy.deepcopy(model)
                            file_dir = os.path.dirname(__file__) + "/chekpoint"
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{r_dir}/{variation_of_clsgr_executed}_{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        r_dir=args.r_dir,
                                                        variation_of_clsgr_executed=variation_of_clsgr_executed,
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            torch.save(model.state_dict(), filename)
                            shutil.copyfile(filename,
                                            "{r_dir}/{variation_of_clsgr_executed}_{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                r_dir=args.r_dir,
                                                variation_of_clsgr_executed=variation_of_clsgr_executed,
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
                    else:
                        if task >= 2:
                            ade_previous = 0
                            val_dataset = data_loader(
                                args, val_datasets[task - 1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(
                                args, model, val_dataset, epoch)
                            for i in range(task - 1):
                                val_dataset_ = data_loader(
                                    args, val_datasets[i], args.batch_size)
                                ade_, _ = utils.validate_cl(
                                    args, model, val_dataset_, epoch)
                                ade_previous += ade_
                        else:
                            val_dataset = data_loader(
                                args, val_datasets[task - 1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(
                                args, model, val_dataset, epoch)
                            ade_previous = 0
                        loss_val_dict_main = {'loss_val': loss_val_current}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch,
                                            loss_val_dict_main, task=task)
                        ade_val = ade_current + ade_previous
                        is_best = ade_val < best_ade
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            previous_model = copy.deepcopy(model)
                            file_dir = os.path.dirname(__file__) + "/chekpoint"
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{r_dir}/{variation_of_clsgr_executed}_{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        r_dir=args.r_dir,
                                                        variation_of_clsgr_executed=variation_of_clsgr_executed,
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            torch.save(model.state_dict(), filename)
                            shutil.copyfile(filename,
                                            "{r_dir}/{variation_of_clsgr_executed}_{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                r_dir=args.r_dir,
                                                variation_of_clsgr_executed=variation_of_clsgr_executed,
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
                if args.val_class == 'replay':
                    if generator is None:
                        val_dataset = data_loader(
                            args, val_datasets[task - 1], args.batch_size)
                        ade_current, loss_val = utils.validate_cl(
                            args, model, val_dataset, epoch)
                        loss_val_dict_main = {'loss_val': loss_val}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch,
                                            loss_val_dict_main, task=task)
                        ade_val = ade_current
                        is_best = ade_val < best_ade
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            previous_model = copy.deepcopy(model)
                            file_dir = os.path.dirname(__file__) + "/chekpoint"
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{r_dir}/{variation_of_clsgr_executed}_{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        r_dir=args.r_dir,
                                                        variation_of_clsgr_executed=variation_of_clsgr_executed,
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            torch.save(model.state_dict(), filename)
                            shutil.copyfile(filename,
                                            "{r_dir}/{variation_of_clsgr_executed}_{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                r_dir=args.r_dir,
                                                variation_of_clsgr_executed=variation_of_clsgr_executed,
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
                    else:
                        if task >= 2:
                            ade_previous = 0
                            val_dataset = data_loader(
                                args, val_datasets[task - 1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(
                                args, model, val_dataset, epoch)
                            ade_previous = utils.validate_cl_replay(
                                args, model, x_rel_val, y_rel_val, seq_start_end_val)
                        else:
                            val_dataset = data_loader(
                                args, val_datasets[task - 1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(
                                args, model, val_dataset, epoch)
                            ade_previous = 0
                        loss_val_dict_main = {'loss_val': loss_val_current}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch,
                                            loss_val_dict_main, task=task)
                        ade_val = ade_current + ade_previous
                        is_best = ade_val < best_ade
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            previous_model = copy.deepcopy(model)
                            file_dir = os.path.dirname(__file__) + "/chekpoint"
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{r_dir}/{variation_of_clsgr_executed}_{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        r_dir=args.r_dir,
                                                        variation_of_clsgr_executed=variation_of_clsgr_executed,
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            torch.save(model.state_dict(), filename)
                            shutil.copyfile(filename,
                                            "{r_dir}/{variation_of_clsgr_executed}_{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                r_dir=args.r_dir,
                                                variation_of_clsgr_executed=variation_of_clsgr_executed,
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
            else:
                print(
                    f"        args.val is False, therefore, compute validation loss only using the validation split of the dataset for task ({task-1}: {val_datasets[task-1].dataset_name})")

                val_dataset = data_loader(
                    args, val_datasets[task - 1], args.batch_size)

                _, _, loss_val = utils.validate_cl(
                    args, model, val_dataset, epoch)

                loss_val_dict_main = {'loss_val': loss_val}

                for val_loss_cb in val_loss_cbs:
                    if val_loss_cb is not None:
                        val_loss_cb(progress, epoch,
                                    loss_val_dict_main, task=task)
                
                if generator is not None:
                    print(f"        Computing the losses (reconstruction and variational) for the generative model (type: {type(generator)}) only using the validation split of the dataset for task ({task-1}: {val_datasets[task-1].dataset_name})")
                    
                    val_loss_total = 0
                    val_recon_total = 0
                    val_variat_total = 0

                    for i, batch in enumerate(val_dataset):
                        batch = [tensor.cuda() for tensor in batch]
                        (
                            obs_traj,
                            pred_traj_gt,
                            obs_traj_rel,
                            pred_traj_gt_rel,
                            non_linear_ped,
                            loss_mask,
                            seq_start_end,
                            t_embeddings,
                        ) = batch

                        val_logs = generator.validate_a_batch(obs_traj_rel, seq_start_end)

                        val_loss_total += val_logs['loss_val']
                        val_recon_total += val_logs['reconL_val']
                        val_variat_total += val_logs['variatL_val']

                    num_batches_val = len(val_dataset)
                    avg_val_loss_gen = val_loss_total / num_batches_val
                    avg_val_recon_gen = val_recon_total / num_batches_val
                    avg_val_variat_gen = val_variat_total / num_batches_val
                    print(f"        Generative Model Validation Loss: {avg_val_loss_gen:.4f}")

            if generator is not None and losses_dict_generative['loss_total']:
                # Calcula médias de Treino
                avg_gen_loss_total = np.mean(losses_dict_generative['loss_total'])
                avg_reconL = np.mean(losses_dict_generative['reconL'])
                avg_variatL = np.mean(losses_dict_generative['variatL'])
                avg_reconL_r = np.mean(losses_dict_generative['reconL_r'])
                avg_variatL_r = np.mean(losses_dict_generative['variatL_r'])

                # Cabeçalho (se arquivo não existir)
                if not os.path.exists(vae_losses_file):
                    with open(vae_losses_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        # Cabeçalho inclui as colunas de validação
                        writer.writerow([
                            'Method', 'Task', 'Epoch', 
                            'Loss_Total', 'ReconL', 'VariatL', 
                            'ReconL_R', 'VariatL_R', 
                            'ReconL_Validation', 'VariatL_Validation', 'Loss_Total_Validation'
                        ])

                # Escreve a linha com dados de Treino E Validação
                with open(vae_losses_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        variation_of_clsgr_executed,
                        task,
                        epoch,
                        avg_gen_loss_total, # Treino
                        avg_reconL,         # Treino
                        avg_variatL,        # Treino
                        avg_reconL_r,       # Treino
                        avg_variatL_r,      # Treino
                        avg_val_recon_gen,  # Validação
                        avg_val_variat_gen, # Validação
                        avg_val_loss_gen    # Validação
                    ])
            
            validation_loss_per_epoch.append(loss_val_dict_main['loss_val'])

            # Main model
            # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, epoch, losses_dict_main, task=task)
            if args.val:
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(previous_model, epoch, task=task)
            else:
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, epoch, task=task)
            if model.label == "VAE":
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(model, epoch, task=task)

            # Generative model
            # Fire callbacks on each iteration
            for loss_cb in gen_loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress_gen, epoch,
                            losses_dict_generative, task=task)
            for sample_cb in sample_cbs:
                if sample_cb is not None:
                    sample_cb(generator, epoch, task=task)

        # ----> UPON FINISHING EACH TASK...
        # --- Avaliacao Pos-Treino (Preencher linha da matriz) ---
        model.eval()
        print(f"Atualizando matriz de resultados após tarefa {task}...")
        for i in range(num_tasks):
            path = utils.get_dset_path(test_order[i], "test")
            dset = data_dset(args, path, dataset_name=test_order[i], split_name="test")
            loader = data_loader(args, dset, args.batch_size)
            ade, fde = evaluate.validate(model, loader)
            ade_matrix[task, i] = ade
            fde_matrix[task, i] = fde

            print(f"ADE: {ade:.2f}")
            print(f"FDE: {fde:.2f}")
        model.train()
        ###########################################################

        elapsed_time_for_this_task = progress.format_dict['elapsed']
        elpased_time_for_each_task.append(elapsed_time_for_this_task)

        completed_dataset_names.append(current_dataset_name)
        print(
            f"    Completed task {task}, dataset {current_dataset_name} in {elapsed_time_for_this_task:.2f} seconds")

        # Close progress-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()

        if args.val:
            model = copy.deepcopy(previous_model)

        # Calculate statistics required for metrics
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                metric_cb(model, iters, task=task)

        if args.val is False:
            print(
                f"    Saving model after finishing task {task}, dataset {current_dataset_name}")

            # REPLAY: update source for replay
            previous_model = copy.deepcopy(model)
            print(
                f"    'previous_model' (type: {type(previous_model)}) becomes a deep copy of the current 'model' (type: {type(model)}) for replay purposes")

            file_dir = os.path.dirname(__file__)
            filename = os.path.join(
                file_dir, f"{args.r_dir}/{variation_of_clsgr_executed}_{args.method}_{args.replay}_{task}_model_{args.dataset_order}_{batch_size}_{args.seed}_{args.val}_{args.val_class}_{args.si}_{args.si_c}.path")

            torch.save(model.state_dict(), filename)

        best_ade = 200

        # SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and (model.si_c > 0):
            model.update_omega(W, model.epsilon)

        if replay_model == 'generative':
            print(f"    Generative replay will be performed during the next task")

            print(f"    Updating 'Generative' flag to True")
            Generative = True

            previous_generator = copy.deepcopy(generator).eval(
            ) if generator is not None else previous_model

            print(
                f"    'previous_generator' (type: {type(previous_generator)}) becomes a deep copy of the current 'generator' (type: {type(generator)}) for generative replay purposes")

        # EXEMPLARS: update exemplar sets
        if replay_model == "exemplars":
            # based on this dataset, construct new exemplar-set for this class
            # model.construct_exemplar_set(dataset=train_dataset, n=model.memory_budget)
            # model.compute_means = True
            # previous_dataset = model.exemplar_sets
            # previous_datasets.append(previous_dataset)

            print(f"    Updating 'Exact' flag to True")
            Exact = True

    nova_coluna_nome = 'Loss_Validation'
    log_file_path_main = main_losses_file
    if os.path.exists(log_file_path_main):
        linhas_atualizadas = []

        with open(log_file_path_main, 'r', newline='') as f:
            reader = csv.reader(f)
            linhas = list(reader)

            if linhas:
            # Modifica o cabeçalho (primeira linha)
                cabecalho = linhas[0]
                if nova_coluna_nome not in cabecalho:
                    cabecalho.append(nova_coluna_nome)
                    
                    # Modifica as linhas de dados
                    for i in range(1, len(linhas)):
                        linhas[i].append(validation_loss_per_epoch[i-1])
                    
                    linhas_atualizadas = linhas

        # 2. Sobrescreve o arquivo com os novos dados
        if linhas_atualizadas:
            with open(log_file_path_main, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(linhas_atualizadas)
    
    print("Matriz de erros R (ADE) dentro da funcao train_cl(), primeira linha eh o random_baseline:")
    with np.printoptions(precision=2):
        print(ade_matrix)

    if args.metrics:
        if args.time:
            return ade_matrix, fde_matrix, elpased_time_for_each_task
        else:
            return ade_matrix, fde_matrix, None
    elif args.time:
        return None, None, elpased_time_for_each_task
    else:
        return None, None, None
