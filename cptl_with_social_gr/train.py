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
from data.trajectories import SceneBatch

###################################
# Functions
###################################
def write_main_model_losses_file_csv(
        dict_of_average_losses: dict,
        filename: str,
        current_task: int,
        current_epoch: int,
        variation_of_clsgr_executed: str):

    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    'Method',
                    'Task',
                    'Epoch',
                    'Loss_Total',
                    'Loss_Current',
                    'Loss_Replay',
                    'Pred_Traj',
                    'Pred_Traj_R'
                ]
            )

    # Anexar dados
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            variation_of_clsgr_executed,
            current_task,
            current_epoch,
            dict_of_average_losses.get('loss_total', 0),
            dict_of_average_losses.get('loss_current', 0),
            dict_of_average_losses.get('loss_replay', 0),
            dict_of_average_losses.get('pred_traj', 0),
            dict_of_average_losses.get('pred_traj_r', 0)
        ])

def write_generative_model_losses_file_csv(
        dict_of_average_losses: dict,
        filename: str,
        current_task: int,
        current_epoch: int,
        variation_of_clsgr_executed: str,
        total_number_of_tasks: int):

    columns = [
        'Method',
        'Task',
        'Epoch',
        'Loss_Total',
        'ReconL',
        'VariatL',
        'ReconL_R',
        'VariatL_R',
        'ReconL_Validation',
        'VariatL_Validation',
        'Loss_Total_Validation',
    ]
    
    columns.extend([f'ReconL_Validation_Task{t}' for t in range(1, total_number_of_tasks+1)])
    
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            
    row = [
        variation_of_clsgr_executed,
        current_task,
        current_epoch,
        dict_of_average_losses['loss_total'],
        dict_of_average_losses['reconL'],
        dict_of_average_losses['variatL'],
        dict_of_average_losses['reconL_r'],
        dict_of_average_losses['variatL_r'],
        dict_of_average_losses['loss_val'],
        dict_of_average_losses['reconL_val'],
        dict_of_average_losses['variatL_val'],
    ]
    
    for t in range(1, total_number_of_tasks + 1):
        row.append(dict_of_average_losses.get(f'reconL_val_task{t}', 0))

    # Escreve a linha com dados de Treino E Validação
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def train(
        args,
        model,
        train_loader,
        optimizer,
        epoch,
        writer):
    
    losses = utils.AverageMeter("Loss", ":.6f")

    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )

    model.train()

    for batch_idx, batch in enumerate(train_loader):
        batch = SceneBatch(batch, device='cuda')

        optimizer.zero_grad()

        loss = torch.zeros(1).to(batch.pred_traj)
        l2_loss_rel = []
        loss_mask = batch.loss_mask[:, args.obs_len:]

        model_input = torch.cat((batch.obs_traj_rel, batch.pred_traj_rel), dim=0)

        pred_traj_fake_rel = model(model_input, batch.seq_start_end)

        l2_loss_rel = utils.l2_loss(
            pred_traj_fake_rel,
            model_input[-args.pred_len:],
            loss_mask,
            mode="average",
        )

        loss = l2_loss_rel

        losses.update(loss.item(), batch.obs_traj.shape[1])

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

    variation_of_clsgr_executed = utils.variation_of_clsgr_being_executed(args)

    # Losses files for main and generative models:
    vae_losses_file = os.path.join(
        args.r_dir, f"{variation_of_clsgr_executed}_losses_vae_{args.batch_size}_{args.iters}.csv")

    main_model_losses_filename = os.path.join(
        args.r_dir, f"{variation_of_clsgr_executed}_losses_main_{args.batch_size}_{args.iters}.csv")

    if os.path.exists(vae_losses_file):
        os.remove(vae_losses_file)

    if os.path.exists(main_model_losses_filename):
        os.remove(main_model_losses_filename)

    # Set model to training mode:
    model.train()

    # Initialize matrices to store ADE and FDE results:
    total_number_of_tasks = len(train_datasets)
    ade_matrix = np.zeros((total_number_of_tasks + 1, total_number_of_tasks))
    fde_matrix = np.zeros((total_number_of_tasks + 1, total_number_of_tasks))
    elpased_time_for_each_task = []
    completed_dataset_names = []
    validation_loss_per_epoch = []

    # Evaluation of randomly initialized models (FWT):
    print("Evaluation of randomly initialized models (FWT)...")
    model.eval()
    for i in range(total_number_of_tasks):
        emb_path = None
        if args.adapt_architecture_to_include_sequence_embedding:
            # Usa {} como default para permitir o encadeamento seguro
            emb_path = args.llm_sequences_embeddings_mapping.get(test_order[i], {}).get("test", None)
        
        path = utils.get_dset_path(test_order[i], "test")
        dset = data_dset(
            args,
            path,
            sequences_embeddings_path=emb_path,
            dataset_name=test_order[i],
            split_name="test"
        )
        loader = data_loader(args, dset)

        ade, fde = evaluate.validate(model, loader)
        ade_matrix[0, i] = ade
        fde_matrix[0, i] = fde

    model.train()

    # Model's device:
    device = model._device()

    # Initiate replay flags (no replay for 1st task):
    Exact = Generative = Current = False
    previous_model = None

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c > 0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer(
                    '{}_SI_prev_task'.format(n), p.data.clone())

    # Tasks loop:
    for task, train_dataset in enumerate(train_datasets, start=1):
        current_dataset_name = train_dataset.dataset_name

        print(f"\nIterating over task {task}, dataset {current_dataset_name}")

        training_dataset = data_loader(args, train_dataset)
        total_number_of_batches = len(training_dataset)

        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (model.si_c > 0):
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        # Initialize the number of iterations left on current data-loader(s):
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)

        progress = tqdm.tqdm(range(1, iters_to_use + 1))
        if generator is not None:
            progress_gen = tqdm.tqdm(
                range(1, gen_iters + 1)
            )

        if fake_generator is not None:
            progress_gen = tqdm.tqdm(
                range(1, gen_iters * total_number_of_batches + 1)
            )

        if args.val:
            x_rel_val = None
            y_rel_val = None
            seq_start_end_val = None

        replay_data_loader = data_loader(
            args, train_dataset, args.replay_batch_size
        )

        replay_data_loader = iter(replay_data_loader)

        ## -----REPLAYED BATCH------##
        if not Exact and not Generative and not Current:
            print(f"    No replay will be performed during this task")

        # Epochs loop:
        for epoch in range(1, iters_to_use+1):
            print(f"    Epoch {epoch}/{iters_to_use}")
            if args.use_kl_annealing and generator is not None:
                # Calcula o novo beta para esta época
                new_beta = utils.get_cyclical_beta(
                    epoch,
                    iters_to_use,
                    n_cycles=4,
                    ratio=0.5,
                    shape="linear"
                )

                # Atualiza o parâmetro dentro do modelo gerador
                # Nota: Se generator for DataParallel, use generator.module.lamda_vl
                generator.lamda_vl = new_beta * 0.01

                if epoch % 50 == 0:  # Log ocasional
                    print(
                        f"    [KL Annealing] Epoch {epoch}: Beta (lamda_vl) updated to {generator.lamda_vl:.4f}")

            dictionary_training_losses_main_model = {
                'loss_total': [],
                'loss_current': [],
                'loss_replay': [],
                'pred_traj': [],
                'pred_traj_r': []
            }

            dictionary_training_losses_generative_model = {
                'loss_total': [],
                'reconL': [],
                'variatL': [],
                'reconL_r': [],
                'variatL_r': []
            }

            # batches loop:
            for batch_index, batch in enumerate(training_dataset, start=0):
                print(
                    f"        Batch {batch_index+1}/{total_number_of_batches}")

                batch = SceneBatch(batch, device='cuda')

                print(
                    f"            Number of trajectories in current task-specific batch: {batch.obs_traj.shape[1]}")
                #print(
                    #f"            Number of embeddings in current task-specific batch: {t_embeddings.shape[0]}")

                # Collect/extract data from current batch:
                x_rel = batch.obs_traj_rel
                y_rel = batch.pred_traj_rel
                seq_start_end = batch.seq_start_end
                loss_mask = batch.loss_mask
                sequence_embeddings = batch.sequence_embeddings
                # Initialize replayed data variables:
                x_rel_ = y_rel_ = seq_start_end_ = sequence_embeddings_ = None

                # Exact replay:
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

                        x_rel_ = memory_seq['obs_traj_rel'].cuda()
                        y_rel_ = memory_seq['pred_traj_rel'].cuda()
                        seq_start_end_ = memory_seq['seq_start_end'].cuda()
                        sequence_embeddings_ = memory_seq['sequence_embedding'].cuda()

                # ----Generative / Current Replay----#
                if Generative:
                    print(
                        f"            Generative replay will be performed for this batch")

                    # Get replayed data (i.e., [x_]) -- either current data or use previous generator

                    try:
                        replay_out = next(replay_data_loader)
                    except StopIteration:
                        replay_data_loader = iter(
                            data_loader(
                                args, train_dataset, args.replay_batch_size
                            )
                        )

                        replay_out = next(replay_data_loader)

                    if args.replay_model == 'lstm':
                        print(f"            LSTM replay model selected")

                        print(
                            f"            'previous_generator' (type: {type(previous_generator)}) is sampling replay data from 'replay_out' (type: {type(replay_out)}), built on dataset {train_dataset.dataset_name}")

                        replay_traj = previous_generator.sample(
                            replay_out['obs_traj_rel'].to(device),
                            replay_out['obs_traj'].to(device),
                            replay_out['seq_start_end'].to(device),
                            replay_out['sequence_embedding'].to(device)
                        )

                        x_ = replay_traj[0]
                        x_rel_ = replay_traj[1]
                        seq_start_end_ = replay_traj[2]
                        sequence_embeddings_ = replay_traj[3]

                        print(
                            f"            Number of trajectories replayed: {x_rel_.shape[1]}")
                        #print(
                            #f"            Number of embeddings replayed: {sequence_embeddings_.shape[0]}")

                    if args.replay_model == 'vrnn':
                        replay_traj = previous_generator.sample(
                            replay_out['obs_traj_rel'].to(device),
                            replay_out['obs_traj'].to(device),
                            replay_out['seq_start_end'].to(device)
                        )

                        x_rel_ = replay_traj.cuda()
                        seq_start_end_ = seq_start_end
                        sequence_embeddings_ = sequence_embeddings

                    if "CL_SGR" in variation_of_clsgr_executed:
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
                if batch_index <= iters * total_number_of_batches:
                    print(
                        f"            Training main model (type: {type(model)}) for this batch")

                    # Train the main model with this batch
                    loss_dict_main = model.train_a_batch(
                        x_rel,
                        y_rel,
                        seq_start_end,
                        x_rel_=x_rel_,
                        y_rel_=y_rel_,
                        seq_start_end_=seq_start_end_,
                        loss_mask=loss_mask,
                        rnt=1./task
                    )

                    dictionary_training_losses_main_model['loss_total'].append(
                        loss_dict_main['loss_total'])

                    dictionary_training_losses_main_model['loss_current'].append(
                        loss_dict_main['loss_current'])

                    dictionary_training_losses_main_model['loss_replay'].append(
                        loss_dict_main['loss_replay'])

                    dictionary_training_losses_main_model['pred_traj'].append(
                        loss_dict_main['pred_traj'])

                    dictionary_training_losses_main_model['pred_traj_r'].append(
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
                if generator is not None and batch_index <= gen_iters * total_number_of_batches:
                    print(
                        f"            Training generative model (type: {type(generator)}) for this batch")

                    # Train the generator with this batch
                    loss_dict_generative = generator.train_a_batch(
                        x_rel,
                        y_rel,
                        seq_start_end,
                        sequence_embedding=sequence_embeddings,
                        x_=x_rel_,
                        y_=y_rel_,
                        seq_start_end_=seq_start_end_,
                        sequence_embedding_=sequence_embeddings_,
                        rnt=1./task
                    )

                    dictionary_training_losses_generative_model['loss_total'].append(
                        loss_dict_generative['loss_total'])

                    dictionary_training_losses_generative_model['reconL'].append(
                        loss_dict_generative['reconL'])

                    dictionary_training_losses_generative_model['variatL'].append(
                        loss_dict_generative['variatL'])

                    dictionary_training_losses_generative_model['reconL_r'].append(
                        loss_dict_generative['reconL_r'])

                    dictionary_training_losses_generative_model['variatL_r'].append(
                        loss_dict_generative['variatL_r'])

            print(
                f"        Computing the average of the losses from main model (type: {type(model)}) for all batches processed during this epoch")

            average_of_losses_main_model = {
                'loss_total': np.mean(dictionary_training_losses_main_model['loss_total']),
                'loss_current': np.mean(dictionary_training_losses_main_model['loss_current']),
                'loss_replay': np.mean(dictionary_training_losses_main_model['loss_replay']),
                'pred_traj': np.mean(dictionary_training_losses_main_model['pred_traj']),
                'pred_traj_r': np.mean(dictionary_training_losses_main_model['pred_traj_r'])
            }

            write_main_model_losses_file_csv(
                average_of_losses_main_model,
                main_model_losses_filename,
                task,
                epoch,
                variation_of_clsgr_executed
            )

            if args.val:
                print(
                    f"        Validating main model at epoch {epoch} for task {task}")

                file_dir = os.path.dirname(__file__) + "/chekpoint"
                if os.path.exists(file_dir) is False:
                    os.mkdir(file_dir)
                filename = os.path.join(
                    file_dir, f"{args.r_dir}/{variation_of_clsgr_executed}_{args.method}_{args.replay}_{task}_model_{args.dataset}_{batch_size}_{args.seed}_{epoch}_{args.val}_{args.val_class}_{args.si}_{args.si_c}.path")

                if args.val_class == 'current':
                    val_dataset = data_loader(
                        args, val_datasets[task-1])
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

                        torch.save(model.state_dict(), filename)

                        shutil.copyfile(
                            filename, f"{args.r_dir}/{variation_of_clsgr_executed}_{args.method}_{args.replay}_{task}_model_{args.dataset}_{batch_size}_{args.seed}_{args.val}_{args.val_class}_{args.si}_{args.si_c}.path")

                if args.val_class == 'all':
                    if generator is None:
                        val_dataset = data_loader(
                            args, val_datasets[task - 1])
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

                            torch.save(model.state_dict(), filename)

                            shutil.copyfile(filename,
                                            f"{args.r_dir}/{variation_of_clsgr_executed}_{args.method}_{args.replay}_{task}_model_{args.dataset}_{batch_size}_{args.seed}_{args.val}_{args.val_class}_{args.si}_{args.si_c}.path")
                    else:
                        if task >= 2:
                            ade_previous = 0
                            val_dataset = data_loader(
                                args, val_datasets[task - 1])
                            ade_current, loss_val_current = utils.validate_cl(
                                args, model, val_dataset, epoch)
                            for i in range(task - 1):
                                val_dataset_ = data_loader(
                                    args, val_datasets[i])
                                ade_, _ = utils.validate_cl(
                                    args, model, val_dataset_, epoch)
                                ade_previous += ade_
                        else:
                            val_dataset = data_loader(
                                args, val_datasets[task - 1])
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

                            torch.save(model.state_dict(), filename)

                            shutil.copyfile(filename,
                                            f"{args.r_dir}/{variation_of_clsgr_executed}_{args.method}_{args.replay}_{task}_model_{args.dataset}_{batch_size}_{args.seed}_{args.val}_{args.val_class}_{args.si}_{args.si_c}.path")

                if args.val_class == 'replay':
                    if generator is None:
                        val_dataset = data_loader(
                            args, val_datasets[task - 1])
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

                            torch.save(model.state_dict(), filename)

                            shutil.copyfile(filename,
                                            f"{args.r_dir}/{variation_of_clsgr_executed}_{args.method}_{args.replay}_{task}_model_{args.dataset}_{batch_size}_{args.seed}_{args.val}_{args.val_class}_{args.si}_{args.si_c}.path")

                    else:
                        if task >= 2:
                            ade_previous = 0
                            val_dataset = data_loader(
                                args, val_datasets[task - 1])
                            ade_current, loss_val_current = utils.validate_cl(
                                args, model, val_dataset, epoch)
                            ade_previous = utils.validate_cl_replay(
                                args, model, x_rel_val, y_rel_val, seq_start_end_val)
                        else:
                            val_dataset = data_loader(
                                args, val_datasets[task - 1])
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

                            torch.save(model.state_dict(), filename)

                            shutil.copyfile(filename,
                                            f"{args.r_dir}/{variation_of_clsgr_executed}_{args.method}_{args.replay}_{task}_model_{args.dataset}_{batch_size}_{args.seed}_{args.val}_{args.val_class}_{args.si}_{args.si_c}.path")

            else:
                print(
                    f"        args.val is False, therefore, compute validation loss only using the validation split of the dataset for task ({task-1}: {val_datasets[task-1].dataset_name})")

                validation_data_loader_current_task = data_loader(
                    args, val_datasets[task - 1])

                _, _, validation_loss_main_model = utils.validate_cl(
                    args, model, validation_data_loader_current_task, epoch)

                loss_val_dict_main = {'loss_val': validation_loss_main_model}

                for val_loss_cb in val_loss_cbs:
                    if val_loss_cb is not None:
                        val_loss_cb(progress, epoch,
                                    loss_val_dict_main, task=task)

                if generator is not None:
                    print(
                        f"        Computing the losses (reconstruction and variational) for the generative model (type: {type(generator)}) only using the validation split of the dataset for task ({task-1}: {val_datasets[task-1].dataset_name})")

                    dictionary_of_average_validation_losses_generative_model = evaluate.validate_generative_model(
                        generator, validation_data_loader_current_task)

                    dictionary_training_losses_generative_model.update(
                        dictionary_of_average_validation_losses_generative_model)
                    
                    if len(completed_dataset_names) > 0:
                        print(f"        There is at least one previous task, therefore, the current generative model will try to reconstruct the trajectories from the validation split of those previous tasks.")
                        
                        for t, _ in enumerate(completed_dataset_names):
                            val_dataset_t = data_loader(
                                args, val_datasets[t])
                            
                            losses_val_task_t = evaluate.validate_generative_model(
                                generator, val_dataset_t)
                            
                            if f'reconL_val_task{t+1}' not in dictionary_training_losses_generative_model:
                                dictionary_training_losses_generative_model[f'reconL_val_task{t+1}'] = []
                                
                            dictionary_training_losses_generative_model[f'reconL_val_task{t+1}'].append(
                                losses_val_task_t['reconL_val']
                            )

            if generator is not None and dictionary_training_losses_generative_model['loss_total']:
                average_of_losses_generative_model = {
                    'loss_total': np.mean(dictionary_training_losses_generative_model['loss_total']),
                    'reconL': np.mean(dictionary_training_losses_generative_model['reconL']),
                    'variatL': np.mean(dictionary_training_losses_generative_model['variatL']),
                    'reconL_r': np.mean(dictionary_training_losses_generative_model['reconL_r']),
                    'variatL_r': np.mean(dictionary_training_losses_generative_model['variatL_r']),
                    'loss_val': np.mean(dictionary_training_losses_generative_model['loss_val']),
                    'reconL_val': np.mean(dictionary_training_losses_generative_model['reconL_val']),
                    'variatL_val': np.mean(dictionary_training_losses_generative_model['variatL_val'])
                }
                
                for t, _ in enumerate(completed_dataset_names):
                    key = f'reconL_val_task{t+1}'
                    average_of_losses_generative_model[key] = np.mean(
                        dictionary_training_losses_generative_model[key]
                    )

                write_generative_model_losses_file_csv(
                    average_of_losses_generative_model,
                    vae_losses_file,
                    task,
                    epoch,
                    variation_of_clsgr_executed,
                    total_number_of_tasks
                )

            # validation_loss_per_epoch.append(loss_val_dict_main['loss_val'])

            # Main model
            # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, epoch,
                            dictionary_training_losses_main_model, task=task)
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
                            dictionary_training_losses_generative_model, task=task)
            for sample_cb in sample_cbs:
                if sample_cb is not None:
                    sample_cb(generator, epoch, task=task)

        # ----> UPON FINISHING EACH TASK...
        # --- Avaliacao Pos-Treino (Preencher linha da matriz) ---
        model.eval()
        print(
            f"Updating the error matrices (for ADE and FDE) R after finished task {task}...")
        for i in range(total_number_of_tasks):
            emb_path = None
            if args.adapt_architecture_to_include_sequence_embedding:
                # Usa {} como default para permitir o encadeamento seguro
                emb_path = args.llm_sequences_embeddings_mapping.get(test_order[i], {}).get("test", None)
            
            path = utils.get_dset_path(test_order[i], "test")
            dset = data_dset(
                args,
                path,
                sequences_embeddings_path=emb_path,
                dataset_name=test_order[i],
                split_name="test"
            )
            loader = data_loader(args, dset)
            ade, fde = evaluate.validate(model, loader)
            ade_matrix[task, i] = ade
            fde_matrix[task, i] = fde

        model.train()

        elapsed_time_for_this_task = progress.format_dict['elapsed']
        elpased_time_for_each_task.append(elapsed_time_for_this_task)

        completed_dataset_names.append(current_dataset_name)
        print(
            f"    Completed task {task}, dataset {current_dataset_name} in {elapsed_time_for_this_task:.2f} seconds")

        elapsed_time_for_this_task = progress.format_dict['elapsed']
        elpased_time_for_each_task.append(elapsed_time_for_this_task)

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
                file_dir, f"{args.r_dir}/{variation_of_clsgr_executed}_{args.method}_{args.replay}_{task}_model_{args.dataset}_{batch_size}_{args.seed}_{args.val}_{args.val_class}_{args.si}_{args.si_c}.path")

            torch.save(model.state_dict(), filename)
            
            if generator is not None:
                print(
                    f"    Saving generative model after finishing task {task}, dataset {current_dataset_name}")

                filename_generator = filename.replace("_model_", "_generativeModelCheckpoint_")

                torch.save(generator.state_dict(), filename_generator)

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
            print(f"    Updating 'Exact' flag to True")
            Exact = True

    print("R error matrix (ADE) inside funcion train_cl(), the first line is the random baseline:")
    with np.printoptions(precision=2):
        print(ade_matrix)

    if args.metrics:
        np.savetxt(
            os.path.join(
                args.r_dir,
                f"{variation_of_clsgr_executed}_ADE_matrix_{args.batch_size}_{args.iters}.txt"
            ),
            ade_matrix
        )
        
        np.savetxt(
            os.path.join(
                args.r_dir,
                f"{variation_of_clsgr_executed}_FDE_matrix_{args.batch_size}_{args.iters}.txt"
            ),
            fde_matrix
        )
        
        if args.time:
            return ade_matrix, fde_matrix, elpased_time_for_each_task
        else:
            return ade_matrix, fde_matrix, None
    elif args.time:
        return None, None, elpased_time_for_each_task
    else:
        return None, None, None
