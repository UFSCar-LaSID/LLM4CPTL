#!/usr/bin/env python3

###################################
# Imports and packages
###################################

import torch
from helper import utils
from helper import visual_visdom
from data.trajectories import SceneBatch
from generative_model.vae_models_scratch import CVAE

import numpy as np

###################################
# Functions
###################################
def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_

def initiate_metrics_dict(n_tasks):
    metrics_dict = {}
    metrics_dict["average_ade"] = []
    metrics_dict["average_fde"] = []
    metrics_dict["x_iteration"] = []
    metrics_dict["x_task"] = []
    metrics_dict["ade per task"] = {}
    metrics_dict["fde per task"] = {}
    for i in range(n_tasks):
        metrics_dict["ade per task"]["task {}".format(i+1)] = []
        metrics_dict["fde per task"]["task {}".format(i+1)] = []
    return metrics_dict


def intial_accuracy(model, datasets, metric_dict, test_size=None, verbose=False, no_task_mask=False):
    n_tasks = len(datasets)
    ades = []
    fdes = []

    for i in range(n_tasks):
        ade, fde = validate(model, datasets[i])
        ades.append(ade)
        fdes.append(fde)

    metric_dict["initial ade per task"] = ades
    metric_dict["initial fde per task"] = fdes
    return metric_dict


def metric_statistics(model, datasets, current_task, iteration,
                      metrics_dict=None, test_size=None, verbose=False):
    n_tasks = len(datasets)
    ades_all_classes = []
    fdes_all_classes = []
    ades_all_classes_ = []
    fdes_all_classes_ = []

    for i in range(n_tasks):
        ade_, fde_ = validate(model, datasets[i])
        ades_all_classes_.append(ade_)
        fdes_all_classes_.append(fde_)

    average_ades = sum([ades_all_classes_[task_id]
                       for task_id in range(current_task)]) / current_task
    average_fdes = sum([fdes_all_classes_[task_id]
                       for task_id in range(current_task)]) / current_task

    for task_id in range(n_tasks):
        metrics_dict["ade per task"]["task {}".format(
            task_id+1)].append(ades_all_classes_[task_id])
        metrics_dict["fde per task"]["task {}".format(
            task_id+1)].append(fdes_all_classes_[task_id])

    metrics_dict["average_ade"].append(average_ades)
    metrics_dict["average_fde"].append(average_fdes)
    metrics_dict["x_iteration"].append(iteration)
    metrics_dict["x_task"].append(current_task)

    return metrics_dict

def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade_ = utils.displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde_ = utils.final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade_, fde_


def evaluate(loader, predictor):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = SceneBatch(batch, device='cuda')

            ade, fde = [], []
            total_traj += batch.pred_traj.size(1)
            pred_len = batch.pred_traj.size(0)

            for _ in range(1):
                pred_traj_fake_rel = predictor(batch.obs_traj_rel, batch.seq_start_end)
                pred_traj_fake = utils.relative_to_abs(pred_traj_fake_rel, batch.obs_traj[-1])

                ade_, fde_ = cal_ade_fde(batch.pred_traj, pred_traj_fake)

                ade.append(ade_)
                fde.append(fde_)

            ade_sum = evaluate_helper(ade, batch.seq_start_end)
            fde_sum = evaluate_helper(fde, batch.seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer).item() / (total_traj * pred_len)
        fde = sum(fde_outer).item() / (total_traj)
        
        return ade, fde


def validate(model, dataset_loader):
    '''
    Evaluate precision (ADE and FDE) of a predictor ([model]) on [dataset].
    '''

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Loop over batches in [dataset]
    ade, fde = evaluate(dataset_loader, model)

    # Set model back to its initial mode and return it
    model.train(mode=mode)

    return ade, fde


def validate_generative_model(generator: CVAE, loader):
    average_losses_generative_model = {
        'loss_val': [],
        'reconL_val': [],
        'variatL_val': []
    }

    for i, batch in enumerate(loader):
        batch = SceneBatch(batch, device='cuda')

        losses_dictionary_for_this_batch = generator.evaluate_a_batch(
            batch.obs_traj_rel,
            batch.seq_start_end,
            sequence_embedding=batch.sequence_embeddings
        )

        average_losses_generative_model['loss_val'].append(
            losses_dictionary_for_this_batch['loss_val'])
        average_losses_generative_model['reconL_val'].append(
            losses_dictionary_for_this_batch['reconL_val'])
        average_losses_generative_model['variatL_val'].append(
            losses_dictionary_for_this_batch['variatL_val'])

    for key in average_losses_generative_model.keys():
        average_losses_generative_model[key] = np.mean(average_losses_generative_model[key])

    return average_losses_generative_model

def precision(model, datasets, current_task, iteration, classes_per_task=None, scenario="domain",
              test_size=None, visdom=None, verbose=False, summary_graph=True):
    n_tasks = len(datasets)
    ades = []
    fdes = []
    for i in range(n_tasks):
        if i+1 <= current_task:
            ade, fde = validate(model, datasets[i])
            ades.append(ade)
            fdes.append(fde)
        else:
            ades.append(0)
            fdes.append(0)

    average_ades = sum([ades[task_id]
                       for task_id in range(current_task)]) / current_task
    average_fdes = sum([fdes[task_id]
                       for task_id in range(current_task)]) / current_task

    # Send results to visdom server
    names = ['task {}'.format(i+1) for i in range(n_tasks)]
    if visdom is not None:
        visual_visdom.visualize_scalars(
            ades, names=names, title="ADE on validation set (CL_{})".format(
                visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylable="ADE precision"
        )
        visual_visdom.visualize_scalars(
            fdes, names=names, title="FDE on validation set (CL_{})".format(
                visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylable="FDE precision"
        )
        if n_tasks > 1 and summary_graph:
            visual_visdom.visualize_scalars(
                [average_ades], names=["ADE"], title="Average ADE on validation set (CL_{})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylable="ADE precision"
            )
            visual_visdom.visualize_scalars(
                [average_fdes], names=["FDE"], title="Average FDE on validation set (CL_{})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylable="FDE precision"
            )

def calculate_cl_metrics(matrix, G=None, error_metric="ADE"):
    """
    Calcula métricas de Continual Learning para matrizes de ERRO (ADE/FDE).
    
    Args:
        matrix: numpy array (T+1, T). Linha 0 é avaliação inicial.
        G (G): numpy array (T,). Performance de um modelo treinado 
                           isoladamente em cada tarefa (para TBWT). 
                           Se None, usa a diagonal da matriz (R_ii) como proxy.
    """
    # Remove a linha 0 (avaliação inicial) para facilitar contas de BWT/Forgetting
    # R[i, j] agora será: Após treinar tarefa i+1, performance na tarefa j+1
    # Shape passa a ser (T, T)
    R = matrix[1:, :] 

    print(f"Matriz de erros R ({error_metric}) dentro da funcao calculate_cl_metrics():")
    with np.printoptions(precision=2):
        print(R)
    
    # Random/Initial baseline (Linha 0 da matriz original)
    Random_Baseline = matrix[0, :]
    print(f"Vetor de erros Random_Baseline ({error_metric}) dentro da funcao calculate_cl_metrics():")
    with np.printoptions(precision=2):
        print(Random_Baseline)
    
    T = R.shape[0] # Número de tarefas
    
    # ---------------------------------------------------------
    # 1. BWT (Backward Transfer)
    # ---------------------------------------------------------
    # Fórmula: Média da diferença entre erro final e erro logo após o treino.
    # Para ERRO: Se (Final - Inicial) > 0, o erro aumentou (Esquecimento).
    # Se (Final - Inicial) < 0, o erro diminuiu (Transferência Positiva).
    
    bwt_sum = 0
    for i in range(T - 1): # Para tarefas 0 até T-2 (penúltima)
        # Erro na tarefa i após treinar T (última linha)
        err_final = R[T-1, i] 
        # Erro na tarefa i logo após treinar i (diagonal)
        err_original = R[i, i]
        bwt_sum += (err_final - err_original)
        
    BWT = bwt_sum / (T - 1) if T > 1 else 0.0

    # ---------------------------------------------------------
    # 2. FWT (Forward Transfer)
    # ---------------------------------------------------------
    # Influência do treino de tarefas passadas em uma tarefa futura k antes de treiná-la.
    # Comparamos o erro "Zero-shot" (antes de treinar k) com o Baseline Aleatório.
    # Para ERRO: Se (Random - ZeroShot) > 0, o erro caiu (FWT Positivo).
    
    fwt_sum = 0
    for i in range(1, T): # Para tarefas 1 até T-1 (exclui a primeira)
        # Erro na tarefa i usando modelo treinado até i-1
        err_zero_shot = R[i-1, i]
        # Erro da tarefa i com pesos aleatórios
        err_random = Random_Baseline[i]
        
        # Quanto o erro diminuiu graças ao conhecimento prévio?
        fwt_sum += (err_random - err_zero_shot)
        
    FWT = fwt_sum / (T - 1) if T > 1 else 0.0

    # ---------------------------------------------------------
    # 4. TBWT (Transfer-based Backward Transfer)
    # ---------------------------------------------------------
    # Requer o Gold Standard (G). Se não fornecido, BWT padrão é retornado.
    # Fórmula: 1/(T-1) * Sum( R_{T,i} - G_{i} )
    # Note: Para erro, queremos saber se R_{T,i} está longe de G_{i}.
    # Como G é o "melhor possível", R geralmente é maior.
    
    if G is not None:
        tbwt_sum = 0
        for i in range(T - 1):
            err_final = R[T-1, i]
            err_gold = G[i]
            tbwt_sum += (err_final - err_gold)
        TBWT = tbwt_sum / (T - 1) if T > 1 else 0.0
    else:
        TBWT = None

    # ---------------------------------------------------------
    # 5. CBWT (Cumulative BWT)
    # ---------------------------------------------------------
    # Calcula o BWT sofrido pela tarefa t ao longo do tempo (t+1 até T).
    # Retorna uma lista/vetor com o CBWT para cada tarefa t.
    
    cbwt_per_task = {}
    
    for t in range(T - 1): # Para cada tarefa t (menos a última que não tem futuro)
        sum_diff = 0
        steps = 0
        
        # Somatório de t+1 até T
        for i in range(t + 1, T):
            # R_{i,t}: Erro na tarefa t após treinar tarefa i
            err_current = R[i, t]
            # R_{t,t}: Erro na tarefa t logo após treinar t
            err_orig = R[t, t]
            
            sum_diff += (err_current - err_orig)
            steps += 1
            
        if steps > 0:
            cbwt_per_task[t] = sum_diff / steps
            print(f"CBWT ({error_metric}) para tarefa {t}: {cbwt_per_task[t]}")

    return BWT, FWT, TBWT, cbwt_per_task 