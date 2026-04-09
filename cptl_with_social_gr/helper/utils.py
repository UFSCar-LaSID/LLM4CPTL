#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################
# Imports and packages
###################################
import math
import pickle
import os
import re
import logging
import torch
import shutil

from data.trajectories import SceneBatch

###################################
# Functions
###################################
# Data-handling functions
def get_data_loader():
    pass

# Get dataset path
def get_dset_path(dset_name: str, dset_type: str):
    _dir = os.path.dirname(__file__)
    return os.path.join(_dir, '../datasets', dset_name, dset_type)

# relative to absolute


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


## -------------------------------------------------------------------------------------------------------------------##

################################
## Model-inspection functions ##
################################

def count_parameters(model, verbose=True):
    '''Count number of parameters, print to screen.'''
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims == 0 else n_params*dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print("--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
        print("    of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                   round(learnable_params / 1000000, 1)))
        print("              - fixed: {} (~{} million)".format(fixed_params,
              round(fixed_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params


def print_model_info(model, title: str = "MODEL"):
    '''Print information on [model] onto the screen.'''
    print("\n" + 40*"-" + title + 40*"-")
    print(model)
    print(90*"-")
    _ = count_parameters(model)
    print(90*"-")


# Calculate l2_loss
def l2_loss(pred_traj, pred_traj_gt, mode: str = "average"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, size = pred_traj.size()
    # equation below , the first part do noing, can be delete

    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / (seq_len*batch*size)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)

# Calculate ADE metric


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode: str = "sum"):
    '''
    Input:
    :param pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory. [12, person_num, 2]
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth predictions.
    :param consider_ped: Tensor of shape (batch)
    :param mode: Can be one of sum, raw
    :return:
    - loss: gives the eculidian displacement error
    '''

    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)

    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss

# Calculate FDE metric


def final_displacement_error(pred_pos, pred_pos_gt, condiser_ped=None, mode="sum"):
    '''
    Input:
    :param pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    :param pred_pos_gt: Tensor of shape (batch, 2). Ground truth last pos.
    :param condiser_ped: Tensor of shape (batch).
    :param mode:
    :return: loss --> gives the eculidian displacement error
    '''

    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if condiser_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * condiser_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)


##################
## Batch learning##
##################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def int_tuple(s):
    return tuple(int(i) for i in s.split(","))

# for batch learning training


def save_checkpoint(args, state, is_best, filename="checkpoint.pth.tar", model_name=None):
    torch.save(state, filename)
    if is_best:
        logging.info("-------------- lower ade ----------------")
        shutil.copyfile(filename, "{}/IL_{}_{}_{}_best.pth.tar".format(
            args.log_dir, model_name, args.log_dir, args.aug))


# for evaluation model
def save_dict(obj, name):
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_dict_txt(obj, name):
    with open(name+'.txt', 'w') as f:
        for k, v in obj.items():
            f.write(str(k)+':'+str(v)+'\n')
    f.close()


def validate_cl(args, model, val_loader, epoch, writer=None):
    ade = AverageMeter("ADE", ":.6f")
    fde = AverageMeter("FDE", ":.6f")
    losses_val = AverageMeter("Loss", ":.6f")

    #total_traj = 0
    #ade_outer, fde_outer = [], []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = SceneBatch(batch, device="cuda")

            #ade, fde = [], []
            #loss_val = torch.zeros(1).to(pred_traj_gt)
            #total_traj += pred_traj_gt.size(1)

            pred_traj_fake_rel = model(batch.obs_traj_rel, batch.seq_start_end)
            pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len:]

            pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, batch.obs_traj[-1])

            ade_, fde_ = cal_ade_fde(batch.pred_traj, pred_traj_fake)
            ade_per_ped = ade_ / args.pred_len

            num_trajs = batch.pred_traj.size(1)

            ade.update(torch.mean(ade_per_ped).item(), num_trajs)
            fde.update(torch.mean(fde_).item(), num_trajs)

            loss_val = l2_loss(pred_traj_fake_rel_predpart, batch.pred_traj_rel, mode="average")

            losses_val.update(loss_val.item(), num_trajs)

    return ade.avg, fde.avg, losses_val.avg

def validate_cl_replay(args, model, x_rel_val, y_rel_val, seq_start_end_val):
    ade = AverageMeter("ADE", ":.6f")
    fde = AverageMeter("FDE", ":.6f")
    total_traj = 0
    # mode = model.training
    model.eval()
    with torch.no_grad():
        total_traj = y_rel_val.size(1)
        pred_len = y_rel_val.size(0)
        pred_traj_fake_rel = model(x_rel_val, seq_start_end_val)
        pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len:]
        pred_traj_fake = relative_to_abs(
            pred_traj_fake_rel_predpart, x_rel_val[-1])
        ade_, _ = cal_ade_fde(y_rel_val, pred_traj_fake)
        ade = sum(ade_).item() / (total_traj * pred_len)
        return ade


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


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade_ = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde_ = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade_, fde_


def validate(args, model, val_loader, epoch, writer=None):
    ade = AverageMeter("ADE", ":.6f")
    fde = AverageMeter("FDE", ":.6f")
    progress = ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")
    total_traj = 0
    ade_outer, fde_outer = [], []
    mode = model.training
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = SceneBatch(batch, device="cuda")
            
            ade, fde = [], []
            total_traj += batch.pred_traj.size(1)
            pred_len = batch.pred_traj.size(0)
            loss_mask = batch.loss_mask[:, args.obs_len:]
            pred_traj_fake_rel = model(batch.obs_traj_rel, batch.seq_start_end)

            pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len:]
            pred_traj_fake = relative_to_abs(
                pred_traj_fake_rel_predpart, batch.obs_traj[-1])
            ade_, fde_ = cal_ade_fde(batch.pred_traj, pred_traj_fake)
            ade_sum = sum(ade_)
            fde_sum = sum(fde_)
            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer).item() / (total_traj * pred_len)
        fde = sum(fde_outer).item() / (total_traj)
        # if i % args.print_every == 0:
        #     progress.display(i)

        logging.info(
            " * ADE  {ade:.3f} FDE  {fde:.3f}".format(ade=ade, fde=fde)
        )
        writer.add_scalar("val_ade", ade, epoch)
    model.train(mode=mode)
    return ade

def variation_of_clsgr_being_executed(args):
    variation_of_clsgr_executed = "IL"
    if args.method == "continual_learning":
        variation_of_clsgr_executed = "CL_"
        if args.replay == "none":
            variation_of_clsgr_executed += "NR"
        elif args.replay == "exemplars":
            variation_of_clsgr_executed += "ER"
        elif args.replay == "generative" and args.replay_model == "condition":
            variation_of_clsgr_executed += "CGR"
        else:
            variation_of_clsgr_executed += "SGR"
            
            if args.use_kl_annealing:
                variation_of_clsgr_executed += "eKLAN"
            if args.adapt_architecture_to_include_sequence_embedding:
                variation_of_clsgr_executed += "eModalLLM"
            if args.use_embeddings_buffer:
                variation_of_clsgr_executed += "Buffer"
            if args.use_uncertainty_filter:
                variation_of_clsgr_executed += "eUncFilter"
            if args.use_prior_adaptation:
                variation_of_clsgr_executed += "eBoo"
            if args.use_dc_vampprior:
                variation_of_clsgr_executed += "eVamp"

    return variation_of_clsgr_executed


def clean_text(text: str) -> str:
    text = re.sub(r'\(.*?\)', '', text).strip()
    text = re.sub(r'\s+([,.:;])', r'\1', text)
    text = re.sub(r'\$.*?([-+]?\d*\.?\d+).*?\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub('\s+', ' ', text)
    
    # --- A Feature de Limpeza de Sentença ---
    # Explicação do Regex:
    # ^          -> Começo da string
    # .* -> Pega tudo (greedy), garantindo que pegamos até a ÚLTIMA pontuação
    # (?<!\d)    -> Lookbehind: O caractere antes da pontuação NÃO pode ser um dígito
    # [.!?]      -> Aceita ponto, exclamação ou interrogação
    # (?:\s|$)   -> Seguido por um espaço ou pelo fim da linha

    match = re.search(r'^(.*(?<!\d)[.!?](?:\s|$))', text, re.DOTALL)

    if match:
        text = match.group(1).strip()
    
    return text


def get_cyclical_beta(current_step, total_steps, n_cycles=4, ratio=0.5, shape="linear", start=0.0, stop=1.0):
    """
    Calcula o valor de Beta (KL weight) usando agendamento cíclico sob demanda.
    
    Args:
        current_step (int): Época ou iteração atual.
        total_steps (int): Total de épocas ou iterações planejadas.
        n_cycles (int): Quantos ciclos 'sobe-mantém' acontecerão durante o treino.
        ratio (float): Proporção do ciclo usada para subir de start a stop (R). 
                       O restante (1-R) mantém em stop.
        shape (str): Formato da curva de subida ('linear', 'sigmoid', 'quadratic', 'cosine').
        start (float): Valor mínimo do peso KL no início do ciclo.
        stop (float): Valor máximo do peso KL no platô do ciclo.
    
    Returns:
        float: Valor de beta para o passo atual.
    """
    # Evita divisão por zero ou ratio nulo, retornando o valor máximo por segurança
    if total_steps == 0 or ratio <= 1e-9:
        return stop

    # Tamanho de um ciclo em passos
    period = total_steps / n_cycles

    # Posição relativa dentro do ciclo atual (tau vai de 0.0 a 1.0)
    step_in_cycle = current_step % period
    tau = step_in_cycle / period

    # Fase de Platô: mantém no valor máximo (stop)
    if tau > ratio:
        return stop

    # Fase de Subida: calcula o progresso de 0.0 a 1.0
    progress = tau / ratio

    # Define o formato da curva (base_beta vai de 0.0 a 1.0)
    if shape == 'linear':
        base_beta = progress
    elif shape == 'sigmoid':
        # Sigmoide ajustada para ir de ~0 a ~1 suavemente no intervalo
        base_beta = 1 / (1 + math.exp(-12 * (progress - 0.5)))
    elif shape == 'quadratic':
        # Sobe lentamente no início e acelera
        base_beta = progress ** 2
    elif shape == 'cosine':
        # Metade de uma onda cosseno (suave no início e no fim)
        base_beta = 0.5 - 0.5 * math.cos(progress * math.pi)
    else:
        base_beta = progress

    # Escala o valor de [0, 1] para o intervalo [start, stop]
    beta = start + (stop - start) * base_beta

    return beta
