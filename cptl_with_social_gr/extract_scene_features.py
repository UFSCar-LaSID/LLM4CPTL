#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################
# Imports and packages
###################################
import argparse
import numpy as np
import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm

from scipy.spatial import ConvexHull
from scipy.stats import entropy
from sklearn.decomposition import PCA
from args import get_all_args

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent.resolve()
if str(parent_dir) not in sys.path:
    print(f"Adding {parent_dir} to sys.path")
    sys.path.insert(0, str(parent_dir))

try:
    from helper import utils
    from data.loader import data_dset
    print("Successfully imported CPTL-SGR modules.")
except ModuleNotFoundError as e:
    print(f"Error importing CPTL-SGR modules: {e}")
    exit(1)

###################################
# Classes
###################################


class SceneFeatureExtractor:
    def __init__(self, obs_traj, target_features):
        """
        obs_traj: Numpy array de shape (Num_Peds, 2, Obs_Len)
        target_features: Lista de strings com as features desejadas.
        """
        self.target_features = target_features

        # Transpor para (Num_Peds, Obs_Len, 2)
        self.traj = np.transpose(obs_traj, (0, 2, 1))
        self.num_peds = self.traj.shape[0]
        self.seq_len = self.traj.shape[1]

        # Calcular velocidades: (N, T-1, 2)
        self.vel = self.traj[:, 1:, :] - self.traj[:, :-1, :]
        self.speeds = np.linalg.norm(self.vel, axis=2)

    def get_topology_and_structure(self):
        """Infere a estrutura física (Corredor vs Praça)."""
        if self.num_peds < 2:
            return {
                "structure_type": "Sparse/Empty",
                "flow_linearity": 0.0,
                "heading_entropy": 0.0,
                "walkable_area_density": 0.0
            }

        all_vecs = self.vel.reshape(-1, 2)
        
        if len(all_vecs) < 2 or np.all(np.abs(all_vecs) < 1e-5):
            linearity = 0.0
            heading_entropy = 0.0
            structure = "Static/Empty"
        else:
            # Entropia de Direção
            angles = np.arctan2(all_vecs[:, 1], all_vecs[:, 0])
            hist, _ = np.histogram(
                angles, bins=18, range=(-np.pi, np.pi), density=True)
            heading_entropy = entropy(hist + 1e-10)

            # Linearidade (PCA)
            try:
                pca = PCA(n_components=2)
                pca.fit(all_vecs)
                explained_var = pca.explained_variance_ratio_
                # Se a variância total for 0, explained_variance_ratio_ pode ter NaNs
                if np.isnan(explained_var).any():
                    linearity = 0.0
                else:
                    linearity = explained_var[0]
            except:
                linearity = 0.0

            # Classificação Heurística
            if linearity > 0.85:
                structure = "Structured Corridor/Lane"
            elif heading_entropy > 2.0:
                structure = "Open Plaza/Intersection"
            else:
                structure = "Semi-structured Walkway"

        # 2. Área Andável (Convex Hull)
        all_points = self.traj.reshape(-1, 2)
        try:
            # Precisa de pelo menos 3 pontos não colineares para ter área
            if len(all_points) > 2:
                hull = ConvexHull(all_points)
                area = hull.volume
                density = self.num_peds / (area + 1e-6)
            else:
                area = 0.0
                density = 0.0
        except:
            area = 0.0
            density = 0.0

        return {
            "structure_type": structure,
            "flow_linearity": float(linearity),
            "heading_entropy": float(heading_entropy),
            "walkable_area_est": float(area),
            "spatial_density": float(density)
        }

    def get_social_interactions(self):
        """Identifica grupos e risco de colisão."""
        if self.num_peds < 2:
            return {"group_ratio": 0.0, "collision_risk_ratio": 0.0, "average_group_size": 1.0}

        pos_t0 = self.traj[:, 0, :]
        vel_t0 = self.vel[:, 0, :]

        dist_mat = np.linalg.norm(
            pos_t0[:, None, :] - pos_t0[None, :, :], axis=2)
        norms = np.linalg.norm(vel_t0, axis=1, keepdims=True) + 1e-6
        vel_norm = vel_t0 / norms
        sim_mat = np.dot(vel_norm, vel_norm.T)

        # Grupos
        is_group_pair = (dist_mat < 1.5) & (sim_mat > 0.9)
        np.fill_diagonal(is_group_pair, False)
        has_group = np.any(is_group_pair, axis=1)
        group_ratio = has_group.sum() / self.num_peds

        if has_group.sum() > 0:
            avg_size = (is_group_pair[has_group].sum(axis=1) + 1).mean()
        else:
            avg_size = 1.0

        # Colisões (Projeção)
        future_pos = pos_t0 + (vel_t0 * 8)
        fut_dist_mat = np.linalg.norm(
            future_pos[:, None, :] - future_pos[None, :, :], axis=2)
        is_risk_pair = (dist_mat > 2.0) & (fut_dist_mat < 0.5)
        np.fill_diagonal(is_risk_pair, False)
        risk_ratio = np.any(is_risk_pair, axis=1).sum() / self.num_peds

        return {
            "group_ratio": float(group_ratio),
            "average_group_size": float(avg_size),
            "collision_risk_ratio": float(risk_ratio)
        }

    def get_dynamics(self):
        """Velocidade e comportamento de parada."""
        avg_speed = np.mean(self.speeds)
        stop_frames = np.sum(self.speeds < 0.05)
        total_frames = self.speeds.size
        stop_ratio = stop_frames / total_frames

        return {
            "average_speed": float(avg_speed),
            "stop_ratio": float(stop_ratio)
        }

    def extract_all(self):
        """
        Extrai apenas as features solicitadas em target_features.
        Sempre inclui 'num_agents' como base.
        """
        stats = {"num_agents": int(self.num_peds)}

        # Mapeamento para evitar recálculo se 'group' e 'collision' forem pedidos juntos
        computed_social = False

        if "topology" in self.target_features:
            stats.update(self.get_topology_and_structure())

        if "dynamics" in self.target_features:
            stats.update(self.get_dynamics())

        if "group" in self.target_features or "collision" in self.target_features:
            # Calcula o bloco social se qualquer sub-feature for necessária
            social_stats = self.get_social_interactions()
            stats.update(social_stats)

        return stats
        
###################################
# Functions
###################################


def classify_topology(stats):
    """
    Classifica a topologia ASSUMINDO APENAS PEDESTRES.
    Distingue:
    1. Corridor/Sidewalk (Fluxo Linear)
    2. Open Plaza (Fluxo Livre Contínuo)
    3. Intersection/Crosswalk (Fluxo Estruturado com Espera)
    """
    # Extrair métricas chave
    entropy = stats["average_entropy"]      # Caos direcional
    # Se todos vão pro mesmo lado (Global PCA)
    linearity = stats["average_linearity"]
    stop_ratio = stats.get("stop_ratios", 0)  # Se não tiver, assume 0

    # Se stop_ratio não estiver direto no dict final, pegar da média calculada
    # (No script anterior chamamos de 'avg_stop_ratio' se tiver calculado,
    # ou calculamos agora caso tenha vindo da lista crua)
    if "average_stop_ratio" in stats:
        stop_ratio = stats["average_stop_ratio"]
    else:
        # Fallback seguro
        stop_ratio = 0.0

    # --- LÓGICA DE DECISÃO ---
    if entropy > 1.5:
        return "Open plaza or shared space with unstructured flow"
    
    if linearity > 0.8 or entropy < 1.0:
        return "Linear pedestrian walkway or corridor"

    elif stop_ratio > 0.15: 
        return "Signalized intersection or crosswalk, structured flow with waiting"

    else:
        return "Undefined"


def preprocess_dataset(
        args: argparse.Namespace,
        dataset_name: str,
        dataset_split_name: str,
        coord_system="meter",
        use_scene_context=False,
        target_features=["topology", "group", "collision", "dynamics"]):

    # Configuração de saída
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Note que o nome do arquivo agora reflete que é por CENA
    output_file = os.path.join(
        args.output_dir, f"{dataset_name}_{dataset_split_name}_scene_features.json")

    # Carregar Dataset
    dset_path = utils.get_dset_path(dataset_name, dataset_split_name)
    dset = data_dset(
        args,
        dset_path,
        dataset_name=dataset_name,
        split_name=dataset_split_name
    )

    # --- MUDANÇA PRINCIPAL: Dicionário de Acumuladores ---
    # Estrutura: { "nome_da_cena": { "speeds": [], "group_ratios": [], ... } }
    scenes_accumulators = {}

    print(f"--- Scanning {dataset_name} ({dataset_split_name}) ---")

    for i in tqdm(range(len(dset)), desc="Processing windows"):
        batch_data = dset[i]

        # 1. Recuperar o ID da Cena
        # Assumindo que você aplicou a correção no TrajectoryDataset,
        # o scene_id deve ser o último elemento da lista retornada.
        # Ele vem como um array numpy de strings (repetido para cada pedestre)
        # 1. Recuperar o ID da Cena
        try:
            # O Dataset retorna 9 itens (índices 0 a 8).
            # O item -1 (último) deve ser a lista de nomes das cenas.
            raw_scene_list = batch_data[-1]

            # raw_scene_list é uma tupla ou lista, ex: ('seq_hotel', 'seq_hotel', ...)
            # Como estamos processando janelas, todos os pedestres na janela pertencem à mesma cena.
            # Pegamos o primeiro elemento.

            if isinstance(raw_scene_list, (list, tuple, np.ndarray)):
                scene_name_item = raw_scene_list[0]
            elif isinstance(raw_scene_list, torch.Tensor):
                # Se caiu aqui, você pegou o tensor de embeddings por engano!
                # Provavelmente o seq_collate não foi atualizado ou o índice está errado.
                raise ValueError(
                    f"Esperava string, recebeu Tensor: {raw_scene_list}")
            else:
                scene_name_item = raw_scene_list

            # Limpeza final da string
            scene_name = str(scene_name_item)

            # Remove caracteres de bytes se necessário (b'eth' -> eth)
            if scene_name.startswith("b'") or scene_name.startswith('b"'):
                scene_name = scene_name[2:-1]

        except Exception as e:
            print(f"\nERRO DE DEBUG:")
            print(f"Tipo do batch_data[-1]: {type(batch_data[-1])}")
            print(f"Conteúdo do batch_data[-1]: {batch_data[-1]}")
            print(f"Tamanho do batch_data: {len(batch_data)}")
            raise e

        # 2. Inicializar o balde dessa cena se for a primeira vez que a vemos
        if scene_name not in scenes_accumulators:
            scenes_accumulators[scene_name] = {
                "speeds": [], "group_ratios": [], "risk_ratios": [],
                "linearities": [], "entropies": [], "densities": [],
                "stop_ratios": [], "num_agents_per_frame": []
            }

        # 3. Extração de Features (Igual ao anterior)
        obs_traj_numpy = batch_data[0].numpy()
        extractor = SceneFeatureExtractor(obs_traj_numpy, target_features)
        stats = extractor.extract_all()

        # 4. Acumular NO BALDE CORRETO
        acc = scenes_accumulators[scene_name]
        acc["speeds"].append(stats.get("average_speed", 0))
        acc["group_ratios"].append(stats.get("group_ratio", 0))
        acc["risk_ratios"].append(stats.get("collision_risk_ratio", 0))
        acc["linearities"].append(stats.get("flow_linearity", 0))
        acc["entropies"].append(stats.get("heading_entropy", 0))
        acc["densities"].append(stats.get("spatial_density", 0))
        acc["stop_ratios"].append(stats.get("stop_ratio", 0))
        acc["num_agents_per_frame"].append(stats.get("num_agents", 0))

    # --- PÓS-PROCESSAMENTO: Iterar sobre cada cena encontrada ---
    final_records = []

    print(
        f"\nGenerating descriptions for {len(scenes_accumulators)} unique scenes found...")

    for scene_name, stats_list in scenes_accumulators.items():
        # Calcular médias específicas da cena
        scene_final_stats = {
            "average_speed_global": float(np.nanmean(stats_list["speeds"])),
            "standard_deviation_speed_global": float(np.nanstd(stats_list["speeds"])),

            "average_stop_ratio": float(np.nanmean(stats_list["stop_ratios"])),

            "average_group_ratio": float(np.nanmean(stats_list["group_ratios"])),
            "max_group_ratio": float(np.nanmax(stats_list["group_ratios"])) if stats_list["group_ratios"] else 0.0,

            "average_risk_ratio": float(np.nanmean(stats_list["risk_ratios"])),

            "average_linearity": float(np.nanmean(stats_list["linearities"])),
            "average_entropy": float(np.nanmean(stats_list["entropies"])),

            "average_density": float(np.nanmean(stats_list["densities"])),
            "average_agents_per_scene": float(np.nanmean(stats_list["num_agents_per_frame"]))
        }

        # Classificar Topologia desta cena específica
        topology_desc = classify_topology(scene_final_stats)
        scene_final_stats["topology_description"] = topology_desc

        # Criar registro
        record = {
            "dataset": dataset_name,
            "split": dataset_split_name,
            "scene_name": scene_name,  # Identificador crucial
            "raw_aggregated_stats": scene_final_stats
        }
        final_records.append(record)
        print(f" > {scene_name}: {topology_desc}")

    # Salvar JSON (Lista de Cenas)
    with open(output_file, encoding="utf-8", mode='w') as f:
        json.dump(final_records, f, indent=2)

    print(f"Extracted scene features saved to file: {output_file}")

def main(args: argparse.Namespace):
    for d in args.dataset:
        for s in args.split:
            print(f"Processing dataset: {d}, split: {s}")
            preprocess_dataset(args, d, s)            

###################################
# Main program
###################################
if __name__ == "__main__":
    args = get_all_args()
    main(args)
    print("End!")