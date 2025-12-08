#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################
## Imports and packages
###################################

import argparse
from helper import utils
from data.loader import data_loader, data_dset

parser = argparse.ArgumentParser('./main.py', description='Run experiment.')
parser.add_argument('--obs_len', default=8, type=int, help="the observed frame of trajectory")
parser.add_argument('--pred_len', default=12, type=int, help="the predicted frame of trajectory")
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--delim', default='\t')
parser.add_argument('--t_embedding_path', default="/home/matheus/LLM4CPTL/cptl_with_social_gr/data_preprocessed/llm_motion_cues_qwen3-0.6b_qwen3-embedding-0.6b_ETH_1024_0.8_512_none_10.json", type=str)
parser.add_argument('--zc_id_map_path', default="/home/matheus/LLM4CPTL/cptl_with_social_gr/data_preprocessed/clusters_id_map_ETH.json", type=str)

if __name__ == "__main__":
    print("Start")
    args = parser.parse_args()
    train_path = utils.get_dset_path("ETH", "train")
    train_dset = data_dset(args, train_path, t_embedding_path=args.t_embedding_path, zc_id_map_path=args.zc_id_map_path, dataset_name="ETH")
    
    print(train_dset.embeddings_loaded)
    print(train_dset.t_embeddings[0])
    
    print("End!")