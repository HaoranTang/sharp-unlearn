import copy
import os
from collections import OrderedDict
from rich import print as rich_print
import matplotlib.pyplot as plt
import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import unlearn
import utils
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from trainer import validate
from imagenet import prepare_data
import pdb

if __name__ == "__main__":
    args = arg_parser.parse_args()
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=12,
            pin_memory=True,
            shuffle=shuffle,
        )
    
    loaders = prepare_data(
        dataset="imagenet", batch_size=1, data_path="/your/path/to/data"
    )
    train_loader_full = loaders['train']
    val_loader = loaders['val']
    train_loader = DataLoader(
            train_loader_full.dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
    loaded_results = np.load('imagenet_index.npz', allow_pickle=True)
    loaded_memorization = loaded_results['tr_mem']
    loaded_filenames_raw = loaded_results['tr_filenames']
    # Normalize the loaded filenames (everything before the ".")
    loaded_filenames = [f.decode('utf-8').split('.')[0] for f in loaded_filenames_raw]
    # Create memorization-filename pairs and sort them
    mem_filename_pairs = list(zip(loaded_memorization, loaded_filenames))
    
    # Get high memorization filenames (sorted by memorization, highest first)
    mem_filename_pairs.sort(key=lambda x: x[0], reverse=True)
    high_mem_filenames = [f for _, f in mem_filename_pairs[:args.num_indexes_to_replace]]
    high_mem_scores = [m for m, _ in mem_filename_pairs[:args.num_indexes_to_replace]]
    # Get low memorization filenames (sorted by memorization, lowest first)
    low_mem_filenames = [f for _, f in mem_filename_pairs[-args.num_indexes_to_replace:]]
    low_mem_scores = [m for m, _ in mem_filename_pairs[-args.num_indexes_to_replace:]]
    # Get medium memorization filenames (closest to 0.5)
    mem_filename_pairs.sort(key=lambda x: abs(x[0] - 0.5))
    med_mem_filenames = [f for _, f in mem_filename_pairs[:args.num_indexes_to_replace]]
    med_mem_scores = [m for m, _ in mem_filename_pairs[:args.num_indexes_to_replace]]
    
    print(f'check: first 100 h_mem: {high_mem_scores[:100]}, total mean: {sum(high_mem_scores)/len(high_mem_scores)}')
    print(f'check: first 100 l_mem: {low_mem_scores[:100]}, total mean: {sum(low_mem_scores)/len(low_mem_scores)}')
    print(f'check: first 100 m_mem: {med_mem_scores[:100]}, total mean: {sum(med_mem_scores)/len(med_mem_scores)}')
    # Create a filename to index mapping for the train_set
    high_mem = []
    med_mem = []
    low_mem = []
    print("Building filename to index mapping...")
    for i, example in tqdm(enumerate(train_loader.dataset), desc="iterating raw dataset", total=len(train_loader.dataset)):
        filename = example['filename']
        # Handle the case like n01981276_163_n01981276.JPEG - extract n01981276_163
        norm_filename = '_'.join(filename.split('_')[:2])
        if norm_filename in high_mem_filenames:
            high_mem.append(i)
        elif norm_filename in med_mem_filenames:
            med_mem.append(i)
        elif norm_filename in low_mem_filenames:
            low_mem.append(i)
    assert len(high_mem) == args.num_indexes_to_replace
    assert len(med_mem) == args.num_indexes_to_replace
    assert len(low_mem) == args.num_indexes_to_replace
    indices_cache_file = f'imagenet_filenames_indices_{args.num_indexes_to_replace}_{args.seed}.npz'
    # Save to cache for future runs
    np.savez(indices_cache_file, 
                high_mem_indices=np.array(high_mem), 
                med_mem_indices=np.array(med_mem),
                low_mem_indices=np.array(low_mem))
    
    