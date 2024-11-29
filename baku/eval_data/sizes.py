#!/usr/bin/env python3

import hydra
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict

def get_dataset_stats(dataset):
    """Analyze the structure and statistics of the dataset"""
    stats = defaultdict(dict)
    
    # Basic dataset properties
    stats['general']['total_samples'] = len(dataset)
    
    # Sample the first item to understand structure
    first_item = dataset[0]
    stats['structure']['keys'] = list(first_item.keys())
    
    # Analyze each key in the dataset
    for key in first_item.keys():
        if isinstance(first_item[key], (np.ndarray, torch.Tensor)):
            stats['shapes'][key] = first_item[key].shape
            if hasattr(first_item[key], 'dtype'):
                stats['dtypes'][key] = str(first_item[key].dtype)
    
    # Get episode information if available
    if hasattr(dataset, '_max_episode_len'):
        stats['episodes']['max_length'] = dataset._max_episode_len
    if hasattr(dataset, '_max_state_dim'):
        stats['state']['max_dim'] = dataset._max_state_dim
    if hasattr(dataset, '_max_action_dim'):
        stats['action']['max_dim'] = dataset._max_action_dim
    if hasattr(dataset, 'envs_till_idx'):
        stats['environments']['count'] = dataset.envs_till_idx
        
    return stats

@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    # Load dataset
    dataset = hydra.utils.call(cfg.expert_dataset)
    
    # Get and print statistics
    stats = get_dataset_stats(dataset)
    
    print("\n=== Dataset Analysis ===")
    print("\nGeneral Information:")
    print(f"Total samples: {stats['general']['total_samples']}")
    
    print("\nDataset Structure:")
    print("Available keys:", stats['structure']['keys'])
    
    print("\nShape Information:")
    for key, shape in stats['shapes'].items():
        print(f"{key}: {shape}")
        
    print("\nData Types:")
    for key, dtype in stats['dtypes'].items():
        print(f"{key}: {dtype}")
    
    print("\nEpisode Information:")
    if 'max_length' in stats['episodes']:
        print(f"Maximum episode length: {stats['episodes']['max_length']}")
    if 'max_dim' in stats['state']:
        print(f"Maximum state dimension: {stats['state']['max_dim']}")
    if 'max_dim' in stats['action']:
        print(f"Maximum action dimension: {stats['action']['max_dim']}")
    if 'count' in stats['environments']:
        print(f"Number of environments: {stats['environments']['count']}")
    
    # Print sample item if dataset isn't empty
    if stats['general']['total_samples'] > 0:
        print("\nSample Data Item:")
        sample = dataset[0]
        for key, value in sample.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                print(f"{key}: {type(value)} with shape {value.shape}")
            else:
                print(f"{key}: {type(value)}")

if __name__ == "__main__":
    main()