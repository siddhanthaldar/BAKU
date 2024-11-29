#!/usr/bin/env python3

import hydra
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict

def get_dataset_stats(dataset):
    """Analyze the structure and statistics of the dataset"""
    stats = defaultdict(dict)
    
    # Get dataset properties from BCDataset attributes
    stats['general']['total_samples'] = dataset._num_samples
    stats['general']['num_environments'] = dataset.envs_till_idx
    stats['general']['max_episode_length'] = dataset._max_episode_len
    stats['general']['min_episode_len'] = dataset._min_episode_len
    stats['general']['max_state_dim'] = dataset._max_state_dim
    
    # Get observation type and prompt settings
    stats['config'] = {
        'observation_type': dataset._obs_type,
        'prompt_type': dataset._prompt,
        'history_enabled': dataset._history,
        'history_length': dataset._history_len,
        'temporal_aggregation': dataset._temporal_agg
    }
    
    # Sample one item to understand structure
    sample_item = next(iter(dataset))
    stats['structure'] = {
        'keys': list(sample_item.keys()),
        'shapes': {k: tuple(v.shape) if isinstance(v, (np.ndarray, torch.Tensor)) else type(v) 
                  for k, v in sample_item.items()},
        'dtypes': {k: str(v.dtype) if isinstance(v, (np.ndarray, torch.Tensor)) else type(v) 
                  for k, v in sample_item.items()}
    }
    
    # Get stats about loaded episodes
    stats['episodes'] = {
        'num_episodes_per_env': {env_idx: len(episodes) 
                               for env_idx, episodes in dataset._episodes.items()}
    }
    
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
    print(f"Number of environments: {stats['general']['num_environments']}")
    print(f"Maximum episode length: {stats['general']['max_episode_length']}")
    print(f"Minimum episode length: {stats['general']['min_episode_len']}")
    print(f"Maximum state dimension: {stats['general']['max_state_dim']}")
    
    print("\nConfiguration:")
    for key, value in stats['config'].items():
        print(f"{key}: {value}")
    
    print("\nData Structure:")
    print("Available keys:", stats['structure']['keys'])
    
    print("\nShape Information:")
    for key, shape in stats['structure']['shapes'].items():
        print(f"{key}: {shape}")
        
    print("\nData Types:")
    for key, dtype in stats['structure']['dtypes'].items():
        print(f"{key}: {dtype}")
    
    print("\nEpisodes per Environment:")
    for env_idx, num_episodes in stats['episodes']['num_episodes_per_env'].items():
        print(f"Environment {env_idx}: {num_episodes} episodes")

    # Print preprocessing stats
    print("\nPreprocessing Statistics:")
    for key, value in dataset.stats.items():
        print(f"{key}:")
        print(f"  min: {value['min']}")
        print(f"  max: {value['max']}")

if __name__ == "__main__":
    main()