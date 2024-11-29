#!/usr/bin/env python3

import hydra
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import seaborn as sns
from tqdm import tqdm
import torch
from collections import defaultdict

def collect_task_actions(dataset, env_idx):
    """Collect actions for a specific task/environment index"""
    actions = []
    
    # Get all episodes for this task
    episodes = dataset._episodes[env_idx]
    for episode in episodes:
        # Get raw actions before preprocessing
        episode_actions = episode['action']
        actions.extend(episode_actions)
    
    return np.array(actions)

def compute_task_statistics(actions):
    """Compute basic statistics for a task's actions"""
    return {
        'mean': np.mean(actions, axis=0),
        'std': np.std(actions, axis=0),
        'min': np.min(actions, axis=0),
        'max': np.max(actions, axis=0)
    }

def compute_wasserstein_matrix(task_actions, action_dim):
    """
    Compute Wasserstein distance matrix between all pairs of tasks
    for each action dimension
    """
    num_tasks = len(task_actions)
    distances = np.zeros((action_dim, num_tasks, num_tasks))
    
    # Compute distances for each action dimension
    for dim in range(action_dim):
        for i in range(num_tasks):
            for j in range(i+1, num_tasks):
                # Extract the specific dimension of actions for both tasks
                actions_i = task_actions[i][:, dim]
                actions_j = task_actions[j][:, dim]
                
                try:
                    # Compute Wasserstein distance
                    dist = wasserstein_distance(actions_i, actions_j)
                    distances[dim, i, j] = dist
                    distances[dim, j, i] = dist
                except Exception as e:
                    print(f"Warning: Could not compute distance for tasks {i}, {j}, dim {dim}: {e}")
                    distances[dim, i, j] = np.nan
                    distances[dim, j, i] = np.nan
                    
    return distances

def plot_wasserstein_analysis(distances, save_dir, dim_names=None):
    """Create visualizations of the Wasserstein distance analysis"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    num_dims = distances.shape[0]
    num_tasks = distances.shape[1]
    
    # Plot heatmap for each action dimension
    for dim in range(num_dims):
        plt.figure(figsize=(12, 10))
        sns.heatmap(distances[dim], 
                   cmap='viridis',
                   xticklabels=range(num_tasks),
                   yticklabels=range(num_tasks))
        
        dim_name = dim_names[dim] if dim_names is not None else f"Dimension {dim}"
        plt.title(f'Wasserstein Distances - {dim_name}')
        plt.xlabel('Task Index')
        plt.ylabel('Task Index')
        
        plt.savefig(save_dir / f'wasserstein_heatmap_dim_{dim}.png')
        plt.close()
    
    # Plot average distance across all dimensions
    plt.figure(figsize=(12, 10))
    avg_distances = np.nanmean(distances, axis=0)
    sns.heatmap(avg_distances,
                cmap='viridis',
                xticklabels=range(num_tasks),
                yticklabels=range(num_tasks))
    plt.title('Average Wasserstein Distances Across All Dimensions')
    plt.xlabel('Task Index')
    plt.ylabel('Task Index')
    plt.savefig(save_dir / 'wasserstein_heatmap_average.png')
    plt.close()


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    # Load dataset
    dataset = hydra.utils.call(cfg.expert_dataset)
    
    print("Collecting actions for each task...")
    task_actions = []
    task_stats = []
    
    # Collect actions and compute statistics for each task
    for env_idx in range(dataset.envs_till_idx):
        actions = collect_task_actions(dataset, env_idx)
        task_actions.append(actions)
        task_stats.append(compute_task_statistics(actions))
    
    action_dim = task_actions[0].shape[1]
    
    print("\nComputing Wasserstein distances...")
    distances = compute_wasserstein_matrix(task_actions, action_dim)
    
    # Create visualizations
    save_dir = Path.cwd() / "wasserstein_analysis"
    
    # If available, use actual dimension names
    dim_names = None
    if hasattr(dataset, 'action_dim_names'):
        dim_names = dataset.action_dim_names
    
    plot_wasserstein_analysis(distances, save_dir, dim_names)
    
if __name__ == "__main__":
    main()