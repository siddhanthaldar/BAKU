#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy.signal import find_peaks
from scipy.cluster import hierarchy
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

def analyze_trajectory_segments(similarity_matrices, n_segments=5, window_size=10):
    """
    Analyze trajectory similarities in temporal segments using similarity matrices directly
    """
    # n_windows = 194, n_tasks = 10, n_windows suggests the 
    # number of windows of window_size with stride 1
    # that moves 
    # through the entire trajectory of the minimum length task
    # thus the timesteps would
    n_windows, n_tasks, _ = similarity_matrices.shape
    # segment_size = 38
    segment_size = n_windows // n_segments
    segment_analyses = []
    
    for i in range(n_segments):
        # start_idx = 0
        start_idx = i * segment_size
        # end_idx = 38
        end_idx = start_idx + segment_size if i < n_segments-1 else n_windows
        import ipdb; ipdb.set_trace()
        
        # Get segment similarities
        segment_similarities = similarity_matrices[start_idx:end_idx]
        
        # Average over time steps in this segment
        segment_avg = np.mean(segment_similarities, axis=0)
        
        # Find most and least similar pairs
        # Add a mask for diagonal elements to ignore self-similarities
        mask = np.eye(n_tasks)
        masked_similarities = segment_avg + mask * float('inf')
        most_similar = np.unravel_index(np.argmin(masked_similarities), (n_tasks, n_tasks))
        least_similar = np.unravel_index(np.argmax(segment_avg), (n_tasks, n_tasks))
        
        segment_analyses.append({
            'segment': (start_idx * window_size, end_idx * window_size),  # Convert to actual timesteps
            'most_similar_pair': most_similar,
            'most_similar_distance': segment_avg[most_similar],
            'least_similar_pair': least_similar,
            'least_similar_distance': segment_avg[least_similar],
            'mean_distance': np.mean(segment_avg[~np.eye(n_tasks, dtype=bool)]),  # Exclude diagonal
            'std_distance': np.std(segment_avg[~np.eye(n_tasks, dtype=bool)])
        })
    
    return segment_analyses

def plot_trajectory_segments(episode_actions, task_ids, save_dir):
    """Plot action trajectories by dimension with temporal segmentation"""
    save_dir = Path(save_dir)
    action_dims = episode_actions[0].shape[1]
    dim_names = ['X Position', 'Y Position', 'Z Position', 
                 'Roll', 'Pitch', 'Yaw', 
                 'Gripper'][:action_dims]
    
    # Plot each dimension separately
    for dim in range(action_dims):
        plt.figure(figsize=(15, 5))
        for i, actions in enumerate(episode_actions):
            plt.plot(actions[:, dim], label=f'Task {task_ids[i]}', alpha=0.7)
        
        # Add vertical lines for key transition points
        transitions = find_action_transitions(episode_actions, dim)
        for t in transitions:
            plt.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
            
        plt.title(f'Action Trajectories - {dim_names[dim]}')
        plt.xlabel('Timestep')
        plt.ylabel('Action Value')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(save_dir / f'trajectory_dimension_{dim}.png')
        plt.close()

def find_action_transitions(episode_actions, dim):
    """Find significant transitions in action trajectories"""
    # Combine all trajectories for this dimension
    all_actions = np.concatenate([actions[:, dim] for actions in episode_actions])
    
    # Find peaks in the derivative (significant changes)
    derivatives = np.abs(np.diff(all_actions))
    peaks, _ = find_peaks(derivatives, height=np.std(derivatives)*2)
    
    return peaks

def create_temporal_similarity_matrix(episode_actions, window_size=10):
    """Create a similarity matrix showing when different tasks are most similar"""
    n_tasks = len(episode_actions)
    print(f"Number of tasks: {n_tasks}")
    min_length = min(len(actions) for actions in episode_actions)
    n_windows = min_length - window_size + 1
    
    # Initialize similarity matrix for each time window
    similarity_matrices = np.zeros((n_windows, n_tasks, n_tasks))
    
    for t in range(n_windows):
        for i in range(n_tasks):
            for j in range(i+1, n_tasks):
                window1 = episode_actions[i][t:t+window_size].flatten()
                window2 = episode_actions[j][t:t+window_size].flatten()
                
                distance = wasserstein_distance(window1, window2)
                similarity_matrices[t, i, j] = distance
                similarity_matrices[t, j, i] = distance
    
    return similarity_matrices

def visualize_temporal_patterns(similarity_matrices, save_dir, window_size=10):
    """Create visualizations of temporal similarity patterns"""
    save_dir = Path(save_dir)
    n_windows = similarity_matrices.shape[0]
    
    # Create time-averaged similarity matrix
    avg_similarity = np.mean(similarity_matrices, axis=0)
    
    # Plot average similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_similarity, cmap='viridis')
    plt.title('Average Task Similarity')
    plt.xlabel('Task ID')
    plt.ylabel('Task ID')
    plt.savefig(save_dir / 'average_similarity.png')
    plt.close()
    
    # Plot temporal evolution of similarities
    # Select a few key task pairs based on overall similarity
    most_similar_pairs = np.unravel_index(np.argmin(avg_similarity + np.eye(avg_similarity.shape[0]) * float('inf')), avg_similarity.shape)
    least_similar_pairs = np.unravel_index(np.argmax(avg_similarity), avg_similarity.shape)
    
    plt.figure(figsize=(15, 5))
    time_points = np.arange(n_windows) * window_size
    plt.plot(time_points, similarity_matrices[:, most_similar_pairs[0], most_similar_pairs[1]], 
             label=f'Most Similar (Tasks {most_similar_pairs[0]}-{most_similar_pairs[1]})')
    plt.plot(time_points, similarity_matrices[:, least_similar_pairs[0], least_similar_pairs[1]], 
             label=f'Least Similar (Tasks {least_similar_pairs[0]}-{least_similar_pairs[1]})')
    
    # Add key transition points
    transitions = find_similarity_transitions(similarity_matrices)
    for t in transitions:
        plt.axvline(x=t*window_size, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel('Timestep')
    plt.ylabel('Similarity')
    plt.title('Temporal Evolution of Task Similarities')
    plt.legend()
    plt.savefig(save_dir / 'temporal_evolution.png')
    plt.close()

def find_similarity_transitions(similarity_matrices):
    """Find points where task similarities change significantly"""
    # Average similarity over all task pairs at each timestep
    avg_similarities = np.mean(similarity_matrices, axis=(1,2))
    
    # Find significant changes
    derivatives = np.abs(np.diff(avg_similarities))
    peaks, _ = find_peaks(derivatives, height=np.std(derivatives)*2)
    
    return peaks

def summarize_temporal_patterns(segment_analyses, similarity_matrices, save_dir):
    """Create a textual and visual summary of temporal patterns"""
    summary_file = save_dir / 'temporal_analysis_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("Temporal Analysis Summary\n")
        f.write("========================\n\n")
        
        # Summarize each segment
        for i, analysis in enumerate(segment_analyses):
            f.write(f"Segment {i+1} (Steps {analysis['segment'][0]}-{analysis['segment'][1]}):\n")
            f.write(f"  Most similar tasks: {analysis['most_similar_pair']} (distance: {analysis['most_similar_distance']:.3f})\n")
            f.write(f"  Least similar tasks: {analysis['least_similar_pair']} (distance: {analysis['least_similar_distance']:.3f})\n")
            f.write(f"  Average distance: {analysis['mean_distance']:.3f} ± {analysis['std_distance']:.3f}\n\n")
        
        # Add overall statistics
        f.write("\nOverall Statistics:\n")
        f.write("=================\n")
        avg_similarity = np.mean(similarity_matrices, axis=0)
        most_similar = np.unravel_index(np.argmin(avg_similarity + np.eye(avg_similarity.shape[0]) * float('inf')), avg_similarity.shape)
        least_similar = np.unravel_index(np.argmax(avg_similarity), avg_similarity.shape)
        
        f.write(f"Most similar tasks overall: {most_similar}\n")
        f.write(f"Least similar tasks overall: {least_similar}\n")


import hydra
@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    # Load dataset
    print("Loading dataset...")
    dataset = hydra.utils.call(cfg.expert_dataset)
    
    # Create save directory
    save_dir = Path.cwd() / "temporal_analysis"
    save_dir.mkdir(exist_ok=True)
    
    # Collect one representative trajectory from each task
    print("Collecting task trajectories...")
    episode_actions = []
    task_ids = []
    
    for env_idx in range(dataset.envs_till_idx):
        # Get the first episode for each task (assuming it's representative)
        # Could be modified to use mean/median of all episodes if needed
        actions = dataset._episodes[env_idx][0]['action']
        
        # Store normalized actions
        if hasattr(dataset, 'preprocess'):
            actions = dataset.preprocess['actions'](actions)
        
        episode_actions.append(actions)
        print(f"Task {env_idx}: {actions.shape[0]} timesteps")
        task_ids.append(env_idx)
    
    # Analyze trajectory segments
    print("Analyzing trajectory segments...")
    window_size = 10  # Size of sliding window for temporal analysis
    n_segments = 5    # Number of segments to divide trajectories into
    
    # print("Computing similarity matrices...")
    similarity_matrices = create_temporal_similarity_matrix(
        episode_actions, 
        window_size=window_size
    )
    
    # # Analyze segments with fixed function
    # print("Analyzing trajectory segments...")
    segment_analyses = analyze_trajectory_segments(
        similarity_matrices, 
        n_segments=n_segments, 
        window_size=window_size
    )
    
    # # Plot trajectory analysis
    # print("Creating visualizations...")
    
    # # Plot individual dimension trajectories
    # plot_trajectory_segments(episode_actions, task_ids, save_dir)
    
    # # Create temporal similarity visualizations
    # visualize_temporal_patterns(
    #     similarity_matrices,
    #     save_dir,
    #     window_size=window_size
    # )
    
    # # Generate summary
    # print("Generating analysis summary...")
    # summarize_temporal_patterns(
    #     segment_analyses,
    #     similarity_matrices,
    #     save_dir
    # )
    
    # # Additional analysis: Find key alignments between tasks
    # print("Analyzing task alignments...")
    
    # # Create task alignment summary
    # alignment_summary = {}
    # for i in range(len(episode_actions)):
    #     for j in range(i+1, len(episode_actions)):
    #         # Get temporal similarity pattern for this pair
    #         pair_similarity = similarity_matrices[:, i, j]
    #         
    #         # Find points of high similarity (low distance)
    #         similarity_threshold = np.mean(pair_similarity) - np.std(pair_similarity)
    #         high_similarity_regions = np.where(pair_similarity < similarity_threshold)[0]
    #         
    #         # Group consecutive points into regions
    #         if len(high_similarity_regions) > 0:
    #             regions = np.split(high_similarity_regions, 
    #                              np.where(np.diff(high_similarity_regions) != 1)[0] + 1)
    #             
    #             # Store significant regions
    #             significant_regions = []
    #             for region in regions:
    #                 if len(region) * window_size >= 20:  # At least 20 timesteps
    #                     significant_regions.append({
    #                         'start': region[0] * window_size,
    #                         'end': region[-1] * window_size,
    #                         'avg_similarity': np.mean(pair_similarity[region])
    #                     })
    #             
    #             if significant_regions:
    #                 alignment_summary[(i,j)] = significant_regions
    
    # # Save alignment summary
    # with open(save_dir / 'task_alignments.txt', 'w') as f:
    #     f.write("Task Alignment Analysis\n")
    #     f.write("=====================\n\n")
    #     
    #     for (task1, task2), regions in alignment_summary.items():
    #         f.write(f"Tasks {task1}-{task2}:\n")
    #         for region in regions:
    #             f.write(f"  Similar motion during steps {region['start']}-{region['end']}"
    #                     f" (similarity: {region['avg_similarity']:.3f})\n")
    #         f.write("\n")
    
    # # Create visualization of task alignments
    # plt.figure(figsize=(15, 10))
    # for (task1, task2), regions in alignment_summary.items():
    #     for region in regions:
    #         plt.plot([region['start'], region['end']], 
    #                 [task1, task2],
    #                 'b-',
    #                 alpha=0.5,
    #                 linewidth=2)
    
    # plt.title('Task Motion Alignments')
    # plt.xlabel('Timestep')
    # plt.ylabel('Task ID')
    # plt.grid(True)
    # plt.savefig(save_dir / 'task_alignments.png')
    # plt.close()
    
    # print(f"Analysis complete. Results saved to {save_dir}")

if __name__ == "__main__":
    main()