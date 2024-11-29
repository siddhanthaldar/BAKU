#!/usr/bin/env python3

import hydra
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import seaborn as sns
from tqdm import tqdm

def collect_actions(dataset):
    """Collect all actions from the dataset into a single array"""
    actions = []
    
    # Collect actions from all episodes
    for env_idx in range(dataset.envs_till_idx):
        for episode in dataset._episodes[env_idx]:
            # Get raw actions before preprocessing
            episode_actions = episode['action']
            actions.extend(episode_actions)
    
    return np.array(actions)

def analyze_action_space(actions):
    """Perform PCA and UMAP analysis on the action space"""
    # Standardize the data
    scaler = StandardScaler()
    actions_scaled = scaler.fit_transform(actions)
    
    # PCA Analysis
    pca = PCA()
    actions_pca = pca.fit_transform(actions_scaled)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # UMAP Analysis
    reducer = umap.UMAP(random_state=42)
    actions_umap = reducer.fit_transform(actions_scaled)
    
    return {
        'pca': actions_pca,
        'umap': actions_umap,
        'explained_variance': explained_variance_ratio,
        'cumulative_variance': cumulative_variance_ratio
    }

def plot_analysis(results, save_dir):
    """Create visualizations of the analysis results"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Plot PCA explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(results['explained_variance']) + 1), 
            results['cumulative_variance'], 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig(save_dir / 'pca_explained_variance.png')
    plt.close()
    
    # Plot first two PCA components
    plt.figure(figsize=(10, 10))
    plt.scatter(results['pca'][:, 0], results['pca'][:, 1], 
               alpha=0.5, s=1)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA: First Two Components')
    plt.savefig(save_dir / 'pca_scatter.png')
    plt.close()
    
    # Plot UMAP embedding
    plt.figure(figsize=(10, 10))
    plt.scatter(results['umap'][:, 0], results['umap'][:, 1], 
               alpha=0.5, s=1)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('UMAP Embedding')
    plt.savefig(save_dir / 'umap_scatter.png')
    plt.close()
    
    # Create density plots
    plt.figure(figsize=(10, 10))
    sns.kdeplot(data={'UMAP1': results['umap'][:, 0], 
                     'UMAP2': results['umap'][:, 1]},
                x='UMAP1', y='UMAP2', cmap='viridis')
    plt.title('UMAP Density Plot')
    plt.savefig(save_dir / 'umap_density.png')
    plt.close()

def analyze_clusters(actions_umap):
    """Basic cluster analysis using UMAP embeddings"""
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    
    # Try different eps values for DBSCAN
    eps_values = np.linspace(0.1, 2, 20)
    best_score = -1
    best_labels = None
    best_eps = None
    
    for eps in tqdm(eps_values, desc="Finding optimal clusters"):
        clusterer = DBSCAN(eps=eps, min_samples=5)
        labels = clusterer.fit_predict(actions_umap)
        
        # Only calculate score if we have more than one cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            score = silhouette_score(actions_umap, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_eps = eps
    
    return best_labels, best_eps, best_score

@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    # Load dataset
    dataset = hydra.utils.call(cfg.expert_dataset)
    
    print("Collecting actions from dataset...")
    actions = collect_actions(dataset)
    print(f"Collected {len(actions)} actions with dimension {actions.shape[1]}")
    
    print("Performing PCA and UMAP analysis...")
    results = analyze_action_space(actions)
    
    print("\nPCA Analysis Results:")
    for i, var in enumerate(results['explained_variance']):
        print(f"Component {i+1}: {var:.4f} explained variance ratio")
        if results['cumulative_variance'][i] > 0.95:
            print(f"95% variance explained with {i+1} components")
            break
    
    print("\nPerforming cluster analysis...")
    cluster_labels, best_eps, silhouette = analyze_clusters(results['umap'])
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Found {n_clusters} clusters with eps={best_eps:.3f}")
    print(f"Silhouette score: {silhouette:.3f}")
    
    # Plot results with cluster colors
    save_dir = Path.cwd() / "action_analysis"
    plot_analysis(results, save_dir)
    
    # Plot clustered UMAP
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(results['umap'][:, 0], results['umap'][:, 1], 
                         c=cluster_labels, cmap='tab20', alpha=0.5, s=1)
    plt.colorbar(scatter)
    plt.title(f'UMAP Embedding with {n_clusters} Clusters')
    plt.savefig(save_dir / 'umap_clusters.png')
    plt.close()

if __name__ == "__main__":
    main()