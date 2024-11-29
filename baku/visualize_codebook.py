#!/usr/bin/env python3

import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from replay_buffer import make_expert_replay_loader
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import einops

def get_code(vqvae_model, actions):
    # Visualize the usage frequency of each codebook entry
    codes = []
    with torch.no_grad():
        for batch in torch.split(torch.tensor(actions).cuda(), 1024):
            # Reshape the batch to match the expected input shape
            if len(batch.shape) == 2:
                batch = batch.unsqueeze(1)  # Add time dimension if needed
            
            # Forward pass through the VQ-VAE to get indices
            _, vq_code = vqvae_model.get_code(batch)
            # get the indices of the code
            codes.append(vq_code.cpu().numpy())
    
    codes = np.concatenate(codes, axis=0)  # shape: [batch, seq_len, num_quantizers]
    return codes

def preprocess_actions(actions, preprocess, num_queries):
    # preprocess the actions
    reshaped_actions = []
    for action in actions:
        action = preprocess["actions"](action)
        action = np.lib.stride_tricks.sliding_window_view(
            action,
            (num_queries, action.shape[-1]),
        )[:, 0]
        action = einops.rearrange(action, "n t a -> n (t a)")
        reshaped_actions.extend(action)
    reshaped_actions = np.array(reshaped_actions)
    return reshaped_actions


def topk_codes(codes, k, bottom=False):
    # Get the top-k most frequent codebook entries
    # codes: [N, num_quantizers]
    topk = []
    for i in range(codes.shape[1]):
        code_frequencies = np.bincount(codes[:, i])
        if bottom:
            topk.append(np.argsort(code_frequencies)[:k])
        else:
            topk.append(np.argsort(code_frequencies)[::-1][:k])
    return topk

def visualize_code_combinations(codes, num_codes_per_quantizer, save_path):
    """
    Create a heatmap showing the frequency of code combinations between quantizers.
    
    Args:
        codes: numpy array of shape [N, num_quantizers] containing the codes
        num_codes_per_quantizer: int, number of possible codes per quantizer
        save_path: Path object or string, where to save the visualization
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    
    # Ensure save_path is a Path object
    save_path = Path(save_path)
    
    # Create a matrix to store co-occurrence frequencies
    cooccurrence_matrix = np.zeros((num_codes_per_quantizer, num_codes_per_quantizer))
    
    # Count co-occurrences between quantizer 0 and quantizer 1
    for i in range(len(codes)):
        q0_code = codes[i, 0]
        q1_code = codes[i, 1]
        cooccurrence_matrix[q0_code, q1_code] += 1
    
    # Create a heatmap using seaborn
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cooccurrence_matrix,
        annot=True,  # Show numbers in cells
        fmt='.0f',   # Format as integer
        cmap='viridis',
        xticklabels=range(num_codes_per_quantizer),
        yticklabels=range(num_codes_per_quantizer),
        cbar_kws={'label': 'Frequency'}
    )
    
    plt.title('Code Combination Frequencies Between Quantizers')
    plt.xlabel('Quantizer 2 Code Index')
    plt.ylabel('Quantizer 1 Code Index')
    
    # Add percentage annotations
    total_occurrences = cooccurrence_matrix.sum()
    for i in range(num_codes_per_quantizer):
        for j in range(num_codes_per_quantizer):
            percentage = (cooccurrence_matrix[i, j] / total_occurrences) * 100
            if percentage > 0:  # Only show non-zero percentages
                plt.text(
                    j + 0.5, 
                    i + 0.7,  # Offset to show below the count
                    f'({percentage:.1f}%)',
                    ha='center',
                    va='center',
                    color='white' if cooccurrence_matrix[i, j] > cooccurrence_matrix.max()/2 else 'black'
                )
    
    plt.tight_layout()
    plt.savefig(save_path / 'code_combinations_heatmap.png')
    plt.close()
    
    # Print summary statistics
    print("\nCode Combination Statistics:")
    print(f"Total number of samples: {int(total_occurrences)}")
    print(f"Number of unique combinations used: {np.count_nonzero(cooccurrence_matrix)}")
    print(f"Most common combination: Q1={np.unravel_index(cooccurrence_matrix.argmax(), cooccurrence_matrix.shape)[0]}, "
          f"Q2={np.unravel_index(cooccurrence_matrix.argmax(), cooccurrence_matrix.shape)[1]} "
          f"(used {int(cooccurrence_matrix.max())} times, {(cooccurrence_matrix.max()/total_occurrences)*100:.1f}%)")
    

def visualize_top_actions(actor_model, top_codes, env, save_path):
    """
    Visualize the actions generated from top VQ codes by executing them in the environment
    and saving the resulting videos. Includes both decoded actions and offsets.
    
    Args:
        actor_model: The actor model that contains both VQ-VAE and offset prediction
        top_codes: List of two 1D tensors containing top codes for each quantizer
        env: Environment instance to execute actions
        save_path: Path to save the visualization videos
    """
    from video import VideoRecorder
    import torch
    import numpy as np
    from pathlib import Path
    
    # Ensure save_path is a Path object
    save_path = Path(save_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_top_codes = len(top_codes[0])
    combined_codes = torch.stack([
        torch.tensor(top_codes[0].copy(), device=device), 
        torch.tensor(top_codes[1].copy(), device=device)
    ], dim=1)  # Shape: (N, 2)
    
    print(f"Combined codes shape: {combined_codes.shape}")
    print(f"Combined codes:\n{combined_codes}")
    
    # Create a video recorder
    video_recorder = VideoRecorder(save_path)
    
    # Process each combination of codes
    for code_idx in range(num_top_codes):
        print(f"Processing code combination {code_idx}: {combined_codes[code_idx]}")
        
        # Create encoding indices tensor with codes from both quantizers
        encoding_indices = torch.zeros(
            (1, actor_model._vqvae_model.vqvae_groups),
            dtype=torch.long,
            device=device
        )
        encoding_indices[0, 0] = combined_codes[code_idx, 0]  # First quantizer
        encoding_indices[0, 1] = combined_codes[code_idx, 1]  # Second quantizer
        
        with torch.no_grad():
            # Get the decoded action (base action from codebook)
            latent = actor_model._vqvae_model.draw_code_forward(encoding_indices)
            decoded_action = actor_model._vqvae_model.get_action_from_latent(latent).squeeze(0)
            
            # Get the predicted action (with offsets)
            # We need to prepare input for the cbet prediction
            x = decoded_action.clone()  # Shape: [1, action_dim]
            x = einops.rearrange(x, "N A -> N 1 A")  # Add time dimension
            
            # Get the offsets using the actor model
            result = actor_model._map_to_cbet_preds_offset(x)
            result = einops.rearrange(
                result, "(NT) (G C WA) -> (NT) G C WA", 
                G=actor_model._G, 
                C=actor_model._C
            )
            
            # Get the specific offsets for our codes
            NT = result.shape[0]
            indices = (
                torch.zeros(NT, dtype=torch.long, device=device),  # batch indices
                torch.arange(actor_model._G, device=device),      # quantizer indices
                encoding_indices[0]                               # our selected codes
            )
            sampled_offsets = result[indices].sum(dim=0)  # Sum across quantizers
            
            # Add offsets to decoded action
            predicted_action = decoded_action + sampled_offsets.view_as(decoded_action)
            
            # Convert to numpy
            predicted_action = predicted_action.cpu().numpy()
            decoded_action = decoded_action.cpu().numpy()
            
            print(f"Full action (with offsets) for combination {code_idx}:\n{predicted_action}")
            print(f"Decoded action (without offsets) for combination {code_idx}:\n{decoded_action}")
        
        # Reset environment and record execution with predicted action (including offsets)
        time_step = env[0].reset()
        video_recorder.init(env[0], enabled=True)
        
        # Execute the action sequence
        done = False
        step = 0
        max_steps = 100  # Limit the number of steps to avoid infinite loops
        # repeat the action steps 10 times
        predicted_action_repeated = np.repeat(predicted_action, 10, axis=1)
        
        while not done and step < max_steps:
            current_action = predicted_action_repeated[0, step*7:(step+1)*7]
            time_step = env[0].step(current_action)
            video_recorder.record(env[0])
            done = time_step.last()
            step += 1
        
        # Save video with descriptive name
        video_path = f"top_combination_{code_idx}_q0_{combined_codes[code_idx, 0]}_q1_{combined_codes[code_idx, 1]}.mp4"
        video_recorder.save(str(save_path / video_path))
            
        print(f"Saved video for code combination {code_idx}")
        
        # Save comparison video using just the decoded actions (without offsets)
        decoded_action_repeated = np.repeat(decoded_action, 10, axis=1)
        time_step = env[0].reset()
        video_recorder.init(env[0], enabled=True)
        
        step = 0
        done = False
        while not done and step < max_steps:
            current_action = decoded_action_repeated[0, step*7:(step+1)*7]
            time_step = env[0].step(current_action)
            video_recorder.record(env[0])
            done = time_step.last()
            step += 1
            
        video_path = f"top_combination_{code_idx}_q0_{combined_codes[code_idx, 0]}_q1_{combined_codes[code_idx, 1]}_decoded_only.mp4"
        video_recorder.save(str(save_path / video_path))
        
        print(f"Saved comparison video for code combination {code_idx}")

    
def visualize_codebook(agent, actions, save_path, preprocess, num_queries):
    """Visualize the learned codebook embeddings using PCA and t-SNE."""
    print(f"The shape of the actions is: {len(actions)}")
    print(f"The shape of the actions is: {actions[0].shape}")
    # Get the embeddings from the ResidualVQ model
    vqvae_model = agent._vqvae_model
    reshaped_actions = preprocess_actions(actions, preprocess, num_queries)

    # Get codebooks from all layers
    codebooks = vqvae_model.vq_layer.codebooks.detach().cpu().numpy()  # shape: [num_quantizers, num_codes, dim]
    num_quantizers, num_codes, embedding_dim = codebooks.shape
    
    # Create visualizations for each quantizer
    for q in range(num_quantizers):
        embeddings = codebooks[q]  # shape: [num_codes, dim]
        
        # Create a figure with two subplots for PCA and t-SNE
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA visualization
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)
        
        sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], ax=ax1)
        ax1.set_title(f'PCA Visualization - Quantizer {q+1}')
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Second Principal Component')
        
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=15)
        print(f"Embeddings shape: {embeddings.shape}")
        embeddings_tsne = tsne.fit_transform(embeddings)
        
        sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], ax=ax2)
        ax2.set_title(f't-SNE Visualization - Quantizer {q+1}')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        
        plt.tight_layout()
        plt.savefig(save_path / f'codebook_embeddings_quantizer_{q+1}.png')
        plt.close()
    codes = get_code(vqvae_model, reshaped_actions)
    num_codes = vqvae_model.vq_layer.codebooks.shape[1]    
    # Plot frequencies for each quantizer
    plt.figure(figsize=(15, 5 * num_quantizers))
    for q in range(num_quantizers):
        quantizer_codes = codes[..., q].reshape(-1)
        code_frequencies = np.zeros(num_codes)
        for i in range(num_codes):
            code_frequencies[i] = np.sum(quantizer_codes == i)
        
        plt.subplot(num_quantizers, 1, q + 1)
        plt.bar(range(num_codes), code_frequencies)
        plt.title(f'Codebook Entry Usage Frequency - Quantizer {q+1}')
        plt.xlabel('Codebook Index')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path / 'codebook_frequencies.png')
    plt.close()

    # Visualize the action space coverage
    plt.figure(figsize=(10, 10))
    
    # Sample random subset of actions for visualization if dataset is too large
    max_samples = 10000
    if len(actions) > max_samples:
        idx = np.random.choice(reshaped_actions.shape[0], max_samples, replace=False)
        actions_subset = reshaped_actions[idx]
    else:
        actions_subset = reshaped_actions
        
    # Project actions to 2D
    actions_flat = actions_subset.reshape(-1, actions_subset.shape[-1])
    pca = PCA(n_components=2)
    actions_2d = pca.fit_transform(actions_flat)
    
    # Plot original actions
    plt.scatter(actions_2d[:, 0], actions_2d[:, 1], alpha=0.5, label='Original Actions')
    
    # Get reconstructed actions
    # with torch.no_grad():
    #     actions_tensor = torch.tensor(actions_subset).cuda()
    #     if len(actions_tensor.shape) == 2:
    #         actions_tensor = actions_tensor.unsqueeze(1)
    #     
    #     # Forward pass through VQ-VAE
    #     reconstructed_actions, _, _ = vqvae_model(actions_tensor)
    #     reconstructed_actions = reconstructed_actions.cpu().numpy()
    #     
    #     if len(reconstructed_actions.shape) == 3:
    #         reconstructed_actions = reconstructed_actions.reshape(-1, reconstructed_actions.shape[-1])
    #     
    # # Project reconstructed actions to same 2D space
    # reconstructed_2d = pca.transform(reconstructed_actions)
    # plt.scatter(reconstructed_2d[:, 0], reconstructed_2d[:, 1], alpha=0.5, label='Reconstructed Actions')
    
    plt.title('Action Space Coverage')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.savefig(save_path / 'action_space_coverage.png')
    plt.close()

    # Visualize the hierarchical structure
    plt.figure(figsize=(15, 10))
    
    # Get a small subset of actions for visualization
    subset_size = min(100, len(actions))
    random_indices = np.random.choice(len(actions), subset_size, replace=False)
    subset_actions = torch.tensor(actions[random_indices]).cuda()
    
    if len(subset_actions.shape) == 2:
        subset_actions = subset_actions.unsqueeze(1)
    
    with torch.no_grad():
        # Get reconstructions at each level
        residual = subset_actions
        reconstructions = []
        for layer in vqvae_model.layers:
            quantized, *_ = layer(residual)
            reconstructions.append(quantized.cpu().numpy())
            residual = residual - quantized.detach()
    
    # Project all reconstructions to 2D
    pca = PCA(n_components=2)
    original_2d = pca.fit_transform(subset_actions.cpu().numpy().reshape(-1, subset_actions.shape[-1]))
    
    plt.scatter(original_2d[:, 0], original_2d[:, 1], alpha=0.8, label='Original', marker='o')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(reconstructions)))
    for i, (recon, color) in enumerate(zip(reconstructions, colors)):
        recon_2d = pca.transform(recon.reshape(-1, recon.shape[-1]))
        plt.scatter(recon_2d[:, 0], recon_2d[:, 1], alpha=0.5, 
                   label=f'Quantizer {i+1}', color=color, marker='x')
    
    plt.title('Hierarchical Reconstruction Visualization')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.savefig(save_path / 'hierarchical_reconstruction.png')
    plt.close()


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    # Set up paths
    work_dir = Path.cwd()
    save_path = work_dir / "codebook_visualization"
    save_path.mkdir(exist_ok=True)
    
    # Load dataset
    dataset_iterable = hydra.utils.call(cfg.expert_dataset)
    expert_replay_loader = make_expert_replay_loader(dataset_iterable, cfg.batch_size)
    dataset = expert_replay_loader.dataset
    cfg.suite.task_make_fn.max_episode_len = (
        expert_replay_loader.dataset._max_episode_len
    )
    cfg.suite.task_make_fn.max_state_dim = (
        expert_replay_loader.dataset._max_state_dim
    )
 
    # Create environment and agent
    print("Creating environment and agent...")
    env, task_descriptions = hydra.utils.call(cfg.suite.task_make_fn)
    print(f"Length of the environment is: {len(env)}")
    
    # Create agent
    from train import make_agent
    agent = make_agent(env[0].observation_spec(), env[0].action_spec(), cfg)
    
    # Load snapshot
    if cfg.load_bc:
        snapshot_path = Path(cfg.bc_weight)
        if not snapshot_path.exists():
            raise FileNotFoundError(f"BC weight not found: {snapshot_path}")
        print(f"Loading BC weight: {snapshot_path}")
        
        with snapshot_path.open('rb') as f:
            payload = torch.load(f)
        
        agent_payload = {k: v for k, v in payload.items() if k not in ['timer', '_global_step', '_global_episode', 'stats']}
        agent.load_snapshot(agent_payload, eval=True)
    num_queries = agent.num_queries
    # Get the VQ-VAE model from the agent
    print("Getting VQ-VAE model from the agent...", type(agent))
    # print the attributes of the agent
    if hasattr(agent, 'actor') and agent.actor._policy_head == "vqbet":
        vqvae_model = agent.actor._action_head
        print("VQ-VAE model found in the actor's policy head.")
    else:
        raise AttributeError("Could not find VQ-VAE model in the agent")
    
    # Get all actions from the dataset
    all_actions = dataset.actions
    
    # Visualize the codebook
    print("Visualizing codebook embeddings...")
    preprocess = expert_replay_loader.dataset.preprocess
    reshaped_actions = preprocess_actions(all_actions, preprocess, num_queries)
    codes = get_code(vqvae_model._vqvae_model, reshaped_actions)
    top_codes = topk_codes(codes, 30, bottom=False)
    # visualize the first 5 top codes
    # print(f"Top 10 codes are: {top_codes}")
    visualize_top_actions(vqvae_model, top_codes, env, save_path=".")

    # visualize_code_combinations(codes, 16, save_path=".")

    # visualize_codebook(vqvae_model, all_actions, save_path, preprocess, num_queries)
    
    # # Print codebook statistics
    # print("\nCodebook Statistics:")
    # print(f"Number of groups: {vqvae_model.vqvae_groups}")
    # print(f"Number of embeddings per group: {vqvae_model.vqvae_n_embed}")
    # print(f"Embedding dimension: {vqvae_model.embedding_dim}")
    
    # # Save statistics to a file
    # with open(save_path / "codebook_stats.txt", "w") as f:
    #     f.write(f"Number of groups: {vqvae_model.vqvae_groups}\n")
    #     f.write(f"Number of embeddings per group: {vqvae_model.vqvae_n_embed}\n")
    #     f.write(f"Embedding dimension: {vqvae_model.embedding_dim}\n")

if __name__ == "__main__":
    main()