import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

def plot_trajectories(actions_file):
    # Load the actions
    with open(actions_file, 'rb') as f:
        actions_taken = pickle.load(f)
    
    # Create a new figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for different episodes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(actions_taken[0])))
    
    for env_idx in actions_taken.keys():
        for episode, episode_color in zip(actions_taken[env_idx].keys(), colors):
            # Extract x, y, z positions from actions
            # Assuming actions contain end effector positions - adjust indices based on your action space
            x_pos = []
            y_pos = []
            z_pos = []
            
            for step in sorted(actions_taken[env_idx][episode].keys()):
                action = actions_taken[env_idx][episode][step]
                # Assuming first 3 elements of action are x,y,z positions
                # Modify these indices based on your action space structure
                x_pos.append(action[0])
                y_pos.append(action[1])
                z_pos.append(action[2])
            
            # Plot trajectory for this episode
            ax.plot3D(x_pos, y_pos, z_pos, 
                     color=episode_color, 
                     label=f'Env {env_idx}, Episode {episode}',
                     alpha=0.7)
            
            # Plot start point (green) and end point (red)
            ax.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', marker='o')
            ax.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='red', marker='x')
    
    # Customize the plot
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('End Effector Trajectories')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(True)
    
    # Adjust the view
    ax.view_init(elev=20, azim=45)
    
    # Save the plot
    plt.savefig('trajectories.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Create separate plots for each environment
    for env_idx in actions_taken.keys():
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for episode, episode_color in zip(actions_taken[env_idx].keys(), colors):
            x_pos = []
            y_pos = []
            z_pos = []
            
            for step in sorted(actions_taken[env_idx][episode].keys()):
                action = actions_taken[env_idx][episode][step]
                x_pos.append(action[0])
                y_pos.append(action[1])
                z_pos.append(action[2])
            
            ax.plot3D(x_pos, y_pos, z_pos, 
                     color=episode_color, 
                     label=f'Episode {episode}',
                     alpha=0.7)
            
            ax.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', marker='o')
            ax.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='red', marker='x')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'End Effector Trajectories - Environment {env_idx}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        ax.view_init(elev=20, azim=45)
        
        plt.savefig(f'trajectories_env_{env_idx}.png', bbox_inches='tight', dpi=300)
        plt.close()

# Function to create an animated view of the trajectories
def create_animated_view(actions_file):
    import matplotlib.animation as animation
    
    with open(actions_file, 'rb') as f:
        actions_taken = pickle.load(f)
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(actions_taken[0])))
    
    def update(frame):
        ax.view_init(elev=20, azim=frame)
        return fig,
    
    # Plot all trajectories
    for env_idx in actions_taken.keys():
        for episode, episode_color in zip(actions_taken[env_idx].keys(), colors):
            x_pos = []
            y_pos = []
            z_pos = []
            
            for step in sorted(actions_taken[env_idx][episode].keys()):
                action = actions_taken[env_idx][episode][step]
                x_pos.append(action[0])
                y_pos.append(action[1])
                z_pos.append(action[2])
            
            ax.plot3D(x_pos, y_pos, z_pos, 
                     color=episode_color, 
                     label=f'Env {env_idx}, Episode {episode}',
                     alpha=0.7)
            
            ax.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', marker='o')
            ax.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='red', marker='x')
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('End Effector Trajectories (Rotating View)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=360, interval=50, blit=True)
    anim.save('trajectories_rotating.gif', writer='pillow')
    plt.close()

# Usage
actions_file = "/home/bpatil/workspace/BAKU/baku/exp_local/eval/2024.10.31_eval/vqbet/162049_hidden_dim_256/0_actions_taken.pkl"  # Replace with your actual file name
plot_trajectories(actions_file)
create_animated_view(actions_file)