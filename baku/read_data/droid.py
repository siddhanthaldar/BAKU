import random
import numpy as np
import tensorflow_datasets as tfds
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset


class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        num_demos,
        obs_type,
        history,
        history_len,
        prompt,
        temporal_agg,
        num_queries,
        img_size,
        intermediate_goal_step=30,
        store_actions=False,
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self._img_size = img_size
        self._intermediate_goal_step = intermediate_goal_step
        self._store_actions = store_actions

        # temporal aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # Load DROID dataset
        self.ds = tfds.load("droid_100", data_dir=path, split="train")
        
        # Store episodes
        self._episodes = []
        self._max_episode_len = 0
        self._num_samples = 0
        
        # Process and store episodes
        for episode in self.ds.take(num_demos):
            observations = []
            actions = []
            
            # Generate a random task embedding for each episode
            # Using similar dimensionality as typical language models (e.g., 768)
            task_emb = np.random.normal(0, 1, (768,))
            
            for step in episode["steps"]:
                # Store observations
                obs = {
                    "exterior_image_1": step["observation"]["exterior_image_1_left"].numpy(),
                    "exterior_image_2": step["observation"]["exterior_image_2_left"].numpy(),
                    "wrist_image": step["observation"]["wrist_image_left"].numpy(),
                    "cartesian_position": step["observation"]["cartesian_position"].numpy(),
                    "gripper_position": step["observation"]["gripper_position"].numpy(),
                    "joint_position": step["observation"]["joint_position"].numpy(),
                }
                observations.append(obs)
                
                # Store actions
                action = step["action"].numpy()
                actions.append(action)
            
            episode_dict = {
                "observation": observations,
                "action": np.array(actions),
                "task_emb": task_emb,  # Add task embedding
            }
            
            self._episodes.append(episode_dict)
            self._max_episode_len = max(self._max_episode_len, len(observations))
            self._num_samples += len(observations)

        # Augmentation
        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self._img_size, self._img_size)),
            transforms.ToTensor(),
        ])

        # Action and proprioceptive normalization
        self.stats = {
            "actions": {
                "min": -1,
                "max": 1,
            },
            "proprioceptive": {
                "min": -1,
                "max": 1,
            },
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"]) 
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
            "proprioceptive": lambda x: (x - self.stats["proprioceptive"]["min"]) 
            / (self.stats["proprioceptive"]["max"] - self.stats["proprioceptive"]["min"] + 1e-5),
        }

    def _sample_episode(self):
        episode = random.choice(self._episodes)
        return episode

    def _sample(self):
        episode = self._sample_episode()
        observations = episode["observation"]
        actions = episode["action"]
        task_emb = episode["task_emb"]
        
        # Sample random index
        # When history is False, we only need 1 frame regardless of history_len
        effective_history_len = 1 if not self._history else self._history_len
        sample_idx = np.random.randint(0, len(observations) - effective_history_len)
        
        if self._obs_type == "pixels":
            # Process images
            sampled_pixel = []  # third-person view
            sampled_pixel_egocentric = []  # egocentric view
            for i in range(effective_history_len):
                obs = observations[sample_idx + i]
                # Process third-person view (exterior camera 1)
                sampled_pixel.append(self.aug(obs["exterior_image_1"]))
                # Process egocentric view (wrist camera)
                sampled_pixel_egocentric.append(self.aug(obs["wrist_image"]))
            
            sampled_pixel = torch.stack(sampled_pixel)
            sampled_pixel_egocentric = torch.stack(sampled_pixel_egocentric)
            
            # Process proprioceptive states
            sampled_proprioceptive_state = np.array([
                np.concatenate([
                    observations[sample_idx + i]["cartesian_position"],
                    observations[sample_idx + i]["gripper_position"],
                    observations[sample_idx + i]["joint_position"],
                ]) for i in range(effective_history_len)
            ])

            if self._temporal_agg:
                # Handle temporal aggregation
                sampled_action = np.zeros((effective_history_len, self._num_queries, actions.shape[-1]))
                num_actions = effective_history_len + self._num_queries - 1
                act = np.zeros((num_actions, actions.shape[-1]))
                act[:min(len(actions), sample_idx + num_actions) - sample_idx] = \
                    actions[sample_idx:sample_idx + num_actions]
                sampled_action = np.lib.stride_tricks.sliding_window_view(
                    act, (self._num_queries, actions.shape[-1])
                )
                sampled_action = sampled_action[:, 0]
            else:
                sampled_action = actions[sample_idx:sample_idx + effective_history_len]

            return {
                "pixels": sampled_pixel,
                "pixels_egocentric": sampled_pixel_egocentric,
                "proprioceptive": self.preprocess["proprioceptive"](sampled_proprioceptive_state),
                "actions": self.preprocess["actions"](sampled_action),
                "task_emb": task_emb,  # Return task embedding instead of instruction
            }

        else:
            raise NotImplementedError("Only 'pixels' observation type is supported for DROID dataset")

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples

    def get_observation_spec(self):
        """Returns the observation specification for the DROID dataset.
        Returns specs in the same format as environment observation specs."""
        # State dim = cartesian (3) + gripper () + joint (9) = 13 (14? somehow?)
        self._max_state_dim = 14
        
        # Convert shapes to tuples instead of torch.Size
        return {
            'pixels': torch.zeros((3, self._img_size, self._img_size)).numpy(),  # Convert to numpy array
            'pixels_egocentric': torch.zeros((3, self._img_size, self._img_size)).numpy(),
            'proprioceptive': torch.zeros(14).numpy(),
            'task_emb': torch.zeros(768).numpy(),  # Add task embedding spec
            'features': torch.zeros(self._max_state_dim).numpy()
        }

    def get_action_spec(self):
        """Returns the action specification for the DROID dataset."""
        return torch.zeros(7).numpy()  # Convert to numpy array

def test_droid_dataset():
    """Test function to verify the DROID dataset loader."""
    try:
        history = True
        history_len = 2
        num_demos = 2
        num_queries = 10
        img_size = 224
        # Initialize dataset
        dataset = BCDataset(
            path="/home/bpatil/workspace/skill_seg/skill_seg/data/",  # Update with your actual path
            num_demos=num_demos,  # Small number for testing
            obs_type="pixels",
            history=history,
            history_len=history_len,
            prompt=None,
            temporal_agg=True,
            num_queries=num_queries,
            img_size=img_size,
        )
        
        # Get a sample
        sample = next(iter(dataset))
        
        # Check if all expected keys are present
        expected_keys = ["pixels", "pixels_egocentric", "proprioceptive", "actions", "task_emb"]
        assert all(key in sample for key in expected_keys), f"Missing keys in sample. Expected {expected_keys}, got {list(sample.keys())}"
        
        # Check shapes
        assert len(sample["pixels"].shape) == 4, f"Expected 4D tensor for pixels, got shape {sample['pixels'].shape}"
        assert len(sample["pixels_egocentric"].shape) == 4, f"Expected 4D tensor for pixels_egocentric, got shape {sample['pixels_egocentric'].shape}"
        assert sample["pixels"].shape == sample["pixels_egocentric"].shape, f"Expected same shape for both views, got {sample['pixels'].shape} and {sample['pixels_egocentric'].shape}"
        assert sample["pixels"].shape[0] == history_len, f"Expected history_len={history_len}, got {sample['pixels'].shape[0]}"
        assert sample["pixels"].shape[1] == 3, f"Expected 3 channels, got {sample['pixels'].shape[1]}"
        assert sample["pixels"].shape[2] == img_size, f"Expected height={img_size}, got {sample['pixels'].shape[2]}"
        assert sample["pixels"].shape[3] == img_size, f"Expected width={img_size}, got {sample['pixels'].shape[3]}"
        
        # Check proprioceptive state
        assert len(sample["proprioceptive"].shape) == 2, f"Expected 2D array for proprioceptive, got shape {sample['proprioceptive'].shape}"
        assert sample["proprioceptive"].shape[0] == history_len, f"Expected history_len={history_len}, got {sample['proprioceptive'].shape[0]}"
        
        # Check actions
        if dataset._temporal_agg:
            expected_action_shape = (history_len, num_queries, 7)  # (history_len, num_queries, action_dim)
        else:
            expected_action_shape = (history_len, 7)  # (history_len, action_dim)
        assert sample["actions"].shape == expected_action_shape, f"Expected actions shape {expected_action_shape}, got {sample['actions'].shape}"
        
        # Check task embedding
        assert isinstance(sample["task_emb"], np.ndarray), f"Expected numpy array for task_emb, got {type(sample['task_emb'])}"
        assert sample["task_emb"].shape == (768,), f"Expected task_emb shape (768,), got {sample['task_emb'].shape}"
        
        print("All tests passed! Dataset is working as expected.")
        print("\nSample data:")
        print(f"Third-person view shape: {sample['pixels'].shape}")
        print(f"Egocentric view shape: {sample['pixels_egocentric'].shape}")
        print(f"Proprioceptive shape: {sample['proprioceptive'].shape}")
        print(f"Actions shape: {sample['actions'].shape}")
        print(f"Task embedding shape: {sample['task_emb'].shape}")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    test_droid_dataset()
