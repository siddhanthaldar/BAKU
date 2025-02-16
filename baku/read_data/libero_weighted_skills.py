import random
import numpy as np
import pickle as pkl
from pathlib import Path
from typing import Dict, Optional, Union
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset


class WeightedSkillDataset(IterableDataset):
    def __init__(
        self,
        path,
        suite,
        skills,
        obs_type,
        history,
        history_len,
        prompt,
        temporal_agg,
        num_queries,
        img_size,
        weights: Optional[Dict[str, float]] = None,
        batch_size: int = 32,
        intermediate_goal_step=50,
        store_actions: bool=False
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self.img_size = img_size
        self.intermediate_goal_step = intermediate_goal_step
        self.batch_size = batch_size

        # temporal_aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # Get data paths for each skill
        self._paths = {}
        for skill in skills:
            skill_path = Path(path) / f"{skill}.pkl"
            if skill_path.exists():
                self._paths[skill] = skill_path
            else:
                print(f"Warning: No data found for skill {skill}")

        # Initialize data structures
        self._demonstrations = {}
        self._max_episode_len = 0
        self._min_episode_len = float("inf")
        self._max_state_dim = 0
        self._num_samples = 0
        self._skill_sizes = {}  # Track number of demonstrations per skill

        # Load data for each skill
        for skill_name, skill_path in self._paths.items():
            print(f"Loading skill data from {str(skill_path)}")
            with open(str(skill_path), "rb") as f:
                demonstrations = pkl.load(f)
            
            self._demonstrations[skill_name] = demonstrations
            self._skill_sizes[skill_name] = len(demonstrations)
            
            # Update statistics
            for demo in demonstrations:
                episode_len = len(demo['actions'])
                self._max_episode_len = max(self._max_episode_len, episode_len)
                self._min_episode_len = min(self._min_episode_len, episode_len)
                self._max_state_dim = max(self._max_state_dim, demo['states'].shape[-1] if 'states' in demo else 0)
                self._num_samples += episode_len

        # Initialize sampling weights
        self._initialize_weights(weights)

        # Calculate number of samples per skill in a batch
        self._calculate_batch_distribution()

        # Define data statistics for normalization
        self.stats = {
            "actions": {"min": 0, "max": 1},
            "proprioceptive": {"min": 0, "max": 1},
        }

        # Define preprocessing functions
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
            "proprioceptive": lambda x: (x - self.stats["proprioceptive"]["min"])
            / (
                self.stats["proprioceptive"]["max"]
                - self.stats["proprioceptive"]["min"]
                + 1e-5
            ),
        }

        # Image augmentation pipeline
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        # Store total number of skills
        self.num_skills = len(self._demonstrations)

    def _calculate_batch_distribution(self):
        """Calculate how many samples from each skill should be in a batch."""
        self._batch_counts = {}
        remaining = self.batch_size
        
        # First pass: Calculate initial counts and handle rounding
        for skill, weight in self._weights.items():
            count = int(self.batch_size * weight)
            self._batch_counts[skill] = count
            remaining -= count
        
        # Distribute remaining samples
        if remaining > 0:
            skills = list(self._weights.keys())
            probs = list(self._weights.values())
            extra_skills = np.random.choice(
                skills, 
                size=remaining, 
                p=probs,
                replace=True
            )
            for skill in extra_skills:
                self._batch_counts[skill] += 1

    def _initialize_weights(self, weights: Optional[Dict[str, float]] = None):
        """Initialize sampling weights for skills."""
        if weights is None:
            # Default to uniform weights
            total_demos = sum(self._skill_sizes.values())
            self._weights = {skill: size/total_demos for skill, size in self._skill_sizes.items()}
        else:
            # Normalize provided weights
            total_weight = sum(weights.values())
            self._weights = {skill: weight/total_weight for skill, weight in weights.items()}
            
            # Verify all skills have weights
            missing_skills = set(self._demonstrations.keys()) - set(weights.keys())
            if missing_skills:
                raise ValueError(f"Missing weights for skills: {missing_skills}")
        
        # Create a flattened list of (skill, demo) pairs for efficient sampling
        self._sample_pool = []
        for skill, demos in self._demonstrations.items():
            weight = self._weights[skill]
            # Calculate number of copies based on weight
            num_copies = max(1, int(weight * 1000))  # Scale up for better granularity
            self._sample_pool.extend([(skill, demo_idx) for demo_idx in range(len(demos))] * num_copies)
        
        # Shuffle the pool
        random.shuffle(self._sample_pool)
        self._pool_index = 0

    def update_weights(self, new_weights: Dict[str, float]):
        """Update sampling weights and recreate the sample pool."""
        self._initialize_weights(new_weights)

    def _get_next_sample(self):
        """Get next sample from the pool."""
        if self._pool_index >= len(self._sample_pool):
            random.shuffle(self._sample_pool)
            self._pool_index = 0
        
        skill_name, demo_idx = self._sample_pool[self._pool_index]
        self._pool_index += 1
        
        demo = self._demonstrations[skill_name][demo_idx]
        return self._process_sample(demo, skill_name)

    def __iter__(self):
        while True:
            valid_samples = []
            attempts = 0
            max_attempts = self.batch_size * 2  # Limit total attempts
            
            while len(valid_samples) < self.batch_size and attempts < max_attempts:
                sample = self._get_next_sample()
                if sample is not None:
                    valid_samples.append(sample)
                attempts += 1
            
            if valid_samples:  # Yield whatever valid samples we got
                for sample in valid_samples:
                    yield sample


    def get_weights(self) -> Dict[str, float]:
        """Return current sampling weights."""
        return self._weights.copy()

    def get_batch_distribution(self) -> Dict[str, int]:
        """Return current batch distribution."""
        return self._batch_counts.copy()

    def _sample_demonstration(self, skill_name: str):
        """Sample a demonstration from a specific skill."""
        return random.choice(self._demonstrations[skill_name])

    def _process_sample(self, demo, skill_name):
        """Process a single demonstration into a sample."""
        if self._obs_type == "pixels":
            observations = demo["observations"]
            actions = demo["actions"]
            task_emb = demo["task_emb"]
            # Sample observation frames
            max_start_idx = len(observations["pixels"]) - self._history_len
            if max_start_idx < 1:
                sample_idx = 0
                try:
                    # Handle padding for short demonstrations
                    pad_len = self._history_len - len(observations["pixels"])
                    if pad_len > 0:
                        observations["pixels"] = np.pad(observations["pixels"], 
                            ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')
                        observations["pixels_egocentric"] = np.pad(observations["pixels_egocentric"],
                            ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')
                        observations["joint_states"] = np.pad(observations["joint_states"],
                            ((0, pad_len), (0, 0)), mode='edge')
                        observations["gripper_states"] = np.pad(observations["gripper_states"],
                            ((0, pad_len), (0, 0)), mode='edge')
                        actions = np.pad(actions, ((0, pad_len), (0, 0)), mode='edge')
                except ValueError as e:
                    # print(f"Skipping problematic demonstration for skill {skill_name}: {str(e)}")
                    return None
            else:
                sample_idx = np.random.randint(0, max_start_idx)
            
            # Process and return sample data
            # [Rest of the processing logic remains the same as in original _sample]
            # Process image observations
            sampled_pixel = observations["pixels"][
                sample_idx : sample_idx + self._history_len
            ]
            sampled_pixel_egocentric = observations["pixels_egocentric"][
                sample_idx : sample_idx + self._history_len
            ]
            
            # Apply augmentations
            sampled_pixel = torch.stack(
                [self.aug(sampled_pixel[i]) for i in range(len(sampled_pixel))]
            )
            sampled_pixel_egocentric = torch.stack(
                [
                    self.aug(sampled_pixel_egocentric[i])
                    for i in range(len(sampled_pixel_egocentric))
                ]
            )
            
            # Process proprioceptive states
            sampled_proprioceptive_state = np.concatenate(
                [
                    observations["joint_states"][
                        sample_idx : sample_idx + self._history_len
                    ],
                    observations["gripper_states"][
                        sample_idx : sample_idx + self._history_len
                    ],
                ],
                axis=-1,
            )

            # Handle temporal aggregation if enabled
            if self._temporal_agg:
                sampled_action = np.zeros(
                    (self._history_len, self._num_queries, actions.shape[-1])
                )
                num_actions = self._history_len + self._num_queries - 1
                act = np.zeros((num_actions, actions.shape[-1]))
                act[
                    : min(len(actions), sample_idx + num_actions) - sample_idx
                ] = actions[sample_idx : sample_idx + num_actions]
                sampled_action = np.lib.stride_tricks.sliding_window_view(
                    act, (self._num_queries, actions.shape[-1])
                )
                sampled_action = sampled_action[:, 0]
            else:
                sampled_action = actions[sample_idx : sample_idx + self._history_len]

            # Return based on prompt type
            # print(f"Skill Names: {skill_name}")
            base_dict = {
                "pixels": sampled_pixel,
                "pixels_egocentric": sampled_pixel_egocentric,
                "proprioceptive": self.preprocess["proprioceptive"](
                    sampled_proprioceptive_state
                ),
                "actions": self.preprocess["actions"](sampled_action),
                "task_emb": task_emb,
                "skill_name": skill_name,
                "language_instruction": demo.get("language_instruction", None),
                "initial_gripper": demo.get("initial_gripper", None),
                "final_gripper": demo.get("final_gripper", None),
            }

            if self._prompt == "text":
                return base_dict
            
            elif self._prompt in ["goal", "intermediate_goal"]:
                # Handle goal-based prompts
                if self._prompt == "goal":
                    prompt_demo = self._sample_demonstration(skill_name)
                    goal_idx = -1
                else:  # intermediate_goal
                    prompt_demo = demo
                    intermediate_goal_step = (
                        self.intermediate_goal_step + np.random.randint(-30, 30)
                    )
                    goal_idx = min(
                        sample_idx + intermediate_goal_step,
                        len(prompt_demo["observations"]["pixels"]) - 1,
                    )

                # Process prompt observations
                prompt_pixel = self.aug(
                    prompt_demo["observations"]["pixels"][goal_idx]
                )[None]
                prompt_pixel_egocentric = self.aug(
                    prompt_demo["observations"]["pixels_egocentric"][goal_idx]
                )[None]
                prompt_proprioceptive_state = np.concatenate(
                    [
                        prompt_demo["observations"]["joint_states"][goal_idx:goal_idx + 1],
                        prompt_demo["observations"]["gripper_states"][goal_idx:goal_idx + 1],
                    ],
                    axis=-1,
                )
                prompt_action = prompt_demo["actions"][goal_idx:goal_idx + 1]

                return {
                    **base_dict,
                    "prompt_pixels": prompt_pixel,
                    "prompt_pixels_egocentric": prompt_pixel_egocentric,
                    "prompt_proprioceptive": self.preprocess["proprioceptive"](
                        prompt_proprioceptive_state
                    ),
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                }

        elif self._obs_type == "features":
            states = demo["states"]
            actions = demo["actions"]
            task_emb = demo["task_emb"]

            # Sample observation features
            max_start_idx = len(states) - self._history_len
            if max_start_idx < 1:
                sample_idx = 0
                # Pad states and actions if needed
                pad_len = self._history_len - len(states)
                if pad_len > 0:
                    states = np.pad(states, ((0, pad_len), (0, 0)), mode='edge')
                    actions = np.pad(actions, ((0, pad_len), (0, 0)), mode='edge')
            else:
                sample_idx = np.random.randint(0, max_start_idx)

            sampled_obs = np.array(states[sample_idx : sample_idx + self._history_len])
            sampled_action = actions[sample_idx : sample_idx + self._history_len]

            # Pad observations to match max_state_dim
            obs = np.zeros((self._history_len, self._max_state_dim))
            state_dim = sampled_obs.shape[-1]
            obs[:, :state_dim] = sampled_obs
            sampled_obs = obs

            base_dict = {
                "features": sampled_obs,
                "actions": self.preprocess["actions"](sampled_action),
                "task_emb": task_emb,
                "skill_name": skill_name,
                "language_instruction": demo.get("language_instruction", None),
                "initial_gripper": demo.get("initial_gripper", None),
                "final_gripper": demo.get("final_gripper", None),
            }

            if self._prompt == "text":
                return base_dict
            
            elif self._prompt in ["goal", "intermediate_goal"]:
                if self._prompt == "goal":
                    prompt_demo = self._sample_demonstration(skill_name)
                    prompt_obs = np.array(prompt_demo["states"][-1:])
                else:  # intermediate_goal
                    goal_idx = min(
                        sample_idx + self.intermediate_goal_step,
                        len(states) - 1,
                    )
                    prompt_obs = np.array(states[goal_idx:goal_idx + 1])
                
                prompt_action = prompt_demo["actions"][goal_idx:goal_idx + 1]
                
                return {
                    **base_dict,
                    "prompt_obs": prompt_obs,
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                }

    def __len__(self):
        return self._num_samples
    

import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
import argparse


def test_dataset(args):
    print("\n=== Starting WeightedSkillDataset Test ===\n")

    # Initialize dataset
    print("Initializing dataset...")
    try:
        dataset = WeightedSkillDataset(
            path=args.data_path,
            skills=args.skills,
            obs_type=args.obs_type,
            history=True,
            history_len=args.history_len,
            prompt=args.prompt,
            temporal_agg=True,
            num_queries=10,
            img_size=224,
            batch_size=args.batch_size,
        )
        print("✓ Dataset initialized successfully")
        print(f"Total number of skills loaded: {dataset.num_skills}")
        print(f"Max episode length: {dataset._max_episode_len}")
        print(f"Min episode length: {dataset._min_episode_len}")
        print(f"Total number of samples: {len(dataset)}")
        print(f"Initial weights: {dataset.get_weights()}")
        print(f"Initial batch distribution: {dataset.get_batch_distribution()}")

    except Exception as e:
        print(f"ERROR during dataset initialization: {str(e)}")
        return

    # Test data loading
    print("\nTesting data loading...")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers) # batch_size=1 for simpler checks
    iterator = iter(dataloader)

    try:
        # Time the data loading
        start_time = time.time()
        batch = next(iterator)
        load_time = time.time() - start_time
        print(f"✓ Successfully loaded first batch in {load_time:.3f} seconds")

        # Print batch information
        print("\nBatch contents:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"  {key}: list of length {len(value)}")
            else:
                print(f"  {key}: type={type(value)}")

        # Test a few more batches for consistency and weight-based sampling
        print("\nTesting multiple batch loads and batch distribution...")
        batch_times = []
        skill_counts = defaultdict(int) # Track skill distribution in batches
        for i in range(args.num_test_batches):
            start_time = time.time()
            batch = next(iterator)
            batch_times.append(time.time() - start_time)

            # Basic validation
            if args.obs_type == "pixels":
                assert "pixels" in batch, "Missing pixels in batch"
                assert "pixels_egocentric" in batch, "Missing egocentric pixels in batch"
            else:
                assert "features" in batch, "Missing features in batch"

            assert "actions" in batch, "Missing actions in batch"
            assert "task_emb" in batch, "Missing task embeddings in batch"
            skill_counts[batch['skill_name'][0]] += 1 # Count skill occurrences

        avg_time = np.mean(batch_times)
        std_time = np.std(batch_times)
        print(f"✓ Successfully loaded {args.num_test_batches} additional batches")
        print(f"  Average batch load time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Skill distribution in test batches: {skill_counts}")

        # Memory usage check (rough estimate)
        if args.obs_type == "pixels":
            pixels_mem = batch["pixels"].element_size() * batch["pixels"].nelement()
            print(f"\nApproximate memory usage per batch:")
            print(f"  Pixels: {pixels_mem / 1024 / 1024:.2f} MB")

        # Test weight updating and batch distribution recalculation
        print("\nTesting weight updating...")
        initial_weights = dataset.get_weights()
        # new_weights = {skill: weight * 2 for skill, weight in initial_weights.items()} # Example: double weights
        new_weights = initial_weights.copy()
        new_weights['rotating'] = 0.5
        new_weights['picking'] = 0.25
        new_weights['placing'] = 0.25
        dataset.update_weights(new_weights)
        updated_weights = dataset.get_weights()
        updated_distribution = dataset.get_batch_distribution()
        print(f"✓ Weights updated successfully")
        print(f"  Initial weights: {initial_weights}")
        print(f"  Updated weights: {updated_weights}")
        print(f"  Updated batch distribution: {updated_distribution}")


    except AssertionError as e:
        print(f"ERROR: Validation failed - {str(e)}")
    except Exception as e:
        print(f"ERROR during data loading: {str(e)}")

    print("\n=== WeightedSkillDataset Test Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the WeightedSkillDataset loader")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the skill data directory")
    parser.add_argument("--skills", nargs="+", required=True, help="List of skills to load")
    parser.add_argument("--obs_type", type=str, default="pixels", choices=["pixels", "features"],
                        help="Type of observations to use")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for testing (dataset batch size, not dataloader)")
    parser.add_argument("--history_len", type=int, default=10, help="History length for sequential data")
    parser.add_argument("--prompt", type=str, default="text",
                        choices=["text", "goal", "intermediate_goal"], help="Type of prompt to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num_test_batches", type=int, default=5,
                        help="Number of batches to test after the first one")

    args = parser.parse_args()
    test_dataset(args)