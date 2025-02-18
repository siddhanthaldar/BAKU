import random
import numpy as np
import pickle as pkl
from pathlib import Path
from typing import Dict, Optional, Union, List
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
        batch_size: int = 32,
        temperature: float = 3.5,  # Added temperature parameter
        weights: Optional[Dict[str, float]] = None,
        intermediate_goal_step=50,
        store_actions: bool=False,
        validation_split: float = 0.1  # Added validation split parameter
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self.img_size = img_size
        self.intermediate_goal_step = intermediate_goal_step
        self.batch_size = batch_size
        self.temperature = temperature  # Store temperature

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

        # Initialize sampling weights and probabilities
        self._initialize_weights(weights)
        
        # Calculate number of batches
        self.n_batches = self._num_samples // batch_size
        if self._num_samples % batch_size != 0:
            self.n_batches += 1

        # Calculate number of samples per skill in a batch
        # self._calculate_batch_distribution()

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

        # Create validation set
        self._validation_demos = {}
        self._train_demos = {}
        
        for skill_name, demonstrations in self._demonstrations.items():
            # Determine split index
            n_demos = len(demonstrations)
            n_val = max(1, int(n_demos * validation_split))
            
            # Create validation set (fixed seed for reproducibility)
            rng = np.random.RandomState(42)
            val_indices = rng.choice(n_demos, n_val, replace=False)
            train_indices = np.array([i for i in range(n_demos) if i not in val_indices])
            
            self._validation_demos[skill_name] = [demonstrations[i] for i in val_indices]
            self._train_demos[skill_name] = [demonstrations[i] for i in train_indices]
            
            print(f"Skill {skill_name}: {len(self._train_demos[skill_name])} train, {len(self._validation_demos[skill_name])} validation demonstrations")
        
        # Use training demos as default
        self._demonstrations = self._train_demos

    def _compute_sampling_probs(self) -> Dict[str, float]:
        """
        Compute temperature-adjusted sampling probabilities.
        p(i;τ) = |Di|^(1/τ) / Σ_j |Dj|^(1/τ)
        """
        # Convert sizes to numpy array for vectorized operations
        skills = list(self._skill_sizes.keys())
        sizes = np.array([self._skill_sizes[skill] for skill in skills])
        
        # Apply temperature scaling
        temp_adjusted = np.power(sizes, 1/self.temperature)
        probs = temp_adjusted / temp_adjusted.sum()
        
        # Convert back to dictionary
        return dict(zip(skills, probs))
    
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
    

    def _initialize_weights(self, weights: Optional[Dict[str, float]] = None):
        """Initialize sampling weights and compute temperature-adjusted probabilities."""
        if weights is None:
            # If no weights provided, use dataset sizes
            self._base_weights = {
                skill: size/sum(self._skill_sizes.values()) 
                for skill, size in self._skill_sizes.items()
            }
        else:
            # Normalize provided weights
            total_weight = sum(weights.values())
            self._base_weights = {
                skill: weight/total_weight 
                for skill, weight in weights.items()
            }
            
            # Verify all skills have weights
            missing_skills = set(self._demonstrations.keys()) - set(weights.keys())
            if missing_skills:
                raise ValueError(f"Missing weights for skills: {missing_skills}")

        # Compute temperature-adjusted probabilities
        self._sampling_probs = self._compute_sampling_probs()

    def sample_batch_distribution(self) -> Dict[str, int]:
        """
        Sample batch sizes for each skill based on temperature-adjusted probabilities.
        """
        skills = list(self._sampling_probs.keys())
        probs = np.array([self._sampling_probs[skill] for skill in skills])
        
        # Use multinomial distribution to sample batch sizes
        counts = np.random.multinomial(self.batch_size, probs)
        return dict(zip(skills, counts))

    def set_temperature(self, new_temp: float):
        """Update the sampling temperature and recompute probabilities."""
        self.temperature = new_temp
        self._sampling_probs = self._compute_sampling_probs()

    def __iter__(self):
        while True:
            # Sample batch distribution based on temperature-adjusted probabilities
            batch_distribution = self.sample_batch_distribution()
            batch_samples = []
            
            # Sample according to batch distribution
            for skill, count in batch_distribution.items():
                attempts = 0
                max_attempts = count * 2  # Limit retries per skill
                
                while len([s for s in batch_samples if s['skill_name'] == skill]) < count and attempts < max_attempts:
                    demo = self._sample_demonstration(skill)
                    sample = self._process_sample(demo, skill)
                    
                    if sample is not None:
                        batch_samples.append(sample)
                    attempts += 1
            
            # Skip empty batches
            if not batch_samples:
                continue
                
            # Shuffle samples within the batch
            random.shuffle(batch_samples)
            
            # Yield samples
            for sample in batch_samples:
                yield sample

    # def get_weights(self) -> Dict[str, float]:
    #    """Return current sampling weights."""
    #    return self._weights.copy()


    def update_weights(self, new_weights: Dict[str, float]):
        """Update base weights and recompute sampling probabilities."""
        self._initialize_weights(new_weights)

    def get_sampling_probs(self) -> Dict[str, float]:
        """Return current temperature-adjusted sampling probabilities."""
        return self._sampling_probs.copy()

    def get_base_weights(self) -> Dict[str, float]:
        """Return current base weights before temperature adjustment."""
        return self._base_weights.copy()

    def get_validation_batch(self, batch_size: Optional[int] = None) -> List[Dict]:
        """Get a batch of validation samples."""
        if batch_size is None:
            batch_size = self.batch_size
            
        batch_samples = []
        batch_distribution = self.sample_batch_distribution()
        
        for skill, count in batch_distribution.items():
            if not self._validation_demos[skill]:
                continue
                
            for _ in range(count):
                demo = random.choice(self._validation_demos[skill])
                sample = self._process_sample(demo, skill)
                if sample is not None:
                    batch_samples.append(sample)
                    
                if len(batch_samples) >= batch_size:
                    break
                    
            if len(batch_samples) >= batch_size:
                break
                
        return batch_samples[:batch_size]

    def get_validation_batch_per_skill(self, batch_size_per_skill: int = 32) -> Dict[str, List[Dict]]:
        """Get validation batches separately for each skill.
        
        Args:
            batch_size_per_skill: Number of samples to get for each skill
            
        Returns:
            Dict mapping skill names to lists of validation samples
        """
        validation_batches = {}
        
        for skill_name, demos in self._validation_demos.items():
            skill_samples = []
            attempts = 0
            max_attempts = batch_size_per_skill * 2
            
            while len(skill_samples) < batch_size_per_skill and attempts < max_attempts:
                demo = random.choice(demos)
                sample = self._process_sample(demo, skill_name)
                if sample is not None:
                    skill_samples.append(sample)
                attempts += 1
            
            if skill_samples:  # Only include skills with valid samples
                validation_batches[skill_name] = skill_samples
                
        return validation_batches

    def get_skill_names(self) -> List[str]:
        """Return list of all skill names."""
        return list(self._demonstrations.keys())


def test_dataset(args):
    print("\n=== Starting Temperature-based WeightedSkillDataset Test ===\n")

    # Initialize dataset
    print("Initializing dataset...")
    try:
        dataset = WeightedSkillDataset(
            path=args.data_path,
            suite=args.suite,
            skills=args.skills,
            obs_type=args.obs_type,
            history=True,
            history_len=args.history_len,
            prompt=args.prompt,
            temporal_agg=True,
            num_queries=10,
            img_size=224,
            batch_size=args.batch_size,
            temperature=args.temperature,  # Added temperature parameter
        )
        print("✓ Dataset initialized successfully")
        print(f"Total number of skills loaded: {dataset.num_skills}")
        print(f"Max episode length: {dataset._max_episode_len}")
        print(f"Min episode length: {dataset._min_episode_len}")
        print(f"Total number of samples: {len(dataset)}")
        print(f"Initial base weights: {dataset.get_base_weights()}")
        print(f"Temperature-adjusted probabilities: {dataset.get_sampling_probs()}")

    except Exception as e:
        print(f"ERROR during dataset initialization: {str(e)}")
        return

    # Test data loading
    print("\nTesting data loading...")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
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

        # Test sampling distribution at different temperatures
        print("\nTesting sampling distribution at different temperatures...")
        temperatures = [0.1, 1.0, 5.0, 10.0]  # Test various temperatures
        for temp in temperatures:
            print(f"\nTesting temperature τ = {temp}")
            dataset.set_temperature(temp)
            print(f"Sampling probabilities at τ = {temp}:")
            print(dataset.get_sampling_probs())
            
            # Collect samples to verify distribution
            skill_counts = defaultdict(int)
            batch_times = []
            
            for i in range(args.num_test_batches):
                start_time = time.time()
                batch = next(iterator)
                batch_times.append(time.time() - start_time)
                skill_counts[batch['skill_name'][0]] += 1

                # Basic validation
                if args.obs_type == "pixels":
                    assert "pixels" in batch, "Missing pixels in batch"
                    assert "pixels_egocentric" in batch, "Missing egocentric pixels in batch"
                else:
                    assert "features" in batch, "Missing features in batch"

                assert "actions" in batch, "Missing actions in batch"
                assert "task_emb" in batch, "Missing task embeddings in batch"

            avg_time = np.mean(batch_times)
            std_time = np.std(batch_times)
            print(f"✓ Successfully loaded {args.num_test_batches} batches")
            print(f"  Average batch load time: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"  Observed skill distribution: {dict(skill_counts)}")

        # Memory usage check (rough estimate)
        if args.obs_type == "pixels":
            pixels_mem = batch["pixels"].element_size() * batch["pixels"].nelement()
            print(f"\nApproximate memory usage per batch:")
            print(f"  Pixels: {pixels_mem / 1024 / 1024:.2f} MB")

        # Test weight updating with temperature
        print("\nTesting weight updating with temperature...")
        initial_probs = dataset.get_sampling_probs()
        
        # Test updating base weights
        new_weights = {
            'rotating': 0.5,
            'picking': 0.25,
            'placing': 0.25
        }
        dataset.update_weights(new_weights)
        
        print("After updating base weights:")
        print(f"  New base weights: {dataset.get_base_weights()}")
        print(f"  New sampling probabilities: {dataset.get_sampling_probs()}")
        
        # Test changing temperature with new weights
        new_temp = 2.0
        dataset.set_temperature(new_temp)
        print(f"\nAfter changing temperature to {new_temp}:")
        print(f"  Updated sampling probabilities: {dataset.get_sampling_probs()}")

    except AssertionError as e:
        print(f"ERROR: Validation failed - {str(e)}")
    except Exception as e:
        print(f"ERROR during data loading: {str(e)}")

    print("\n=== Temperature-based WeightedSkillDataset Test Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Temperature-based WeightedSkillDataset loader")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the skill data directory")
    parser.add_argument("--skills", nargs="+", required=True, help="List of skills to load")
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--obs_type", type=str, default="pixels", choices=["pixels", "features"],
                        help="Type of observations to use")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for testing")
    parser.add_argument("--history_len", type=int, default=10, help="History length for sequential data")
    parser.add_argument("--prompt", type=str, default="text",
                        choices=["text", "goal", "intermediate_goal"], help="Type of prompt to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num_test_batches", type=int, default=5,
                        help="Number of batches to test after the first one")
    parser.add_argument("--temperature", type=float, default=5.0,
                        help="Initial sampling temperature (default: 5.0)")

    args = parser.parse_args()
    test_dataset(args)