import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset


class SkillDataset(IterableDataset):
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
        store_actions,
        intermediate_goal_step=50,
    ):
        print(f"Skills are {skills}")
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self.img_size = img_size
        self.intermediate_goal_step = intermediate_goal_step

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

        # Load data for each skill
        for skill_name, skill_path in self._paths.items():
            print(f"Loading skill data from {str(skill_path)}")
            with open(str(skill_path), "rb") as f:
                demonstrations = pkl.load(f)
            
            self._demonstrations[skill_name] = demonstrations
            
            # Update statistics
            for demo in demonstrations:
                episode_len = len(demo['actions'])
                self._max_episode_len = max(self._max_episode_len, episode_len)
                self._min_episode_len = min(self._min_episode_len, episode_len)
                self._max_state_dim = max(self._max_state_dim, demo['states'].shape[-1])
                self._num_samples += episode_len

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

    def _sample_demonstration(self, skill_name=None):
        if skill_name is None:
            skill_name = random.choice(list(self._demonstrations.keys()))
            demo = random.choice(self._demonstrations[skill_name])
            return demo, skill_name
        else:
            return random.choice(self._demonstrations[skill_name])

    def _sample(self):
        try:
            demo, skill_name = self._sample_demonstration()
            # print(f"Successfully sampled demonstration from skill: {skill_name}")
        except Exception as e:
            print(f"Error in _sample: {str(e)}")
            raise
        if self._obs_type == "pixels":
            observations = demo["observations"]
            actions = demo["actions"]
            task_emb = demo["task_emb"]

            # Sample observation frames
            max_start_idx = len(observations["pixels"]) - self._history_len
            if max_start_idx < 1:
                # Handle edge case where demonstration is shorter than history_len
                sample_idx = 0
                # Pad observations and actions if needed
                pad_len = self._history_len - len(observations["pixels"])
                if pad_len > 0:
                    # Pad with copies of the last frame
                    observations["pixels"] = np.pad(observations["pixels"], 
                        ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')
                    observations["pixels_egocentric"] = np.pad(observations["pixels_egocentric"],
                        ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')
                    observations["joint_states"] = np.pad(observations["joint_states"],
                        ((0, pad_len), (0, 0)), mode='edge')
                    observations["gripper_states"] = np.pad(observations["gripper_states"],
                        ((0, pad_len), (0, 0)), mode='edge')
                    actions = np.pad(actions, ((0, pad_len), (0, 0)), mode='edge')
            else:
                sample_idx = np.random.randint(0, max_start_idx)
            
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

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples
    

import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
import argparse


def test_dataset(args):
    print("\n=== Starting Dataset Test ===\n")

    # Initialize dataset
    print("Initializing dataset...")
    try:
        dataset = SkillDataset(
            path=args.data_path,
            suite='libero_90',
            skills=args.skills,
            obs_type=args.obs_type,
            history=True,
            history_len=args.history_len,
            prompt=args.prompt,
            temporal_agg=True,
            num_queries=10,
            store_actions=True,
            img_size=224,
        )
        print("✓ Dataset initialized successfully")
        print(f"Total number of skills loaded: {dataset.num_skills}")
        print(f"Max episode length: {dataset._max_episode_len}")
        print(f"Min episode length: {dataset._min_episode_len}")
        print(f"Total number of samples: {len(dataset)}")
    except Exception as e:
        print(f"ERROR during dataset initialization: {str(e)}")
        return

    # Test data loading
    print("\nTesting data loading...")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
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

        # Test a few more batches for consistency
        print("\nTesting multiple batch loads...")
        batch_times = []
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
            
        avg_time = np.mean(batch_times)
        std_time = np.std(batch_times)
        print(f"✓ Successfully loaded {args.num_test_batches} additional batches")
        print(f"  Average batch load time: {avg_time:.3f}s ± {std_time:.3f}s")

        # Memory usage check (rough estimate)
        if args.obs_type == "pixels":
            pixels_mem = batch["pixels"].element_size() * batch["pixels"].nelement()
            print(f"\nApproximate memory usage per batch:")
            print(f"  Pixels: {pixels_mem / 1024 / 1024:.2f} MB")

    except AssertionError as e:
        print(f"ERROR: Validation failed - {str(e)}")
    except Exception as e:
        print(f"ERROR during data loading: {str(e)}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Skill Dataset loader")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the skill data directory")
    parser.add_argument("--skills", nargs="+", required=True, help="List of skills to load")
    parser.add_argument("--obs_type", type=str, default="pixels", choices=["pixels", "features"], 
                        help="Type of observations to use")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size for testing")
    parser.add_argument("--history_len", type=int, default=10, help="History length for sequential data")
    parser.add_argument("--prompt", type=str, default="text", 
                        choices=["text", "goal", "intermediate_goal"], help="Type of prompt to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num_test_batches", type=int, default=5, 
                        help="Number of batches to test after the first one")

    args = parser.parse_args()
    test_dataset(args)