import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset


class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        suite,
        scenes,
        tasks,
        num_demos_per_task,
        obs_type,
        history,
        history_len,
        prompt,
        temporal_agg,
        num_queries,
        img_size,
        subsample,
        intermediate_goal_step=50,
        store_actions=False,
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self.subsample = subsample
        self.img_size = img_size
        self.intermediate_goal_step = intermediate_goal_step

        # intermediate steps for inverse model
        self.inverse = True
        self.inverse_step_gap = 50  # 5
        self.num_intermediate_steps = 1  # 7

        # temporal_aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # Convert task_names, which is a list, to a dictionary
        tasks = {task_name: scene[task_name] for scene in tasks for task_name in scene}

        # Get relevant task names
        task_name = []
        for scene in scenes:
            task_name.extend([task_name for task_name in tasks[scene]])

        # get data paths
        self._paths = []
        # for suite in suites:
        self._paths.extend(list((Path(path) / suite).glob("*")))

        if task_name is not None:
            paths = {}
            idx2name = {}
            for path in self._paths:
                task = str(path).split(".")[0].split("/")[-1]
                if task in task_name:
                    # get idx of task in task_name
                    idx = task_name.index(task)
                    paths[idx] = path
                    idx2name[idx] = task
            del self._paths
            self._paths = paths

        # store actions
        if store_actions:
            self.actions = []

        # read data
        self._episodes = {}
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = (
                data["observations"] if self._obs_type == "pixels" else data["states"]
            )
            actions = data["actions"]
            task_emb = data["task_emb"]
            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                episode = dict(
                    observation=observations[i],
                    action=actions[i],
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(observations[i])
                        if not isinstance(observations[i], dict)
                        else len(observations[i]["pixels"])
                    ),
                )
                # if obs_type == 'features':
                self._max_state_dim = max(
                    self._max_state_dim, data["states"][i].shape[-1]
                )
                self._num_samples += (
                    len(observations[i])
                    if self._obs_type == "features"
                    else len(observations[i]["pixels"])
                )

                # store actions
                if store_actions:
                    self.actions.append(actions[i])

        self.stats = {
            "actions": {
                "min": 0,
                "max": 1,
            },
            "proprioceptive": {
                "min": 0,
                "max": 1,
            },
        }
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

        # augmentation
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        # Samples from envs
        self.envs_till_idx = len(self._episodes)

    def _sample_episode(self, env_idx=None):
        idx = random.randint(0, self.envs_till_idx - 1) if env_idx is None else env_idx
        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        episodes, env_idx = self._sample_episode()
        observations = episodes["observation"]
        actions = episodes["action"]
        task_emb = episodes["task_emb"]

        if self._obs_type == "pixels":
            # Sample obs, action
            sample_idx = np.random.randint(
                0, len(observations["pixels"]) - self._history_len
            )
            sampled_pixel = observations["pixels"][
                sample_idx : sample_idx + self._history_len
            ]
            sampled_pixel_egocentric = observations["pixels_egocentric"][
                sample_idx : sample_idx + self._history_len
            ]
            sampled_pixel = torch.stack(
                [self.aug(sampled_pixel[i]) for i in range(len(sampled_pixel))]
            )
            sampled_pixel_egocentric = torch.stack(
                [
                    self.aug(sampled_pixel_egocentric[i])
                    for i in range(len(sampled_pixel_egocentric))
                ]
            )
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
            if self._temporal_agg:
                # arrange sampled action to be of shape (history_len, num_queries, action_dim)
                sampled_action = np.zeros(
                    (self._history_len, self._num_queries, actions.shape[-1])
                )
                num_actions = (
                    self._history_len + self._num_queries - 1
                )  # -1 since its num_queries including the last action of the history
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

            # intermediate steps for inverse model
            if self.inverse:
                intermediate_frames = []
                for i in range(1, self.num_intermediate_steps + 1):
                    idx = sample_idx + i * self.inverse_step_gap
                    idx = min(idx, len(observations["pixels"]) - 1)
                    intermediate_frames.append(self.aug(observations["pixels"][idx]))
                intermediate_frames = torch.stack(intermediate_frames)

            # prompt
            if self._prompt == "text":
                return {
                    "pixels": sampled_pixel,
                    "pixels_egocentric": sampled_pixel_egocentric,
                    "proprioceptive": self.preprocess["proprioceptive"](
                        sampled_proprioceptive_state
                    ),
                    "actions": self.preprocess["actions"](sampled_action),
                    "intermediate_frames": intermediate_frames
                    if self.inverse
                    else None,
                    "task_emb": task_emb,
                }
            elif self._prompt == "goal":
                prompt_episode = self._sample_episode(env_idx)
                prompt_observations = prompt_episode["observation"]
                prompt_pixel = self.aug(prompt_observations["pixels"][-1])[None]
                prompt_pixel_egocentric = self.aug(
                    prompt_observations["pixels_egocentric"][-1]
                )[None]
                prompt_proprioceptive_state = np.concatenate(
                    [
                        prompt_observations["joint_states"][-1:],
                        prompt_observations["gripper_states"][-1:],
                    ],
                    axis=-1,
                )
                prompt_action = prompt_episode["action"][-1:]
                return {
                    "pixels": sampled_pixel,
                    "pixels_egocentric": sampled_pixel_egocentric,
                    "proprioceptive": self.preprocess["proprioceptive"](
                        sampled_proprioceptive_state
                    ),
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_pixels": prompt_pixel,
                    "prompt_pixels_egocentric": prompt_pixel_egocentric,
                    "prompt_proprioceptive": self.preprocess["proprioceptive"](
                        prompt_proprioceptive_state
                    ),
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "intermediate_frames": intermediate_frames
                    if self.inverse
                    else None,
                    "task_emb": task_emb,
                }
            elif self._prompt == "intermediate_goal":
                prompt_episode = episodes
                prompt_observations = prompt_episode["observation"]
                intermediate_goal_step = (
                    self.intermediate_goal_step + np.random.randint(-30, 30)
                )
                goal_idx = min(
                    sample_idx + intermediate_goal_step,
                    len(prompt_observations["pixels"]) - 1,
                )
                prompt_pixel = self.aug(prompt_observations["pixels"][goal_idx])[None]
                prompt_pixel_egocentric = self.aug(
                    prompt_observations["pixels_egocentric"][goal_idx]
                )[None]
                prompt_proprioceptive_state = np.concatenate(
                    [
                        prompt_observations["joint_states"][goal_idx : goal_idx + 1],
                        prompt_observations["gripper_states"][goal_idx : goal_idx + 1],
                    ],
                    axis=-1,
                )
                prompt_action = prompt_episode["action"][goal_idx : goal_idx + 1]
                return {
                    "pixels": sampled_pixel,
                    "pixels_egocentric": sampled_pixel_egocentric,
                    "proprioceptive": self.preprocess["proprioceptive"](
                        sampled_proprioceptive_state
                    ),
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_pixels": prompt_pixel,
                    "prompt_pixels_egocentric": prompt_pixel_egocentric,
                    "prompt_proprioceptive": self.preprocess["proprioceptive"](
                        prompt_proprioceptive_state
                    ),
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "intermediate_frames": intermediate_frames
                    if self.inverse
                    else None,
                    "task_emb": task_emb,
                }

        elif self._obs_type == "features":
            # Sample obs, action
            sample_idx = np.random.randint(0, len(observations) - self._history_len)
            sampled_obs = np.array(
                observations[sample_idx : sample_idx + self._history_len]
            )
            sampled_action = actions[sample_idx : sample_idx + self._history_len]
            # pad obs to match self._max_state_dim
            obs = np.zeros((self._history_len, self._max_state_dim))
            state_dim = sampled_obs.shape[-1]
            obs[:, :state_dim] = sampled_obs
            sampled_obs = obs

            # prompt obs, action
            if self._prompt == "text":
                return {
                    "features": sampled_obs,
                    "actions": self.preprocess["actions"](sampled_action),
                    "task_emb": task_emb,
                }
            elif self._prompt == "goal":
                prompt_episode = self._sample_episode(env_idx)
                prompt_obs = np.array(prompt_episode["observation"][-1:])
                prompt_action = prompt_episode["action"][-1:]
                return {
                    "features": sampled_obs,
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_obs": prompt_obs,
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "task_emb": task_emb,
                }
            elif self._prompt == "intermediate_goal":
                prompt_episode = self._sample_episode(env_idx)
                goal_idx = min(
                    sample_idx + self.intermediate_goal_step,
                    len(prompt_episode["observation"]) - 1,
                )
                prompt_obs = np.array(
                    prompt_episode["observation"][goal_idx : goal_idx + 1]
                )
                prompt_action = prompt_episode["action"][goal_idx : goal_idx + 1]
                return {
                    "features": sampled_obs,
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_obs": prompt_obs,
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "task_emb": task_emb,
                }

    def sample_test(self, env_idx, step=None):
        episode = self._sample_episode(env_idx)
        observations = episode["observation"]
        actions = episode["action"]
        task_emb = episode["task_emb"]

        if self._obs_type == "pixels":
            pixels_shape = observations["pixels"].shape

            # observation
            if self._prompt == None or self._prompt == "text":
                prompt_pixel = None
                prompt_pixel_egocentric = None
                prompt_proprioceptive_state = None
                prompt_action = None
            elif self._prompt == "goal":
                prompt_pixel = np.transpose(observations["pixels"][-1:], (0, 3, 1, 2))
                prompt_pixel_egocentric = np.transpose(
                    observations["pixels_egocentric"][-1:], (0, 3, 1, 2)
                )
                prompt_proprioceptive_state = np.concatenate(
                    [
                        observations["joint_states"][-1:],
                        observations["gripper_states"][-1:],
                    ],
                    axis=-1,
                )
                prompt_action = None
            elif self._prompt == "intermediate_goal":
                goal_idx = min(
                    step + self.intermediate_goal_step, len(observations["pixels"]) - 1
                )
                prompt_pixel = np.transpose(
                    observations["pixels"][goal_idx : goal_idx + 1], (0, 3, 1, 2)
                )
                prompt_pixel_egocentric = np.transpose(
                    observations["pixels_egocentric"][goal_idx : goal_idx + 1],
                    (0, 3, 1, 2),
                )
                prompt_proprioceptive_state = np.concatenate(
                    [
                        observations["joint_states"][goal_idx : goal_idx + 1],
                        observations["gripper_states"][goal_idx : goal_idx + 1],
                    ],
                    axis=-1,
                )
                prompt_action = None

            return {
                "prompt_pixels": prompt_pixel,
                "prompt_pixels_egocentric": prompt_pixel_egocentric,
                "prompt_proprioceptive": (
                    self.preprocess["proprioceptive"](prompt_proprioceptive_state)
                    if prompt_proprioceptive_state is not None
                    else None
                ),
                "prompt_actions": (
                    self.preprocess["actions"](prompt_action)
                    if prompt_action is not None
                    else None
                ),
                "task_emb": task_emb,
            }

        elif self._obs_type == "features":
            # observation
            if self._prompt == None or self._prompt == "text":
                prompt_obs, prompt_action = None, None
            elif self._prompt == "goal":
                prompt_obs = np.array(observations[-1:])
                prompt_action = None

            return {
                "prompt_features": prompt_obs,
                "prompt_actions": self.preprocess["actions"](prompt_action),
                "task_emb": task_emb,
            }

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples
