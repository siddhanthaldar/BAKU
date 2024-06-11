import cv2
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
        tasks,
        num_demos_per_task,
        obs_type,
        history,
        history_len,
        prompt,
        temporal_agg,
        num_queries,
        img_size,
        intermediate_goal_step=30,
        store_actions=False,
        pixel_keys=["pixels"],
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self._img_size = img_size
        self._intermediate_goal_step = intermediate_goal_step
        self._store_actions = store_actions
        self._pixel_keys = pixel_keys

        # temporal aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # get data paths
        self._paths = []
        self._paths.extend([Path(path) / f"{task}.pkl" for task in tasks])

        paths = {}
        idx = 0
        for path in self._paths:
            paths[idx] = path
            idx += 1
        del self._paths
        self._paths = paths

        # store actions
        if self._store_actions:
            self.actions = []

        # read data
        self._episodes = {}
        self._max_episode_len = 1000
        self._max_state_dim = 0
        self._max_action_dim = 0
        self._num_samples = 0
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = (
                data["observations"] if self._obs_type == "pixels" else data["states"]
            )
            actions = np.array(data["actions"])
            task_emb = data["task_emb"]
            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                # store
                episode = dict(
                    observation=observations[i],
                    action=actions[i],
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._max_state_dim = max(
                    self._max_state_dim, data["states"][i].shape[-1]
                )
                self._max_action_dim = max(self._max_action_dim, actions[i].shape[-1])
                self._num_samples += (
                    len(observations[i])
                    if self._obs_type == "features"
                    else len(observations[i][self._pixel_keys[0]])
                )

        # Store actions projected to max action dim
        if self._store_actions:
            for _path_idx in self._paths:
                for episode in self._episodes[_path_idx]:
                    actions = episode["action"]
                    new_actions = np.zeros(
                        (actions.shape[0], self._max_action_dim)
                    ).astype(np.float32)
                    # repeat actions to max_action_dim
                    repeat = self._max_action_dim // actions.shape[-1]
                    new_actions[:, : actions.shape[-1] * repeat] = np.tile(
                        actions, (1, repeat)
                    )
                    episode["action"] = new_actions

                    self.actions.append(new_actions)

        self.stats = {
            "actions": {
                "min": 0,
                "max": 1,
            },
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
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
            NotImplementedError

        elif self._obs_type == "features":
            # Sample obs, action
            sample_idx = np.random.randint(0, len(observations) - self._history_len)

            feat = observations[sample_idx : sample_idx + self._history_len]
            sampled_feature = np.zeros((self._history_len, self._max_state_dim))
            # Repeat feat to max_state_dim
            repeat = self._max_state_dim // feat[0].shape[-1]
            sampled_feature[:, : feat[0].shape[-1] * repeat] = np.tile(
                feat, (1, repeat)
            )

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

            return_dict = {}
            return_dict["features"] = sampled_feature
            return_dict["actions"] = self.preprocess["actions"](sampled_action)
            return_dict["task_emb"] = task_emb

            if self._prompt == None or self._prompt == "text":
                return return_dict
            elif self._prompt == "goal":
                NotImplementedError
            elif self._prompt == "intermediate_goal":
                NotImplementedError

    def sample_test(self, env_idx, step=None):
        episode = self._sample_episode(env_idx)
        task_emb = episode["task_emb"]

        if self._obs_type == "pixels":
            NotImplementedError

        elif self._obs_type == "features":
            # observation
            if self._prompt == None or self._prompt == "text":
                prompt_feature = None
                prompt_action = None
            elif self._prompt == "goal":
                NotImplementedError
            elif self._prompt == "intermediate_goal":
                NotImplementedError

            return {
                "prompt_features": prompt_feature,
                "prompt_actions": prompt_action,
                "task_emb": task_emb,
            }

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples
