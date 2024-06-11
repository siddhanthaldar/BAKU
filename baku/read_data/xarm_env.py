import cv2
import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
from scipy.spatial.transform import Rotation as R


def get_relative_action(actions, action_after_steps):
    """
    Convert absolute axis angle actions to relative axis angle actions
    Action has both position and orientation. Convert to transformation matrix, get
    relative transformation matrix, convert back to axis angle
    """

    relative_actions = []
    for i in range(len(actions)):
        ####### Get relative transformation matrix #######
        # previous pose
        pos_prev = actions[i, :3]
        ori_prev = actions[i, 3:6]
        r_prev = R.from_rotvec(ori_prev).as_matrix()
        matrix_prev = np.eye(4)
        matrix_prev[:3, :3] = r_prev
        matrix_prev[:3, 3] = pos_prev
        # current pose
        next_idx = min(i + action_after_steps, len(actions) - 1)
        pos = actions[next_idx, :3]
        ori = actions[next_idx, 3:6]
        gripper = actions[next_idx, 6:]
        r = R.from_rotvec(ori).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = r
        matrix[:3, 3] = pos
        # relative transformation
        matrix_rel = np.linalg.inv(matrix_prev) @ matrix
        # relative pose
        pos_rel = pos - pos_prev
        r_rel = R.from_matrix(matrix_rel[:3, :3]).as_rotvec()
        # add to list
        relative_actions.append(np.concatenate([pos_rel, r_rel, gripper]))

    # last action
    last_action = np.zeros_like(actions[-1])
    last_action[-1] = actions[-1][-1]
    while len(relative_actions) < len(actions):
        relative_actions.append(last_action)
    return np.array(relative_actions, dtype=np.float32)


def get_quaternion_orientation(cartesian):
    """
    Get quaternion orientation from axis angle representation
    """
    new_cartesian = []
    for i in range(len(cartesian)):
        pos = cartesian[i, :3]
        ori = cartesian[i, 3:]
        quat = R.from_rotvec(ori).as_quat()
        new_cartesian.append(np.concatenate([pos, quat], axis=-1))
    return np.array(new_cartesian, dtype=np.float32)


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
        action_after_steps,
        intermediate_goal_step,
        store_actions,
        pixel_keys,
        subsample,
        skip_first_n,
        relative_actions,
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self._img_size = img_size
        self._action_after_steps = action_after_steps
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
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        min_stat, max_stat = None, None
        min_act, max_act = None, None
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = (
                data["observations"] if self._obs_type == "pixels" else data["states"]
            )
            task_emb = data["task_emb"]
            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                # compute actions
                # absolute actions
                actions = np.concatenate(
                    [
                        observations[i]["cartesian_states"],
                        observations[i]["gripper_states"][:, None],
                    ],
                    axis=1,
                )
                if len(actions) == 0:
                    continue
                # skip first n
                if skip_first_n is not None:
                    for key in observations[i].keys():
                        observations[i][key] = observations[i][key][skip_first_n:]
                    actions = actions[skip_first_n:]
                # subsample
                if subsample is not None:
                    for key in observations[i].keys():
                        observations[i][key] = observations[i][key][::subsample]
                    actions = actions[::subsample]
                # action after steps
                if relative_actions:
                    actions = get_relative_action(actions, self._action_after_steps)
                else:
                    actions = actions[self._action_after_steps :]
                # Convert cartesian states to quaternion orientation
                observations[i]["cartesian_states"] = get_quaternion_orientation(
                    observations[i]["cartesian_states"]
                )
                # Repeat last dimension of each observation for history_len times
                for key in observations[i].keys():
                    observations[i][key] = np.concatenate(
                        [
                            observations[i][key],
                            [observations[i][key][-1]] * self._history_len,
                        ],
                        axis=0,
                    )
                # Repeat last action for history_len times
                remaining_actions = actions[-1]
                if relative_actions:
                    pos = remaining_actions[:-1]
                    ori_gripper = remaining_actions[-1:]
                    remaining_actions = np.concatenate(
                        [np.zeros_like(pos), ori_gripper]
                    )
                actions = np.concatenate(
                    [
                        actions,
                        [remaining_actions] * self._history_len,
                    ],
                    axis=0,
                )

                # store
                episode = dict(
                    observation=observations[i],
                    action=actions,
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(observations[i])
                        if not isinstance(observations[i], dict)
                        else len(observations[i][self._pixel_keys[0]])
                    ),
                )
                self._max_state_dim = 7
                self._num_samples += (
                    len(observations[i])
                    if self._obs_type == "features"
                    else len(observations[i][self._pixel_keys[0]])
                )

                # max, min action
                if min_act is None:
                    min_act = np.min(actions, axis=0)
                    max_act = np.max(actions, axis=0)
                else:
                    min_act = np.minimum(min_act, np.min(actions, axis=0))
                    max_act = np.maximum(max_act, np.max(actions, axis=0))

                # store actions
                if self._store_actions:
                    self.actions.append(actions)

            # keep record of max and min stat
            max_cartesian = data["max_cartesian"]
            min_cartesian = data["min_cartesian"]
            max_cartesian = np.concatenate(
                [data["max_cartesian"][:3], [1] * 4]
            )  # for quaternion
            min_cartesian = np.concatenate(
                [data["min_cartesian"][:3], [-1] * 4]
            )  # for quaternion
            max_gripper = data["max_gripper"]
            min_gripper = data["min_gripper"]
            max_val = np.concatenate([max_cartesian, max_gripper[None]], axis=0)
            min_val = np.concatenate([min_cartesian, min_gripper[None]], axis=0)
            if max_stat is None:
                max_stat = max_val
                min_stat = min_val
            else:
                max_stat = np.maximum(max_stat, max_val)
                min_stat = np.minimum(min_stat, min_val)

        min_act[3:6], max_act[3:6] = 0, 1  # nullify action orientation normalization
        self.stats = {
            "actions": {
                "min": min_act,
                "max": max_act,
            },
            "proprioceptive": {
                "min": min_stat,
                "max": max_stat,
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
                transforms.RandomCrop(self._img_size, padding=4),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
            ]
        )

        # Samples from envs
        self.envs_till_idx = len(self._episodes)

    def _sample_episode(self, env_idx=None):
        idx = random.randint(0, self.envs_till_idx - 1) if env_idx is None else env_idx

        # sample idx with probability
        idx = np.random.choice(list(self._episodes.keys()))

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
                0, len(observations[self._pixel_keys[0]]) - self._history_len
            )
            sampled_pixel = {}
            for key in self._pixel_keys:
                sampled_pixel[key] = observations[key][
                    sample_idx : sample_idx + self._history_len
                ]
                sampled_pixel[key] = torch.stack(
                    [
                        self.aug(sampled_pixel[key][i])
                        for i in range(len(sampled_pixel[key]))
                    ]
                )
            sampled_proprioceptive_state = np.concatenate(
                [
                    observations["cartesian_states"][
                        sample_idx : sample_idx + self._history_len
                    ],
                    observations["gripper_states"][
                        sample_idx : sample_idx + self._history_len
                    ][:, None],
                ],
                axis=1,
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
                if len(actions) < sample_idx + num_actions:
                    act[len(actions) - sample_idx :] = actions[-1]
                sampled_action = np.lib.stride_tricks.sliding_window_view(
                    act, (self._num_queries, actions.shape[-1])
                )
                sampled_action = sampled_action[:, 0]
            else:
                sampled_action = actions[sample_idx : sample_idx + self._history_len]

            return_dict = {}
            for key in self._pixel_keys:
                return_dict[key] = sampled_pixel[key]
            return_dict["proprioceptive"] = self.preprocess["proprioceptive"](
                sampled_proprioceptive_state
            )
            return_dict["actions"] = self.preprocess["actions"](sampled_action)
            return_dict["task_emb"] = task_emb

            # prompt
            if self._prompt == "text":
                return return_dict
            elif self._prompt == "goal":
                prompt_episode = self._sample_episode(env_idx)
                prompt_observations = prompt_episode["observation"]
                # pixels
                for pixel_key in self._pixel_keys:
                    prompt_pixel = self.aug(prompt_observations[pixel_key][-1])[None]
                    return_dict["prompt_" + pixel_key] = prompt_pixel
                # proprio
                prompt_proprioceptive_state = np.concatenate(
                    [
                        prompt_observations["cartesian_states"][-1:],
                        prompt_observations["gripper_states"][-1:][:, None],
                    ],
                    axis=1,
                )
                return_dict["prompt_proprioceptive"] = self.preprocess[
                    "proprioceptive"
                ](prompt_proprioceptive_state)
                # actions
                prompt_action = prompt_episode["action"][-1:]
                return_dict["prompt_actions"] = self.preprocess["actions"](
                    prompt_action
                )
                return return_dict
            elif self._prompt == "intermediate_goal":
                prompt_episode = episodes
                prompt_observations = prompt_episode["observation"]
                intermediate_goal_step = (
                    self._intermediate_goal_step
                    + np.random.randint(
                        -self._intermediate_goal_step // 10 * 3,
                        self._intermediate_goal_step // 10 * 3,
                    )
                )
                goal_idx = min(
                    sample_idx + intermediate_goal_step,
                    len(prompt_observations[self._pixel_keys[0]]) - 1,
                )
                # pixels
                for pixel_key in self._pixel_keys:
                    prompt_pixel = self.aug(prompt_observations[pixel_key][goal_idx])[
                        None
                    ]
                    return_dict["prompt_" + pixel_key] = prompt_pixel
                prompt_proprioceptive_state = np.concatenate(
                    [
                        prompt_observations["cartesian_states"][
                            goal_idx : goal_idx + 1
                        ],
                        prompt_observations["gripper_states"][goal_idx : goal_idx + 1][
                            :, None
                        ],
                    ],
                    axis=1,
                )
                return_dict["prompt_proprioceptive"] = self.preprocess[
                    "proprioceptive"
                ](prompt_proprioceptive_state)
                # actions
                prompt_action = prompt_episode["action"][goal_idx : goal_idx + 1]
                return_dict["prompt_actions"] = self.preprocess["actions"](
                    prompt_action
                )
                return return_dict

        elif self._obs_type == "features":
            raise NotImplementedError

    def sample_test(self, env_idx, step=None):
        episode = self._sample_episode(env_idx)
        observations = episode["observation"]
        task_emb = episode["task_emb"]

        return_dict = {}
        return_dict["task_emb"] = task_emb

        if self._obs_type == "pixels":
            # observation
            if self._prompt == "text":
                for key in self._pixel_keys:
                    return_dict["prompt_" + key] = None
                return_dict["prompt_" + "proprioceptive"] = None
                return_dict["prompt_actions"] = None
            elif self._prompt == "goal":
                for key in self._pixel_keys:
                    prompt_pixel = np.transpose(observations[key][-1:], (0, 3, 1, 2))
                    return_dict["prompt_" + key] = prompt_pixel
                prompt_proprioceptive_state = np.concatenate(
                    [
                        observations["cartesian_states"][-1:],
                        observations["gripper_states"][-1:][:, None],
                    ],
                    axis=1,
                )
                return_dict["prompt_proprioceptive"] = self.preprocess[
                    "proprioceptive"
                ](prompt_proprioceptive_state)
                return_dict["prompt_actions"] = None
            elif self._prompt == "intermediate_goal":
                goal_idx = min(
                    step + self._intermediate_goal_step,
                    len(observations[self._pixel_keys[0]]) - 1,
                )
                for key in self._pixel_keys:
                    prompt_pixel = np.transpose(
                        observations[key][goal_idx : goal_idx + 1], (0, 3, 1, 2)
                    )
                    return_dict["prompt_" + key] = prompt_pixel
                prompt_proprioceptive_state = np.concatenate(
                    [
                        observations["cartesian_states"][goal_idx : goal_idx + 1],
                        observations["gripper_states"][goal_idx : goal_idx + 1][
                            :, None
                        ],
                    ],
                    axis=1,
                )
                return_dict["prompt_proprioceptive"] = self.preprocess[
                    "proprioceptive"
                ](prompt_proprioceptive_state)
                return_dict["prompt_actions"] = None

            return return_dict

        elif self._obs_type == "features":
            raise NotImplementedError

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples
