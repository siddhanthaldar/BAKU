from collections import deque
from typing import Any, NamedTuple

import gym
from gym import Wrapper, spaces
from gym.wrappers import FrameStack

import dm_env
import numpy as np
from dm_env import StepType, specs, TimeStep

import os
import cv2
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from sentence_transformers import SentenceTransformer

sentence_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class RGBArrayAsObservationWrapper(dm_env.Environment):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides

    From: https://github.com/hill-a/stable-baselines/issues/915
    """

    def __init__(
        self, env, width=84, height=84, max_episode_len=300, max_state_dim=100
    ):
        self._env = env
        self._width = width
        self._height = height
        self._max_episode_len = max_episode_len
        self._max_state_dim = max_state_dim
        obs = self._env.reset()
        dummy_obs = obs["agentview_image"]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype
        )

        # task emb
        self.task_emb = sentence_encoder.encode(self._env.language_instruction)

        # Action spec
        action_spec = self._env.env.action_spec
        self._action_spec = specs.BoundedArray(
            action_spec[0].shape, np.float32, action_spec[0], action_spec[1], "action"
        )
        # Observation spec
        robot_state = np.concatenate(
            [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        )
        self._obs_spec = {}
        self._obs_spec["pixels"] = specs.BoundedArray(
            shape=dummy_obs.shape, dtype=np.uint8, minimum=0, maximum=255, name="pixels"
        )
        self._obs_spec["pixels_egocentric"] = specs.BoundedArray(
            shape=dummy_obs.shape,
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="pixels_egocentric",
        )
        self._obs_spec["proprioceptive"] = specs.BoundedArray(
            shape=robot_state.shape,
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="proprioceptive",
        )
        self._obs_spec["features"] = specs.BoundedArray(
            shape=(self._max_state_dim,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="features",
        )

        self.render_image = None

    def reset(self, **kwargs):
        self._step = 0
        obs = self._env.reset(**kwargs)
        self.render_image = obs["agentview_image"][::-1, :]

        observation = {}
        observation["pixels"] = obs["agentview_image"][::-1, :]
        observation["pixels_egocentric"] = obs["robot0_eye_in_hand_image"][::-1, :]
        observation["proprioceptive"] = np.concatenate(
            [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        )
        # get state
        observation["features"] = np.zeros(self._max_state_dim)
        state = self._env.get_sim_state()  # TODO: Change to robot state
        observation["features"][: state.shape[0]] = state
        observation["task_emb"] = self.task_emb
        observation["goal_achieved"] = False
        return observation

    def step(self, action):
        self._step += 1
        obs, reward, done, info = self._env.step(action)
        self.render_image = obs["agentview_image"][::-1, :]

        observation = {}
        observation["pixels"] = obs["agentview_image"][::-1, :]
        observation["pixels_egocentric"] = obs["robot0_eye_in_hand_image"][::-1, :]
        observation["proprioceptive"] = np.concatenate(
            [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        )
        # get state
        observation["features"] = np.zeros(self._max_state_dim)
        state = self._env.get_sim_state()  # TODO: Change to robot state
        observation["features"][: state.shape[0]] = state
        observation["task_emb"] = self.task_emb
        observation["goal_achieved"] = done
        return observation, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        return cv2.resize(self.render_image, (width, height))

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._frames_egocentric = deque([], maxlen=num_frames)

        wrapped_obs_spec = env.observation_spec()["pixels"]

        pixels_shape = wrapped_obs_spec.shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = {}
        self._obs_spec["features"] = self._env.observation_spec()["features"]
        self._obs_spec["proprioceptive"] = self._env.observation_spec()[
            "proprioceptive"
        ]
        self._obs_spec["pixels"] = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="pixels",
        )
        self._obs_spec["pixels_egocentric"] = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="pixels_egocentric",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        assert len(self._frames_egocentric) == self._num_frames
        obs = {}
        obs["features"] = time_step.observation["features"]
        obs["pixels"] = np.concatenate(list(self._frames), axis=0)
        obs["pixels_egocentric"] = np.concatenate(list(self._frames_egocentric), axis=0)
        obs["proprioceptive"] = time_step.observation["proprioceptive"]
        obs["task_emb"] = time_step.observation["task_emb"]
        obs["goal_achieved"] = time_step.observation["goal_achieved"]
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation["pixels"]
        pixels_egocentric = time_step.observation["pixels_egocentric"]

        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        if len(pixels_egocentric.shape) == 4:
            pixels_egocentric = pixels_egocentric[0]
        return (
            pixels.transpose(2, 0, 1).copy(),
            pixels_egocentric.transpose(2, 0, 1).copy(),
        )

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        pixels, pixels_egocentric = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
            self._frames_egocentric.append(pixels_egocentric)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels, pixels_egocentric = self._extract_pixels(time_step)
        self._frames.append(pixels)
        self._frames_egocentric.append(pixels_egocentric)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0

        # Action spec
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            np.float32,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        # Make time step for action space
        observation, reward, done, info = self._env.step(action)
        step_type = (
            StepType.LAST
            if (
                self._env._step == self._env._max_episode_len
                or observation["goal_achieved"]
            )
            else StepType.MID
        )
        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return TimeStep(
            step_type=StepType.FIRST, reward=0, discount=self._discount, observation=obs
        )

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def _replace(
        self, time_step, observation=None, action=None, reward=None, discount=None
    ):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount
        return ExtendedTimeStep(
            observation=observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=discount,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(
    suite,
    scenes,
    tasks,
    frame_stack,
    action_repeat,
    seed,
    height,
    width,
    max_episode_len,
    max_state_dim,
    eval,
):
    # Convert task_names, which is a list, to a dictionary
    tasks = {task_name: scene[task_name] for scene in tasks for task_name in scene}

    envs = []
    task_descriptions = []
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()
    idx2name = {}
    idx = 0
    for scene in scenes:
        for task_name in tasks[scene]:
            if task_name in task_suite.get_task_names():
                # get task id from list of task names
                task_id = task_suite.get_task_names().index(task_name)
                # create environment
                task = task_suite.get_task(task_id)
                task_name = task.name
                task_descriptions.append(task.language)
                task_bddl_file = os.path.join(
                    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
                )
                env_args = {
                    "bddl_file_name": task_bddl_file,
                    "camera_heights": 128,
                    "camera_widths": 128,
                }
                env = OffScreenRenderEnv(**env_args)
                env.seed(seed)
                print(f"Initialized environment: {task_name}")

                # apply wrappers
                env = RGBArrayAsObservationWrapper(
                    env,
                    height=height,
                    width=width,
                    max_episode_len=max_episode_len,
                    max_state_dim=max_state_dim,
                )
                env = ActionDTypeWrapper(env, np.float32)
                env = ActionRepeatWrapper(env, action_repeat)
                env = FrameStackWrapper(env, frame_stack)
                env = ExtendedTimeStepWrapper(env)

                envs.append(env)

                idx2name[idx] = task_name
                idx += 1
            else:
                for task_id in range(task_suite.get_num_tasks()):
                    task = task_suite.get_task(task_id)
                    task_name = task.name
                    task_descriptions.append(task.language)
                    task_bddl_file = os.path.join(
                        get_libero_path("bddl_files"),
                        task.problem_folder,
                        task.bddl_file,
                    )
                    env_args = {
                        "bddl_file_name": task_bddl_file,
                        "camera_heights": 128,
                        "camera_widths": 128,
                    }
                    env = OffScreenRenderEnv(**env_args)
                    env.seed(seed)

                    # apply wrappers
                    env = RGBArrayAsObservationWrapper(
                        env,
                        height=height,
                        width=width,
                        max_episode_len=max_episode_len,
                        max_state_dim=max_state_dim,
                    )
                    env = ActionDTypeWrapper(env, np.float32)
                    env = ActionRepeatWrapper(env, action_repeat)
                    env = FrameStackWrapper(env, frame_stack)
                    env = ExtendedTimeStepWrapper(env)

                    envs.append(env)

                    if not eval:
                        break

            if not eval:
                break
        if not eval:
            break

        # write task descriptions to file
        if eval:
            with open("task_names_env.txt", "w") as f:
                for idx in idx2name:
                    f.write(f"{idx}: {idx2name[idx]}\n")

    return envs, task_descriptions
