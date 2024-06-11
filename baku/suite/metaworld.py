from data_generation.mw_info import CAMERA
from data_generation.mw_info import NUM_STEPS as MAX_PATH_LENGTH

from collections import deque
from typing import Any, NamedTuple

import gym
from gym import Wrapper, spaces

import dm_env
import numpy as np
from dm_env import StepType, specs, TimeStep
from dm_control.utils import rewards

import cv2
import random
import metaworld

from sentence_transformers import SentenceTransformer

sentence_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class RGBArrayAsObservationWrapper(dm_env.Environment):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides

    From: https://github.com/hill-a/stable-baselines/issues/915
    """

    def __init__(
        self,
        env,
        name,
        ml1,
        width=84,
        height=84,
        max_path_length=125,
        max_state_dim=0,
    ):
        name = name.replace("_", " ")
        self.name = name

        self._env = env
        self.ml1 = ml1
        self._width = width
        self._height = height
        self.max_path_length = max_path_length
        self.max_state_dim = max_state_dim
        # dummy render to init opengl context
        dummy_obs = self.get_frame()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype
        )
        self.action_space = self._env.action_space

        # task emb
        self.task_emb = sentence_encoder.encode(name)

        # Action spec
        wrapped_action_spec = self.action_space
        if not hasattr(wrapped_action_spec, "minimum"):
            wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
        if not hasattr(wrapped_action_spec, "maximum"):
            wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            np.float32,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )
        # Observation spec
        self._obs_spec = {}
        self._obs_spec["pixels"] = specs.BoundedArray(
            shape=self.observation_space.shape,
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )
        self._obs_spec["features"] = specs.BoundedArray(
            shape=(self.max_state_dim,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="features",
        )

    def reset(self, **kwargs):
        # Set random goal
        task = random.choice(self.ml1.train_tasks)
        self._env.set_task(task)  # Set task

        # Set episode step to 0
        self.episode_step = 0

        obs = {}
        obs["features"] = np.zeros(self.max_state_dim, dtype=np.float32)
        state = self._env.reset(**kwargs)[0].astype(np.float32)
        obs["features"][: state.shape[0]] = state
        obs["pixels"] = self.get_frame()
        obs["task_emb"] = self.task_emb
        obs["goal_achieved"] = False
        return obs

    def step(self, action):
        observation, reward, _, done, info = self._env.step(action)
        obs = {}
        obs["features"] = observation.astype(np.float32)
        obs["pixels"] = self.get_frame()
        obs["task_emb"] = self.task_emb
        obs["goal_achieved"] = info["success"]
        self.episode_step += 1
        if self.episode_step == self.max_path_length:
            done = True
        return obs, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        if mode == "rgb_array":
            frame = self.get_frame(width, height)
            return frame
        else:
            self._env.render_mode = "human"
            self._env.render()

    def get_frame(self, width=None, height=None):
        width = self._width if width is None else width
        height = self._height if height is None else height
        self._env.mujoco_renderer.viewer.make_context_current()
        frame = self._env.render()[::-1]
        frame = cv2.resize(frame, (width, height))
        return frame

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

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)

        wrapped_obs_spec = env.observation_spec()["pixels"]

        pixels_shape = wrapped_obs_spec.shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = {}
        self._obs_spec["pixels"] = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )
        self._obs_spec["features"] = env.observation_spec()["features"]

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = {}
        obs["features"] = time_step.observation["features"]
        obs["pixels"] = np.concatenate(list(self._frames), axis=0)
        obs["task_emb"] = time_step.observation["task_emb"]
        obs["goal_achieved"] = time_step.observation["goal_achieved"]
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation["pixels"]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
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
        wrapped_action_spec = env.action_space
        if not hasattr(wrapped_action_spec, "minimum"):
            wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
        if not hasattr(wrapped_action_spec, "maximum"):
            wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            np.float32,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )
        # Observation spec
        self._obs_spec = env.observation_spec()

    def step(self, action):
        action = action.astype(self._env.action_space.dtype)
        # Make time step for action space
        observation, reward, done, info = self._env.step(action)
        reward = reward + 1
        step_type = StepType.LAST if done else StepType.MID
        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def reset(self):
        obs = self._env.reset()
        return TimeStep(
            step_type=StepType.FIRST, reward=0, discount=self._discount, observation=obs
        )

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
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
    idx2name = {}
    idx = 0

    for scene in scenes:
        name = tasks[scene]
        ml1 = metaworld.ML1(name)  # Construct the benchmark, sampling tasks
        env = ml1.train_classes[name](
            render_mode="rgb_array"
        )  # Create an environment with task
        env.seed(seed)

        # Set a random task to be able to use env
        task = random.choice(ml1.train_tasks)
        env.set_task(task)  # Set task
        env.camera_name = CAMERA[name]

        # add wrappers
        env = RGBArrayAsObservationWrapper(
            env,
            scene,
            ml1,
            width=width,
            height=height,
            max_path_length=max_episode_len,
            max_state_dim=max_state_dim,
        )
        env = ActionDTypeWrapper(env, np.float32)
        env = ActionRepeatWrapper(env, action_repeat)
        env = FrameStackWrapper(env, frame_stack)
        env = ExtendedTimeStepWrapper(env)

        envs.append(env)
        task_descriptions.append(scene)
        idx2name[idx] = scene
        idx += 1

        if not eval:
            break

    # write task descriptions to file
    if eval:
        with open("task_names_env.txt", "w") as f:
            for idx in idx2name:
                f.write(f"{idx}: {idx2name[idx]}\n")

    return envs, task_descriptions
