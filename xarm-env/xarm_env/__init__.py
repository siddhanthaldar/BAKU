from gym.envs.registration import register

register(
    id="Robot-v1",
    entry_point="xarm_env.envs:RobotEnv",
    max_episode_steps=400,
)
