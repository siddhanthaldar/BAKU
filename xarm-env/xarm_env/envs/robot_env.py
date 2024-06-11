import gym
from gym import spaces
import cv2
import numpy as np

# import pybullet
# import pybullet_data
import pickle
from scipy.spatial.transform import Rotation as R

from openteach.utils.network import create_request_socket, ZMQCameraSubscriber
from xarm_env.envs.constants import *


def get_quaternion_orientation(cartesian):
    """
    Get quaternion orientation from axis angle representation
    """
    pos = cartesian[:3]
    ori = cartesian[3:]
    r = R.from_rotvec(ori)
    quat = r.as_quat()
    return np.concatenate([pos, quat], axis=-1)


class RobotEnv(gym.Env):
    def __init__(
        self,
        height=224,
        width=224,
        use_robot=True,  # True when robot used
        use_egocentric=False,  # True when egocentric camera used
    ):
        super(RobotEnv, self).__init__()
        self.height = height
        self.width = width
        self.use_robot = use_robot
        self.feature_dim = 8
        self.action_dim = 7

        self.n_channels = 3
        self.reward = 0

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        if self.use_robot:
            # camera subscribers
            self.image_subscribers = []
            for cam_idx in list(CAM_SERIAL_NUMS.keys()):
                port = CAMERA_PORT_OFFSET + cam_idx
                self.image_subscribers.append(
                    ZMQCameraSubscriber(
                        host=HOST_ADDRESS,
                        port=port,
                        topic_type="RGB",
                    )
                )

            for fish_eye_cam_idx in list(FISH_EYE_CAM_SERIAL_NUMS.keys()):
                port = FISH_EYE_CAMERA_PORT_OFFSET + fish_eye_cam_idx
                self.image_subscribers.append(
                    ZMQCameraSubscriber(
                        host=HOST_ADDRESS,
                        port=port,
                        topic_type="RGB",
                    )
                )

            # action request port
            self.action_request_socket = create_request_socket(
                HOST_ADDRESS, DEPLOYMENT_PORT
            )

    def step(self, action):
        print("current step's action is: ", action)

        action = np.array(action)

        action_dict = {
            "xarm": {
                "cartesian": action[:-1],
                "gripper": action[-1:],
            }
        }

        # send action
        self.action_request_socket.send(pickle.dumps(action_dict, protocol=-1))
        ret = self.action_request_socket.recv()
        ret = pickle.loads(ret)
        if ret == "Command failed!":
            print("Command failed!")
            self.action_request_socket.send(b"get_state")
            robot_state = pickle.loads(self.action_request_socket.recv())["xarm"]
        else:
            robot_state = ret["xarm"]

        cartesian = robot_state[:6]
        quat_cartesian = get_quaternion_orientation(cartesian)
        robot_state = np.concatenate([quat_cartesian, robot_state[6:]], axis=0)

        # subscribe images
        image_list = []
        for subscriber in self.image_subscribers:
            image_list.append(subscriber.recv_rgb_image()[0])

        obs = {}
        obs["features"] = np.array(robot_state, dtype=np.float32)
        for idx, image in enumerate(image_list):
            obs[f"pixels{idx}"] = cv2.resize(image, (self.width, self.height))

        return obs, self.reward, False, None

    def reset(self):  # currently same positions, with gripper opening
        if self.use_robot:
            print("resetting")
            self.action_request_socket.send(b"reset")
            reset_state = pickle.loads(self.action_request_socket.recv())

            # subscribe robot state
            self.action_request_socket.send(b"get_state")
            robot_state = pickle.loads(self.action_request_socket.recv())["xarm"]
            cartesian = robot_state[:6]
            quat_cartesian = get_quaternion_orientation(cartesian)
            robot_state = np.concatenate([quat_cartesian, robot_state[6:]], axis=0)

            # subscribe images
            image_list = []
            for subscriber in self.image_subscribers:
                image_list.append(subscriber.recv_rgb_image()[0])

            obs = {}
            obs["features"] = robot_state
            for idx, image in enumerate(image_list):
                obs[f"pixels{idx}"] = cv2.resize(image, (self.width, self.height))

            return obs
        else:
            obs = {}
            obs["features"] = np.zeros(self.feature_dim)
            obs["pixels"] = np.zeros((self.height, self.width, self.n_channels))
            return obs

    def render(self, mode="rgb_array", width=640, height=480):
        print("rendering")
        # subscribe images
        image_list = []
        for subscriber in self.image_subscribers:
            image = subscriber.recv_rgb_image()[0]
            image_list.append(cv2.resize(image, (width, height)))

        obs = np.concatenate(image_list, axis=1)
        return obs


if __name__ == "__main__":
    env = RobotEnv()
    obs = env.reset()

    for i in range(30):
        action = obs["features"]
        action[0] += 2
        obs, reward, done, _ = env.step(action)

    for i in range(30):
        action = obs["features"]
        action[1] += 2
        obs, reward, done, _ = env.step(action)

    for i in range(30):
        action = obs["features"]
        action[2] += 2
        obs, reward, done, _ = env.step(action)
