import os
import h5py
import pickle as pkl
import numpy as np
from pathlib import Path

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from sentence_transformers import SentenceTransformer

DATASET_PATH = Path("/path/to/datasets")
BENCHMARKS = ["libero_10", "libero_90"]
SAVE_DATA_PATH = Path("../../expert_demos/libero")
img_size = (128, 128)

# create save directory
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# benchmark for suite
benchmark_dict = benchmark.get_benchmark_dict()

# load sentence transformer
lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Total number of tasks
num_tasks = 0
for benchmark in BENCHMARKS:
    benchmark_path = DATASET_PATH / benchmark
    num_tasks += len(list(benchmark_path.glob("*.hdf5")))

tasks_stored = 0
for benchmark in BENCHMARKS:
    print(f"############################# {benchmark} #############################")
    benchmark_path = DATASET_PATH / benchmark

    save_benchmark_path = SAVE_DATA_PATH / benchmark
    save_benchmark_path.mkdir(parents=True, exist_ok=True)

    # Init env benchmark suite
    task_suite = benchmark_dict[benchmark]()

    for task_file in benchmark_path.glob("*.hdf5"):
        print(f"Processing {tasks_stored+1}/{num_tasks}: {task_file}")
        data = h5py.File(task_file, "r")["data"]

        # Init env
        task_name = str(task_file).split("/")[-1][:-10]
        # get task id from list of task names
        task_id = task_suite.get_task_names().index(task_name)
        # create environment
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": img_size[0],
            "camera_widths": img_size[1],
        }
        env = OffScreenRenderEnv(**env_args)
        obs = env.reset()
        render_image = obs["agentview_image"][::-1, :]

        observations = []
        states = []
        actions = []
        rewards = []

        for demo in data.keys():
            print(f"Processing {demo}")
            demo_data = data[demo]

            observation = {}
            observation["robot_states"] = np.array(
                demo_data["robot_states"], dtype=np.float32
            )

            # render image offscreen
            pixels, pixels_ego = [], []
            joint_states, eef_states, gripper_states = [], [], []
            for i in range(len(demo_data["states"])):
                obs = env.regenerate_obs_from_state(demo_data["states"][i])
                img = obs["agentview_image"][::-1]
                img_ego = obs["robot0_eye_in_hand_image"][::-1]
                joint_state = obs["robot0_joint_pos"]
                eef_state = np.concatenate(
                    [obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
                )
                gripper_state = obs["robot0_gripper_qpos"]
                # append
                pixels.append(img)
                pixels_ego.append(img_ego)
                joint_states.append(joint_state)
                eef_states.append(eef_state)
                gripper_states.append(gripper_state)
            observation["pixels"] = np.array(pixels, dtype=np.uint8)
            observation["pixels_egocentric"] = np.array(pixels_ego, dtype=np.uint8)
            observation["joint_states"] = np.array(joint_states, dtype=np.float32)
            observation["eef_states"] = np.array(eef_states, dtype=np.float32)
            observation["gripper_states"] = np.array(gripper_states, dtype=np.float32)

            observations.append(observation)
            states.append(np.array(demo_data["states"], dtype=np.float32))
            actions.append(np.array(demo_data["actions"], dtype=np.float32))
            rewards.append(np.array(demo_data["rewards"], dtype=np.float32))

        # save data
        save_data_path = save_benchmark_path / (
            str(task_file).split("/")[-1][:-10] + ".pkl"
        )
        with open(save_data_path, "wb") as f:
            pkl.dump(
                {
                    "observations": observations,
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                    "task_emb": lang_model.encode(env.language_instruction),
                },
                f,
            )
        print(f"Saved to {str(save_data_path)}")

        tasks_stored += 1
