## Installation Instructions

- Clone the repository and create a conda environment using the provided `conda_env.yml` file.
```
conda env create -f conda_env.yml
```
- Activate the environment using `conda activate baku`.
- To install LIBERO, follow the instructions in the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO).
- To install Meta-World, install the version added to this repository as a submodule.
```
git submodule update --init --recursive
cd Metaworld
pip install -e .
```

## Instructions for Real Robot Experiments

- For running experiments on an xArm robot, install the [xArm-Python-SDK](https://github.com/xArm-Developer/xArm-Python-SDK) using the instructions provided in the repository.
- Install the xarm environment using the following command.
```
cd xarm_env
pip install -e .
```
- For teleoperating the robot using [Open Teach](https://open-teach.github.io/), use the version of Open Teach added to this repository as a submodule. Move inside the directory using `cd Open-Teach` and install the package using the instructions provided in the official [Open-Teach repository](https://github.com/aadhithya14/Open-Teach).
    - Instructions for robot teleoperation and data collection are provided [here](https://github.com/siddhanthaldar/Open-Teach/blob/baku/instructions.md).
- To run the experiments, use the following command.

## Instructions for Demonstrations
- For LIBERO, download the demonstrations using instructions provided in the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO).
    - Convert these demonstrations into a `.pkl` format using the following commands.
    ```
    cd baku/data_generation
    python generate_libero.py
    ```
    - Make sure to change the `DATASET_PATH` in `baku/data_generation/generate_libero.py` to the path where the demonstrations are stored.
    - The script stores the demonstration pkl files in `path/to/repo/expert_demos/libero/`. Set `root_dir` in `cfg/config.yaml` to `path/to/repo`.
- To access the datasets for Meta-World, DMControl, and the real world xArm Kitchen, please send an email to sh6474@nyu.edu.

## Train BAKU
- Remember to set `root_dir` in `cfg/config.yaml` to `path/to/repo`.
- To train BAKU on LIBERO-90, use the following command.
```
python train.py agent=baku suite=libero dataloader=libero suite/task=libero_90 suite.hidden_dim=256
```
- To train BAKU on Meta-World, use the following command.
```
python train.py agent=baku suite=metaworld dataloader=metaworld suite.hidden_dim=256 use_proprio=false
```
- To train BAKU on DMControl, use the following command.
```
python train.py agent=baku suite=dmc dataloader=dmc suite.hidden_dim=256 obs_type=features use_proprio=false
```

## Evaluate BAKU
- We provide the weights for BAKU trained on the simulated benchmarks [here](https://osf.io/3x8v5/?view_only=fb8285f025e84d23a41a0eef683a7e6d). Please download this weights to evaluate BAKU on the benchmarks.
- We provide 2 variants of evaluation scripts - one for sequential evaluation (`eval.py`) and one for parallel evaluation (`eval_ray.py` to use which you will have to install [Ray](https://docs.ray.io/en/latest/ray-overview/getting-started.html)). 
- Before evaluating, make sure to set the `root_dir` in `cfg/config_eval.yaml` to `path/to/repo`.
- For evaluation on LIBERO-90, use one of the the following commands.
```
python eval.py agent=baku suite=libero dataloader=libero suite/task=libero_90 suite.hidden_dim=256 bc_weight=/path/to/weight
```
```
python eval_ray.py agent=baku suite=libero dataloader=libero suite/task=libero_90 suite.hidden_dim=256 bc_weight=/path/to/weight
```

Follow the same pattern for evaluation on Meta-World and DMControl.

## Train Baselines

### MT-ACT

- To train MT-ACT on LIBERO-90, use the following command.
```
python train.py agent=mtact suite=libero dataloader=libero suite/task=libero_90
```
- To train MT-ACT on Meta-World, use the following command.
```
python train.py agent=mtact suite=metaworld dataloader=metaworld use_proprio=false
```
- To train MT-ACT on DMControl, use the following command.
```
python train.py agent=baku suite=dmc dataloader=dmc obs_type=features use_proprio=false
```

### RT-1

- To train RT-1 on LIBERO-90, use the following command.
```
python train.py agent=rt1 suite=libero dataloader=libero suite/task=libero_90 suite.hidden_dim=512 suite.history=true suite.history_len=6 temporal_agg=false
```
- To train RT-1 on Meta-World, use the following command.
```
python train.py agent=rt1 suite=metaworld dataloader=metaworld suite.hidden_dim=512 use_proprio=false suite.history=true suite.history_len=6 temporal_agg=false
```
- To train RT-1 on DMControl, use the following command.
```
python train.py agent=rt1 suite=dmc dataloader=dmc suite.hidden_dim=512 obs_type=features use_proprio=false suite.history=true suite.history_len=6 temporal_agg=false
```