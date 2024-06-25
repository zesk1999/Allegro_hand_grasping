# Master Thesis: Autonomous Grasping and Force Compensation for Object Manipulation using Robotic Hand

This repository contains the implementation and results of Master thesis of Željko Jovanović, Intelligent Field Robotic Systems MSc. The goal of this project was to develop techniques that enable robotic hands to reliably and adaptively grasp objects in diverse and unpredictable environments without dropping or breaking them, while compensating for external forces.


## Overview

1. **Adaptive Grasping**: Developing techniques for robots to adaptively grasp objects of varying shapes, sizes, and materials.
2. **Force Control**: Balancing the force exerted by the robotic hand to secure the object without causing damage.
3. **Reinforcement Learning**: Integrating reinforcement learning (RL) to enhance the adaptability and robustness of the grasping strategy.


## Function: `compute_hand_reward`

The core function of this implementation is `compute_hand_reward`, which calculates the reward for each step in the reinforcement learning process based on various factors including distance, velocity, contact forces, and successful grasping.


### Reward Calculation

1. **DOF Velocity Reward**: Penalty for higher velocities of the joints.
2. **Object Velocity Reward**: Penalty for higher object velocities, with an additional reward for very small velocities.
3. **Grasp Reward**: Reward for successful grasp based on contact forces.
4. **Action Penalty**: Reward based on the actions taken.
5. **Fall Penalty**: Penalty if the distance to the goal is larger than a threshold.
6. **Timed Out Penalty**: Penalty for not reaching the goal within the maximum episode length.
7. **Success Reward**: Bonus for reaching the goal.
8. **Force Penalty**: Penalty for net force stronger than a defined threshold.


### Environment Resets

- **Fall Reset**: Resets the environment if the distance to the goal is larger than a threshold.
- **Goal Achieved Reset**: Resets the environment if the goal is reached.
- **Exploration Reset**: Resets the environment after a defined number of consecutive successes to encourage exploration.
- **Timed Out Reset**: Resets the environment if the episode length exceeds the maximum allowed length.


### Installation

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. We highly recommend using a conda environment 
to simplify set up.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

```bash
pip install -e .
```


## How to Use

**Run the Simulation**:
    ```bash
    python3 train.py task=AllegroHand
    ```


## Results

To further illustrate the performance and robustness of the reward function designed for Allegro hand v4.0, qualitative evaluations were conducted and can be viewed in the following YouTube videos:

1. [Object grasping with Allegro hand v4.0 (static characteristics)](https://www.youtube.com/watch?v=uPX2_LV7XBQ)
2. [Object grasping with Allegro hand v4.0 (dynamic characteristics)](https://www.youtube.com/watch?v=xswUvcy5ilE)


## Domain Randomization

IsaacGymEnvs includes a framework for Domain Randomization to improve Sim-to-Real transfer of trained
RL policies. You can read more about it [here](docs/domain_randomization.md).


## Reproducibility and Determinism

If deterministic training of RL policies is important for your work, you may wish to review our [Reproducibility and Determinism Documentation](docs/reproducibility.md).


## Multi-GPU Training

You can run multi-GPU training using `torchrun` (i.e., `torch.distributed`) using this repository.

Here is an example command for how to run in this way -
`torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py multi_gpu=True task=AllegroHand <OTHER_ARGS>`

Where the `--nproc_per_node=` flag specifies how many processes to run and note the `multi_gpu=True` flag must be set on the train script in order for multi-GPU training to run.


## Population Based Training

You can run population based training to help find good hyperparameters or to train on very difficult environments which would otherwise
be hard to learn anything on without it.


## WandB support

You can run [WandB](https://wandb.ai/) with Isaac Gym Envs by setting `wandb_activate=True` flag from the command line. You can set the group, name, entity, and project for the run by setting the `wandb_group`, `wandb_name`, `wandb_entity` and `wandb_project` set. Make sure you have WandB installed with `pip install wandb` before activating.


## Capture videos

We implement the standard `env.render(mode='rgb_rray')` `gym` API to provide an image of the simulator viewer. Additionally, we can leverage `gym.wrappers.RecordVideo` to help record videos that shows agent's gameplay. Consider running the following file which should produce a video in the `videos` folder.

```python
import gym
import isaacgym
import isaacgymenvs
import torch

num_envs = 64

envs = isaacgymenvs.make(
	seed=0, 
	task="AllegroHand", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
	headless=False,
	multi_gpu=False,
	virtual_screen_capture=True,
	force_render=False,
)
envs.is_vector_env = True
envs = gym.wrappers.RecordVideo(
	envs,
	"./videos",
	step_trigger=lambda step: step % 10000 == 0, # record the videos every 10000 steps
	video_length=100  # for each video record up to 100 steps
)
envs.reset()
print("the image of Isaac Gym viewer is an array of shape", envs.render(mode="rgb_array").shape)
for _ in range(100):
	actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
	envs.step(actions)
```


## Capture videos during training

You can automatically capture the videos of the agents gameplay by toggling the `capture_video=True` flag and tune the capture frequency `capture_video_freq=1500` and video length via `capture_video_len=100`. You can set `force_render=False` to disable rendering when the videos are not captured.

```
python train.py capture_video=True capture_video_freq=1500 capture_video_len=100 force_render=False
```

##  Citing

(https://github.com/isaac-sim/IsaacGymEnvs/tree/main)

@misc{makoviychuk2021isaac,
      title={Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning}, 
      author={Viktor Makoviychuk and Lukasz Wawrzyniak and Yunrong Guo and Michelle Lu and Kier Storey and Miles Macklin and David Hoeller and Nikita Rudin and Arthur Allshire and Ankur Handa and Gavriel State},
      year={2021},
      journal={arXiv preprint arXiv:2108.10470}
}
