Reference Repo Used

- Reference: [external/mujoco-demo](C:/physical-ai-challange-2026/external/mujoco-demo)
- Original upstream: [vmanoj1996/mujoco-demo](https://github.com/vmanoj1996/mujoco-demo)

What we took from it

- It confirms a stable MuJoCo pick-and-place scene can be built around:
  - a fixed industrial arm
  - camera-based object localization
  - scripted planning for grasp and place
- Its most useful assets for this workspace are:
  - [external/mujoco-demo/examples/models/factory.xml](C:/physical-ai-challange-2026/external/mujoco-demo/examples/models/factory.xml)
  - [external/mujoco-demo/examples/models/scene.xml](C:/physical-ai-challange-2026/external/mujoco-demo/examples/models/scene.xml)
  - [external/mujoco-demo/examples/generateImagesForTraining.m](C:/physical-ai-challange-2026/external/mujoco-demo/examples/generateImagesForTraining.m)

What it does not give us directly

- It is MATLAB / Simulink based.
- It does not ship a Python policy learner for ACT, SmolVLA, or diffusion policy.
- It does not match the SO-101 robot and ROS2 bridge in this repo.

What we built here instead

- A lightweight behavioral cloning baseline that can train directly on our recorded MuJoCo demos:
  - [train_bc_policy.py](C:/physical-ai-challange-2026/workshop/dev/docker/workspace/train_bc_policy.py)
  - [run_bc_policy.py](C:/physical-ai-challange-2026/workshop/dev/docker/workspace/run_bc_policy.py)

Current baseline model

- Inputs:
  - 6 joint positions
  - 6 joint velocities
  - red box position
  - red box orientation
- Outputs:
  - 6 commanded joint targets

Why this is the right first model

- It works with the data we already record.
- It needs only `numpy`, so it can run immediately in the current container.
- It gives us a deployable closed-loop baseline before we upgrade to ACT or SmolVLA.

Training

```bash
cd /home/hacker/workspace
python3 train_bc_policy.py --demos-root demos --output models/bc_policy_latest.npz --include-failures
```

Running the trained policy

```bash
cd /home/hacker/workspace
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 run_bc_policy.py --model models/bc_policy_latest.npz
```

Recommended next step

- Record 20-30 good demos before judging the policy.
- Once that baseline moves the arm toward the box reliably, export the same dataset into ACT / LeRobot format.
