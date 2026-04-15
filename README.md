# Physical AI Challenge 2026

This repository contains our SO-101 simulation workspace, automation scripts, starter assets, and demo material for the Physical AI Challenge 2026 project.

## Demo

![Judge demo](demo_assets/judge_demo_working.gif)

The current working judge path demonstrates deterministic pick-and-place in MuJoCo using the ROS 2 bridge and the autonomous pick-and-place controller.

## Repository Layout

- `workshop/dev/docker/workspace/` - main Docker/ROS 2/MuJoCo workspace and project code
- `workshop/dev/docker/workspace/src/so101_mujoco/` - MuJoCo scene, bridge, and simulation scripts
- `workshop/dev/docker/workspace/auto_pick_place.py` - deterministic pick-and-place controller used in the judge demo
- `starter_from_urdf/` - starter URDF-based simulation assets
- `tools/` - host-side helper utilities
- `demo_assets/` - demo GIFs, videos, and images
- `snapshots/` - generated snapshots and related artifacts

## Main Workflows

### 1. Bring up the environment

From the project root:

```bash
docker compose up -d
docker exec -it lerobot_hackathon_env bash
```

Inside the container:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### 2. Run the MuJoCo bridge

```bash
cd /home/hacker/workspace
source /opt/ros/humble/setup.bash
if [ -f install/setup.bash ]; then source install/setup.bash; fi
python3 src/so101_mujoco/scripts/so101_mujoco_bridge.py --model src/so101_mujoco/mujoco/scene.xml
```

### 3. Run the judge demo pick-and-place flow

```bash
cd /home/hacker/workspace
source /opt/ros/humble/setup.bash
if [ -f install/setup.bash ]; then source install/setup.bash; fi
python3 -u auto_pick_place.py --exact-known-pose --place-x 0.32 --place-y 0.20 --speed 1.2
```

### 4. Verify the final object pose

```bash
source /opt/ros/humble/setup.bash
if [ -f /home/hacker/workspace/install/setup.bash ]; then source /home/hacker/workspace/install/setup.bash; fi
ros2 topic echo /mujoco/red_box_pose --once
```

Expected final pose:

```text
x ~= 0.32
y ~= 0.20
z ~= 0.025
```

## Extra Notes

- `workshop/dev/docker/workspace/JUDGE_DEMO.md` contains the exact judge-demo steps and talking points.
- `workshop/dev/docker/workspace/HACKATHON_GUIDE.md` covers simulation setup and keyboard teleoperation.
- The repository currently includes both host-side launch scripts and the in-container workspace used during development.
