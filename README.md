# Physical AI Challenge 2026

This repository contains our simulation workspace, automation scripts, demo assets, and final presentation material for the Physical AI Challenge 2026 project.

## Project Overview

The project is centered on an SO-101 MuJoCo and ROS 2 workflow for autonomous pick-and-place experimentation, demo preparation, and hackathon deliverables.

Key areas in this repository:

- `workshop/dev/docker/workspace/` - main Docker, ROS 2, and MuJoCo workspace
- `workshop/dev/docker/workspace/src/so101_mujoco/` - MuJoCo scene, bridge, and simulation scripts
- `workshop/dev/docker/workspace/auto_pick_place.py` - autonomous pick-and-place flow used in demos
- `starter_from_urdf/` - starter URDF-based simulation assets
- `tools/` - helper utilities used during asset and presentation preparation
- `demo_assets/` - demo GIFs, videos, and supporting visuals
- `snapshots/` - generated snapshots and image references
- `deliverables/` - presentation and reference files added for GitHub sharing

## Demo

![Judge demo](demo_assets/judge_demo_working.gif)

The current judge demo path shows deterministic pick-and-place behavior in MuJoCo using the ROS 2 bridge and the autonomous controller flow.

## Deliverables Added To This Repo

The following files were added under `deliverables/`:

- `deliverables/KAIROS_final_final.pptx`
- `deliverables/IRJAEH-03-05-0301-2504579-2063-2069.pdf`
- `deliverables/layer 2.json`
- `deliverables/Industrial IoT Predictive Maintenance (17).zip`

## Main Workflow

### 1. Start the environment

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

### 3. Run the pick-and-place demo

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

## Notes

- `workshop/dev/docker/workspace/JUDGE_DEMO.md` contains the judge-demo steps and talking points.
- `workshop/dev/docker/workspace/HACKATHON_GUIDE.md` covers simulation setup and teleoperation.
- The repo includes both host-side launch scripts and the in-container workspace used during development.
