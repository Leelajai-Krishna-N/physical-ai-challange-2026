#!/usr/bin/env bash
set -eo pipefail

cd /home/hacker/workspace
source /opt/ros/humble/setup.bash
source install/setup.bash

exec /usr/bin/python3 src/so101_mujoco/scripts/so101_mujoco_bridge.py \
  --model src/so101_mujoco/mujoco/scene.xml "$@"
