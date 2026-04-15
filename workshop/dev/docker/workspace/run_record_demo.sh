#!/usr/bin/env bash
set -eo pipefail

cd /home/hacker/workspace
source /opt/ros/humble/setup.bash
source install/setup.bash

exec /usr/bin/python3 record_demos.py "$@"
