# Starter From URDF

This folder is now a minimal self-contained SO-101 simulation starter bundle.

It includes:

- `so101_description/` for the URDF and description meshes
- `so101_mujoco/` for the MuJoCo XML scene and meshes
- `so101_mujoco/scripts/so101_mujoco_bridge.py` so the scene can actually run
- `so101_mujoco/launch/starter_mujoco_minimal.launch.py` as a minimal launch entrypoint

## Simplest direct run inside the container

```bash
cd /home/hacker/workspace/starter_from_urdf
source /opt/ros/humble/setup.bash
source /home/hacker/workspace/install/setup.bash
export DISPLAY=host.docker.internal:0.0
export QT_X11_NO_MITSHM=1
python3 so101_mujoco/scripts/so101_mujoco_bridge.py --model so101_mujoco/mujoco/scene.xml --viewer --enable-camera
```

## Minimal ROS launch from source

```bash
cd /home/hacker/workspace/starter_from_urdf
source /opt/ros/humble/setup.bash
source /home/hacker/workspace/install/setup.bash
python3 -c "from launch import LaunchService; from launch.launch_description_sources import PythonLaunchDescriptionSource; from pathlib import Path; ls=LaunchService(); ls.include_launch_description(PythonLaunchDescriptionSource(str(Path('so101_mujoco/launch/starter_mujoco_minimal.launch.py').resolve()))); raise SystemExit(ls.run())"
```

## Important note

This bundle is self-contained for the robot description and MuJoCo simulation path, but it is still intended to run inside the existing Docker/ROS environment because MuJoCo, ROS 2, and Python dependencies live there.
