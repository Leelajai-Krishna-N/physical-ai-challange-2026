# Judge Demo

## What To Show

This demo shows deterministic known-location pick-and-place in MuJoCo using:

- `src/so101_mujoco/scripts/so101_mujoco_bridge.py`
- `auto_pick_place.py`

The stable judge path is:

1. Run the MuJoCo bridge in headless mode.
2. Run the pick-place script in `--exact-known-pose` mode.
3. Verify the final box pose from ROS.

## Commands

Bridge:

```bash
cd /home/hacker/workspace
source /opt/ros/humble/setup.bash
if [ -f install/setup.bash ]; then source install/setup.bash; fi
python3 src/so101_mujoco/scripts/so101_mujoco_bridge.py --model src/so101_mujoco/mujoco/scene.xml
```

Pick and place:

```bash
cd /home/hacker/workspace
source /opt/ros/humble/setup.bash
if [ -f install/setup.bash ]; then source install/setup.bash; fi
python3 -u auto_pick_place.py --exact-known-pose --place-x 0.32 --place-y 0.20 --speed 1.2
```

Verify final pose:

```bash
source /opt/ros/humble/setup.bash
if [ -f /home/hacker/workspace/install/setup.bash ]; then source /home/hacker/workspace/install/setup.bash; fi
ros2 topic echo /mujoco/red_box_pose --once
```

## Expected Result

The script prints:

- `Explicit attach accepted on attempt 1`
- `=== DONE ===`

Final pose should be approximately:

```text
x ~= 0.32
y ~= 0.20
z ~= 0.025
```

## What To Say

Suggested short explanation:

> We first solved exact deterministic pick-and-place in simulation. The bridge exposes robot and object state through ROS2, and the controller executes a fixed grasp, lift, transfer, and release sequence. After the run, we verify the placed object pose directly from `/mujoco/red_box_pose`.

## Important Note

Do not use the old `pick_place_demo.mp4` as proof of the final fix. It was generated before the current working pick-place path.
