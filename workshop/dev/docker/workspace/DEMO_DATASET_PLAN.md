MuJoCo Demo Recording Plan

Goal:
- Record 60-80 pick-and-place demonstrations in MuJoCo.
- Save them in a simple episode format that can be converted later for ACT, SmolVLA, or diffusion-policy style training.

Working snapshot:
- Safe checkpoint folder:
  [snapshots/mujoco_pick_working_2026-04-14](C:/physical-ai-challange-2026/snapshots/mujoco_pick_working_2026-04-14)

Key files:
- [record_demos.py](C:/physical-ai-challange-2026/workshop/dev/docker/workspace/record_demos.py)
- [auto_pick_place.py](C:/physical-ai-challange-2026/workshop/dev/docker/workspace/auto_pick_place.py)
- [so101_mujoco_bridge.py](C:/physical-ai-challange-2026/workshop/dev/docker/workspace/src/so101_mujoco/scripts/so101_mujoco_bridge.py)

What the recorder captures:
- `/joint_states`
- `/commanded_joint_targets`
- `/mujoco/red_box_pose`
- `/d435i/image` when camera publishing is enabled

Episode format:
- `demos/<episode_name>/meta.json`
- `demos/<episode_name>/steps.csv`
- `demos/<episode_name>/rgb/000000.png ...`

Recommended workflow:

1. Start the Docker container:

```powershell
cd C:\physical-ai-challange-2026\workshop\dev\docker
docker compose up -d
```

2. Start the MuJoCo bridge with viewer and camera publishing:

```powershell
docker exec lerobot_hackathon_env bash -lc "cd /home/hacker/workspace && source /opt/ros/humble/setup.bash && source install/setup.bash && export DISPLAY=host.docker.internal:0.0 && export QT_X11_NO_MITSHM=1 && python3 src/so101_mujoco/scripts/so101_mujoco_bridge.py --model src/so101_mujoco/mujoco/scene.xml --viewer --enable-camera"
```

3. In a second terminal, start recording:

```powershell
docker exec lerobot_hackathon_env bash -lc "cd /home/hacker/workspace && source /opt/ros/humble/setup.bash && source install/setup.bash && python3 record_demos.py --episode-name demo_001 --sample-hz 10 --notes 'pick and place red box'"
```

4. In a third terminal, run either:
- manual teleop, or
- scripted pick and place

Scripted example:

```powershell
docker exec lerobot_hackathon_env bash -lc "cd /home/hacker/workspace && python3 auto_pick_place.py --object-x 0.25 --object-y 0.00 --place-x 0.32 --place-y 0.20 --speed 1.2"
```

5. Stop the recorder with `Ctrl+C` when the episode is done.

6. Mark successful episodes:
- easiest way today is to rerun the recorder command with `--success` for episodes you know are good
- or edit `meta.json` and set `"success": true`

Quality rules:
- Record 10 episodes first.
- Open the saved `steps.csv` and verify:
  - joint positions change over time
  - commanded targets are present
  - red box pose is present
  - RGB frames exist if camera was enabled
- Only then scale to 60-80 episodes.

Suggested collection split:
- 40-50 successful demos
- 10-20 recovery / imperfect demos
- 10 varied object and place positions

Recommended metadata to vary across episodes:
- object x/y
- place x/y
- speed
- success / failure
- short note about what went wrong when it fails

Next conversion step for policy training:
- Keep this raw episode format as the source of truth.
- Add a separate export script later to convert into:
  - ACT dataset format
  - LeRobot / SmolVLA format
  - diffusion-policy sequence format

Do not overwrite the working MuJoCo snapshot before the first 10 episodes are verified.
