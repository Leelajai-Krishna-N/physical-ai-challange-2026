MuJoCo pick-place working snapshot

Date: 2026-04-14
Repo root: C:\physical-ai-challange-2026

Purpose:
- Preserve the currently working MuJoCo pick-place code before further changes.

Files captured here:
- so101_mujoco_bridge.py
- auto_pick_place.py
- scene.xml
- working_diff.patch

Notes:
- `working_diff.patch` may be empty for files that are currently untracked by git.
- Use the file hashes in `HASHES.txt` plus `STATUS.txt` to recover this exact state.

Viewer command that worked:

```powershell
docker exec lerobot_hackathon_env bash -lc "cd /home/hacker/workspace && source /opt/ros/humble/setup.bash && source install/setup.bash && export DISPLAY=host.docker.internal:0.0 && export QT_X11_NO_MITSHM=1 && python3 src/so101_mujoco/scripts/so101_mujoco_bridge.py --model src/so101_mujoco/mujoco/scene.xml --viewer"
```

Demo command:

```powershell
docker exec lerobot_hackathon_env bash -lc "cd /home/hacker/workspace && python3 auto_pick_place.py --object-x 0.25 --object-y 0.00 --place-x 0.32 --place-y 0.20 --speed 1.2"
```
