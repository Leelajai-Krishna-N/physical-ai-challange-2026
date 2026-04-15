$Host.UI.RawUI.WindowTitle = "Keyboard Teleop"
Set-Location "C:\physical-ai-challange-2026\workshop\dev\docker"

docker exec lerobot_hackathon_env bash -lc "cd /home/hacker/workspace && export DISPLAY=host.docker.internal:0.0 && python3 src/so101_mujoco/scripts/so101_keyboard_teleop.py"
