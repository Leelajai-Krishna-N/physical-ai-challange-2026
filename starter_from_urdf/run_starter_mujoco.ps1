$Host.UI.RawUI.WindowTitle = "Starter From URDF MuJoCo"
Set-Location "C:\physical-ai-challange-2026\workshop\dev\docker"

docker exec lerobot_hackathon_env bash -lc "cd /home/hacker/workspace/starter_from_urdf && source /opt/ros/humble/setup.bash && source /home/hacker/workspace/install/setup.bash && export DISPLAY=host.docker.internal:0.0 && export QT_X11_NO_MITSHM=1 && python3 so101_mujoco/scripts/so101_mujoco_bridge.py --model so101_mujoco/mujoco/scene.xml --viewer --enable-camera"
