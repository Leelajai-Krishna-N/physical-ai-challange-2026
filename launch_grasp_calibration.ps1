$Host.UI.RawUI.WindowTitle = "Grasp Calibration"
Set-Location "C:\physical-ai-challange-2026\workshop\dev\docker"

docker exec lerobot_hackathon_env bash -lc "cd /home/hacker/workspace && source /opt/ros/humble/setup.bash && source install/setup.bash && python3 calibrate_live_grasp.py --speed 0.9"
