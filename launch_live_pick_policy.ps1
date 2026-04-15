$Host.UI.RawUI.WindowTitle = "Live Pick Policy"
Set-Location "C:\physical-ai-challange-2026\workshop\dev\docker"

docker exec lerobot_hackathon_env bash -lc "cd /home/hacker/workspace && source /opt/ros/humble/setup.bash && source install/setup.bash && python3 auto_pick_place_live.py --place-x 0.32 --place-y 0.20 --speed 1.2"
