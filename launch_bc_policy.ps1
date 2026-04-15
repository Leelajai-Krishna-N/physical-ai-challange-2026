$Host.UI.RawUI.WindowTitle = "BC Policy"
Set-Location "C:\physical-ai-challange-2026\workshop\dev\docker"

docker exec lerobot_hackathon_env bash -lc "cd /home/hacker/workspace && source /opt/ros/humble/setup.bash && source install/setup.bash && python3 run_bc_policy.py --model models/bc_policy_latest.npz"
