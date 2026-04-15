$Host.UI.RawUI.WindowTitle = "Demo Recorder"
Set-Location "C:\physical-ai-challange-2026\workshop\dev\docker"

$episodeName = if ($args.Length -gt 0) { $args[0] } else { "human_demo_002" }
$notes = if ($args.Length -gt 1) { ($args[1..($args.Length - 1)] -join " ") } else { "human controlled demo" }

docker exec lerobot_hackathon_env bash -lc "cd /home/hacker/workspace && source /opt/ros/humble/setup.bash && source install/setup.bash && python3 record_demos.py --episode-name $episodeName --sample-hz 10 --notes '$notes'"
