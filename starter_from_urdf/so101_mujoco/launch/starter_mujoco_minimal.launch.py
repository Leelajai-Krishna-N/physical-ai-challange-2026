import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = Path(__file__).resolve().parents[1]
    bundle_root = package_dir.parent
    description_dir = bundle_root / "so101_description"

    default_scene = str(package_dir / "mujoco" / "scene.xml")
    default_bridge = str(package_dir / "scripts" / "so101_mujoco_bridge.py")
    default_urdf = str(description_dir / "urdf" / "so101.urdf")

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": Path(default_urdf).read_text()}],
    )

    mujoco_bridge = ExecuteProcess(
        cmd=[
            "python3",
            default_bridge,
            "--model",
            LaunchConfiguration("mujoco_scene"),
            "--startup-pose",
            LaunchConfiguration("startup_pose"),
            "--viewer",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("mujoco_scene", default_value=default_scene),
            DeclareLaunchArgument("startup_pose", default_value="home"),
            robot_state_publisher,
            mujoco_bridge,
        ]
    )
