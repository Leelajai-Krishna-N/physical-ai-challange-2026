import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Paths to package directories
    lerobot_pkg = get_package_share_directory('lerobot_so101')
    mujoco_pkg = get_package_share_directory('so101_mujoco')

    # Path to the MuJoCo scene XML
    # Using the source path instead of share path to ensure the script finds it
    scene_xml_path = '/home/hacker/workspace/src/so101_mujoco/mujoco/scene.xml'
    urdf_path = '/home/hacker/workspace/src/so101_description/urdf/so101.urdf'

    # 1. Robot State Publisher (Loads URDF and publishes TFs)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': open(urdf_path, 'r').read()}] if os.path.exists(urdf_path) else []
    )

    # 2. MuJoCo Bridge (The simulation environment)
    # Using the absolute source path for the bridge script
    mujoco_bridge = ExecuteProcess(
        cmd=['python3', '/home/hacker/workspace/src/so101_mujoco/scripts/so101_mujoco_bridge.py',
             '--model', scene_xml_path,
             '--enable-camera', 'True',
             '--viewer'],
        output='screen'
    )

    # 3. Vision Node (Object detection and 3D projection)
    vision_node = Node(
        package='lerobot_so101',
        executable='vision_node',
        name='vision_node',
        output='screen',
        parameters=[{
            'target_color_lower': [0, 100, 100],
            'target_color_upper': [10, 255, 255],
            'fx': 600.0,
            'fy': 600.0,
            'cx': 320.0,
            'cy': 240.0,
        }]
    )

    # 4. Motion Planner Node (Path planning and execution)
    motion_planner_node = Node(
        package='lerobot_so101',
        executable='motion_planner_node',
        name='motion_planner_node',
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher,
        mujoco_bridge,
        vision_node,
        motion_planner_node
    ])
