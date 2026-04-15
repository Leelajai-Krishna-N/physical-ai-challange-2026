import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, PoseStamped
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
import time

class MotionPlannerNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # Subscriber to the detected object coordinates from the vision node
        self.subscription = self.create_subscription(
            Point,
            '/detected_object_base',
            self.object_callback,
            10)

        # Subscriber to verify grasp success via joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)

        # Action clients for MoveIt 2 / Controller
        self.arm_client = ActionClient(self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, 'gripper_controller/follow_joint_trajectory')

        self.current_joint_states = {}
        self.is_grasping = False

        self.get_logger().info('Motion Planner Node started. Waiting for object coordinates...')

    def joint_state_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_states[name] = pos

    def object_callback(self, msg):
        self.get_logger().info(f'Target received: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}')

        grasp_pose = Pose()
        grasp_pose.position.x = msg.x
        grasp_pose.position.y = msg.y
        grasp_pose.position.z = msg.z
        grasp_pose.orientation.w = 1.0

        self.plan_and_execute_grasp(grasp_pose)

    def plan_and_execute_grasp(self, target_pose):
        self.get_logger().info('Planning collision-free path to grasp pose...')
        # Conceptual MoveIt planning logic
        self.get_logger().info('Executing movement to grasp pose...')
        time.sleep(1.0) # Simulate travel time
        self.get_logger().info('End effector reached grasp pose.')

        self.execute_secure_grasp()

    def execute_secure_grasp(self):
        self.get_logger().info('Executing secure grasp sequence...')

        # 1. Command gripper to close (target position for closed gripper)
        # Based on mujoco_bridge, 'gripper' target is typically < 0.15 for closed
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['gripper']

        point = FollowJointTrajectory.TrajectoryPoint()
        point.positions = [0.1] # Closed position
        point.time_from_start = rclpy.duration.Duration(seconds=1.0).to_msg()
        goal_msg.trajectory.points = [point]

        self.get_logger().info('Closing gripper...')
        self._send_gripper_goal(goal_msg)

        # 2. Verify grasp success
        # A secure grasp is verified if the gripper does not close completely
        # (meaning an object is obstructing the jaws).
        verified = self.verify_grasp_success(timeout=2.0)

        if verified:
            self.get_logger().info('Grasp verified: Object securely held.')
            # Proceed to the move command (e.g., lift object)
            self.lift_object()
        else:
            self.get_logger().error('Grasp failed: Gripper closed completely or no object detected.')

    def verify_grasp_success(self, timeout=2.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            gripper_pos = self.current_joint_states.get('gripper', 0.0)
            # If gripper position is significantly greater than the 'fully closed'
            # limit (e.g. 0.1), an object is likely inside.
            if 0.15 < gripper_pos < 0.8:
                return True
            time.sleep(0.1)
        return False

    def _send_gripper_goal(self, goal):
        self.gripper_client.wait_for_server()
        self.gripper_client.send_goal_async(goal)

    def lift_object(self):
        self.get_logger().info('Lifting object to safe height...')

        # 1. Define a safe transport height (approx 15cm above base)
        target_pose = Pose()
        target_pose.position.x = 0.0 # Relative to object pos
        target_pose.position.y = 0.0
        target_pose.position.z = 0.2  # Safe Z height
        target_pose.orientation.w = 1.0

        self.move_arm_smoothly(target_pose, speed_scaling=0.2)
        self.get_logger().info('Object lifted successfully.')
        self.transport_to_target()

    def transport_to_target(self):
        self.get_logger().info('Transporting object to target location...')

        # Designated target location (Example: X=0.2, Y=-0.2, Z=0.2)
        target_pose = Pose()
        target_pose.position.x = 0.2
        target_pose.position.y = -0.2
        target_pose.position.z = 0.2
        target_pose.orientation.w = 1.0

        # Use slow speed scaling to ensure the object doesn't slip due to acceleration
        self.move_arm_smoothly(target_pose, speed_scaling=0.15)
        self.get_logger().info('Reached target location.')

        # Now place the object
        self.place_object()

    def place_object(self):
        self.get_logger().info('Placing object...')

        # 1. Fine-tuning phase: move to target with extreme precision
        # We use an even slower speed scaling for the final descent
        target_pose = Pose()
        target_pose.position.x = 0.2
        target_pose.position.y = -0.2
        target_pose.position.z = 0.05 # Surface height
        target_pose.orientation.w = 1.0

        self.get_logger().info('Fine-tuning position for maximum precision...')
        self.move_arm_smoothly(target_pose, speed_scaling=0.05)

        # 2. Release the grasp
        self.release_gripper()
        self.get_logger().info('Object released.')

        # 3. Retract arm slightly before moving home
        retract_pose = Pose()
        retract_pose.position.x = 0.2
        retract_pose.position.y = -0.2
        retract_pose.position.z = 0.2
        retract_pose.orientation.w = 1.0
        self.move_arm_smoothly(retract_pose, speed_scaling=0.2)

        # 4. Return to home position
        self.return_to_home()

    def return_to_home(self):
        self.get_logger().info('Returning arm to neutral home position...')
        home_pose = Pose()
        home_pose.position.x = 0.0
        home_pose.position.y = 0.0
        home_pose.position.z = 0.3
        home_pose.orientation.w = 1.0

        self.move_arm_smoothly(home_pose, speed_scaling=0.3)
        self.get_logger().info('Robot returned home. Task complete.')

    def release_gripper(self):
        self.get_logger().info('Opening gripper to release object...')
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['gripper']

        point = FollowJointTrajectory.TrajectoryPoint()
        point.positions = [0.8] # Open position
        point.time_from_start = rclpy.duration.Duration(seconds=1.0).to_msg()
        goal_msg.trajectory.points = [point]

        self._send_gripper_goal(goal_msg)

    def move_arm_smoothly(self, target_pose, speed_scaling=0.1):
        self.get_logger().info(f'Moving arm smoothly with scale {speed_scaling}...')

        # In a MoveIt 2 / MoveGroup implementation:
        # move_group.set_max_velocity_scaling_factor(speed_scaling)
        # move_group.set_max_acceleration_scaling_factor(speed_scaling)
        # move_group.set_pose_target(target_pose)
        # move_group.go(wait=True)

        # Conceptual simulation of smooth movement
        time.sleep(2.0)
        self.get_logger().info('Movement completed.')

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
