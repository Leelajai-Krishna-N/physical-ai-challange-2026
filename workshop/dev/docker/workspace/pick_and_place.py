#!/usr/bin/env python3
"""ROS service based pick-and-place client."""

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from so101_unified_bringup.srv import PickObject, PlaceObject


def make_pose(position):
    """Create a pose with position and a neutral orientation."""
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation.w = 1.0
    return pose


class PickAndPlace(Node):
    def __init__(self):
        super().__init__("pick_and_place")
        self.pick_client = self.create_client(PickObject, "/pick_object")
        self.place_client = self.create_client(PlaceObject, "/place_object")

        if not self.pick_client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("Timed out waiting for /pick_object service")
        if not self.place_client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("Timed out waiting for /place_object service")

        self.get_logger().info("Ready")

    def pick(self, position, attempt=0):
        req = PickObject.Request()
        req.target_pose = make_pose(position)
        req.grip_state = True
        future = self.pick_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            raise RuntimeError("Pick service call failed")
        self.get_logger().info(f"Pick {attempt + 1}: {result.message}")
        return bool(result.success)

    def place(self, position, attempt=0):
        req = PlaceObject.Request()
        req.target_pose = make_pose(position)
        req.grip_state = False
        future = self.place_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            raise RuntimeError("Place service call failed")
        self.get_logger().info(f"Place {attempt + 1}: {result.message}")
        return bool(result.success)


def main():
    rclpy.init()
    node = PickAndPlace()

    obj_pos = (0.2500, 0.0000, 0.0400)
    target_pos = (0.2500, 0.1500, 0.0550)
    approach_offset = (0.0, 0.0, 0.0300)

    try:
        for attempt in range(5):
            print(f"\nAttempt {attempt + 1}: picking...")
            if node.pick(obj_pos, attempt):
                print("Pick SUCCESS - refining pose before placing...")
                refined_target = tuple(
                    value + offset
                    for value, offset in zip(target_pos, approach_offset)
                )
                if node.place(refined_target, attempt):
                    print("Place SUCCESS")
                    break
                print("Retrying place...")
            print("Retrying pick...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
