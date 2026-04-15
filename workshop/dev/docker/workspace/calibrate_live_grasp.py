#!/usr/bin/env python3
"""Search for a working grasp offset around the live red-box pose."""

from __future__ import annotations

import argparse
import time
from itertools import product

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from auto_pick_place import dwell, move_to


HOME_X = 0.20
HOME_Y = 0.00
TRAVEL_Z = 0.25
LIFT_Z = 0.18


class BoxPoseReader(Node):
    def __init__(self):
        super().__init__("grasp_calibrator")
        self.latest_pose: tuple[float, float, float] | None = None
        self.create_subscription(PoseStamped, "/mujoco/red_box_pose", self._cb, 10)

    def _cb(self, msg: PoseStamped):
        self.latest_pose = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )

    def wait_for_pose(self, timeout: float) -> tuple[float, float, float]:
        deadline = time.time() + timeout
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_pose is not None:
                return self.latest_pose
        raise RuntimeError("Timed out waiting for /mujoco/red_box_pose")


def refresh_pose(node: BoxPoseReader) -> tuple[float, float, float]:
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.05)
    assert node.latest_pose is not None
    return node.latest_pose


def run_attempt(node: BoxPoseReader, x: float, y: float, z: float, speed: float) -> tuple[bool, float]:
    start_box = refresh_pose(node)
    move_to(HOME_X, HOME_Y, TRAVEL_Z, True, 1.0 * speed, "Home")
    move_to(x, y, LIFT_Z, True, 1.2 * speed, "Above")
    move_to(x, y, z + 0.02, True, 0.9 * speed, "Pre-grasp")
    move_to(x, y, z, True, 0.8 * speed, "Grasp")
    dwell(0.2)
    move_to(x, y, z, False, 0.6 * speed, "Close")
    dwell(0.4)
    move_to(x, y, LIFT_Z, False, 1.0 * speed, "Lift")
    dwell(0.4)
    lifted_box = refresh_pose(node)
    success = lifted_box[2] > start_box[2] + 0.03
    return success, lifted_box[2] - start_box[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--x-offsets", nargs="+", type=float,
                        default=[-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03])
    parser.add_argument("--y-offsets", nargs="+", type=float,
                        default=[-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03])
    parser.add_argument("--z-offsets", nargs="+", type=float,
                        default=[-0.01, -0.005, 0.0, 0.005, 0.01])
    args = parser.parse_args()

    rclpy.init()
    node = BoxPoseReader()
    try:
        box_x, box_y, box_z = node.wait_for_pose(args.timeout)
        print(f"Live box pose: x={box_x:.4f}, y={box_y:.4f}, z={box_z:.4f}")

        best = None
        for dx, dy, dz in product(args.x_offsets, args.y_offsets, args.z_offsets):
            target_x = box_x + dx
            target_y = box_y + dy
            target_z = box_z + dz
            print(
                f"Trying offset dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f} "
                f"-> target=({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
            )
            success, lift_delta = run_attempt(node, target_x, target_y, target_z, args.speed)
            print(f"  lift_delta={lift_delta:.4f} success={success}")
            if best is None or lift_delta > best[0]:
                best = (lift_delta, dx, dy, dz)
            if success:
                print(f"SUCCESS offset dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f}")
                return 0

        if best is not None:
            print(
                f"No successful grasp found. Best lift delta={best[0]:.4f} "
                f"at dx={best[1]:+.3f} dy={best[2]:+.3f} dz={best[3]:+.3f}"
            )
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
