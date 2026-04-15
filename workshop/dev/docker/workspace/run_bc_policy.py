#!/usr/bin/env python3
"""Run a trained behavioral-cloning policy against the MuJoCo bridge."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState


def ros_time_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class BCPolicyNode(Node):
    def __init__(self, model_path: Path, control_hz: float, udp_host: str, udp_port: int):
        super().__init__("bc_policy_runner")
        raw = np.load(model_path, allow_pickle=False)
        self.weights = raw["weights"]
        self.bias = raw["bias"]
        self.x_mean = raw["x_mean"]
        self.x_std = raw["x_std"]
        self.y_mean = raw["y_mean"]
        self.y_std = raw["y_std"]
        self.joint_limits = raw["joint_limits"]

        self.latest_joint_positions: list[float] | None = None
        self.latest_joint_velocities: list[float] | None = None
        self.latest_box_position: list[float] = [0.0, 0.0, 0.0]
        self.latest_box_orientation: list[float] = [1.0, 0.0, 0.0, 0.0]
        self.last_command: np.ndarray | None = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bridge_addr = (udp_host, udp_port)

        self.create_subscription(JointState, "/joint_states", self._joint_cb, 20)
        self.create_subscription(PoseStamped, "/mujoco/red_box_pose", self._box_cb, 20)
        self.create_timer(1.0 / max(control_hz, 1.0), self._step)

        info = json.loads(str(raw["model_info"]))
        self.get_logger().info(
            f"Loaded BC model from {model_path} with "
            f"{info.get('num_train_steps', 0)} train steps"
        )

    def _joint_cb(self, msg: JointState):
        if len(msg.position) < 6:
            return
        self.latest_joint_positions = [float(v) for v in msg.position[:6]]
        velocities = list(msg.velocity[:6]) if len(msg.velocity) >= 6 else [0.0] * 6
        self.latest_joint_velocities = [float(v) for v in velocities]

    def _box_cb(self, msg: PoseStamped):
        self.latest_box_position = [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ]
        self.latest_box_orientation = [
            float(msg.pose.orientation.w),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
        ]

    def _step(self):
        if self.latest_joint_positions is None or self.latest_joint_velocities is None:
            return

        features = np.asarray(
            self.latest_joint_positions
            + self.latest_joint_velocities
            + self.latest_box_position
            + self.latest_box_orientation,
            dtype=np.float64,
        )
        features_norm = (features - self.x_mean) / self.x_std
        pred_norm = features_norm @ self.weights + self.bias
        pred = pred_norm * self.y_std + self.y_mean
        pred = np.clip(pred, self.joint_limits[:, 0], self.joint_limits[:, 1])

        # A small smoothing term makes the baseline less twitchy on sparse demos.
        if self.last_command is not None:
            pred = 0.35 * pred + 0.65 * self.last_command
        self.last_command = pred

        payload = json.dumps(pred.tolist()).encode("utf-8")
        self.sock.sendto(payload, self.bridge_addr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/bc_policy_latest.npz"))
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=9876)
    args = parser.parse_args()

    rclpy.init()
    node = BCPolicyNode(
        model_path=args.model,
        control_hz=args.control_hz,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
