#!/usr/bin/env python3
import json
import socket
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_LIMITS = np.array(
    [
        (-1.91986, 1.91986),
        (-1.74533, 1.74533),
        (-1.69000, 1.69000),
        (-1.65806, 1.65806),
        (-2.74385, 2.84121),
        (0.0, 1.74),
    ],
    dtype=float,
)

HOME_JOINTS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.74], dtype=float)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class RobotController(Node):
    def __init__(self):
        super().__init__("so101_joint_button_relay")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mujoco_bridge_addr = ("127.0.0.1", 9876)
        self.current = HOME_JOINTS.copy()
        self.last_joint_state_time = 0.0
        self.has_joint_state = False
        self.lock = threading.Lock()

        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 20)

    def _joint_state_cb(self, msg: JointState):
        if len(msg.position) < len(JOINT_NAMES):
            return
        name_to_pos = {name: float(pos) for name, pos in zip(msg.name, msg.position)}
        with self.lock:
            updated = []
            for idx, name in enumerate(JOINT_NAMES):
                if name in name_to_pos:
                    updated.append(name_to_pos[name])
                else:
                    updated.append(float(self.current[idx]))
            self.current = np.array(updated, dtype=float)
            self.has_joint_state = True
            self.last_joint_state_time = time.time()

    def wait_for_joint_state(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self.lock:
                if self.has_joint_state:
                    return True
        return False

    def apply_command(self, payload):
        rclpy.spin_once(self, timeout_sec=0.0)
        kind = payload.get("kind")
        with self.lock:
            current = self.current.copy()

        target = current.copy()
        if kind == "joint_step":
            idx = int(payload.get("joint", -1))
            if 0 <= idx < 5:
                delta = float(payload.get("delta", 0.0))
                target[idx] = float(
                    np.clip(current[idx] + delta, JOINT_LIMITS[idx, 0], JOINT_LIMITS[idx, 1])
                )
            else:
                return
        elif kind == "gripper":
            grip_target = float(payload.get("target", HOME_JOINTS[5]))
            target[5] = float(np.clip(grip_target, JOINT_LIMITS[5, 0], JOINT_LIMITS[5, 1]))
        elif kind == "home":
            target = HOME_JOINTS.copy()
        else:
            return

        self._send_target(target)

    def _send_target(self, target: np.ndarray):
        payload = json.dumps([float(v) for v in target])
        encoded = payload.encode("utf-8")
        # Send a short burst so a single dropped UDP packet doesn't make a click look dead.
        for _ in range(3):
            self.sock.sendto(encoded, self.mujoco_bridge_addr)
            time.sleep(0.01)


def main():
    rclpy.init(args=None)
    controller = RobotController()
    if not controller.wait_for_joint_state():
        print("relay failed to start: no /joint_states received", flush=True)
        controller.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

    print("joint controller relay ready", flush=True)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                rclpy.spin_once(controller, timeout_sec=0.0)
                continue
            try:
                controller.apply_command(json.loads(line))
            except Exception as exc:
                print(f"relay parse error: {exc}", file=sys.stderr, flush=True)
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
