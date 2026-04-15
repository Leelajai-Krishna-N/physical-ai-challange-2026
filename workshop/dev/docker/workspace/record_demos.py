#!/usr/bin/env python3
"""Record MuJoCo pick-and-place demos into episode folders.

This recorder subscribes to:
- /joint_states
- /commanded_joint_targets
- /mujoco/red_box_pose
- /d435i/image (optional)

Each episode is written as:
    demos/<episode_name>/
      meta.json
      steps.csv
      rgb/000000.png ...
"""

import argparse
import csv
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState


@dataclass
class JointSnapshot:
    stamp_sec: float
    names: list[str]
    positions: list[float]
    velocities: list[float]


@dataclass
class BoxSnapshot:
    stamp_sec: float
    position: list[float]
    orientation: list[float]


def ros_time_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class DemoRecorder(Node):
    def __init__(
        self,
        output_dir: Path,
        sample_hz: float,
        save_images: bool,
        target_position: Optional[list[float]] = None,
        success_tolerance_m: float = 0.03,
    ):
        super().__init__("demo_recorder")
        self.output_dir = output_dir
        self.rgb_dir = output_dir / "rgb"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        self.target_position = target_position
        self.success_tolerance_m = float(success_tolerance_m)
        self.sample_period = 1.0 / max(sample_hz, 1.0)
        self.started_at = time.time()
        self.sample_index = 0
        self.image_index = 0
        self.last_image_path: Optional[str] = None
        self.last_joint_state: Optional[JointSnapshot] = None
        self.last_command: Optional[JointSnapshot] = None
        self.last_box_pose: Optional[BoxSnapshot] = None
        self.last_rgb_msg: Optional[Image] = None
        self._stopped = False

        self.steps_file = (self.output_dir / "steps.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.writer = csv.DictWriter(
            self.steps_file,
            fieldnames=[
                "sample_index",
                "wall_time",
                "joint_stamp",
                "command_stamp",
                "box_stamp",
                "joint_names",
                "joint_positions",
                "joint_velocities",
                "command_positions",
                "box_position",
                "box_orientation",
                "rgb_path",
            ],
        )
        self.writer.writeheader()

        self.create_subscription(JointState, "/joint_states", self._joint_cb, 20)
        self.create_subscription(
            JointState, "/commanded_joint_targets", self._command_cb, 20
        )
        self.create_subscription(
            PoseStamped, "/mujoco/red_box_pose", self._box_cb, 20
        )
        self.create_subscription(Image, "/d435i/image", self._rgb_cb, 10)
        self.create_timer(self.sample_period, self._sample_step)

    def placement_error(self) -> Optional[float]:
        if self.target_position is None or self.last_box_pose is None:
            return None
        return math.dist(self.last_box_pose.position, self.target_position)

    def _joint_cb(self, msg: JointState):
        self.last_joint_state = JointSnapshot(
            stamp_sec=ros_time_to_sec(msg.header.stamp),
            names=list(msg.name),
            positions=[float(v) for v in msg.position],
            velocities=[float(v) for v in msg.velocity],
        )

    def _command_cb(self, msg: JointState):
        self.last_command = JointSnapshot(
            stamp_sec=ros_time_to_sec(msg.header.stamp),
            names=list(msg.name),
            positions=[float(v) for v in msg.position],
            velocities=[float(v) for v in msg.velocity],
        )

    def _box_cb(self, msg: PoseStamped):
        self.last_box_pose = BoxSnapshot(
            stamp_sec=ros_time_to_sec(msg.header.stamp),
            position=[
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            orientation=[
                float(msg.pose.orientation.w),
                float(msg.pose.orientation.x),
                float(msg.pose.orientation.y),
                float(msg.pose.orientation.z),
            ],
        )

    def _rgb_cb(self, msg: Image):
        self.last_rgb_msg = msg

    def _write_rgb(self) -> Optional[str]:
        if not self.save_images or self.last_rgb_msg is None:
            return self.last_image_path

        msg = self.last_rgb_msg
        if msg.encoding != "rgb8":
            self.get_logger().warn(
                f"Unsupported image encoding {msg.encoding}; skipping RGB save",
                throttle_duration_sec=10.0,
            )
            return self.last_image_path

        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        image = PILImage.fromarray(rgb, mode="RGB")
        image_name = f"{self.image_index:06d}.png"
        image_path = self.rgb_dir / image_name
        image.save(image_path)
        self.image_index += 1
        self.last_image_path = f"rgb/{image_name}"
        self.last_rgb_msg = None
        return self.last_image_path

    def _sample_step(self):
        if self.last_joint_state is None or self.last_command is None:
            return

        rgb_path = self._write_rgb()
        row = {
            "sample_index": self.sample_index,
            "wall_time": round(time.time() - self.started_at, 6),
            "joint_stamp": self.last_joint_state.stamp_sec,
            "command_stamp": self.last_command.stamp_sec,
            "box_stamp": "" if self.last_box_pose is None else self.last_box_pose.stamp_sec,
            "joint_names": json.dumps(self.last_joint_state.names),
            "joint_positions": json.dumps(self.last_joint_state.positions),
            "joint_velocities": json.dumps(self.last_joint_state.velocities),
            "command_positions": json.dumps(self.last_command.positions),
            "box_position": json.dumps(
                [] if self.last_box_pose is None else self.last_box_pose.position
            ),
            "box_orientation": json.dumps(
                [] if self.last_box_pose is None else self.last_box_pose.orientation
            ),
            "rgb_path": "" if rgb_path is None else rgb_path,
        }
        self.writer.writerow(row)
        self.steps_file.flush()
        self.sample_index += 1

    def close(self, success: bool, notes: str):
        if self._stopped:
            return
        self._stopped = True
        placement_error_m = self.placement_error()
        auto_success = success
        if placement_error_m is not None:
            auto_success = placement_error_m <= self.success_tolerance_m
        meta = {
            "episode_name": self.output_dir.name,
            "created_at_unix": self.started_at,
            "duration_sec": round(time.time() - self.started_at, 3),
            "num_samples": self.sample_index,
            "num_images": self.image_index,
            "save_images": self.save_images,
            "success": auto_success,
            "manual_success_flag": success,
            "notes": notes,
            "target_position": self.target_position,
            "success_tolerance_m": self.success_tolerance_m,
            "placement_error_m": placement_error_m,
            "topics": {
                "joint_states": "/joint_states",
                "commanded_joint_targets": "/commanded_joint_targets",
                "red_box_pose": "/mujoco/red_box_pose",
                "rgb": "/d435i/image",
            },
        }
        (self.output_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        self.steps_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("demos"),
        help="Directory where episode folders will be created",
    )
    parser.add_argument(
        "--episode-name",
        type=str,
        default=None,
        help="Episode folder name. Defaults to demo_YYYYmmdd_HHMMSS",
    )
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=10.0,
        help="How often to write synchronized state/action rows",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip RGB image saving and record only state/action tables",
    )
    parser.add_argument(
        "--success",
        action="store_true",
        help="Mark the recorded episode as successful in meta.json",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-form notes stored in meta.json",
    )
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-z", type=float, default=None)
    parser.add_argument(
        "--success-tolerance-m",
        type=float,
        default=0.03,
        help="Auto-success threshold from final box position to target",
    )
    args = parser.parse_args()

    episode_name = args.episode_name or time.strftime("demo_%Y%m%d_%H%M%S")
    output_dir = args.output_root / episode_name
    output_dir.mkdir(parents=True, exist_ok=True)

    target_position = None
    if (
        args.target_x is not None
        and args.target_y is not None
        and args.target_z is not None
    ):
        target_position = [float(args.target_x), float(args.target_y), float(args.target_z)]

    rclpy.init()
    recorder = DemoRecorder(
        output_dir=output_dir,
        sample_hz=args.sample_hz,
        save_images=not args.no_images,
        target_position=target_position,
        success_tolerance_m=args.success_tolerance_m,
    )

    stop_requested = {"value": False}

    def request_stop(_signum=None, _frame=None):
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    recorder.get_logger().info(f"Recording episode into {output_dir}")
    try:
        while rclpy.ok() and not stop_requested["value"]:
            rclpy.spin_once(recorder, timeout_sec=0.1)
    finally:
        recorder.close(success=args.success, notes=args.notes)
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(output_dir)


if __name__ == "__main__":
    main()
