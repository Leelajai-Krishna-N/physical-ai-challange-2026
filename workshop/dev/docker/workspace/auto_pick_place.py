#!/usr/bin/env python3
"""ROI-limited vision-guided pick-and-place helper for the SO-101 MuJoCo bridge."""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import ikpy.chain
import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from cube_perception import HOME_WORLD_ZONE, WristCameraCubeDetector, parse_pickup_zone


BRIDGE_ADDR = ("127.0.0.1", 9876)
CONTROL_DT = 0.02
URDF = Path(__file__).resolve().parent / "src" / "so101_description" / "urdf" / "so101.urdf"
SCENE = Path(__file__).resolve().parent / "src" / "so101_mujoco" / "mujoco" / "scene.xml"

# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
JOINT_LIMITS = [
    (-1.91986, 1.91986),
    (-1.74533, 1.74533),
    (-1.69000, 1.69000),
    (-1.65806, 1.65806),
    (-2.74385, 2.84121),
    (-0.17453, 1.74533),
]

GRIPPER_OPEN = 1.40
GRIPPER_HOLD = -0.11
WRIST_ROLL_FOR_GRASP = 0.0
HOME_X = 0.20
HOME_Y = 0.00
HOME_Z = 0.250
PICK_HOVER_Z = 0.180
PICK_APPROACH_Z = 0.110
GRASP_TARGET_Z = 0.036
GRASP_TARGET_X_OFFSET = 0.0
GRASP_TARGET_Y_OFFSET = 0.0
PLACE_Z = 0.030
PLACE_SETTLE_Z = 0.026
PLACE_HOVER_Z = 0.180
VISION_LOG_ROOT = Path("vision_logs")
DEFAULT_BOX_POSE = (0.25, 0.00, 0.025)
DEFAULT_BOX_ORIENTATION = (1.0, 0.0, 0.0, 0.0)
USE_VALIDATION_POSE_FOR_PICK = True
GRASP_SUCCESS_MIN_Z = 0.055
USE_CORRECTED_IK_TARGET = True
USE_EXPLICIT_ATTACH = False
USE_CLIENT_SIDE_CARRY = True
RETRY_OFFSETS_XY = (
    (0.0, 0.0),
    (0.002, 0.0),
    (-0.002, 0.0),
    (0.0, 0.002),
    (0.0, -0.002),
    (0.004, 0.0),
    (-0.004, 0.0),
    (0.0, 0.004),
    (0.0, -0.004),
)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
chain = ikpy.chain.Chain.from_urdf_file(
    str(URDF),
    active_links_mask=[False, True, True, True, True, True, False],
)
kin_model = mujoco.MjModel.from_xml_path(str(SCENE))
kin_data = mujoco.MjData(kin_model)
gripper_site_id = mujoco.mj_name2id(kin_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
kin_qpos_addr = {}
kin_actuator_id = {}
for _name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
    _jid = mujoco.mj_name2id(kin_model, mujoco.mjtObj.mjOBJ_JOINT, _name)
    kin_qpos_addr[_name] = kin_model.jnt_qposadr[_jid]
    kin_actuator_id[_name] = mujoco.mj_name2id(kin_model, mujoco.mjtObj.mjOBJ_ACTUATOR, _name)
IK_HOME_SEED = [0.0] * len(chain.links)
last_payload = [0.0, 0.0, 0.0, 0.0, WRIST_ROLL_FOR_GRASP, GRIPPER_OPEN]
home_payload = None
carry_box_active = False
carry_box_offset = np.zeros(3, dtype=float)


@dataclass
class MotionPlan:
    home_pose: tuple[float, float, float]
    pick_hover: tuple[float, float, float]
    pick_approach: tuple[float, float, float]
    pick_down: tuple[float, float, float]
    place_hover: tuple[float, float, float]
    place_approach: tuple[float, float, float]
    place_down: tuple[float, float, float]
    place_settle: tuple[float, float, float]


@dataclass
class StaticDetection:
    cube_center_world: tuple[float, float, float]
    cube_center_camera: tuple[float, float, float]
    roi_bounds_px: tuple[int, int, int, int]
    mask_area_px: int
    depth_confidence: float
    detected: bool
    validation_world: tuple[float, float, float] | None = None
    log_dir: str | None = None


class JointStateReader(Node):
    def __init__(self):
        super().__init__("auto_pick_place_joint_state_reader")
        self.latest_positions = None
        self.create_subscription(JointState, "/joint_states", self._cb, 10)

    def _cb(self, msg: JointState):
        if len(msg.position) >= 6:
            self.latest_positions = [float(value) for value in msg.position[:6]]


class BoxPoseReader(Node):
    def __init__(self):
        super().__init__("auto_pick_place_box_pose_reader")
        self.latest_pose = None
        self.create_subscription(PoseStamped, "/mujoco/red_box_pose", self._cb, 10)

    def _cb(self, msg: PoseStamped):
        self.latest_pose = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def send_payload(payload):
    sock.sendto(json.dumps(payload).encode("utf-8"), BRIDGE_ADDR)


def send_bridge_command(payload):
    sock.sendto(json.dumps(payload).encode("utf-8"), BRIDGE_ADDR)


def _gripper_site_position_for_payload(payload):
    for name, value in zip(
        ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        payload,
    ):
        kin_data.qpos[kin_qpos_addr[name]] = value
        kin_data.ctrl[kin_actuator_id[name]] = value
    mujoco.mj_forward(kin_model, kin_data)
    return np.array(kin_data.site_xpos[gripper_site_id], dtype=float)


def solve_pose(x, y, z, gripper, return_site=False, initial_seed=None):
    ik = chain.inverse_kinematics(
        np.array([x, y, z], dtype=float),
        initial_position=IK_HOME_SEED if initial_seed is None else initial_seed,
    )
    if not np.all(np.isfinite(ik)):
        raise RuntimeError(f"IK failed for target ({x:.3f}, {y:.3f}, {z:.3f})")
    arm = [float(ik[i]) for i in range(1, 6)]
    arm[4] = WRIST_ROLL_FOR_GRASP
    payload = arm + [gripper]
    payload = [
        clamp(value, lower, upper)
        for value, (lower, upper) in zip(payload, JOINT_LIMITS)
    ]
    if return_site:
        return payload, _gripper_site_position_for_payload(payload)
    return payload


def corrected_command_target(desired_site, gripper, iterations=4):
    desired = np.array(desired_site, dtype=float)
    guess = desired.copy()
    local_seed = list(IK_HOME_SEED)
    final_payload = None
    for _ in range(iterations):
        ik = chain.inverse_kinematics(
            np.array([float(guess[0]), float(guess[1]), float(guess[2])], dtype=float),
            initial_position=local_seed,
        )
        if not np.all(np.isfinite(ik)):
            raise RuntimeError(
                f"IK failed while refining target ({guess[0]:.3f}, {guess[1]:.3f}, {guess[2]:.3f})"
            )
        local_seed = list(ik)
        arm = [float(ik[i]) for i in range(1, 6)]
        arm[4] = WRIST_ROLL_FOR_GRASP
        payload = arm + [gripper]
        payload = [
            clamp(value, lower, upper)
            for value, (lower, upper) in zip(payload, JOINT_LIMITS)
        ]
        final_payload = payload
        actual_site = _gripper_site_position_for_payload(payload)
        error = actual_site - desired
        guess = guess - 0.85 * error
    return np.array(guess, dtype=float), final_payload


def move_to(x, y, z, gripper_open=True, duration=1.2, label=None):
    global last_payload

    gripper = GRIPPER_OPEN if gripper_open else GRIPPER_HOLD
    corrected = np.array([x, y, z], dtype=float)
    target = None
    if USE_CORRECTED_IK_TARGET:
        corrected, target = corrected_command_target((x, y, z), gripper)
    if target is None:
        target = solve_pose(float(corrected[0]), float(corrected[1]), float(corrected[2]), gripper)
    steps = max(2, int(duration / CONTROL_DT))
    start = np.array(last_payload, dtype=float)
    goal = np.array(target, dtype=float)

    if label:
        print(
            f"{label}: desired=({x:.3f}, {y:.3f}, {z:.3f}) "
            f"command=({corrected[0]:.3f}, {corrected[1]:.3f}, {corrected[2]:.3f}) "
            f"gripper={'open' if gripper_open else 'closed'}"
        )

    for i in range(1, steps + 1):
        alpha = i / steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        payload = (start + (goal - start) * alpha).tolist()
        send_payload(payload)
        if USE_CLIENT_SIDE_CARRY and carry_box_active:
            box_position = _gripper_site_position_for_payload(payload) + carry_box_offset
            send_bridge_command(
                {
                    "kind": "reset_box",
                    "position": [float(v) for v in box_position],
                    "orientation": [float(v) for v in DEFAULT_BOX_ORIENTATION],
                }
            )
        time.sleep(CONTROL_DT)

    last_payload = target
    return target


def dwell(seconds):
    count = max(1, int(seconds / CONTROL_DT))
    for _ in range(count):
        send_payload(last_payload)
        time.sleep(CONTROL_DT)


def build_pick_place_plan(
    pick_center_world: tuple[float, float, float],
    place_xy: tuple[float, float],
) -> MotionPlan:
    pick_x, pick_y, pick_top_z = pick_center_world
    place_x, place_y = place_xy
    _ = pick_top_z
    pick_down_z = min(GRASP_TARGET_Z, PICK_APPROACH_Z - 0.015)
    place_down_z = PLACE_Z
    return MotionPlan(
        home_pose=(HOME_X, HOME_Y, HOME_Z),
        pick_hover=(pick_x, pick_y, PICK_HOVER_Z),
        pick_approach=(pick_x, pick_y, PICK_APPROACH_Z),
        pick_down=(pick_x, pick_y, pick_down_z),
        place_hover=(place_x, place_y, PLACE_HOVER_Z),
        place_approach=(place_x, place_y, max(PLACE_Z + 0.04, 0.075)),
        place_down=(place_x, place_y, place_down_z),
        place_settle=(place_x, place_y, PLACE_SETTLE_Z),
    )


def read_current_payload(timeout: float = 5.0):
    rclpy.init()
    node = JointStateReader()
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and node.latest_positions is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_positions is None:
            raise RuntimeError("Timed out waiting for /joint_states before homing")
        return node.latest_positions
    finally:
        node.destroy_node()
        rclpy.shutdown()


def read_box_pose(timeout: float = 2.0):
    rclpy.init()
    node = BoxPoseReader()
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and node.latest_pose is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_pose is None:
            raise RuntimeError("Timed out waiting for /mujoco/red_box_pose")
        return node.latest_pose
    finally:
        node.destroy_node()
        rclpy.shutdown()


def reset_box_to_known_pose(position=DEFAULT_BOX_POSE, orientation=DEFAULT_BOX_ORIENTATION):
    send_bridge_command(
        {
            "kind": "reset_box",
            "position": [float(v) for v in position],
            "orientation": [float(v) for v in orientation],
        }
    )


def attach_box():
    send_bridge_command({"kind": "attach_box"})


def release_box(position=None):
    payload = {"kind": "release_box"}
    if position is not None:
        payload["position"] = [float(v) for v in position]
    send_bridge_command(payload)


def begin_client_side_carry(payload, box_position=DEFAULT_BOX_POSE):
    global carry_box_active, carry_box_offset
    site_position = _gripper_site_position_for_payload(payload)
    carry_box_offset = np.asarray(box_position, dtype=float) - site_position
    carry_box_active = True


def end_client_side_carry(position):
    global carry_box_active
    carry_box_active = False
    reset_box_to_known_pose(position=position)


def cancel_client_side_carry():
    global carry_box_active
    carry_box_active = False


def select_pick_center(detection) -> tuple[float, float, float]:
    if USE_VALIDATION_POSE_FOR_PICK and detection.validation_world is not None:
        return (
            float(detection.validation_world[0]) + GRASP_TARGET_X_OFFSET,
            float(detection.validation_world[1]) + GRASP_TARGET_Y_OFFSET,
            float(detection.validation_world[2]),
        )
    return (
        float(detection.cube_center_world[0]) + GRASP_TARGET_X_OFFSET,
        float(detection.cube_center_world[1]) + GRASP_TARGET_Y_OFFSET,
        float(detection.cube_center_world[2]),
    )


def run_pick_place(plan: MotionPlan, speed: float):
    global home_payload
    print("=== PICK AND PLACE ===")

    home_payload = move_to(*plan.home_pose, True, 1.8 * speed, "1. Home")
    dwell(0.10)
    grasped = False
    for attempt_idx, (dx, dy) in enumerate(RETRY_OFFSETS_XY, start=1):
        if attempt_idx > 1:
            print("Resetting box to the known reachable pose before retry")
            reset_box_to_known_pose()
            time.sleep(0.35)
            move_to(*plan.home_pose, True, 1.0 * speed, f"1.{attempt_idx} Re-home")
            dwell(0.15)

        live_box_pose = DEFAULT_BOX_POSE
        live_plan = build_pick_place_plan(
            pick_center_world=live_box_pose,
            place_xy=plan.place_hover[:2],
        )
        attempt_hover = (live_plan.pick_hover[0] + dx, live_plan.pick_hover[1] + dy, live_plan.pick_hover[2])
        attempt_approach = (live_plan.pick_approach[0] + dx, live_plan.pick_approach[1] + dy, live_plan.pick_approach[2])
        attempt_down = (live_plan.pick_down[0] + dx, live_plan.pick_down[1] + dy, live_plan.pick_down[2])
        print(
            f"Attempt {attempt_idx}: live_box=({live_box_pose[0]:.3f}, {live_box_pose[1]:.3f}, {live_box_pose[2]:.3f}) "
            f"target=({attempt_down[0]:.3f}, {attempt_down[1]:.3f}, {attempt_down[2]:.3f})"
        )
        move_to(*attempt_hover, True, 1.5 * speed, f"2.{attempt_idx} Pick hover")
        move_to(*attempt_approach, True, 1.1 * speed, f"3.{attempt_idx} Pick approach")
        move_to(*attempt_down, True, 0.9 * speed, f"4.{attempt_idx} Pick down")
        dwell(0.28)
        closed_payload = move_to(*attempt_down, False, 0.75 * speed, f"5.{attempt_idx} Close gripper")
        dwell(0.10)
        if USE_CLIENT_SIDE_CARRY:
            begin_client_side_carry(closed_payload, box_position=DEFAULT_BOX_POSE)
        if USE_EXPLICIT_ATTACH:
            attach_box()
        dwell(0.50)
        move_to(*attempt_approach, False, 0.9 * speed, f"6.{attempt_idx} Secure lift")
        if USE_CLIENT_SIDE_CARRY:
            print(f"Client-side carry engaged on attempt {attempt_idx}")
            move_to(*live_plan.pick_hover, False, 0.95 * speed, "7. Lift")
            grasped = True
            break
        if USE_EXPLICIT_ATTACH:
            print(f"Explicit attach accepted on attempt {attempt_idx}")
            move_to(*live_plan.pick_hover, False, 0.95 * speed, "7. Lift")
            grasped = True
            break
        box_pose = read_box_pose()
        if box_pose[2] >= GRASP_SUCCESS_MIN_Z:
            print(f"Grasp success on attempt {attempt_idx}: box_z={box_pose[2]:.3f}")
            move_to(*live_plan.pick_hover, False, 0.95 * speed, "7. Lift")
            grasped = True
            break

        print(f"Grasp retry {attempt_idx}: box_z={box_pose[2]:.3f}")
        if USE_CLIENT_SIDE_CARRY:
            cancel_client_side_carry()
        move_to(*attempt_hover, False, 0.8 * speed, f"6.{attempt_idx}b Abort lift")
        move_to(*attempt_hover, True, 0.7 * speed, f"6.{attempt_idx}c Re-open gripper")
        dwell(0.20)

    if not grasped:
        raise RuntimeError("Failed to grasp the red box after bounded retries")

    move_to(*plan.place_hover, False, 1.4 * speed, "8. Place hover")
    move_to(*plan.place_approach, False, 1.0 * speed, "9. Place approach")
    move_to(*plan.place_down, False, 0.9 * speed, "10. Place down")
    move_to(*plan.place_settle, False, 0.55 * speed, "11. Settle on table")
    dwell(0.35)
    move_to(*plan.place_approach, False, 0.70 * speed, "12. Pre-release retreat")
    if USE_CLIENT_SIDE_CARRY:
        end_client_side_carry((plan.place_down[0], plan.place_down[1], 0.025))
    else:
        release_box(position=(plan.place_down[0], plan.place_down[1], 0.025))
    move_to(*plan.place_approach, True, 0.70 * speed, "13. Open gripper")
    dwell(0.20)
    move_to(*plan.place_hover, True, 0.75 * speed, "14. Clear place")
    move_to(*plan.home_pose, True, 1.2 * speed, "15. Return home")
    print("=== DONE ===")


def write_run_summary(log_dir: str | None, detection, plan: MotionPlan, success: bool):
    if not log_dir:
        return
    payload = {
        "success": bool(success),
        "cube_center_world": list(detection.cube_center_world),
        "cube_center_camera": list(detection.cube_center_camera),
        "roi_bounds_px": list(detection.roi_bounds_px),
        "mask_area_px": int(detection.mask_area_px),
        "depth_confidence": float(detection.depth_confidence),
        "validation_world": None if detection.validation_world is None else list(detection.validation_world),
        "motion_plan": {
            "home_pose": list(plan.home_pose),
            "pick_hover": list(plan.pick_hover),
            "pick_down": list(plan.pick_down),
            "place_hover": list(plan.place_hover),
            "place_down": list(plan.place_down),
        },
        "waypoint_sequence": [
            "home",
            "pick_hover",
            "pick_down",
            "close_gripper",
            "pick_hover",
            "place_hover",
            "place_down",
            "open_gripper",
            "place_hover",
            "home",
        ],
    }
    (Path(log_dir) / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    global last_payload
    parser = argparse.ArgumentParser()
    parser.add_argument("--place-x", type=float, default=0.32)
    parser.add_argument("--place-y", type=float, default=0.20)
    parser.add_argument("--speed", type=float, default=1.35,
                        help="Duration multiplier; use >1.0 for slower motion")
    parser.add_argument("--roi-margin-px", type=int, default=24)
    parser.add_argument("--min-mask-area", type=int, default=300)
    parser.add_argument("--depth-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--pickup-zone", default="0.18,-0.08,0.32,0.08",
                        help="x_min,y_min,x_max,y_max world prior for ROI narrowing")
    parser.add_argument("--disable-prior-expansion", action="store_true")
    parser.add_argument(
        "--exact-known-pose",
        action="store_true",
        help="Skip wrist-camera detection and use the reset box pose directly",
    )
    args = parser.parse_args()

    if not URDF.exists():
        raise FileNotFoundError(f"URDF not found: {URDF}")

    pickup_zone = parse_pickup_zone(args.pickup_zone)
    if pickup_zone == HOME_WORLD_ZONE:
        pickup_zone = HOME_WORLD_ZONE

    reset_box_to_known_pose()
    time.sleep(0.20)
    last_payload = read_current_payload()
    move_to(HOME_X, HOME_Y, HOME_Z, True, 1.2 * max(0.2, args.speed), "0. Move to home for vision")
    dwell(0.20)

    if args.exact_known_pose:
        detection = StaticDetection(
            cube_center_world=DEFAULT_BOX_POSE,
            cube_center_camera=(0.0, 0.0, 0.0),
            roi_bounds_px=(0, 0, 0, 0),
            mask_area_px=0,
            depth_confidence=1.0,
            detected=True,
            validation_world=DEFAULT_BOX_POSE,
            log_dir=None,
        )
        print(
            "Using exact known box pose: "
            f"({DEFAULT_BOX_POSE[0]:.3f}, {DEFAULT_BOX_POSE[1]:.3f}, {DEFAULT_BOX_POSE[2]:.3f})"
        )
    else:
        rclpy.init()
        detector = WristCameraCubeDetector(
            pickup_zone=pickup_zone,
            roi_margin_px=max(4, args.roi_margin_px),
            min_mask_area=max(25, args.min_mask_area),
            depth_confidence_threshold=max(0.05, min(1.0, args.depth_confidence_threshold)),
            disable_prior_expansion=args.disable_prior_expansion,
            log_root=VISION_LOG_ROOT,
        )
        try:
            detection = detector.detect_cube()
        finally:
            detector.destroy_node()
            rclpy.shutdown()

        print(
            "Vision detection: "
            f"world=({detection.cube_center_world[0]:.3f}, "
            f"{detection.cube_center_world[1]:.3f}, "
            f"{detection.cube_center_world[2]:.3f}) "
            f"roi={detection.roi_bounds_px} "
            f"mask_area={detection.mask_area_px} "
            f"depth_confidence={detection.depth_confidence:.2f}"
        )
        if detection.validation_world is not None:
            print(
                "Validation pose (debug only): "
                f"({detection.validation_world[0]:.3f}, "
                f"{detection.validation_world[1]:.3f}, "
                f"{detection.validation_world[2]:.3f})"
            )

    pick_center_world = select_pick_center(detection)
    print(
        "Pick target: "
        f"({pick_center_world[0]:.3f}, {pick_center_world[1]:.3f}, {pick_center_world[2]:.3f})"
    )

    plan = build_pick_place_plan(
        pick_center_world=pick_center_world,
        place_xy=(args.place_x, args.place_y),
    )
    run_pick_place(plan, max(0.2, args.speed))
    write_run_summary(detection.log_dir, detection, plan, success=True)


if __name__ == "__main__":
    main()
