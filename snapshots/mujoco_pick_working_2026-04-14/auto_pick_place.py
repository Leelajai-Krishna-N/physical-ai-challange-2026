#!/usr/bin/env python3
"""UDP pick-and-place helper for the SO-101 MuJoCo bridge.

Run the bridge first:
    python3 src/so101_mujoco/scripts/so101_mujoco_bridge.py --model src/so101_mujoco/mujoco/scene.xml

Then run this script from /home/hacker/workspace.
"""

import argparse
import json
import socket
import time
from pathlib import Path

import ikpy.chain
import numpy as np


BRIDGE_ADDR = ("127.0.0.1", 9876)
CONTROL_DT = 0.02
URDF = Path(__file__).resolve().parent / "src" / "so101_description" / "urdf" / "so101.urdf"

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

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
chain = ikpy.chain.Chain.from_urdf_file(
    str(URDF),
    active_links_mask=[False, True, True, True, True, True, False],
)
ik_seed = [0.0] * len(chain.links)
last_payload = [0.0, 0.0, 0.0, 0.0, WRIST_ROLL_FOR_GRASP, GRIPPER_OPEN]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def send_payload(payload):
    sock.sendto(json.dumps(payload).encode("utf-8"), BRIDGE_ADDR)


def solve_pose(x, y, z, gripper):
    global ik_seed

    ik = chain.inverse_kinematics(
        np.array([x, y, z], dtype=float),
        initial_position=ik_seed,
    )
    if not np.all(np.isfinite(ik)):
        raise RuntimeError(f"IK failed for target ({x:.3f}, {y:.3f}, {z:.3f})")

    ik_seed = list(ik)
    arm = [float(ik[i]) for i in range(1, 6)]
    arm[4] = WRIST_ROLL_FOR_GRASP
    payload = arm + [gripper]
    return [
        clamp(value, lower, upper)
        for value, (lower, upper) in zip(payload, JOINT_LIMITS)
    ]


def move_to(x, y, z, gripper_open=True, duration=1.5, label=None):
    global last_payload

    gripper = GRIPPER_OPEN if gripper_open else GRIPPER_HOLD
    target = solve_pose(x, y, z, gripper)
    steps = max(2, int(duration / CONTROL_DT))
    start = np.array(last_payload, dtype=float)
    goal = np.array(target, dtype=float)

    if label:
        print(f"{label}: x={x:.3f}, y={y:.3f}, z={z:.3f}, gripper={'open' if gripper_open else 'closed'}")

    for i in range(1, steps + 1):
        alpha = i / steps
        # Smoothstep avoids abrupt starts/stops that can knock the object away.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        payload = (start + (goal - start) * alpha).tolist()
        send_payload(payload)
        time.sleep(CONTROL_DT)

    last_payload = target
    return True


def dwell(seconds, repeats=10):
    count = max(1, int(seconds / CONTROL_DT))
    for _ in range(count):
        send_payload(last_payload)
        time.sleep(CONTROL_DT)


def run_pick_place(object_xy, place_xy, speed):
    ox, oy = object_xy
    px, py = place_xy

    table_clearance = 0.030
    grasp_z = 0.015
    lift_z = 0.180
    travel_z = 0.250
    place_z = 0.035

    print("=== PICK AND PLACE ===")
    move_to(0.20, 0.00, travel_z, True, 2.0 * speed, "1. Home")
    dwell(0.25)

    move_to(ox, oy, lift_z, True, 1.8 * speed, "2. Above object")
    move_to(ox, oy, grasp_z + table_clearance, True, 1.2 * speed, "3. Pre-grasp")
    move_to(ox, oy, grasp_z, True, 1.0 * speed, "4. Align jaws")
    dwell(0.30)

    move_to(ox, oy, grasp_z, False, 0.8 * speed, "5. Close gripper")
    dwell(0.80)

    move_to(ox, oy, lift_z, False, 1.5 * speed, "6. Lift")
    move_to(px, py, travel_z, False, 2.0 * speed, "7. Transfer")
    move_to(px, py, place_z, False, 1.4 * speed, "8. Lower")
    dwell(0.30)

    move_to(px, py, place_z, True, 0.8 * speed, "9. Release")
    dwell(0.40)
    move_to(px, py, lift_z, True, 1.2 * speed, "10. Retreat")
    move_to(0.20, 0.00, travel_z, True, 1.8 * speed, "11. Home")
    print("=== DONE ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-x", type=float, default=0.22)
    parser.add_argument("--object-y", type=float, default=-0.12)
    parser.add_argument("--place-x", type=float, default=0.32)
    parser.add_argument("--place-y", type=float, default=0.20)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Duration multiplier; use >1.0 for slower motion")
    args = parser.parse_args()

    if not URDF.exists():
        raise FileNotFoundError(f"URDF not found: {URDF}")

    run_pick_place(
        (args.object_x, args.object_y),
        (args.place_x, args.place_y),
        max(0.2, args.speed),
    )


if __name__ == "__main__":
    main()
