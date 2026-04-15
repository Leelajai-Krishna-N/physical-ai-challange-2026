#!/usr/bin/env python3
"""Interactive Z-height tuner for the SO-101 simulated grasp."""

import argparse
from auto_pick_place import dwell, move_to


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0.25)
    parser.add_argument("--y", type=float, default=0.00)
    args = parser.parse_args()

    move_to(0.20, 0.00, 0.25, gripper_open=True, duration=2.0, label="Home")
    move_to(args.x, args.y, 0.13, gripper_open=True, duration=1.5, label="Above object")

    for z in [0.060, 0.055, 0.050, 0.045, 0.040, 0.036, 0.032, 0.028]:
        print(
            f"Testing Z={z:.3f}. Watch the gripper alignment, then press ENTER "
            "to go lower or Ctrl+C to stop."
        )
        move_to(args.x, args.y, z, gripper_open=True, duration=0.9, label="Test")
        dwell(0.3)
        input()


if __name__ == "__main__":
    main()
