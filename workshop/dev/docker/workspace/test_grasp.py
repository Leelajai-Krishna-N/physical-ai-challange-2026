#!/usr/bin/env python3
"""Close-and-lift test for tuning grasp alignment."""

import argparse

from auto_pick_place import dwell, move_to


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0.30)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.036)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    speed = max(0.2, args.speed)

    move_to(0.20, 0.00, 0.25, True, 1.0 * speed, "Home")
    move_to(args.x, args.y, 0.18, True, 1.2 * speed, "Above")
    move_to(args.x, args.y, args.z, True, 1.2 * speed, "Align")
    dwell(0.4)
    move_to(args.x, args.y, args.z, False, 0.8 * speed, "Close")
    dwell(0.8)
    move_to(args.x, args.y, 0.18, False, 1.5 * speed, "Lift")
    dwell(5.0)


if __name__ == "__main__":
    main()
