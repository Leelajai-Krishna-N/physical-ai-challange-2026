#!/usr/bin/env python3
"""Move to a single Cartesian pose for visual tuning."""

import argparse

from auto_pick_place import dwell, move_to


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float)
    parser.add_argument("--closed", action="store_true")
    parser.add_argument("--hold", type=float, default=5.0)
    args = parser.parse_args()

    move_to(args.x, args.y, 0.18, not args.closed, duration=1.5, label="Above")
    move_to(args.x, args.y, args.z, not args.closed, duration=1.5, label="Pose")
    dwell(args.hold)


if __name__ == "__main__":
    main()
