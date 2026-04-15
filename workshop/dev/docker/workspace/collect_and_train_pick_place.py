#!/usr/bin/env python3
"""Collect deterministic pick-place episodes, then train the BC baseline."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> int:
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--demos-root", type=Path, default=Path("demos"))
    parser.add_argument("--model-output", type=Path, default=Path("models/bc_policy_latest.npz"))
    parser.add_argument("--place-x", type=float, default=0.32)
    parser.add_argument("--place-y", type=float, default=0.20)
    parser.add_argument("--place-z", type=float, default=0.025)
    parser.add_argument("--success-tolerance-m", type=float, default=0.03)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=1.35)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    args.demos_root.mkdir(parents=True, exist_ok=True)
    successful = 0
    attempted = 0

    for idx in range(args.episodes):
        episode_name = time.strftime(f"demo_%Y%m%d_%H%M%S_{idx:03d}")
        recorder_cmd = [
            sys.executable,
            "record_demos.py",
            "--output-root",
            str(args.demos_root),
            "--episode-name",
            episode_name,
            "--sample-hz",
            str(args.sample_hz),
            "--target-x",
            str(args.place_x),
            "--target-y",
            str(args.place_y),
            "--target-z",
            str(args.place_z),
            "--success-tolerance-m",
            str(args.success_tolerance_m),
        ]
        if args.no_images:
            recorder_cmd.append("--no-images")

        recorder = subprocess.Popen(recorder_cmd, cwd=str(workspace))
        time.sleep(1.0)

        auto_cmd = [
            sys.executable,
            "auto_pick_place.py",
            "--place-x",
            str(args.place_x),
            "--place-y",
            str(args.place_y),
            "--speed",
            str(args.speed),
        ]
        auto_rc = run(auto_cmd, cwd=workspace)
        recorder.send_signal(signal.SIGINT)
        recorder.wait(timeout=15)
        attempted += 1

        meta_path = args.demos_root / episode_name / "meta.json"
        if meta_path.exists():
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("success"):
                successful += 1

        print(
            f"[episode {idx + 1}/{args.episodes}] auto_rc={auto_rc} "
            f"successes={successful}/{attempted}"
        )

    train_cmd = [
        sys.executable,
        "train_bc_policy.py",
        "--demos-root",
        str(args.demos_root),
        "--output",
        str(args.model_output),
    ]
    if args.include_failures:
        train_cmd.append("--include-failures")
    train_rc = run(train_cmd, cwd=workspace)
    print(
        f"done episodes={attempted} successes={successful} "
        f"model={args.model_output} train_rc={train_rc}"
    )
    return train_rc


if __name__ == "__main__":
    raise SystemExit(main())
