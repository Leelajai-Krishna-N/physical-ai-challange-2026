#!/usr/bin/env python3
"""Train a small behavioral-cloning baseline from recorded MuJoCo demos.

This stays intentionally simple:
- no torch dependency
- uses joint state + box pose as input
- predicts commanded joint targets

The saved model can be deployed with `run_bc_policy.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


JOINT_LIMITS = np.array(
    [
        [-1.91986, 1.91986],
        [-1.74533, 1.74533],
        [-1.69000, 1.69000],
        [-1.65806, 1.65806],
        [-2.74385, 2.84121],
        [-0.17453, 1.74533],
    ],
    dtype=np.float64,
)


@dataclass
class EpisodeData:
    name: str
    features: np.ndarray
    targets: np.ndarray


def parse_json_array(text: str) -> list[float]:
    return list(json.loads(text))


def load_episode(episode_dir: Path, min_command_norm: float) -> EpisodeData | None:
    steps_path = episode_dir / "steps.csv"
    if not steps_path.exists():
        return None

    rows_x: list[list[float]] = []
    rows_y: list[list[float]] = []
    with steps_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            joint_positions = parse_json_array(row["joint_positions"])
            joint_velocities = parse_json_array(row["joint_velocities"])
            command_positions = parse_json_array(row["command_positions"])
            box_position = parse_json_array(row["box_position"]) or [0.0, 0.0, 0.0]
            box_orientation = parse_json_array(row["box_orientation"]) or [1.0, 0.0, 0.0, 0.0]

            if len(joint_positions) != 6 or len(joint_velocities) != 6 or len(command_positions) != 6:
                continue
            if np.linalg.norm(np.asarray(command_positions, dtype=np.float64)) < min_command_norm:
                continue

            features = (
                joint_positions
                + joint_velocities
                + box_position
                + box_orientation
            )
            rows_x.append([float(v) for v in features])
            rows_y.append([float(v) for v in command_positions])

    if not rows_x:
        return None

    return EpisodeData(
        name=episode_dir.name,
        features=np.asarray(rows_x, dtype=np.float64),
        targets=np.asarray(rows_y, dtype=np.float64),
    )


def load_dataset(demos_root: Path, include_failures: bool, min_command_norm: float) -> list[EpisodeData]:
    episodes: list[EpisodeData] = []
    for episode_dir in sorted(p for p in demos_root.iterdir() if p.is_dir()):
        meta_path = episode_dir / "meta.json"
        success = True
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            success = bool(meta.get("success", True))
        if not include_failures and not success:
            continue
        episode = load_episode(episode_dir, min_command_norm=min_command_norm)
        if episode is not None:
            episodes.append(episode)
    return episodes


def split_episodes(
    episodes: list[EpisodeData], val_ratio: float, seed: int
) -> tuple[list[EpisodeData], list[EpisodeData]]:
    if len(episodes) <= 1 or val_ratio <= 0.0:
        return episodes, []
    rng = np.random.default_rng(seed)
    order = list(range(len(episodes)))
    rng.shuffle(order)
    episodes = [episodes[i] for i in order]
    split_index = max(1, int(round(len(episodes) * (1.0 - val_ratio))))
    split_index = min(split_index, len(episodes) - 1)
    return episodes[:split_index], episodes[split_index:]


def normalize(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < 1e-6] = 1.0
    return (data - mean) / std, mean, std


def fit_ridge_regression(features: np.ndarray, targets: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x_aug = np.concatenate([features, np.ones((features.shape[0], 1), dtype=features.dtype)], axis=1)
    reg = ridge * np.eye(x_aug.shape[1], dtype=features.dtype)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ targets)
    return weights[:-1], weights[-1]


def predict(features: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return features @ weights + bias


def clip_targets(targets: np.ndarray) -> np.ndarray:
    return np.clip(targets, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos-root", type=Path, default=Path("demos"))
    parser.add_argument("--output", type=Path, default=Path("models/bc_policy_latest.npz"))
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--min-command-norm", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    episodes = load_dataset(
        args.demos_root,
        include_failures=args.include_failures,
        min_command_norm=args.min_command_norm,
    )
    if not episodes:
        raise SystemExit("No usable episodes found under demos root.")

    train_eps, val_eps = split_episodes(episodes, args.val_ratio, args.seed)
    train_x = np.concatenate([ep.features for ep in train_eps], axis=0)
    train_y = np.concatenate([ep.targets for ep in train_eps], axis=0)

    train_x_norm, x_mean, x_std = normalize(train_x)
    train_y_norm, y_mean, y_std = normalize(train_y)

    weights, bias = fit_ridge_regression(train_x_norm, train_y_norm, ridge=args.ridge)

    train_pred_norm = predict(train_x_norm, weights, bias)
    train_pred = clip_targets(train_pred_norm * y_std + y_mean)

    model_info = {
        "episodes": [ep.name for ep in episodes],
        "train_episodes": [ep.name for ep in train_eps],
        "val_episodes": [ep.name for ep in val_eps],
        "num_train_steps": int(train_x.shape[0]),
        "feature_dim": int(train_x.shape[1]),
        "target_dim": int(train_y.shape[1]),
        "train_mse": mse(train_pred, train_y),
        "ridge": float(args.ridge),
        "min_command_norm": float(args.min_command_norm),
        "seed": int(args.seed),
    }

    if val_eps:
        val_x = np.concatenate([ep.features for ep in val_eps], axis=0)
        val_y = np.concatenate([ep.targets for ep in val_eps], axis=0)
        val_x_norm = (val_x - x_mean) / x_std
        val_pred_norm = predict(val_x_norm, weights, bias)
        val_pred = clip_targets(val_pred_norm * y_std + y_mean)
        model_info["num_val_steps"] = int(val_x.shape[0])
        model_info["val_mse"] = mse(val_pred, val_y)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        weights=weights,
        bias=bias,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        joint_limits=JOINT_LIMITS,
        model_info=json.dumps(model_info, indent=2),
    )

    print(json.dumps(model_info, indent=2))
    print(f"Saved model to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
