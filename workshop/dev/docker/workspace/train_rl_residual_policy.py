#!/usr/bin/env python3
"""Train a stable residual policy for grasp correction in simulation.

This trainer uses a cross-entropy / elite-selection style RL update rather than
high-variance policy gradients. In practice that is much more stable for this
small action space and short episodic task.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from auto_pick_place import run_pick_place


BRIDGE_ADDR = ("127.0.0.1", 9876)
ACTION_SCALE = np.array([0.018, 0.018, 0.010], dtype=np.float64)
BOX_Z_DEFAULT = 0.025


def send_bridge_command(payload):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(json.dumps(payload).encode("utf-8"), BRIDGE_ADDR)
    finally:
        sock.close()


class BoxPoseTracker(Node):
    def __init__(self, topic_name: str):
        super().__init__("rl_box_pose_tracker")
        self.latest_pose: np.ndarray | None = None
        self.latest_orientation: np.ndarray | None = None
        self.create_subscription(PoseStamped, topic_name, self._cb, 20)

    def _cb(self, msg: PoseStamped):
        self.latest_pose = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        self.latest_orientation = np.array(
            [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ],
            dtype=np.float64,
        )

    def wait_for_pose(self, timeout: float) -> tuple[np.ndarray, np.ndarray]:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_pose is not None and self.latest_orientation is not None:
                return self.latest_pose.copy(), self.latest_orientation.copy()
        raise RuntimeError("Timed out waiting for /mujoco/red_box_pose")


@dataclass
class EpisodeResult:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    final_pose: np.ndarray
    success: bool
    moved_distance: float
    place_error: float


class ResidualPolicy:
    def __init__(self, feature_dim: int, action_dim: int, init_std: float):
        self.feature_mean = np.zeros(feature_dim, dtype=np.float64)
        self.feature_std = np.ones(feature_dim, dtype=np.float64)
        self.weights = np.zeros((feature_dim, action_dim), dtype=np.float64)
        self.bias = np.zeros(action_dim, dtype=np.float64)
        self.std = np.full(action_dim, init_std, dtype=np.float64)

    def fit_normalizer(self, observations: list[np.ndarray]):
        data = np.asarray(observations, dtype=np.float64)
        self.feature_mean = data.mean(axis=0)
        self.feature_std = data.std(axis=0)
        self.feature_std[self.feature_std < 1e-6] = 1.0

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.feature_mean) / self.feature_std

    def predict(self, obs: np.ndarray) -> np.ndarray:
        norm_obs = self.normalize(obs)
        return norm_obs @ self.weights + self.bias

    def sample(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean = self.predict(obs)
        action = mean + rng.normal(scale=self.std, size=mean.shape)
        return np.clip(action, -ACTION_SCALE, ACTION_SCALE)

    def refit_from_elites(self, elite_batch: list[EpisodeResult], ridge: float, std_floor: float):
        x = np.asarray([self.normalize(item.observation) for item in elite_batch], dtype=np.float64)
        y = np.asarray([item.action for item in elite_batch], dtype=np.float64)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        reg = ridge * np.eye(x_aug.shape[1], dtype=np.float64)
        reg[-1, -1] = 0.0
        coeff = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
        self.weights = coeff[:-1]
        self.bias = np.clip(coeff[-1], -ACTION_SCALE, ACTION_SCALE)

        pred = x @ self.weights + self.bias
        residuals = np.clip(y - pred, -ACTION_SCALE, ACTION_SCALE)
        self.std = np.maximum(residuals.std(axis=0), std_floor)
        self.std = np.minimum(self.std, ACTION_SCALE)

    def save(self, path: Path, metadata: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "log_std": np.log(self.std).tolist(),
            "action_scale": ACTION_SCALE.tolist(),
            "metadata": metadata,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_reward(
    initial_pose: np.ndarray,
    final_pose: np.ndarray,
    place_xy: tuple[float, float],
    action: np.ndarray,
) -> tuple[float, bool, float, float]:
    place_target = np.array([place_xy[0], place_xy[1], initial_pose[2]], dtype=np.float64)
    place_error = float(np.linalg.norm(final_pose[:2] - place_target[:2]))
    z_error = float(abs(final_pose[2] - place_target[2]))
    moved_distance = float(np.linalg.norm(final_pose[:2] - initial_pose[:2]))
    pickup_proxy = float(final_pose[2] > initial_pose[2] + 0.01)
    success = place_error <= 0.045 and z_error <= 0.04 and moved_distance >= 0.14
    reward = (
        8.0 * float(success)
        + 2.5 * pickup_proxy
        + 3.0 * min(moved_distance, 0.25)
        - 12.0 * place_error
        - 3.0 * z_error
        - 0.35 * float(np.linalg.norm(action / ACTION_SCALE))
    )
    return reward, success, moved_distance, place_error


def reset_box(position: np.ndarray, orientation: np.ndarray):
    send_bridge_command(
        {
            "kind": "reset_box",
            "position": [float(v) for v in position],
            "orientation": [float(v) for v in orientation],
        }
    )


def sample_box_position(rng: np.random.Generator, base_pose: np.ndarray, random_xy: float) -> np.ndarray:
    pose = base_pose.copy()
    pose[0] += rng.uniform(-random_xy, random_xy)
    pose[1] += rng.uniform(-random_xy, random_xy)
    pose[2] = max(float(pose[2]), BOX_Z_DEFAULT)
    return pose


def run_episode(
    tracker: BoxPoseTracker,
    policy: ResidualPolicy,
    rng: np.random.Generator,
    place_xy: tuple[float, float],
    speed: float,
    pose_timeout: float,
    random_xy: float,
    base_pose: np.ndarray,
    base_orientation: np.ndarray,
    force_zero_action: bool = False,
) -> EpisodeResult:
    spawn_pose = sample_box_position(rng, base_pose, random_xy)
    reset_box(spawn_pose, base_orientation)
    time.sleep(0.25)
    observed_pose, _ = tracker.wait_for_pose(timeout=pose_timeout)
    action = np.zeros(3, dtype=np.float64) if force_zero_action else policy.sample(observed_pose, rng)
    run_pick_place(
        (float(observed_pose[0]), float(observed_pose[1])),
        place_xy,
        speed=max(0.2, speed),
        residual_xyz=tuple(float(v) for v in action),
    )
    time.sleep(0.5)
    final_pose, _ = tracker.wait_for_pose(timeout=pose_timeout)
    reward, success, moved_distance, place_error = compute_reward(observed_pose, final_pose, place_xy, action)
    return EpisodeResult(
        observation=observed_pose,
        action=action,
        reward=reward,
        final_pose=final_pose,
        success=success,
        moved_distance=moved_distance,
        place_error=place_error,
    )


def summarize(batch: list[EpisodeResult]) -> tuple[float, float]:
    avg_reward = float(np.mean([item.reward for item in batch]))
    success_rate = float(np.mean([float(item.success) for item in batch]))
    return avg_reward, success_rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/rl_residual_policy.json"))
    parser.add_argument("--topic", default="/mujoco/red_box_pose")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--elite-frac", type=float, default=0.4)
    parser.add_argument("--init-std", type=float, default=0.004)
    parser.add_argument("--std-floor", type=float, default=0.0015)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--place-x", type=float, default=0.32)
    parser.add_argument("--place-y", type=float, default=0.20)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pose-timeout", type=float, default=5.0)
    parser.add_argument("--randomize-xy", type=float, default=0.04)
    parser.add_argument("--baseline-every", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rclpy.init()
    tracker = BoxPoseTracker(args.topic)

    try:
        base_pose, base_orientation = tracker.wait_for_pose(timeout=args.pose_timeout)
        policy = ResidualPolicy(feature_dim=3, action_dim=3, init_std=args.init_std)
        observations: list[np.ndarray] = [base_pose]
        history: list[EpisodeResult] = []
        best_result: EpisodeResult | None = None

        for episode_idx in range(args.episodes):
            force_zero = args.baseline_every > 0 and episode_idx % args.baseline_every == 0
            result = run_episode(
                tracker=tracker,
                policy=policy,
                rng=rng,
                place_xy=(args.place_x, args.place_y),
                speed=args.speed,
                pose_timeout=args.pose_timeout,
                random_xy=args.randomize_xy,
                base_pose=base_pose,
                base_orientation=base_orientation,
                force_zero_action=force_zero,
            )
            history.append(result)
            observations.append(result.observation)
            policy.fit_normalizer(observations)

            if best_result is None or result.reward > best_result.reward:
                best_result = result

            print(
                f"episode={episode_idx + 1:03d} "
                f"reward={result.reward:+.3f} "
                f"success={int(result.success)} "
                f"place_error={result.place_error:.3f} "
                f"moved={result.moved_distance:.3f} "
                f"action={[round(float(v), 4) for v in result.action]}"
                f"{' baseline' if force_zero else ''}"
            )

            if len(history) % max(args.batch_size, 1) == 0:
                batch = history[-args.batch_size:]
                elite_count = max(1, int(round(len(batch) * args.elite_frac)))
                elite_batch = sorted(batch, key=lambda item: item.reward, reverse=True)[:elite_count]
                policy.refit_from_elites(
                    elite_batch=elite_batch,
                    ridge=args.ridge,
                    std_floor=args.std_floor,
                )
                avg_reward, avg_success = summarize(batch)
                elite_reward, elite_success = summarize(elite_batch)
                print(
                    f"batch_update avg_reward={avg_reward:+.3f} avg_success={avg_success:.2f} "
                    f"elite_reward={elite_reward:+.3f} elite_success={elite_success:.2f} "
                    f"std={[round(float(v), 4) for v in policy.std]}"
                )

        metadata = {
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "elite_frac": args.elite_frac,
            "init_std": args.init_std,
            "std_floor": args.std_floor,
            "ridge": args.ridge,
            "seed": args.seed,
            "place_xy": [args.place_x, args.place_y],
            "randomize_xy": args.randomize_xy,
            "avg_reward": float(np.mean([item.reward for item in history])) if history else 0.0,
            "success_rate": float(np.mean([float(item.success) for item in history])) if history else 0.0,
            "best_reward": float(best_result.reward) if best_result is not None else 0.0,
            "best_action": [float(v) for v in best_result.action] if best_result is not None else [0.0, 0.0, 0.0],
        }
        policy.save(args.output, metadata)
        print(json.dumps(metadata, indent=2))
        print(f"Saved RL residual policy to {args.output}")
    finally:
        tracker.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
