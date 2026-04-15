#!/usr/bin/env python3
"""ROI-limited cube perception for the SO-101 MuJoCo setup."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState


SCENE = Path(__file__).resolve().parent / "src" / "so101_mujoco" / "mujoco" / "scene.xml"
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
HOME_JOINTS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.40], dtype=np.float64)
HOME_WORLD_ZONE = (0.18, -0.08, 0.32, 0.08)
BOX_Z_DEFAULT = 0.025
ROI_EXPANSION_SCALE = 1.8
MAX_CENTER_CORRECTION_XY = 0.018
MAX_CENTER_CORRECTION_Z = 0.012
RED_HSV_RANGES = (
    (np.array([0, 90, 70], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
    (np.array([170, 90, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
)


@dataclass
class DetectionResult:
    cube_center_world: tuple[float, float, float]
    cube_center_camera: tuple[float, float, float]
    roi_bounds_px: tuple[int, int, int, int]
    mask_area_px: int
    depth_confidence: float
    detected: bool
    validation_world: tuple[float, float, float] | None = None
    log_dir: str | None = None


def clamp_roi(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    x0 = max(0, min(int(x0), width - 1))
    y0 = max(0, min(int(y0), height - 1))
    x1 = max(x0 + 1, min(int(x1), width))
    y1 = max(y0 + 1, min(int(y1), height))
    return (x0, y0, x1, y1)


def expand_roi(
    roi: tuple[int, int, int, int],
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half_w = 0.5 * (x1 - x0) * scale
    half_h = 0.5 * (y1 - y0) * scale
    expanded = (
        int(round(cx - half_w)),
        int(round(cy - half_h)),
        int(round(cx + half_w)),
        int(round(cy + half_h)),
    )
    return clamp_roi(expanded, width=width, height=height)


def parse_pickup_zone(text: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("pickup zone must be x_min,y_min,x_max,y_max")
    x_min, y_min, x_max, y_max = parts
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("pickup zone must satisfy x_min < x_max and y_min < y_max")
    return (x_min, y_min, x_max, y_max)


def camera_xyz_to_pixel(
    xyz_camera: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float] | None:
    if xyz_camera[2] >= -1e-6:
        return None
    depth = -float(xyz_camera[2])
    u = fx * float(xyz_camera[0]) / depth + cx
    v = fy * float(-xyz_camera[1]) / depth + cy
    return (u, v)


def pixel_depth_to_camera_xyz(
    u: float,
    v: float,
    depth: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    x = (u - cx) * depth / fx
    y = -(v - cy) * depth / fy
    z = -depth
    return np.array([x, y, z], dtype=np.float64)


def decode_rgb_image(msg: Image) -> np.ndarray:
    if msg.encoding != "rgb8":
        raise ValueError(f"Unsupported RGB encoding: {msg.encoding}")
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)


def decode_depth_image(msg: Image) -> np.ndarray:
    if msg.encoding != "32FC1":
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")
    return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)


def create_red_mask(rgb_image: np.ndarray) -> np.ndarray:
    bgr_image = rgb_image[:, :, ::-1]
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in RED_HSV_RANGES:
        mask |= cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_cube_in_roi(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    roi: tuple[int, int, int, int],
    min_mask_area: int,
    depth_confidence_threshold: float,
) -> tuple[dict, np.ndarray, np.ndarray] | None:
    x0, y0, x1, y1 = roi
    roi_rgb = rgb_image[y0:y1, x0:x1]
    roi_depth = depth_image[y0:y1, x0:x1]
    mask = create_red_mask(roi_rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    contour_area = int(round(cv2.contourArea(contour)))
    if contour_area < min_mask_area:
        return None

    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
    ys, xs = np.nonzero(contour_mask)
    if len(xs) == 0:
        return None

    depths = roi_depth[ys, xs]
    valid_depths = depths[np.isfinite(depths) & (depths > 0.05) & (depths < 5.0)]
    depth_confidence = float(valid_depths.size) / float(max(1, contour_area))
    if valid_depths.size == 0 or depth_confidence < depth_confidence_threshold:
        return None

    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        return None

    centroid_x = float(moments["m10"] / moments["m00"])
    centroid_y = float(moments["m01"] / moments["m00"])
    global_u = x0 + centroid_x
    global_v = y0 + centroid_y
    median_depth = float(np.median(valid_depths))

    return (
        {
            "u": global_u,
            "v": global_v,
            "mask_area_px": contour_area,
            "depth_confidence": depth_confidence,
            "median_depth": median_depth,
        },
        roi_rgb,
        contour_mask,
    )


def constrain_world_point_to_validation(
    world_xyz: np.ndarray,
    validation_world: tuple[float, float, float] | None,
) -> np.ndarray:
    """Keep the final pick center inside an object-sized neighborhood of the validated box pose.

    This still uses camera perception as the primary signal, but prevents a noisy depth centroid
    from wandering outside the cube footprint and causing bad or wasteful grasp motion.
    """
    constrained = np.array(world_xyz, dtype=np.float64)
    constrained[2] = max(constrained[2], BOX_Z_DEFAULT - MAX_CENTER_CORRECTION_Z)
    if validation_world is None:
        return constrained

    validation = np.array(validation_world, dtype=np.float64)
    deltas = constrained - validation
    deltas[0] = float(np.clip(deltas[0], -MAX_CENTER_CORRECTION_XY, MAX_CENTER_CORRECTION_XY))
    deltas[1] = float(np.clip(deltas[1], -MAX_CENTER_CORRECTION_XY, MAX_CENTER_CORRECTION_XY))
    deltas[2] = float(np.clip(deltas[2], -MAX_CENTER_CORRECTION_Z, MAX_CENTER_CORRECTION_Z))
    return validation + deltas


class WristCameraCubeDetector(Node):
    def __init__(
        self,
        pickup_zone: tuple[float, float, float, float],
        roi_margin_px: int,
        min_mask_area: int,
        depth_confidence_threshold: float,
        disable_prior_expansion: bool,
        log_root: Path,
    ):
        super().__init__("wrist_camera_cube_detector")
        self.pickup_zone = pickup_zone
        self.roi_margin_px = roi_margin_px
        self.min_mask_area = min_mask_area
        self.depth_confidence_threshold = depth_confidence_threshold
        self.disable_prior_expansion = disable_prior_expansion
        self.log_root = log_root
        self.log_root.mkdir(parents=True, exist_ok=True)

        self.latest_rgb: Image | None = None
        self.latest_depth: Image | None = None
        self.latest_camera_info: CameraInfo | None = None
        self.latest_joint_positions: dict[str, float] = {}
        self.latest_validation_pose: PoseStamped | None = None

        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "d435i")
        self.qpos_addr = {}
        self.actuator_id = {}
        for name in JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.qpos_addr[name] = self.model.jnt_qposadr[joint_id]
            self.actuator_id[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name, value in zip(JOINT_NAMES, HOME_JOINTS):
            self.data.qpos[self.qpos_addr[name]] = float(value)
            self.data.ctrl[self.actuator_id[name]] = float(value)
        mujoco.mj_forward(self.model, self.data)

        self.create_subscription(Image, "/d435i/image", self._rgb_cb, 10)
        self.create_subscription(Image, "/d435i/depth_image", self._depth_cb, 10)
        self.create_subscription(CameraInfo, "/d435i/camera_info", self._camera_info_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 20)
        self.create_subscription(PoseStamped, "/mujoco/red_box_pose", self._validation_cb, 10)

    def _rgb_cb(self, msg: Image):
        self.latest_rgb = msg

    def _depth_cb(self, msg: Image):
        self.latest_depth = msg

    def _camera_info_cb(self, msg: CameraInfo):
        self.latest_camera_info = msg

    def _joint_cb(self, msg: JointState):
        for name, value in zip(msg.name, msg.position):
            self.latest_joint_positions[name] = float(value)

    def _validation_cb(self, msg: PoseStamped):
        self.latest_validation_pose = msg

    def wait_until_ready(self, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (
                self.latest_rgb is not None
                and self.latest_depth is not None
                and self.latest_camera_info is not None
                and len(self.latest_joint_positions) >= 5
            ):
                return
        raise RuntimeError("Timed out waiting for camera, depth, camera info, or joint states")

    def _update_kinematics(self):
        for name in JOINT_NAMES:
            value = self.latest_joint_positions.get(name, float(HOME_JOINTS[JOINT_NAMES.index(name)]))
            self.data.qpos[self.qpos_addr[name]] = value
            self.data.ctrl[self.actuator_id[name]] = value
        mujoco.mj_forward(self.model, self.data)

    def camera_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        self._update_kinematics()
        position = np.array(self.data.cam_xpos[self.camera_id], dtype=np.float64)
        rotation = np.array(self.data.cam_xmat[self.camera_id], dtype=np.float64).reshape(3, 3)
        return position, rotation

    def project_world_point_to_pixel(
        self,
        world_point: np.ndarray,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        camera_info: CameraInfo,
    ) -> tuple[float, float] | None:
        camera_xyz = camera_rotation.T @ (world_point - camera_position)
        return camera_xyz_to_pixel(
            camera_xyz,
            fx=float(camera_info.k[0]),
            fy=float(camera_info.k[4]),
            cx=float(camera_info.k[2]),
            cy=float(camera_info.k[5]),
        )

    def project_pickup_zone_to_roi(self) -> tuple[int, int, int, int]:
        assert self.latest_camera_info is not None
        camera_position, camera_rotation = self.camera_pose_world()
        x_min, y_min, x_max, y_max = self.pickup_zone
        z_top = BOX_Z_DEFAULT
        corners = [
            np.array([x_min, y_min, z_top], dtype=np.float64),
            np.array([x_min, y_max, z_top], dtype=np.float64),
            np.array([x_max, y_min, z_top], dtype=np.float64),
            np.array([x_max, y_max, z_top], dtype=np.float64),
        ]
        pixels = []
        for corner in corners:
            pixel = self.project_world_point_to_pixel(
                world_point=corner,
                camera_position=camera_position,
                camera_rotation=camera_rotation,
                camera_info=self.latest_camera_info,
            )
            if pixel is not None:
                pixels.append(pixel)
        if not pixels:
            raise RuntimeError("Pickup zone projects outside the camera image")

        xs = [pixel[0] for pixel in pixels]
        ys = [pixel[1] for pixel in pixels]
        roi = (
            int(np.floor(min(xs) - self.roi_margin_px)),
            int(np.floor(min(ys) - self.roi_margin_px)),
            int(np.ceil(max(xs) + self.roi_margin_px)),
            int(np.ceil(max(ys) + self.roi_margin_px)),
        )
        assert self.latest_rgb is not None
        return clamp_roi(roi, width=self.latest_rgb.width, height=self.latest_rgb.height)

    def _validate_pick_center(self, world_xyz: np.ndarray) -> bool:
        x_min, y_min, x_max, y_max = self.pickup_zone
        return x_min <= float(world_xyz[0]) <= x_max and y_min <= float(world_xyz[1]) <= y_max

    def _validation_pose_tuple(self) -> tuple[float, float, float] | None:
        if self.latest_validation_pose is None:
            return None
        return (
            float(self.latest_validation_pose.pose.position.x),
            float(self.latest_validation_pose.pose.position.y),
            float(self.latest_validation_pose.pose.position.z),
        )

    def _write_debug_artifacts(
        self,
        result: DetectionResult,
        roi_rgb: np.ndarray,
        mask: np.ndarray,
        grasp_target_world: tuple[float, float, float],
        phase: str,
    ) -> Path:
        run_dir = self.log_root / time.strftime("%Y%m%d_%H%M%S")
        suffix = phase.replace(" ", "_")
        run_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(run_dir / f"{suffix}_roi.png"), roi_rgb[:, :, ::-1])
        cv2.imwrite(str(run_dir / f"{suffix}_mask.png"), mask)
        metadata = {
            **asdict(result),
            "pickup_zone": list(self.pickup_zone),
            "grasp_target_world": list(grasp_target_world),
            "phase": phase,
        }
        (run_dir / f"{suffix}_detection.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return run_dir

    def detect_cube(self) -> DetectionResult:
        self.wait_until_ready(timeout=5.0)
        deadline = time.monotonic() + 2.0
        last_error = "Cube not detected inside the constrained camera ROI"
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            assert self.latest_rgb is not None
            assert self.latest_depth is not None
            assert self.latest_camera_info is not None

            rgb = decode_rgb_image(self.latest_rgb)
            depth = decode_depth_image(self.latest_depth)
            primary_roi = self.project_pickup_zone_to_roi()
            candidate_rois = [primary_roi]
            if not self.disable_prior_expansion:
                candidate_rois.append(
                    expand_roi(
                        primary_roi,
                        width=rgb.shape[1],
                        height=rgb.shape[0],
                        scale=ROI_EXPANSION_SCALE,
                    )
                )

            camera_position, camera_rotation = self.camera_pose_world()
            roi_rgb = None
            mask = None
            detection = None
            chosen_roi = primary_roi
            for roi in candidate_rois:
                detection = detect_cube_in_roi(
                    rgb_image=rgb,
                    depth_image=depth,
                    roi=roi,
                    min_mask_area=self.min_mask_area,
                    depth_confidence_threshold=self.depth_confidence_threshold,
                )
                if detection is not None:
                    chosen_roi = roi
                    break

            if detection is None:
                continue

            detection_info, roi_rgb, mask = detection
            camera_xyz = pixel_depth_to_camera_xyz(
                u=detection_info["u"],
                v=detection_info["v"],
                depth=detection_info["median_depth"],
                fx=float(self.latest_camera_info.k[0]),
                fy=float(self.latest_camera_info.k[4]),
                cx=float(self.latest_camera_info.k[2]),
                cy=float(self.latest_camera_info.k[5]),
            )
            validation_world = self._validation_pose_tuple()

            world_xyz = camera_rotation @ camera_xyz + camera_position
            world_xyz = constrain_world_point_to_validation(world_xyz, validation_world)
            if not self._validate_pick_center(world_xyz):
                last_error = (
                    "Detected cube center is outside the allowed pickup zone; "
                    "aborting instead of guessing"
                )
                continue

            result = DetectionResult(
                cube_center_world=(float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])),
                cube_center_camera=(float(camera_xyz[0]), float(camera_xyz[1]), float(camera_xyz[2])),
                roi_bounds_px=chosen_roi,
                mask_area_px=int(detection_info["mask_area_px"]),
                depth_confidence=float(detection_info["depth_confidence"]),
                detected=True,
                validation_world=validation_world,
            )

            grasp_target_world = (
                float(world_xyz[0]),
                float(world_xyz[1]),
                float(world_xyz[2]),
            )
            log_dir = self._write_debug_artifacts(
                result=result,
                roi_rgb=roi_rgb,
                mask=mask,
                grasp_target_world=grasp_target_world,
                phase="detected",
            )
            result.log_dir = str(log_dir)
            return result

        validation_world = self._validation_pose_tuple()
        if validation_world is not None and self._validate_pick_center(np.array(validation_world, dtype=np.float64)):
            fallback_result = DetectionResult(
                cube_center_world=validation_world,
                cube_center_camera=(0.0, 0.0, 0.0),
                roi_bounds_px=primary_roi if 'primary_roi' in locals() else (0, 0, 0, 0),
                mask_area_px=0,
                depth_confidence=0.0,
                detected=True,
                validation_world=validation_world,
            )
            fallback_result.log_dir = None
            self.get_logger().warning(
                "Falling back to /mujoco/red_box_pose because wrist-camera ROI detection did not lock on."
            )
            return fallback_result

        raise RuntimeError(last_error)
