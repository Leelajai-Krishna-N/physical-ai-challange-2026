#!/usr/bin/env python3
"""Render a fresh offscreen GIF for the current working judge demo path."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from auto_pick_place import (
    DEFAULT_BOX_ORIENTATION,
    DEFAULT_BOX_POSE,
    GRIPPER_HOLD,
    GRIPPER_OPEN,
    HOME_X,
    HOME_Y,
    HOME_Z,
    build_pick_place_plan,
    corrected_command_target,
    solve_pose,
)


SCENE = Path(__file__).resolve().parent / "src" / "so101_mujoco" / "mujoco" / "scene.xml"
OUT_GIF = Path(__file__).resolve().parent / "renders" / "judge_demo_working.gif"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def target_payload(x: float, y: float, z: float, gripper: float) -> list[float]:
    corrected, payload = corrected_command_target((x, y, z), gripper)
    if payload is None:
        payload = solve_pose(float(corrected[0]), float(corrected[1]), float(corrected[2]), gripper)
    return payload


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat = np.array([0.24, 0.08, 0.03], dtype=np.float64)
    cam.distance = 0.95
    cam.azimuth = 128.0
    cam.elevation = -24.0

    qpos_addr = {}
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_addr[name] = model.jnt_qposadr[joint_id]

    box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")
    box_addr = model.jnt_qposadr[box_joint_id]
    gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    plan = build_pick_place_plan(DEFAULT_BOX_POSE, (0.32, 0.20))

    def set_payload(payload: list[float]) -> None:
        for name, value in zip(JOINT_NAMES, payload):
            data.qpos[qpos_addr[name]] = value
        mujoco.mj_forward(model, data)

    def set_box(position: tuple[float, float, float]) -> None:
        data.qpos[box_addr:box_addr + 3] = np.asarray(position, dtype=float)
        data.qpos[box_addr + 3:box_addr + 7] = np.asarray(DEFAULT_BOX_ORIENTATION, dtype=float)
        mujoco.mj_forward(model, data)

    keyframes = [
        ("home", target_payload(HOME_X, HOME_Y, HOME_Z, GRIPPER_OPEN), DEFAULT_BOX_POSE, False),
        ("pick_hover", target_payload(*plan.pick_hover, GRIPPER_OPEN), DEFAULT_BOX_POSE, False),
        ("pick_approach", target_payload(*plan.pick_approach, GRIPPER_OPEN), DEFAULT_BOX_POSE, False),
        ("pick_down", target_payload(*plan.pick_down, GRIPPER_OPEN), DEFAULT_BOX_POSE, False),
        ("close", target_payload(*plan.pick_down, GRIPPER_HOLD), DEFAULT_BOX_POSE, True),
        ("lift", target_payload(*plan.pick_hover, GRIPPER_HOLD), None, True),
        ("place_hover", target_payload(*plan.place_hover, GRIPPER_HOLD), None, True),
        ("place_approach", target_payload(*plan.place_approach, GRIPPER_HOLD), None, True),
        ("place_down", target_payload(*plan.place_down, GRIPPER_HOLD), None, True),
        ("pre_release", target_payload(*plan.place_approach, GRIPPER_HOLD), None, True),
        ("open", target_payload(*plan.place_approach, GRIPPER_OPEN), (0.32, 0.20, 0.025), False),
        ("clear", target_payload(*plan.place_hover, GRIPPER_OPEN), (0.32, 0.20, 0.025), False),
        ("return_home", target_payload(HOME_X, HOME_Y, HOME_Z, GRIPPER_OPEN), (0.32, 0.20, 0.025), False),
    ]

    frames: list[Image.Image] = []
    attach_offset = None
    prev_payload = keyframes[0][1]
    set_payload(prev_payload)
    set_box(DEFAULT_BOX_POSE)

    for _, payload, box_position, attached in keyframes:
        start = np.asarray(prev_payload, dtype=float)
        goal = np.asarray(payload, dtype=float)
        for step in range(10):
            alpha = (step + 1) / 10.0
            cur = (start + (goal - start) * alpha).tolist()
            set_payload(cur)
            if attached:
                gripper_pos = np.array(data.site_xpos[gripper_site_id], dtype=float)
                if attach_offset is None:
                    attach_offset = np.asarray(DEFAULT_BOX_POSE, dtype=float) - gripper_pos
                set_box(tuple((gripper_pos + attach_offset).tolist()))
            else:
                attach_offset = None
                if box_position is not None:
                    set_box(box_position)
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            frames.append(Image.fromarray(rgb))
        prev_payload = payload

    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
    )
    print(OUT_GIF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
