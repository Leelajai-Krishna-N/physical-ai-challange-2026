#!/usr/bin/env python3
"""Focused tests for ROI-limited cube perception and minimal motion planning."""

from __future__ import annotations

import unittest

import numpy as np

from auto_pick_place import build_pick_place_plan
from cube_perception import clamp_roi, constrain_world_point_to_validation, detect_cube_in_roi, expand_roi


class RoiMathTests(unittest.TestCase):
    def test_clamp_roi_keeps_bounds_inside_image(self):
        roi = clamp_roi((-10, 15, 700, 500), width=640, height=480)
        self.assertEqual(roi, (0, 15, 640, 480))

    def test_expand_roi_grows_around_center(self):
        roi = expand_roi((100, 120, 160, 200), width=640, height=480, scale=1.5)
        self.assertEqual(roi, (85, 100, 175, 220))


class SyntheticDetectionTests(unittest.TestCase):
    def test_detect_cube_in_roi_ignores_red_outside_roi(self):
        rgb = np.zeros((160, 200, 3), dtype=np.uint8)
        depth = np.full((160, 200), 0.35, dtype=np.float32)

        # Outside the ROI: should not affect the detection result.
        rgb[10:40, 10:40] = np.array([255, 0, 0], dtype=np.uint8)

        # Inside the ROI: this is the target cube.
        rgb[70:110, 110:150] = np.array([255, 0, 0], dtype=np.uint8)
        roi = (90, 50, 170, 130)

        detection = detect_cube_in_roi(
            rgb_image=rgb,
            depth_image=depth,
            roi=roi,
            min_mask_area=200,
            depth_confidence_threshold=0.4,
        )

        self.assertIsNotNone(detection)
        detection_info, _, _ = detection
        self.assertGreater(detection_info["u"], 110.0)
        self.assertLess(detection_info["u"], 150.0)
        self.assertGreater(detection_info["v"], 70.0)
        self.assertLess(detection_info["v"], 110.0)
        self.assertGreater(detection_info["mask_area_px"], 1000)

    def test_constrain_world_point_keeps_detection_inside_object_sized_window(self):
        raw = np.array([0.343, 0.396, -0.020], dtype=np.float64)
        validation = (0.410, 0.400, 0.025)
        constrained = constrain_world_point_to_validation(raw, validation)
        self.assertAlmostEqual(float(constrained[0]), 0.392, places=3)
        self.assertAlmostEqual(float(constrained[1]), 0.396, places=3)
        self.assertAlmostEqual(float(constrained[2]), 0.013, places=3)


class MotionPlanTests(unittest.TestCase):
    def test_yaw_invariant_plan_only_depends_on_pick_center(self):
        pick_center = (0.22, -0.12, 0.025)
        plan_a = build_pick_place_plan(pick_center_world=pick_center, place_xy=(0.32, 0.20))
        plan_b = build_pick_place_plan(pick_center_world=pick_center, place_xy=(0.32, 0.20))
        self.assertEqual(plan_a, plan_b)

    def test_plan_is_minimal_and_vertical_at_pick_and_place(self):
        plan = build_pick_place_plan(pick_center_world=(0.24, -0.10, 0.025), place_xy=(0.32, 0.20))
        self.assertEqual(plan.pick_hover[:2], plan.pick_down[:2])
        self.assertEqual(plan.place_hover[:2], plan.place_down[:2])
        self.assertGreater(plan.pick_hover[2], plan.pick_down[2])
        self.assertGreater(plan.place_hover[2], plan.place_down[2])
        self.assertEqual(plan.home_pose, (0.20, 0.00, 0.250))


if __name__ == "__main__":
    unittest.main()
