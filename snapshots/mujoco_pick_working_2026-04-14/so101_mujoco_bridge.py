#!/usr/bin/python3
"""
so101_mujoco_bridge.py  –  Fixed version
Changes vs original:
  1. Passive MuJoCo viewer runs INSIDE this process (owns the real MjData).
     Delete viewer.py — it is no longer needed.
  2. Physics loop runs N_SUBSTEPS x mj_step per timer tick so the contact
     solver has enough iterations to stop the gripper phasing through objects.
  3. MUJOCO_GL is only forced to 'egl' when running headless (no viewer).
     When the viewer is enabled the default OpenGL backend is used.
  4. Renderer for camera topics is still on its own thread (EGL via env var
     set per-thread, not globally, so it doesn't fight the viewer context).
"""

import argparse
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2, PointField

# timer at 100 Hz + 5 substeps @ dt=0.005 s → 0.025 s of sim per tick.
# Increase to 10 if the gripper still tunnels through fast-moving objects.
N_SUBSTEPS = 5
GRASP_ATTACH_DISTANCE = 0.04
RED_BOX_HALF_HEIGHT = 0.025

# Set True to open an interactive MuJoCo viewer window.
# Set False for headless / server deployments.
ENABLE_VIEWER = False
# ────────────────────────────────────────────────────────────────────────────


class So101MujocoBridge(Node):
    _CAM_NAME     = 'd435i'
    _CAM_W        = 640
    _CAM_H        = 480
    _CAM_FOVY_DEG = 55.0

    def __init__(self, model_path: str, publish_rate: float,
                 startup_pose: str = 'home'):
        super().__init__('so101_mujoco_bridge')

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)

        self.joint_names = [
            'shoulder_pan', 'shoulder_lift', 'elbow_flex',
            'wrist_flex', 'wrist_roll', 'gripper',
        ]

        self._joint_qpos_addr: dict[str, int] = {}
        self._joint_dof_addr:  dict[str, int] = {}
        self._actuator_id:     dict[str, int] = {}
        for name in self.joint_names:
            j_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id < 0:
                raise RuntimeError(f'Joint not found in MuJoCo model: {name}')
            self._joint_qpos_addr[name] = self.model.jnt_qposadr[j_id]
            self._joint_dof_addr[name] = self.model.jnt_dofadr[j_id]

            a_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a_id < 0:
                raise RuntimeError(
                    f'Actuator not found in MuJoCo model: {name}')
            self._actuator_id[name] = a_id

        # Startup pose
        if startup_pose == 'upright':
            init = {
                'shoulder_pan': 0.0, 'shoulder_lift': -1.57,
                'elbow_flex': 0.0,   'wrist_flex': 0.0,
                'wrist_roll': 0.0,   'gripper': 0.0,
            }
        else:
            init = {n: 0.0 for n in self.joint_names}

        self._lock    = threading.Lock()
        self._targets = init.copy()
        self._attached_box = False
        self._grasp_armed = False
        self._attach_offset = np.zeros(3)
        self._red_box_qpos_addr = self._freejoint_qpos_addr('red_box_joint')
        self._red_box_dof_addr = self._freejoint_dof_addr('red_box_joint')
        self._red_box_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, 'red_box_geom')
        if self._red_box_geom_id < 0:
            raise RuntimeError('Geom not found in MuJoCo model: red_box_geom')
        self._gripper_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
        if self._gripper_site_id < 0:
            raise RuntimeError('Site not found in MuJoCo model: gripperframe')
        self._gripper_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper'),
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, 'moving_jaw_so101_v1'),
        }
        if any(body_id < 0 for body_id in self._gripper_body_ids):
            raise RuntimeError('Required gripper bodies not found in MuJoCo model')
        self._attached_box_quat = np.array([1.0, 0.0, 0.0, 0.0])

        for name, val in init.items():
            self.data.qpos[self._joint_qpos_addr[name]] = val
            self.data.ctrl[self._actuator_id[name]]     = val
        mujoco.mj_forward(self.model, self.data)

        # Camera renderer (lazy, lives on camera thread only)
        self._renderer: mujoco.Renderer | None = None
        self._cam_thread_stop = threading.Event()

        # Publishers
        self.joint_state_pub = self.create_publisher(
            JointState,   '/joint_states',          10)
        self.rgb_pub         = self.create_publisher(
            Image,        '/d435i/image',           10)
        self.depth_pub       = self.create_publisher(
            Image,        '/d435i/depth_image',     10)
        self.cam_info_pub    = self.create_publisher(
            CameraInfo,   '/d435i/camera_info',     10)
        self.points_pub      = self.create_publisher(
            PointCloud2,  '/d435i/points',          10)

        # UDP teleop socket
        import socket
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp_sock.bind(('127.0.0.1', 9876))
        self._udp_sock.settimeout(0.2)
        self._stop_event = threading.Event()
        self._udp_thread = threading.Thread(
            target=self._udp_listener_loop, daemon=True)
        self._udp_thread.start()


        # Physics + joint-state publish timer
        period = 1.0 / max(publish_rate, 1.0)
        self.create_timer(period, self._step_and_publish)

        # Action servers
        self._arm_server = ActionServer(
            self, FollowJointTrajectory,
            'arm_controller/follow_joint_trajectory',
            execute_callback=self._execute_arm,
            goal_callback=self._goal_arm,
            cancel_callback=self._cancel_cb,
        )
        self._gripper_server = ActionServer(
            self, FollowJointTrajectory,
            'gripper_controller/follow_joint_trajectory',
            execute_callback=self._execute_gripper,
            goal_callback=self._goal_gripper,
            cancel_callback=self._cancel_cb,
        )

        self.get_logger().info(
            f'Loaded MuJoCo model: {model_path}  '
            f'startup_pose={startup_pose}  '
            f'substeps={N_SUBSTEPS}  viewer={ENABLE_VIEWER}')

    def _freejoint_qpos_addr(self, joint_name: str) -> int:
        j_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if j_id < 0:
            raise RuntimeError(f'Free joint not found in MuJoCo model: {joint_name}')
        return int(self.model.jnt_qposadr[j_id])

    def _freejoint_dof_addr(self, joint_name: str) -> int:
        j_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if j_id < 0:
            raise RuntimeError(f'Free joint not found in MuJoCo model: {joint_name}')
        return int(self.model.jnt_dofadr[j_id])

    # ── action callbacks ─────────────────────────────────────────────────────

    def _goal_arm(self, goal_request):
        expected = {
            'shoulder_pan', 'shoulder_lift', 'elbow_flex',
            'wrist_flex', 'wrist_roll',
        }
        if not set(goal_request.trajectory.joint_names).issubset(expected):
            self.get_logger().warn('Rejecting arm trajectory')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _goal_gripper(self, goal_request):
        if not set(goal_request.trajectory.joint_names).issubset({'gripper'}):
            self.get_logger().warn('Rejecting gripper trajectory')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _gh):
        return CancelResponse.ACCEPT

    @staticmethod
    def _duration_to_sec(dur: Duration) -> float:
        return float(dur.sec) + float(dur.nanosec) * 1e-9

    def _apply_point_targets(self, joint_names, positions):
        with self._lock:
            for name, pos in zip(joint_names, positions):
                if name in self._targets:
                    self._targets[name] = float(pos)

    def _udp_listener_loop(self):
        import json
        import socket
        names = [
            'shoulder_pan', 'shoulder_lift', 'elbow_flex',
            'wrist_flex', 'wrist_roll', 'gripper',
        ]
        while rclpy.ok() and not self._stop_event.is_set():
            try:
                raw, _ = self._udp_sock.recvfrom(1024)
                positions = json.loads(raw.decode('utf-8'))
                if len(positions) != len(names):
                    self.get_logger().warn(
                        f'UDP: expected {len(names)} values, '
                        f'got {len(positions)} — ignoring')
                    continue
                self._apply_point_targets(names, positions)
            except TimeoutError:
                continue
            except socket.timeout:
                continue
            except Exception as exc:
                self.get_logger().warn(
                    f'UDP error: {exc}', throttle_duration_sec=5.0)

    def _execute_common(self, goal_handle, allowed_joints):
        trajectory  = goal_handle.request.trajectory
        points      = trajectory.points
        joint_names = trajectory.joint_names

        if not points or not joint_names:
            goal_handle.succeed()
            return FollowJointTrajectory.Result()

        start  = time.time()
        prev_t = 0.0
        for point in points:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            t      = self._duration_to_sec(point.time_from_start)
            t      = max(t, prev_t)
            prev_t = t

            if point.positions:
                names_to_set, pos_to_set = [], []
                for name, pos in zip(joint_names, point.positions):
                    if name in allowed_joints:
                        names_to_set.append(name)
                        pos_to_set.append(pos)
                if names_to_set:
                    self._apply_point_targets(names_to_set, pos_to_set)

            while time.time() - start < t:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return FollowJointTrajectory.Result()
                time.sleep(0.002)

        goal_handle.succeed()
        return FollowJointTrajectory.Result()

    def _execute_arm(self, goal_handle):
        return self._execute_common(
            goal_handle,
            {'shoulder_pan', 'shoulder_lift', 'elbow_flex',
             'wrist_flex', 'wrist_roll'},
        )

    def _execute_gripper(self, goal_handle):
        return self._execute_common(goal_handle, {'gripper'})

    def shutdown_bridge(self):
        self._stop_event.set()
        self._cam_thread_stop.set()
        try:
            self._udp_sock.close()
        except OSError:
            pass

    # ── physics step (FIX: N_SUBSTEPS) ──────────────────────────────────────

    def _red_box_position(self) -> np.ndarray:
        return np.array(
            self.data.qpos[self._red_box_qpos_addr:self._red_box_qpos_addr + 3],
            dtype=float,
        )

    def _has_gripper_box_contact(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self._red_box_geom_id:
                other_geom = geom2
            elif geom2 == self._red_box_geom_id:
                other_geom = geom1
            else:
                continue

            other_body = int(self.model.geom_bodyid[other_geom])
            if other_body in self._gripper_body_ids:
                return True

        return False

    def _update_grasp_assist(self):
        gripper_target = self._targets.get('gripper', 0.0)
        gripper_pos = np.array(self.data.site_xpos[self._gripper_site_id])
        box_pos = self._red_box_position()
        box_distance = float(np.linalg.norm(box_pos - gripper_pos))
        touching_box = self._has_gripper_box_contact()

        if gripper_target > 0.8:
            self._grasp_armed = True

        if (
            gripper_target < 0.15
            and self._grasp_armed
            and not self._attached_box
            and (touching_box or box_distance <= GRASP_ATTACH_DISTANCE)
        ):
            self._attached_box = True
            self._grasp_armed = False
            self._attach_offset = box_pos - gripper_pos
            self._attached_box_quat = np.array(
                self.data.qpos[
                    self._red_box_qpos_addr + 3:self._red_box_qpos_addr + 7],
                dtype=float,
            )
            self.get_logger().info('Grasp assist attached red_box')

        if gripper_target > 0.8 and self._attached_box:
            self._attached_box = False
            pos = self._red_box_position()
            quat = np.array(
                self.data.qpos[
                    self._red_box_qpos_addr + 3:self._red_box_qpos_addr + 7],
                dtype=float,
            )
            pos[2] = max(pos[2], RED_BOX_HALF_HEIGHT)
            self.data.qpos[
                self._red_box_qpos_addr:self._red_box_qpos_addr + 3] = pos
            self.data.qpos[
                self._red_box_qpos_addr + 3:self._red_box_qpos_addr + 7] = quat
            self.data.qvel[
                self._red_box_dof_addr:self._red_box_dof_addr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self.get_logger().info('Grasp assist released red_box')

        if self._attached_box:
            target = gripper_pos + self._attach_offset
            target[2] = max(target[2], RED_BOX_HALF_HEIGHT)
            self.data.qpos[
                self._red_box_qpos_addr:self._red_box_qpos_addr + 3] = target
            self.data.qpos[
                self._red_box_qpos_addr + 3:self._red_box_qpos_addr + 7] = self._attached_box_quat
            self.data.qvel[
                self._red_box_dof_addr:self._red_box_dof_addr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def _step_and_publish(self):
        """
        Run N_SUBSTEPS physics steps per timer tick.
        More substeps = better contact resolution = gripper won't phase through
        objects.  Each substep uses the model's own timestep (0.005 s).
        """
        with self._lock:
            for name, target in self._targets.items():
                self.data.ctrl[self._actuator_id[name]] = target

            # ↓ THIS is the core fix for the phasing-through-objects bug
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(self.model, self.data)

            self._update_grasp_assist()

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name         = list(self.joint_names)
            msg.position     = [
                float(self.data.qpos[self._joint_qpos_addr[n]])
                for n in self.joint_names
            ]
            msg.velocity     = [
                float(self.data.qvel[self._joint_dof_addr[n]])
                for n in self.joint_names
            ]
            msg.effort       = []

        self.joint_state_pub.publish(msg)

    # ── passive viewer (FIX: runs on bridge's own MjData) ───────────────────

    def run_viewer(self):
        """
        Launch the passive MuJoCo viewer in the CALLING thread.
        Must be called from the main thread (OpenGL requirement on most OSes).
        The viewer reads directly from self.data — no copy, no desync.
        """
        self.get_logger().info('Launching passive MuJoCo viewer …')
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while rclpy.ok() and viewer.is_running():
                # sync() must be called on the same thread as launch_passive
                with self._lock:
                    viewer.sync()
                time.sleep(0.01)   # ~100 Hz refresh is plenty
        self.get_logger().info('MuJoCo viewer closed.')

    # ── camera ───────────────────────────────────────────────────────────────

    def _init_renderer(self) -> bool:
        try:
            self._renderer = mujoco.Renderer(
                self.model, height=self._CAM_H, width=self._CAM_W)
            return True
        except Exception as exc:
            self.get_logger().warn(
                f'Camera renderer init failed: {exc}',
                throttle_duration_sec=10.0)
            return False

    def _camera_thread_loop(self):
        # Mandatory delay to allow the main thread (viewer) to initialize OpenGL
        time.sleep(5.0)
        self.get_logger().info("Initializing camera renderer (EGL)...")
        period = 1.0 / 30.0
        while not self._cam_thread_stop.is_set():
            t0 = time.monotonic()
            self._publish_camera()
            remaining = period - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)

    def _camera_info(self, stamp) -> CameraInfo:
        fovy_rad = np.deg2rad(self._CAM_FOVY_DEG)
        f  = (self._CAM_H / 2.0) / np.tan(fovy_rad / 2.0)
        cx, cy = self._CAM_W / 2.0, self._CAM_H / 2.0
        info                    = CameraInfo()
        info.header.stamp       = stamp
        info.header.frame_id    = 'd435i_link'
        info.width              = self._CAM_W
        info.height             = self._CAM_H
        info.distortion_model   = 'plumb_bob'
        info.d                  = [0.0] * 5
        info.k                  = [float(f), 0.0, float(cx), 0.0, float(f), float(cy), 0.0, 0.0, 1.0]
        info.r                  = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p                  = [float(f), 0.0, float(cx), 0.0, 0.0, float(f), float(cy), 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _depth_to_pointcloud(self, depth: np.ndarray, stamp) -> PointCloud2:
        fovy_rad = np.deg2rad(self._CAM_FOVY_DEG)
        f        = (self._CAM_H / 2.0) / np.tan(fovy_rad / 2.0)
        cx, cy   = self._CAM_W / 2.0, self._CAM_H / 2.0
        u, v     = np.meshgrid(
            np.arange(self._CAM_W), np.arange(self._CAM_H))
        z        = depth.astype(np.float32)
        mask     = (z > 0.05) & (z < 5.0)
        x        = ((u - cx) * z / f)[mask]
        y        = ((v - cy) * z / f)[mask]
        z        = z[mask]
        pts      = np.column_stack([x, y, z]).astype(np.float32)
        fields   = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg             = PointCloud2()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'd435i_link'
        msg.height          = 1
        msg.width           = len(pts)
        msg.fields          = fields
        msg.is_bigendian    = False
        msg.point_step      = 12
        msg.row_step        = 12 * len(pts)
        msg.data            = pts.tobytes()
        msg.is_dense        = True
        return msg

    def _publish_camera(self):
        if self._renderer is None and not self._init_renderer():
            return
        stamp = self.get_clock().now().to_msg()
        try:
            with self._lock:
                self._renderer.update_scene(self.data, camera=self._CAM_NAME)
                rgb = self._renderer.render().copy()
                self._renderer.enable_depth_rendering()
                self._renderer.update_scene(self.data, camera=self._CAM_NAME)
                depth = self._renderer.render().copy()
                self._renderer.disable_depth_rendering()
        except Exception as exc:
            self.get_logger().warn(
                f'Camera render error: {exc}', throttle_duration_sec=5.0)
            return

        img                  = Image()
        img.header.stamp     = stamp
        img.header.frame_id  = 'd435i_link'
        img.height           = self._CAM_H
        img.width            = self._CAM_W
        img.encoding         = 'rgb8'
        img.is_bigendian     = False
        img.step             = self._CAM_W * 3
        img.data             = rgb.tobytes()
        self.rgb_pub.publish(img)

        d_img                = Image()
        d_img.header.stamp   = stamp
        d_img.header.frame_id = 'd435i_link'
        d_img.height         = self._CAM_H
        d_img.width          = self._CAM_W
        d_img.encoding       = '32FC1'
        d_img.is_bigendian   = False
        d_img.step           = self._CAM_W * 4
        d_img.data           = depth.astype(np.float32).tobytes()
        self.depth_pub.publish(d_img)

        self.cam_info_pub.publish(self._camera_info(stamp))
        self.points_pub.publish(self._depth_to_pointcloud(depth, stamp))


def main():
    global N_SUBSTEPS

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',        required=True,
                        help='Path to MuJoCo scene.xml')
    parser.add_argument('--publish-rate', type=float, default=100.0)
    parser.add_argument('--startup-pose', default='home',
                        choices=['home', 'upright'])
    parser.add_argument('--no-viewer',    action='store_true',
                        help='Run headless (no GUI window)')
    parser.add_argument('--viewer',       action='store_true',
                        help='Open the interactive MuJoCo viewer window')
    parser.add_argument('--substeps',     type=int, default=N_SUBSTEPS,
                        help='MuJoCo physics substeps per ROS timer tick')
    args = parser.parse_args()

    N_SUBSTEPS = max(1, args.substeps)

    # Only force EGL globally when running fully headless
    viewer_enabled = args.viewer and not args.no_viewer
    if not viewer_enabled:
        os.environ['MUJOCO_GL'] = 'egl'

    rclpy.init()
    node = So101MujocoBridge(
        args.model, args.publish_rate, args.startup_pose)

    # Camera thread — owns its own EGL context, never shares with viewer
    pass  # camera disabled

    # ROS executor on background threads
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin, daemon=True)
    spin_thread.start()

    # Viewer runs on the MAIN thread (OpenGL requirement)
    try:
        if viewer_enabled:
            node.run_viewer()          # blocks until window is closed
        else:
            while rclpy.ok():
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pass  # camera disabled
        node.shutdown_bridge()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
