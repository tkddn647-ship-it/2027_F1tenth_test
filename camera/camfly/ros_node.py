"""
camfly.ros_node
===============
실차(Jetson, ROS2) 노드: Orbbec Gemini 2L → 파리 눈 + depth 가짜 스캔 → 커넥톰 정책 → /drive.

  ros2 launch orbbec_camera gemini2L.launch.py depth_registration:=true      # OrbbecSDK_ROS2
  python3 -m camera.camfly.ros_node --ros-args -p model:=best_model.zip

구독 (기본값, OrbbecSDK_ROS2 기준 — 실제 이름은 `ros2 topic list` 로 확인):
  /camera/color/image_raw     sensor_msgs/Image (rgb8/bgr8/mono8)   ← 프레임마다 제어 1회 (30 Hz)
  /camera/color/camera_info   sensor_msgs/CameraInfo                ← K (왜곡 보정된 영상 가정)
  /camera/depth/image_raw     sensor_msgs/Image (16UC1 mm 또는 32FC1 m), color 에 정렬 (depth_registration)
  /imu/data                   sensor_msgs/Imu (차 EBIMU)
  /vehicle/speed_mps          std_msgs/Float64 (control_node)
발행: /drive (AckermannDriveStamped)

모델: SB3 zip (encoder=fly) → torch 없이 numpy 로 실행 (camfly.np_actor).
노출은 고정·짧게 (≤ 2 ms) 하는 것을 권장 — 시뮬은 노출 고정 + 밝기 랜덤화로 학습.
"""

from __future__ import annotations

import time

import numpy as np


def _img_to_np(msg):
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint16 if enc in ("16uc1", "mono16") else
                        (np.float32 if enc == "32fc1" else np.uint8))
    if enc in ("rgb8", "bgr8"):
        a = buf.reshape(msg.height, msg.width, 3).astype(np.float32)
        return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) if enc == "rgb8" else \
               (0.114 * a[..., 0] + 0.587 * a[..., 1] + 0.299 * a[..., 2])
    if enc in ("16uc1", "mono16"):
        return buf.reshape(msg.height, msg.width).astype(np.float32) / 1000.0     # mm → m
    if enc == "32fc1":
        return buf.reshape(msg.height, msg.width).astype(np.float32)
    return buf.reshape(msg.height, -1).astype(np.float32)                        # mono8


def main():
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image, Imu
    from std_msgs.msg import Float64

    from mapless40.obs_builder import IntervalAverager, action_to_command

    from .config import CamFlyConfig
    from .env import CamHistory
    from .eye import adapt, depth_to_scan, eye_pixel_map, resample_gray
    from .np_actor import CamNumpyActor

    class CamFlyNode(Node):
        def __init__(self):
            super().__init__("camfly_drive")
            d = self.declare_parameter
            d("model", "best_model.zip")
            d("color_topic", "/camera/color/image_raw")
            d("info_topic", "/camera/color/camera_info")
            d("depth_topic", "/camera/depth/image_raw")
            d("imu_topic", "/imu/data")
            d("speed_topic", "/vehicle/speed_mps")
            d("drive_topic", "/drive")
            d("max_speed", 99.0)
            d("cam_pitch_deg", float("nan"))        # 실제 장착 각도 (NaN = config 값)
            g = lambda k: self.get_parameter(k).value  # noqa: E731
            self.cf = CamFlyConfig()
            pd = float(g("cam_pitch_deg"))
            self.pitch = None if not np.isfinite(pd) else pd
            self.actor = CamNumpyActor(g("model"))
            self.hist = CamHistory(self.cf)
            self.imu = IntervalAverager(maxlen=64)
            self.spd = IntervalAverager(maxlen=32)
            self.K = None
            self.map = None
            self.depth = None
            self.t_last = None
            self.lat_ms: list[float] = []
            self.max_speed = float(g("max_speed"))
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub = self.create_publisher(AckermannDriveStamped, g("drive_topic"), 10)
            self.create_subscription(CameraInfo, g("info_topic"), self.on_info, 10)
            self.create_subscription(Image, g("depth_topic"), self.on_depth, qos)
            self.create_subscription(Image, g("color_topic"), self.on_color, qos)
            self.create_subscription(Imu, g("imu_topic"), self.on_imu, 50)
            self.create_subscription(Float64, g("speed_topic"), self.on_speed, 50)
            self.create_timer(2.0, self.report)
            self.get_logger().info(f"camfly ready (step {self.actor.num_timesteps:,})")

        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def on_info(self, msg):
            if self.K is None:
                self.K = np.array(msg.k, np.float64).reshape(3, 3)
                self.W, self.H = msg.width, msg.height
                self.map = eye_pixel_map(self.cf.cam, self.K, self.W, self.H, self.pitch)
                self.get_logger().info(f"K fx={self.K[0,0]:.1f} cx={self.K[0,2]:.1f} {self.W}x{self.H}, "
                                       f"파리 눈 유효 칸 {int(self.map[2].sum())}/{self.map[2].size}")

        def on_depth(self, msg):
            self.depth = _img_to_np(msg)

        def on_imu(self, msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.imu.push(t if t > 0 else self._now(), float(np.clip(msg.angular_velocity.z, -8, 8)))

        def on_speed(self, msg):
            self.spd.push(self._now(), float(msg.data))

        def on_color(self, msg):
            if self.map is None:
                return
            t0 = time.perf_counter()
            t = self._now()
            gray = _img_to_np(msg)
            eye = adapt(resample_gray(gray, *self.map)).astype(np.float32)
            if self.depth is not None and self.depth.shape == gray.shape:
                ds = depth_to_scan(self.depth, self.K, self.cf.cam, self.pitch)
            else:
                ds = np.ones(self.cf.cam.n_cols, np.float32)
            frame = (eye, ds)
            if self.t_last is None:
                self.hist.reset(frame, self.spd.latest(), self.imu.latest())
                self.t_last = t
                return
            self.hist.push(frame, self.spd.mean(self.t_last, t), self.imu.mean(self.t_last, t),
                           self.imu.latest(), t - self.t_last)
            self.t_last = t
            a = self.actor(self.hist.observation())
            steer, v = action_to_command(a, self.cf.env.act, self.cf.env.action)
            v = min(v, self.max_speed)
            out = AckermannDriveStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.drive.steering_angle, out.drive.speed = float(steer), float(v)
            self.pub.publish(out)
            self.hist.push_cmd(steer, v)
            self.lat_ms.append((time.perf_counter() - t0) * 1e3)

        def report(self):
            if self.lat_ms:
                a = np.array(self.lat_ms[-120:])
                self.get_logger().info(f"latency ms mean {a.mean():.1f} p99 {np.percentile(a, 99):.1f}")
            elif self.K is None:
                self.get_logger().warn("camera_info 미수신")

    rclpy.init()
    node = CamFlyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
