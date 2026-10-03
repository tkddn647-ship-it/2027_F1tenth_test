"""
depthfly.ros_node
=================
실차(Jetson, ROS2): Orbbec Gemini 2L **depth 만** → 파리 눈 가까움 영상 → 커넥톰 → /drive.  RGB 안 씀.

  ros2 launch orbbec_camera gemini2L.launch.py enable_color:=false depth_fps:=30     # OrbbecSDK_ROS2
  python3 -m depthfly.ros_node --ros-args -p model:=best_model.zip -p max_speed:=2.0

구독 (OrbbecSDK_ROS2 기본 이름 — 실제 이름은 `ros2 topic list` 로 확인):
  /camera/depth/image_raw     sensor_msgs/Image (16UC1 mm 또는 32FC1 m)   ← 프레임마다 제어 1회 (30 Hz)
  /camera/depth/camera_info   sensor_msgs/CameraInfo                      ← depth 내부 파라미터 K
  /imu/data                   sensor_msgs/Imu (차 EBIMU)
  /vehicle/speed_mps          std_msgs/Float64 (control_node)
발행: /drive (AckermannDriveStamped)

계산량: depth 는 카메라 칩이 계산 → Jetson 은 64×16 칸 샘플링 + 회로 (합계 ~1–2 ms/프레임, CPU 1코어).
"""

from __future__ import annotations

import time

import numpy as np


def _depth_to_np(msg):
    enc = msg.encoding.lower()
    if enc in ("16uc1", "mono16"):
        return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
    if enc == "32fc1":
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy()
    raise ValueError(f"depth encoding {msg.encoding} 미지원")


def main():
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image, Imu
    from std_msgs.msg import Float64

    from mapless40.obs_builder import IntervalAverager, action_to_command

    from .config import DepthFlyConfig
    from .env import NearHistory
    from .np_actor import NearNumpyActor
    from .sensor import DepthEyeReal

    class DepthFlyNode(Node):
        def __init__(self):
            super().__init__("depthfly_drive")
            d = self.declare_parameter
            d("model", "best_model.zip")
            d("depth_topic", "/camera/depth/image_raw")
            d("info_topic", "/camera/depth/camera_info")
            d("imu_topic", "/imu/data")
            d("speed_topic", "/vehicle/speed_mps")
            d("drive_topic", "/drive")
            d("max_speed", 99.0)
            d("cam_pitch_deg", float("nan"))        # 실제 장착 각도 (NaN = config 값 −10°)
            g = lambda k: self.get_parameter(k).value  # noqa: E731
            self.cf = DepthFlyConfig()
            pd = float(g("cam_pitch_deg"))
            self.pitch = pd if np.isfinite(pd) else None
            self.actor = NearNumpyActor(g("model"))
            self.hist = NearHistory(self.cf)
            self.imu = IntervalAverager(maxlen=64)
            self.spd = IntervalAverager(maxlen=32)
            self.eye = None
            self.t_last = None
            self.lat_ms: list[float] = []
            self.max_speed = float(g("max_speed"))
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub = self.create_publisher(AckermannDriveStamped, g("drive_topic"), 10)
            self.create_subscription(CameraInfo, g("info_topic"), self.on_info, 10)
            self.create_subscription(Image, g("depth_topic"), self.on_depth, qos)
            self.create_subscription(Imu, g("imu_topic"), self.on_imu, 50)
            self.create_subscription(Float64, g("speed_topic"), self.on_speed, 50)
            self.create_timer(2.0, self.report)
            self.get_logger().info(f"depthfly ready (step {self.actor.num_timesteps:,})")

        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def on_info(self, msg):
            if self.eye is None:
                K = np.array(msg.k, np.float64).reshape(3, 3)
                self.eye = DepthEyeReal(self.cf.cam, K, msg.width, msg.height, self.pitch)
                self.get_logger().info(f"K fx={K[0,0]:.1f} cx={K[0,2]:.1f} {msg.width}x{msg.height}, "
                                       f"파리 눈 유효 칸 {int(self.eye.ok.sum())}/{self.eye.ok.size}")

        def on_imu(self, msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.imu.push(t if t > 0 else self._now(), float(np.clip(msg.angular_velocity.z, -8, 8)))

        def on_speed(self, msg):
            self.spd.push(self._now(), float(msg.data))

        def on_depth(self, msg):
            if self.eye is None:
                return
            t0 = time.perf_counter()
            t = self._now()
            near = self.eye.frame(_depth_to_np(msg))
            if self.t_last is None:
                self.hist.reset(near, self.spd.latest(), self.imu.latest())
                self.t_last = t
                return
            self.hist.push(near, self.spd.mean(self.t_last, t), self.imu.mean(self.t_last, t),
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
            elif self.eye is None:
                self.get_logger().warn("camera_info 미수신")

    rclpy.init()
    node = DepthFlyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
