"""
mapless40.ros_node
==================
실차(Jetson, ROS2) mapless 정책 노드.  인터넷·맵·localization 불필요.

  구독: /scan (sensor_msgs/LaserScan, 원본 — scan_rate_adapter 쓰지 말 것)
        /imu/data (sensor_msgs/Imu)           angular_velocity.z
        /vehicle/speed_mps (std_msgs/Float32)
  발행: /drive (ackermann_msgs/AckermannDriveStamped)

  python3 -m mapless40.ros_node --ros-args -p model:=/path/actor.ts.pt -p meta:=/path/actor_meta.json

타이밍: 스캔 콜백 한 번 = 제어 한 번 (40 Hz).  스캔 d_{n-1} 이 도착하면
        d_{n-4..n-1} + 그 구간들의 v̄, ω̄, Δt + 직전 명령 2개로 a_n 을 계산해 바로 발행.
전처리는 시뮬과 같은 obs_builder 코드를 쓴다.
e-stop / AEB 는 기존 control_node · emergency_brake_node 를 그대로 둔다.
"""

from __future__ import annotations

import json
import time

import numpy as np


def main():
    import rclpy
    import torch
    from ackermann_msgs.msg import AckermannDriveStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu, LaserScan
    from std_msgs.msg import Float32

    from .config import ActionSpec, ActuatorSpec, LidarSpec, NormSpec
    from .obs_builder import IntervalAverager, ObsHistory, action_to_command, preprocess_scan

    class MaplessDriveNode(Node):
        def __init__(self):
            super().__init__("mapless40_drive")
            self.declare_parameter("model", "actor.ts.pt")
            self.declare_parameter("meta", "actor_meta.json")
            self.declare_parameter("scan_topic", "/scan")
            self.declare_parameter("imu_topic", "/imu/data")
            self.declare_parameter("speed_topic", "/vehicle/speed_mps")
            self.declare_parameter("drive_topic", "/drive")
            self.declare_parameter("max_speed", 99.0)       # 초기 시험용 추가 상한 [m/s]
            self.declare_parameter("device", "cuda")
            self.declare_parameter("use_header_stamp", False)  # False: 세 센서 모두 수신 시각 사용
            # 스캔 0° 가 차 정면이 아니면 보정 (sensor_static_tf 의 lidar_yaw 와 같은 의미, 0 또는 π)
            self.declare_parameter("mount_yaw", float("nan"))

            meta = json.loads(open(self.get_parameter("meta").value, encoding="utf-8").read())
            self.lidar = LidarSpec(**meta["lidar"])
            my = float(self.get_parameter("mount_yaw").value)
            if np.isfinite(my):
                self.lidar.mount_yaw = my
            self.norm = NormSpec(**meta["norm"])
            self.act = ActuatorSpec(**meta["act"])
            self.action = ActionSpec(**meta["action"])
            self.hist_len = int(meta["hist"])

            dev = self.get_parameter("device").value
            self.device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")
            self.model = torch.jit.load(self.get_parameter("model").value, map_location=self.device).eval()
            self.max_speed = float(self.get_parameter("max_speed").value)
            self.use_hdr = bool(self.get_parameter("use_header_stamp").value)

            self.hist = ObsHistory(self.lidar, self.norm, self.act, self.action, self.hist_len)
            self.imu = IntervalAverager(maxlen=64)
            self.spd = IntervalAverager(maxlen=32)
            self.t_last_scan: float | None = None
            self.have_imu = self.have_spd = False
            self.lat_ms: list[float] = []

            qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub = self.create_publisher(AckermannDriveStamped, self.get_parameter("drive_topic").value, 10)
            self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self.on_scan, qos)
            self.create_subscription(Imu, self.get_parameter("imu_topic").value, self.on_imu, 50)
            self.create_subscription(Float32, self.get_parameter("speed_topic").value, self.on_speed, 50)
            self.create_timer(2.0, self.report)
            self.get_logger().info(f"mapless40 ready: beams={self.lidar.n_beams} device={self.device}")

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _stamp(self, msg) -> float:
            if self.use_hdr:
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if t > 0:
                    return t
            return self._now()

        def on_imu(self, msg: Imu):
            self.imu.push(self._stamp(msg), msg.angular_velocity.z)
            self.have_imu = True

        def on_speed(self, msg: Float32):
            self.spd.push(self._now(), float(msg.data))
            self.have_spd = True

        def on_scan(self, msg: LaserScan):
            t0 = time.perf_counter()
            t = self._stamp(msg)
            scan = preprocess_scan(np.asarray(msg.ranges), msg.angle_min, msg.angle_increment,
                                   self.lidar, msg.range_min)
            if self.t_last_scan is None:
                self.hist.reset(scan, self.spd.latest(), self.imu.latest())
                self.t_last_scan = t
                return
            dt = t - self.t_last_scan
            v_bar = self.spd.mean(self.t_last_scan, t)
            w_bar = self.imu.mean(self.t_last_scan, t)
            self.t_last_scan = t
            self.hist.push(scan, v_bar, w_bar, self.imu.latest(), dt)
            if not (self.have_imu and self.have_spd):
                return

            obs = self.hist.observation()
            with torch.no_grad():
                s = torch.from_numpy(obs["scan"].astype(np.float32))[None].to(self.device)
                st = torch.from_numpy(obs["state"])[None].to(self.device)
                a = self.model(s, st)[0].cpu().numpy()
            steer, v_cmd = action_to_command(a, self.act, self.action)
            v_cmd = min(v_cmd, self.max_speed)
            out = AckermannDriveStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.drive.steering_angle = float(steer)
            out.drive.speed = float(v_cmd)
            self.pub.publish(out)
            self.hist.push_cmd(steer, v_cmd)
            self.lat_ms.append((time.perf_counter() - t0) * 1e3)

        def report(self):
            if self.lat_ms:
                a = np.array(self.lat_ms[-200:])
                self.get_logger().info(f"latency ms mean {a.mean():.2f} p99 {np.percentile(a, 99):.2f}")

    rclpy.init()
    node = MaplessDriveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
