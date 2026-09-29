"""
mapless40.ros_node
==================
실차(Jetson, ROS2) mapless 정책 노드.  인터넷·맵·localization 불필요.

Roboracer-2026-main 스택 기준 토픽 (기본값):
  구독: /scan               sensor_msgs/LaserScan   sllidar_node 원본 (scan_rate_adapter 쓰지 말 것)
        /imu/data           sensor_msgs/Imu         ebimu_driver, imu_link = base_link 축, ~95 Hz
        /vehicle/speed_mps  std_msgs/Float64        control_node 가 VESC ERPM 으로 50 Hz 발행
  발행: /drive              ackermann_msgs/AckermannDriveStamped  → control_node (AUTO)

  python3 -m mapless40.ros_node --ros-args -p model:=/path/actor.ts.pt -p meta:=/path/actor_meta.json

라이다 방향 (중요):
  sllidar 드라이버는 360° 한 바퀴를 angle = π − raw 로 발행한다. 스캔 0° 가 차 정면인지는
  라이다를 어떻게 달았는지에 달렸다 (2026-08-15 실측 기록: 정면 = 스캔 −177°).
  이 노드는 mount_yaw 를
    1) 파라미터 mount_yaw 가 주어지면 그 값,
    2) 아니면 TF base_link → laser 의 yaw (sensor_static_tf 의 lidar_yaw),
  로 잡고, 시작할 때 정지 상태 스캔으로 **차체에 가린 구간이 차 뒤(±180°)에 있는지** 검사한다.
  가린 구간이 앞쪽이면 방향이 틀린 것이므로 명령을 내지 않는다.

타이밍: 스캔 콜백 한 번 = 제어 한 번 (40 Hz).
  스캔은 수신 시각, IMU 는 header.stamp (ebimu_driver 가 PLL 로 10 ms 간격 stamp 를 찍음 —
  시리얼 배치로 몰려 들어오므로 수신 시각을 쓰면 여러 샘플이 한 시각에 뭉친다).
e-stop / AEB 는 기존 control_node · emergency_brake_node 를 그대로 둔다.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np


def main():
    import os

    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu, LaserScan
    from std_msgs.msg import Float32, Float64

    from .config import ActionSpec, ActuatorSpec, LidarSpec, NormSpec
    from .obs_builder import (IntervalAverager, ObsHistory, action_to_command, blocked_center_deg,
                              invalid_bins, preprocess_scan)

    class MaplessDriveNode(Node):
        def __init__(self):
            super().__init__("mapless40_drive")
            self.declare_parameter("model", "actor.ts.pt")
            self.declare_parameter("meta", "actor_meta.json")
            self.declare_parameter("scan_topic", "/scan")
            self.declare_parameter("imu_topic", "/imu/data")
            self.declare_parameter("speed_topic", "/vehicle/speed_mps")
            self.declare_parameter("speed_msg_type", "float64")   # control_node 는 Float64
            self.declare_parameter("drive_topic", "/drive")
            self.declare_parameter("max_speed", 99.0)       # 초기 시험용 추가 상한 [m/s]
            self.declare_parameter("device", "cuda")
            self.declare_parameter("imu_use_header_stamp", True)
            self.declare_parameter("scan_use_header_stamp", False)
            self.declare_parameter("gyro_clip", 8.0)        # rad/s, orientation-only 스트림 튐 방지
            # 스캔 → 차량 각도 회전 [rad]. NaN 이면 TF base_link→laser yaw 사용.
            self.declare_parameter("mount_yaw", float("nan"))
            self.declare_parameter("base_frame", "base_link")
            self.declare_parameter("laser_frame", "laser")
            # 시작 시 차체 가림 구간 검사 (정지 상태에서 스캔 check_scans 장)
            self.declare_parameter("front_check", True)
            self.declare_parameter("check_scans", 40)

            meta_path = self.get_parameter("meta").value
            if meta_path and os.path.exists(meta_path):
                meta = json.loads(open(meta_path, encoding="utf-8").read())
            else:                                   # zip 만 있을 때: 학습 기본 설정값
                from .config import EnvConfig
                c = EnvConfig().to_dict()
                meta = {"lidar": c["lidar"], "norm": c["norm"], "act": c["act"],
                        "action": c["action"], "hist": c["timing"]["hist"]}
                self.get_logger().warn(f"meta 파일 없음({meta_path}) → config.py 기본값 사용")
            self.lidar = LidarSpec(**meta["lidar"])
            self.norm = NormSpec(**meta["norm"])
            self.act = ActuatorSpec(**meta["act"])
            self.action = ActionSpec(**meta["action"])
            self.hist_len = int(meta["hist"])

            model_path = self.get_parameter("model").value
            if model_path.endswith(".zip"):         # SB3 zip 을 torch 없이 바로 (conv1d)
                from .np_actor import NumpyActor
                self.np_actor = NumpyActor(model_path)
                self.device = "numpy"
                self.get_logger().info(f"numpy actor ← {model_path} (step {self.np_actor.num_timesteps:,})")
            else:
                import torch
                self.torch = torch
                self.np_actor = None
                dev = self.get_parameter("device").value
                self.device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")
                self.model = torch.jit.load(model_path, map_location=self.device).eval()
            self.max_speed = float(self.get_parameter("max_speed").value)
            self.imu_hdr = bool(self.get_parameter("imu_use_header_stamp").value)
            self.scan_hdr = bool(self.get_parameter("scan_use_header_stamp").value)
            self.gyro_clip = float(self.get_parameter("gyro_clip").value)

            self.hist = ObsHistory(self.lidar, self.norm, self.act, self.action, self.hist_len)
            self.imu = IntervalAverager(maxlen=64)
            self.spd = IntervalAverager(maxlen=32)
            self.t_last_scan: float | None = None
            self.have_imu = self.have_spd = False
            self.lat_ms: list[float] = []
            self.scan_dts: list[float] = []
            self.n_beams_seen: list[int] = []

            # --- 라이다 방향 ---
            my = float(self.get_parameter("mount_yaw").value)
            src = "param"
            if not np.isfinite(my):
                my, src = self._tf_yaw()
            self.lidar.mount_yaw = my
            self.get_logger().info(f"mount_yaw = {math.degrees(my):+.1f}° ({src})")
            self.check_left = int(self.get_parameter("check_scans").value) \
                if bool(self.get_parameter("front_check").value) else 0
            self.check_acc = np.zeros(36)
            self.check_n = 0
            self.ready = self.check_left == 0

            qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
            spd_type = Float64 if str(self.get_parameter("speed_msg_type").value).lower() == "float64" else Float32
            self.pub = self.create_publisher(AckermannDriveStamped, self.get_parameter("drive_topic").value, 10)
            self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self.on_scan, qos)
            self.create_subscription(Imu, self.get_parameter("imu_topic").value, self.on_imu, 50)
            self.create_subscription(spd_type, self.get_parameter("speed_topic").value, self.on_speed, 50)
            self.create_timer(2.0, self.report)
            self.get_logger().info(f"mapless40 ready: beams={self.lidar.n_beams} device={self.device} "
                                   f"speed_msg={spd_type.__name__}")

        # ------------------------------------------------------------------
        def _tf_yaw(self) -> tuple[float, str]:
            try:
                import tf2_ros
                buf = tf2_ros.Buffer()
                tf2_ros.TransformListener(buf, self)
                base = self.get_parameter("base_frame").value
                laser = self.get_parameter("laser_frame").value
                t_end = time.time() + 3.0
                while time.time() < t_end:
                    rclpy.spin_once(self, timeout_sec=0.1)
                    if buf.can_transform(base, laser, rclpy.time.Time()):
                        q = buf.lookup_transform(base, laser, rclpy.time.Time()).transform.rotation
                        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
                        if abs(roll) > 1.0:
                            self.get_logger().warn("TF base_link→laser roll ≈ π (뒤집힌 장착). "
                                                   "이 노드는 roll 을 반영하지 않음 — sllidar inverted 로 맞출 것")
                        return yaw, f"TF {base}→{laser}"
            except Exception as e:  # pragma: no cover
                self.get_logger().warn(f"TF 조회 실패: {e}")
            self.get_logger().warn("TF base_link→laser 없음 → mount_yaw 0 가정 (시작 검사로 확인)")
            return 0.0, "default"

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        @staticmethod
        def _hdr(msg) -> float:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        def on_imu(self, msg: Imu):
            t = self._hdr(msg) if self.imu_hdr else 0.0
            if t <= 0.0:
                t = self._now()
            wz = float(np.clip(msg.angular_velocity.z, -self.gyro_clip, self.gyro_clip))
            self.imu.push(t, wz)
            self.have_imu = True

        def on_speed(self, msg):
            self.spd.push(self._now(), float(msg.data))
            self.have_spd = True

        # ------------------------------------------------------------------
        def _front_check(self, msg: LaserScan) -> None:
            self.check_acc += invalid_bins(np.asarray(msg.ranges), msg.angle_min, msg.angle_increment,
                                           self.lidar.mount_yaw)
            self.check_n += 1
            self.check_left -= 1
            if self.check_left > 0:
                return
            center, width = blocked_center_deg(self.check_acc / self.check_n)
            n = int(np.median(self.n_beams_seen)) if self.n_beams_seen else 0
            fov_pts = n * self.lidar.fov_deg / 360.0
            self.get_logger().info(
                f"[시작 검사] 스캔 점수 {n}/rev (270° 안 ≈ {fov_pts:.0f}, 학습 격자 {self.lidar.n_beams}) | "
                f"가린 구간 중심 {center:+.0f}° 폭 {width:.0f}° (정상: ±180° 부근)")
            if np.isfinite(center) and abs(center) < 120.0:
                self.get_logger().error(
                    f"가린 구간이 차 앞/옆({center:+.0f}°)에 있음 → 라이다 방향이 틀림. 명령 안 냄.\n"
                    f"  추정 보정값: -p mount_yaw:="
                    f"{math.radians((math.degrees(self.lidar.mount_yaw) + 180.0 - center + 180.0) % 360.0 - 180.0):.3f}"
                    f"  (라이다 스캔 0° 가 차 뒤를 보면 보통 π = 3.142)")
                self.ready = False
                return
            if not np.isfinite(center):
                self.get_logger().warn("가린 구간을 못 찾음 (주변이 너무 가깝거나 비어 있음) — "
                                       "차 정면에 상자를 두고 RViz 로 방향 한 번 확인할 것")
            self.ready = True
            self.get_logger().info("시작 검사 통과 → 제어 시작")

        def on_scan(self, msg: LaserScan):
            t0 = time.perf_counter()
            t = self._hdr(msg) if self.scan_hdr else self._now()
            self.n_beams_seen.append(len(msg.ranges))
            if len(self.n_beams_seen) > 200:
                self.n_beams_seen = self.n_beams_seen[-100:]
            if self.check_left > 0:
                self._front_check(msg)
            scan = preprocess_scan(np.asarray(msg.ranges), msg.angle_min, msg.angle_increment,
                                   self.lidar, msg.range_min)
            if self.t_last_scan is None:
                self.hist.reset(scan, self.spd.latest(), self.imu.latest())
                self.t_last_scan = t
                return
            dt = t - self.t_last_scan
            self.scan_dts.append(dt)
            v_bar = self.spd.mean(self.t_last_scan, t)
            w_bar = self.imu.mean(self.t_last_scan, t)
            self.t_last_scan = t
            self.hist.push(scan, v_bar, w_bar, self.imu.latest(), dt)
            if not (self.ready and self.have_imu and self.have_spd):
                return

            obs = self.hist.observation()
            if self.np_actor is not None:
                a = self.np_actor(obs)
            else:
                torch = self.torch
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
            parts = []
            if self.scan_dts:
                d = np.array(self.scan_dts[-80:])
                hz = 1.0 / max(np.median(d), 1e-6)
                parts.append(f"scan {hz:.1f} Hz (dt p95 {np.percentile(d, 95) * 1e3:.0f} ms)")
                if hz < 35.0:
                    self.get_logger().warn(f"LiDAR {hz:.0f} Hz — 학습은 40 Hz. lidar_scan_frequency:=40.0 로 켤 것")
                self.scan_dts = self.scan_dts[-80:]
            if not self.have_imu:
                parts.append("IMU 미수신")
            if not self.have_spd:
                parts.append("속도 미수신")
            if self.lat_ms:
                a = np.array(self.lat_ms[-200:])
                parts.append(f"latency ms mean {a.mean():.2f} p99 {np.percentile(a, 99):.2f}")
            if not self.ready and self.check_left <= 0:
                parts.append("정지 (라이다 방향 검사 실패)")
            if parts:
                self.get_logger().info(" | ".join(parts))

    rclpy.init()
    node = MaplessDriveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
