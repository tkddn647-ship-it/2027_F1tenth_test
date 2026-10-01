"""
mapless40.sim_adapter
=====================
f1tenth_gym_ros 시뮬 → mapless40.ros_node 가 기대하는 실차 토픽으로 바꿔 주는 어댑터.

  f1tenth_gym_ros                         mapless40.ros_node
  /scan (≈250 Hz)          ──40 Hz 솎음──▶ /mapless/scan
  /ego_racecar/odom        ──────────────▶ /imu/data          (angular_velocity.z = 요레이트)
                           ──────────────▶ /vehicle/speed_mps (Float64, 종속도)
  /drive  ◀──────────────────────────────── /drive            (그대로, 브리지가 구독)

f1tenth_gym 은 충돌해도 속도만 0 으로 만들고 차를 계속 움직이게 둬서 벽을 뚫고 지나간다.
auto_reset 이면 스캔 최소거리 < collision_dist 일 때 /initialpose 로 (sx, sy, stheta) 에 되돌린다.

  python3 -m mapless40.sim_adapter
"""

from __future__ import annotations


def main():
    import rclpy
    import math

    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu, LaserScan
    from std_msgs.msg import Float64

    class SimAdapter(Node):
        def __init__(self):
            super().__init__("mapless40_sim_adapter")
            self.declare_parameter("scan_in", "/scan")
            self.declare_parameter("scan_out", "/mapless/scan")
            self.declare_parameter("odom_in", "/ego_racecar/odom")
            self.declare_parameter("imu_out", "/imu/data")
            self.declare_parameter("speed_out", "/vehicle/speed_mps")
            self.declare_parameter("scan_hz", 40.0)     # 학습 LiDAR 주기
            self.declare_parameter("odom_hz", 100.0)    # 실차 IMU ~100 Hz
            self.declare_parameter("auto_reset", True)
            self.declare_parameter("collision_dist", 0.10)  # 라이다 → 벽 [m]. 차 반폭 0.155 라 0.18 은 스치기만 해도 걸림
            self.declare_parameter("sx", 0.0)
            self.declare_parameter("sy", 0.0)
            self.declare_parameter("stheta", 0.0)
            self.auto_reset = bool(self.get_parameter("auto_reset").value)
            self.coll_dist = float(self.get_parameter("collision_dist").value)
            self.n_reset = 0
            self.t_reset = -1e9
            self.t_start = None

            self.scan_period = 1.0 / float(self.get_parameter("scan_hz").value)
            self.odom_period = 1.0 / float(self.get_parameter("odom_hz").value)
            self.t_scan = self.t_odom = -1e9

            qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.scan_pub = self.create_publisher(LaserScan, self.get_parameter("scan_out").value, qos)
            self.imu_pub = self.create_publisher(Imu, self.get_parameter("imu_out").value, 50)
            self.spd_pub = self.create_publisher(Float64, self.get_parameter("speed_out").value, 50)
            self.reset_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
            self.create_subscription(LaserScan, self.get_parameter("scan_in").value, self.on_scan, 10)
            self.create_subscription(Odometry, self.get_parameter("odom_in").value, self.on_odom, 50)
            self.get_logger().info("sim adapter ready: scan → %.0f Hz, odom → imu/speed"
                                   % (1.0 / self.scan_period))

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _reset(self, t: float, reason: str):
            self.n_reset += 1
            self.t_reset = t
            lap = f"{t - self.t_start:.1f} s 주행" if self.t_start is not None else ""
            self.get_logger().warn(f"{reason} → 출발점으로 리셋 #{self.n_reset} ({lap})")
            self.t_start = None
            th = float(self.get_parameter("stheta").value)
            m = PoseWithCovarianceStamped()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.pose.pose.position.x = float(self.get_parameter("sx").value)
            m.pose.pose.position.y = float(self.get_parameter("sy").value)
            m.pose.pose.orientation.z = math.sin(th / 2.0)
            m.pose.pose.orientation.w = math.cos(th / 2.0)
            self.reset_pub.publish(m)

        def on_scan(self, msg: LaserScan):
            t = self._now()
            if self.auto_reset and t - self.t_reset > 1.0:
                r = [x for x in msg.ranges if x > 0.0]
                if r and min(r) < self.coll_dist:
                    self._reset(t, f"충돌 (최소거리 {min(r):.2f} m)")
                    return
            # 오차 누적 없이 고정 격자에 맞춰 솎음 (늦게 오면 다음 칸으로 리셋)
            if t - self.t_scan < self.scan_period - 1e-3:
                return
            self.t_scan = t if t - self.t_scan > 2 * self.scan_period else self.t_scan + self.scan_period
            self.scan_pub.publish(msg)

        def on_odom(self, msg: Odometry):
            t = self._now()
            if t - self.t_odom < self.odom_period - 1e-3:
                return
            self.t_odom = t
            imu = Imu()
            imu.header.stamp = self.get_clock().now().to_msg()
            imu.header.frame_id = "base_link"
            imu.orientation = msg.pose.pose.orientation
            imu.angular_velocity.z = msg.twist.twist.angular.z
            self.imu_pub.publish(imu)
            self.spd_pub.publish(Float64(data=float(msg.twist.twist.linear.x)))
            if self.t_start is None and abs(msg.twist.twist.linear.x) > 0.3:
                self.t_start = t

    rclpy.init()
    node = SimAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
