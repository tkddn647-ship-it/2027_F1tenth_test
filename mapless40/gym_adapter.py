"""
mapless40.gym_adapter
=====================
f1tenth_gym_ros (팀 README §7 의 시뮬) ↔ mapless40.ros_node 연결용 어댑터.

gym_bridge 는 실차와 토픽이 다르다:
  /scan               ~250 Hz, 1080빔 269°          → /mapless/scan  40 Hz 로 솎아서 (실차 LiDAR 주기)
  /ego_racecar/odom   twist.linear.x, angular.z     → /vehicle/speed_mps (Float64, 50 Hz)
                                                     → /imu/data (Imu angular_velocity.z, 100 Hz)
  /drive              ← ros_node 가 그대로 발행 (gym_bridge 가 구독)

  ros2 launch f1tenth_gym_ros gym_bridge_launch.py
  python3 -m mapless40.gym_adapter
  python3 -m mapless40.ros_node --ros-args -p model:=best_model.zip -p meta:=none \\
      -p scan_topic:=/mapless/scan -p mount_yaw:=0.0 -p front_check:=false

"""

from __future__ import annotations


def main():
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu, LaserScan
    from std_msgs.msg import Float64

    class GymAdapter(Node):
        def __init__(self):
            super().__init__("mapless40_gym_adapter")
            self.declare_parameter("scan_in", "/scan")
            self.declare_parameter("odom_in", "/ego_racecar/odom")
            self.declare_parameter("scan_out", "/mapless/scan")
            self.declare_parameter("scan_hz", 40.0)
            self.declare_parameter("imu_hz", 100.0)
            self.declare_parameter("speed_hz", 50.0)
            g = lambda k: self.get_parameter(k).value  # noqa: E731
            self.scan_dt = 1.0 / float(g("scan_hz"))
            self.imu_dt = 1.0 / float(g("imu_hz"))
            self.spd_dt = 1.0 / float(g("speed_hz"))
            self.t_scan = self.t_imu = self.t_spd = -1e9
            qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub_scan = self.create_publisher(LaserScan, g("scan_out"), qos)
            self.pub_imu = self.create_publisher(Imu, "/imu/data", 20)
            self.pub_spd = self.create_publisher(Float64, "/vehicle/speed_mps", 20)
            self.create_subscription(LaserScan, g("scan_in"), self.on_scan, 10)
            self.create_subscription(Odometry, g("odom_in"), self.on_odom, 20)
            self.n_scan = 0
            self.create_timer(2.0, self.report)
            self.get_logger().info(f"{g('scan_in')} → {g('scan_out')} @ {g('scan_hz')} Hz, "
                                   f"{g('odom_in')} → /imu/data, /vehicle/speed_mps")

        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def on_scan(self, msg: LaserScan):
            t = self._now()
            if t - self.t_scan < self.scan_dt - 0.002:
                return
            self.t_scan = t
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_scan.publish(msg)
            self.n_scan += 1

        def on_odom(self, msg: Odometry):
            t = self._now()
            if t - self.t_imu >= self.imu_dt - 0.001:
                self.t_imu = t
                imu = Imu()
                imu.header.stamp = self.get_clock().now().to_msg()
                imu.header.frame_id = "imu_link"
                imu.angular_velocity.z = float(msg.twist.twist.angular.z)
                self.pub_imu.publish(imu)
            if t - self.t_spd >= self.spd_dt - 0.001:
                self.t_spd = t
                self.pub_spd.publish(Float64(data=float(msg.twist.twist.linear.x)))

        def report(self):
            self.get_logger().info(f"scan out {self.n_scan / 2.0:.1f} Hz")
            self.n_scan = 0

    rclpy.init()
    node = GymAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
