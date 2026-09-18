"""
roboracer_connectome_node.py
==============================
실차(Roboracer-2026) 드롭인 스케치.

역할: stanley_waypoint_follow_node 대신
  구독: /scan, /vehicle/speed_mps (있으면)
  발행: /drive  (ackermann_msgs/AckermannDriveStamped)

학습된 Connectome/MLP 정책 zip 을 로드해 LiDAR→조향·속도.

이 파일은 ROS2 의존성이 있는 Jetson에서 실행한다.
PC 학습 환경에서는 import만 참고용.

  ros2 run ...  (패키지화 후)
  또는:
  python3 roboracer_connectome_node.py --model connectome_ppo_lidar_ajou.zip
"""

from __future__ import annotations
import argparse

# --- ROS2 블록: Jetson에서만 활성화 ---------------------------------
def main_ros():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from ackermann_msgs.msg import AckermannDriveStamped
    from std_msgs.msg import Float32
    import numpy as np
    from stable_baselines3 import PPO

    class ConnectomeDriveNode(Node):
        def __init__(self, model_path: str, n_rays: int = 20, max_steer=0.3735,
                     min_speed=2.0, max_speed=7.0):
            super().__init__("connectome_drive")
            self.n_rays = n_rays
            self.max_steer = max_steer
            self.min_speed = min_speed
            self.max_speed = max_speed
            self.speed_meas = min_speed
            self.model = PPO.load(model_path, device="cpu")
            self.pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
            self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
            self.create_subscription(Float32, "/vehicle/speed_mps", self.on_speed, 10)
            self.get_logger().info(f"Connectome drive ready: {model_path}")

        def on_speed(self, msg: Float32):
            self.speed_meas = float(msg.data)

        def on_scan(self, msg: LaserScan):
            ranges = np.asarray(msg.ranges, dtype=np.float32)
            ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max)
            idx = np.linspace(0, len(ranges) - 1, self.n_rays).astype(int)
            d = np.clip(ranges[idx] / max(msg.range_max, 1e-3), 0, 1)
            yaw_rate = 0.0  # 있으면 IMU에서 채움
            span = max(self.max_speed - self.min_speed, 1e-6)
            v_norm = float(np.clip((self.speed_meas - self.min_speed) / span, 0, 1))
            obs = np.concatenate([d, [v_norm, yaw_rate]])
            action, _ = self.model.predict(obs.astype(np.float32), deterministic=True)
            steer = float(np.clip(action[0], -1, 1)) * self.max_steer
            u = float(np.clip(action[1], 0, 1))
            speed = self.min_speed + u * span
            out = AckermannDriveStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.drive.steering_angle = steer
            out.drive.speed = speed
            self.pub.publish(out)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = ConnectomeDriveNode(args.model)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main_ros()
    except ImportError as e:
        print("ROS2/ackermann 미설치 — 이 노드는 Jetson(Roboracer-2026)에서 실행하세요.")
        print("학습은 PC에서: python train_connectome.py --env lidar --map ajou --use-cache")
        print(f"(import error: {e})")
