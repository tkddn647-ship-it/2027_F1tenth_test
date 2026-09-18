#!/usr/bin/env python3
"""
FGM (Gap Follow Method) 노드.
/scan + /static_obstacles 구독 → Safety Bubble → Max Gap → /fgm_target, /fgm_gap_marker 발행.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker


class FGMNode(Node):
    def __init__(self):
        super().__init__("fgm_node")

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.obstacle_sub = self.create_subscription(
            Float32MultiArray, "/static_obstacles", self.obstacle_callback, 10
        )

        self.target_pub = self.create_publisher(PointStamped, "/fgm_target", 10)
        self.debug_scan_pub = self.create_publisher(LaserScan, "/fgm_debug_scan", 10)
        self.gap_marker_pub = self.create_publisher(Marker, "/fgm_gap_marker", 10)

        self.bubble_radius = 0.20
        self.preprocess_dist = 5.0  # 장애물 인식 거리(m). 이 거리 안만 갭/버블 계산에 사용
        self.latest_obstacles = []
        self.scan_frame_id = "laser"

    def obstacle_callback(self, msg):
        self.latest_obstacles = list(msg.data)

    def scan_callback(self, scan_msg):
        if scan_msg.header.frame_id:
            self.scan_frame_id = scan_msg.header.frame_id
        ranges = np.array(scan_msg.ranges, dtype=np.float64)
        ranges = np.where(np.isinf(ranges), self.preprocess_dist, ranges)
        ranges = np.where(np.isnan(ranges), 0.0, ranges)
        ranges[ranges > self.preprocess_dist] = self.preprocess_dist

        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment

        closest_idx = int(np.argmin(ranges))
        min_dist = float(ranges[closest_idx])
        if min_dist < self.preprocess_dist:
            self._create_bubble(ranges, closest_idx, min_dist, angle_inc)

        if len(self.latest_obstacles) > 0:
            num_obs = len(self.latest_obstacles) // 4
            for i in range(num_obs):
                obs_x = self.latest_obstacles[4 * i + 1]
                obs_y = self.latest_obstacles[4 * i + 2]
                obs_r = self.latest_obstacles[4 * i + 3]
                obs_dist = math.sqrt(obs_x * obs_x + obs_y * obs_y)
                obs_angle = math.atan2(obs_y, obs_x)
                obs_idx = int((obs_angle - angle_min) / angle_inc)
                if 0 <= obs_idx < len(ranges):
                    effective_radius = self.bubble_radius + obs_r
                    self._create_bubble(
                        ranges, obs_idx, obs_dist, angle_inc, radius_override=effective_radius
                    )

        gap_threshold = 1.2
        threshold_indices = np.where(ranges > gap_threshold)[0]
        if len(threshold_indices) == 0:
            gap_threshold = 0.5
            threshold_indices = np.where(ranges > gap_threshold)[0]
            if len(threshold_indices) == 0:
                return

        splits = np.where(np.diff(threshold_indices) > 1)[0] + 1
        gaps = np.split(threshold_indices, splits)
        max_gap = max(gaps, key=len)
        gap_start_idx = int(max_gap[0])
        gap_end_idx = int(max_gap[-1])
        best_idx = int(max_gap[len(max_gap) // 2])
        target_angle = angle_min + best_idx * angle_inc
        target_dist = 3.0
        target_x = target_dist * math.cos(target_angle)
        target_y = target_dist * math.sin(target_angle)

        point_msg = PointStamped()
        point_msg.header = scan_msg.header
        point_msg.header.frame_id = self.scan_frame_id
        point_msg.point.x = float(target_x)
        point_msg.point.y = float(target_y)
        point_msg.point.z = 0.0
        self.target_pub.publish(point_msg)

        debug_msg = LaserScan()
        debug_msg.header = scan_msg.header
        debug_msg.angle_min = scan_msg.angle_min
        debug_msg.angle_max = scan_msg.angle_max
        debug_msg.angle_increment = scan_msg.angle_increment
        debug_msg.range_min = scan_msg.range_min
        debug_msg.range_max = scan_msg.range_max
        debug_msg.ranges = [float(r) for r in ranges]
        self.debug_scan_pub.publish(debug_msg)

        self._publish_gap_marker(
            gap_start_idx, gap_end_idx, ranges, angle_min, angle_inc, scan_msg.header
        )

    def _publish_gap_marker(self, start_idx, end_idx, ranges, angle_min, angle_inc, header):
        marker = Marker()
        marker.header = header
        marker.header.frame_id = self.scan_frame_id
        marker.ns = "fgm_gap"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        gap_marker_offset = 1.5  # 차량에서 1.5m 더 떨어진 위치에 갭마커(빨간선) 표시
        start_angle = angle_min + start_idx * angle_inc
        start_dist = float(ranges[start_idx]) + gap_marker_offset
        p1 = Point()
        p1.x = start_dist * math.cos(start_angle)
        p1.y = start_dist * math.sin(start_angle)
        p1.z = 0.0

        end_angle = angle_min + end_idx * angle_inc
        end_dist = float(ranges[end_idx]) + gap_marker_offset
        p2 = Point()
        p2.x = end_dist * math.cos(end_angle)
        p2.y = end_dist * math.sin(end_angle)
        p2.z = 0.0

        marker.points.append(p1)
        marker.points.append(p2)
        self.gap_marker_pub.publish(marker)

    def _create_bubble(self, ranges, center_idx, dist, angle_inc, radius_override=None):
        radius = radius_override if radius_override is not None else self.bubble_radius
        safe_theta = math.atan(radius / (dist + 0.001))
        idx_radius = int(safe_theta / angle_inc)
        start_idx = max(0, center_idx - idx_radius)
        end_idx = min(len(ranges), center_idx + idx_radius)
        ranges[start_idx:end_idx] = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = FGMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
