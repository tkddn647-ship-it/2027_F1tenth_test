#정적 장애물 인식 노드

#!/usr/bin/env python3
"""
정적 장애물 노드 (최초 버전): /scan 클러스터링 → 크기·위치 필터 → /static_obstacles 발행.
"""
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration


class StaticObstacleNode(Node):
    def __init__(self):
        super().__init__("static_obstacle_node")

        self.subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.listener_callback,
            10,
        )
        self.marker_pub = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)
        self.obstacle_pub = self.create_publisher(Float32MultiArray, "/static_obstacles", 10)

    def listener_callback(self, msg):
        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        points = []
        for i, r in enumerate(ranges):
            if r < float("inf") and r > 0.0:
                points.append((i, r))

        if not points:
            return

        THRESHOLD = 0.20
        clusters = []
        current_cluster = []
        current_cluster.append(points[0])

        for k in range(1, len(points)):
            prev_idx, prev_r = points[k - 1]
            curr_idx, curr_r = points[k]
            prev_theta = angle_min + prev_idx * angle_increment
            curr_theta = angle_min + curr_idx * angle_increment
            prev_x = prev_r * math.cos(prev_theta)
            prev_y = prev_r * math.sin(prev_theta)
            curr_x = curr_r * math.cos(curr_theta)
            curr_y = curr_r * math.sin(curr_theta)
            dist = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            if dist > THRESHOLD:
                clusters.append(current_cluster)
                current_cluster = [points[k]]
            else:
                current_cluster.append(points[k])

        if current_cluster:
            clusters.append(current_cluster)

        valid_clusters = [c for c in clusters if len(c) > 5]

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        obstacle_data_list = []
        final_obstacle_count = 0

        for idx, cluster in enumerate(valid_clusters):
            x_coords = []
            y_coords = []
            closest_dist_sq = float("inf")
            closest_point = (0.0, 0.0)

            for p in cluster:
                theta = angle_min + p[0] * angle_increment
                r = p[1]
                px = r * math.cos(theta)
                py = r * math.sin(theta)
                x_coords.append(px)
                y_coords.append(py)
                dist_sq = px * px + py * py
                if dist_sq < closest_dist_sq:
                    closest_dist_sq = dist_sq
                    closest_point = (px, py)

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            size_x = max_x - min_x
            size_y = max_y - min_y
            vis_center_x = (min_x + max_x) / 2.0
            vis_center_y = (min_y + max_y) / 2.0
            logic_x = closest_point[0]
            logic_y = closest_point[1]

            if size_x > 1.0 or size_y > 1.0:
                continue
            if abs(logic_y) > 1.2:
                continue
            if logic_x > 5.0 or logic_x < 0.0:
                continue
            if size_x < 0.2 or size_y < 0.2:
                continue

            radius = max(size_x, size_y) / 2.0
            obstacle_data_list.extend([float(idx), logic_x, logic_y, radius])

            vis_size_x = size_x if size_x >= 0.1 else 0.1
            vis_size_y = size_y if size_y >= 0.1 else 0.1

            marker = Marker()
            marker.header.frame_id = "ego_racecar/laser"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = vis_center_x
            marker.pose.position.y = vis_center_y
            marker.pose.position.z = 0.0
            marker.scale.x = vis_size_x
            marker.scale.y = vis_size_y
            marker.scale.z = 0.2
            marker.color.a = 0.8
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.lifetime = Duration(sec=0, nanosec=200000000)
            marker_array.markers.append(marker)
            final_obstacle_count += 1

        self.marker_pub.publish(marker_array)
        obs_msg = Float32MultiArray()
        obs_msg.data = obstacle_data_list
        self.obstacle_pub.publish(obs_msg)

        if final_obstacle_count > 0:
            self.get_logger().info(
                f"장애물 {final_obstacle_count}개 감지 → /static_obstacles 발행"
            )


def main(args=None):
    rclpy.init(args=args)
    node = StaticObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
