#경로 추적 노드

#!/usr/bin/env python3
"""
센터라인 경로 노드: CSV 로드 후 차량 위치 기준 앞쪽 N개만 슬라이딩 윈도우로 /recommended_path
 발행. 로컬플래너가 이 토픽을 구독해 /local_path로 전달.
"""
import csv
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener, TransformException


def load_centerline_csv(path: str) -> List[Tuple[float, float]]:
    """CSV 로드. 'x,y' 또는 'x_m, y_m, ...' 형식 지원."""
    pts: List[Tuple[float, float]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    start = 0
    for i, r in enumerate(rows):
        if not r or r[0].strip().startswith("#"):
            start = i + 1
            continue
        if len(r) < 2:
            continue
        try:
            x = float(r[0].strip())
            y = float(r[1].strip())
            pts.append((x, y))
        except ValueError:
            if i == start and (
                "x" in (r[0] + r[1]).lower() or "m" in (r[0] + r[1]).lower()
            ):
                start = i + 1
            continue
    return pts


class CenterlinePathNode(Node):
    def __init__(self):
        super().__init__("centerline_path")

        self.declare_parameter(
            "csv_path",
            "/home/jaewoong/sim_ws/src/my_control/config/Spielberg_raceline.csv",
        )
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("path_topic", "/recommended_path")
        self.declare_parameter("marker_topic", "/centerline_marker")
        self.declare_parameter("publish_hz", 2.0)
        self.declare_parameter("publish_marker", True)
        # Path 메시지가 너무 크면 DDS 제한으로 잘려 반만 그려짐 → 포인트 수 상한으로 맵 전체가 보이게
        self.declare_parameter("path_downsample", 1)  # 1=전부, 2=2개마다 1개, ...
        # 3948포인트 전체 전송 시 Path 메시지 ~400KB → DDS/커널 조각 한계(256KB 등)로 잘릴 수 있음. 2000 이하 권장.
        self.declare_parameter("path_max_poses", 2000)
        self.declare_parameter("waypoint_offset_x", 0.0)
        self.declare_parameter("waypoint_offset_y", 0.0)
        self.declare_parameter("waypoint_flip_y", False)
        self.declare_parameter("path_window_size", 50)  # >0이면 차량 위치 기준 앞쪽 N개만 발행(슬라이딩 윈도우)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "ego_racecar/base_link")

        csv_path = self.get_parameter("csv_path").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        marker_topic = (
            self.get_parameter("marker_topic").get_parameter_value().string_value
        )
        self.hz = float(self.get_parameter("publish_hz").value)
        self.publish_marker = bool(self.get_parameter("publish_marker").value)
        self.path_downsample = max(1, int(self.get_parameter("path_downsample").value))
        self.path_max_poses = max(10, int(self.get_parameter("path_max_poses").value))
        self.waypoint_offset_x = float(self.get_parameter("waypoint_offset_x").value)
        self.waypoint_offset_y = float(self.get_parameter("waypoint_offset_y").value)
        v = self.get_parameter("waypoint_flip_y").value
        self.waypoint_flip_y = v if isinstance(v, bool) else (str(v).lower() in ("true", "1"))
        self.path_window_size = max(0, int(self.get_parameter("path_window_size").value))
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.points = load_centerline_csv(csv_path)
        if len(self.points) < 2:
            raise RuntimeError(
                f"Centerline CSV must have at least 2 points, got {len(self.points)} from {csv_path}"
            )
        self._apply_offset()
        self.get_logger().info(
            f"Loaded {len(self.points)} centerline points from {csv_path}"
            + (f", path_window_size={self.path_window_size}" if self.path_window_size > 0 else "")
        )

        if self.path_window_size > 0:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        # rviz 기본 구독이 Volatile이라 동일하게 맞춤 (안 맞으면 경로 안 뜸)
        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.VOLATILE
        qos.reliability = QoSReliabilityPolicy.RELIABLE

        self.path_pub = self.create_publisher(Path, path_topic, qos)
        if self.publish_marker:
            self.marker_pub = self.create_publisher(Marker, marker_topic, qos)

        self.timer = self.create_timer(
            1.0 / max(self.hz, 0.2), self.publish_path
        )

    def _apply_offset(self) -> None:
        if self.waypoint_offset_x == 0.0 and self.waypoint_offset_y == 0.0 and not self.waypoint_flip_y:
            return
        new_pts: List[Tuple[float, float]] = []
        for x, y in self.points:
            x = x + self.waypoint_offset_x
            y = y + self.waypoint_offset_y
            if self.waypoint_flip_y:
                y = -y
            new_pts.append((x, y))
        self.points = new_pts
        self.get_logger().info(
            f"Applied offset=({self.waypoint_offset_x},{self.waypoint_offset_y}) flip_y={self.waypoint_flip_y}"
        )

    def publish_path(self):
        now = self.get_clock().now().to_msg()
        n = len(self.points)
        path_msg = Path()
        path_msg.header.frame_id = self.frame_id
        path_msg.header.stamp = now

        if self.path_window_size > 0 and hasattr(self, "tf_buffer"):
            try:
                t = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.base_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.15),
                )
            except TransformException:
                return
            mx = t.transform.translation.x
            my = t.transform.translation.y
            best_i = 0
            best_d2 = float("inf")
            for i in range(n):
                x, y = self.points[i]
                d2 = (x - mx) ** 2 + (y - my) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            window_size = min(self.path_window_size, n)
            for k in range(window_size):
                idx = (best_i + k) % n
                x, y = self.points[idx]
                pose = PoseStamped()
                pose.header.frame_id = self.frame_id
                pose.header.stamp = now
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
        else:
            if n <= self.path_max_poses:
                indices = list(range(0, n, self.path_downsample))
            else:
                indices = [
                    int(i * (n - 1) / (self.path_max_poses - 1))
                    for i in range(self.path_max_poses)
                ]
            for i in indices:
                x, y = self.points[i]
                pose = PoseStamped()
                pose.header.frame_id = self.frame_id
                pose.header.stamp = now
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

        # 시뮬에서 라인 확인용: 전체 경로를 점(POINTS)으로 발행
        if self.publish_marker:
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = "centerline"
            m.id = 0
            m.type = Marker.POINTS  # 선 대신 점으로 표시
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.08  # 점 크기 (POINTS에서 width)
            m.scale.y = 0.08  # 점 크기 (height)
            m.scale.z = 0.02
            m.color.a = 1.0
            m.color.r = 0.1
            m.color.g = 0.6
            m.color.b = 1.0
            # 전체 경로 표시 (전부 넣거나, 너무 많으면 균등 샘플)
            max_display = 5000
            if n <= max_display:
                marker_indices = list(range(0, n))
            else:
                marker_indices = [int(i * (n - 1) / (max_display - 1)) for i in range(max_display)]
            for i in marker_indices:
                x, y = self.points[i]
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                m.points.append(p)
            self.marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = CenterlinePathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
