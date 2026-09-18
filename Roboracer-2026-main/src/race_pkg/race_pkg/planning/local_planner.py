# 로컬 플래너 노드 

#!/usr/bin/env python3
"""
로컬 플래너: waypoint 기본 추종 + 장애물 근접 시에만 FGM으로 회피 후 추천경로 재합류.

- 장애물 없거나 멀면: 추천경로 그대로 발행 → waypoint 추종.
- 장애물이 avoid_threshold 안에 들어오면: [현재위치 → FGM타겟(회피)] + 추천경로 앞구간 발행
  → 딱 피한 뒤 다시 waypoint 따라가도록.

- /recommended_path 구독(센터라인패스가 위치 기준 200개 윈도우 발행) → /local_path로 그대로 또는 회피 경로로 발행.
"""
import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException


def _closest_obstacle_distance(
    obstacle_data: list,
    forward_cone_rad: float | None = None,
) -> float:
    """
    /static_obstacles 포맷: [id, x, y, radius, ...]. 가장 가까운 장애물까지 거리(m).
    forward_cone_rad가 주어지면 전방 콘 안의 장애물만 셈 (벽·측면 제외). laser 기준 x=전방, y=좌측.
    """
    if len(obstacle_data) < 4:
        return float("inf")
    n = len(obstacle_data) // 4
    best = float("inf")
    for i in range(n):
        x = obstacle_data[4 * i + 1]
        y = obstacle_data[4 * i + 2]
        r = obstacle_data[4 * i + 3]
        if forward_cone_rad is not None:
            if x <= 0.0:
                continue
            angle = math.atan2(y, x)
            if abs(angle) > forward_cone_rad:
                continue
        d = math.sqrt(x * x + y * y) - r
        if d < best:
            best = d
    return max(0.0, best) if best != float("inf") else float("inf")


def _point_laser_to_map(
    px: float, py: float,
    tx: float, ty: float, qw: float, qx: float, qy: float, qz: float,
) -> Tuple[float, float]:
    """laser 프레임 점 (px,py)를 map 프레임으로. tf는 map->laser (rotation R, translation t). p_map = R @ p_laser + t."""
    # quaternion to rotation matrix (2D만: z 회전)
    # w,x,y,z -> R_2x2
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    cx = math.cos(yaw)
    sx = math.sin(yaw)
    mx = cx * px - sx * py + tx
    my = sx * px + cx * py + ty
    return (mx, my)


class LocalPlannerNode(Node):
    def __init__(self):
        super().__init__("local_planner")

        self.declare_parameter("recommended_path_topic", "/recommended_path")
        self.declare_parameter("static_obstacles_topic", "/static_obstacles")
        self.declare_parameter("fgm_target_topic", "/fgm_target")
        self.declare_parameter("local_path_topic", "/local_path")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("avoid_threshold", 0.7)  # 이 거리(m) 안에 장애물 있으면 FGM 경로로 /local_path 발행 후 다시 추천경로
        self.declare_parameter("use_fgm", True)  # False면 FGM 미사용, 항상 추천경로만 발행(주행 확인용)
        self.declare_parameter("forward_cone_deg", 80.0)  # 전방 콘 각도(deg). FGM과 맞춰 넓히면 곡선 앞 장애물 인식 좋아짐
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("laser_frame", "ego_racecar/laser")
        self.declare_parameter("base_frame", "ego_racecar/base_link")
        rec_topic = self.get_parameter("recommended_path_topic").value
        obs_topic = self.get_parameter("static_obstacles_topic").value
        fgm_topic = self.get_parameter("fgm_target_topic").value
        out_topic = self.get_parameter("local_path_topic").value
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.avoid_threshold = float(self.get_parameter("avoid_threshold").value)
        _use_fgm = self.get_parameter("use_fgm").value
        self.use_fgm = _use_fgm if isinstance(_use_fgm, bool) else str(_use_fgm).lower() in ("1", "true", "yes")
        cone_deg = float(self.get_parameter("forward_cone_deg").value)
        self.forward_cone_rad = math.radians(cone_deg)
        self.map_frame = self.get_parameter("map_frame").value
        self.laser_frame = self.get_parameter("laser_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self._recommended_path: Path | None = None
        self._obstacle_data: list = []
        self._fgm_target: PointStamped | None = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub_rec = self.create_subscription(Path, rec_topic, self.cb_recommended_path, 10)
        self.sub_obs = self.create_subscription(
            Float32MultiArray, obs_topic, self.cb_static_obstacles, 10
        )
        self.sub_fgm = self.create_subscription(
            PointStamped, fgm_topic, self.cb_fgm_target, 10
        )
        self.pub_path = self.create_publisher(Path, out_topic, 10)

        self.timer = self.create_timer(1.0 / max(self.publish_hz, 1.0), self.timer_publish)
        self.get_logger().info(
            f"Local planner: rec={rec_topic}, obs={obs_topic}, fgm={fgm_topic} -> {out_topic}, "
            f"avoid_threshold={self.avoid_threshold}m, use_fgm={self.use_fgm}, forward_cone={cone_deg}deg"
        )

    def cb_recommended_path(self, msg: Path):
        self._recommended_path = msg

    def cb_static_obstacles(self, msg: Float32MultiArray):
        self._obstacle_data = list(msg.data)

    def cb_fgm_target(self, msg: PointStamped):
        self._fgm_target = msg

    def _path_from_points(self, pts: List[Tuple[float, float]]) -> Path:
        """(x,y) 리스트를 map frame Path 메시지로 변환."""
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in pts:
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.position.z = 0.0
            p.pose.orientation.w = 1.0
            path.poses.append(p)
        return path

    def _get_current_pose_map(self) -> PoseStamped | None:
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException:
            return None
        p = PoseStamped()
        p.header.frame_id = self.map_frame
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = t.transform.translation.x
        p.pose.position.y = t.transform.translation.y
        p.pose.position.z = t.transform.translation.z
        p.pose.orientation = t.transform.rotation
        return p

    def _get_fgm_target_in_map(self) -> Tuple[float, float] | None:
        if self._fgm_target is None:
            return None
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self._fgm_target.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException:
            return None
        px = self._fgm_target.point.x
        py = self._fgm_target.point.y
        tx = t.transform.translation.x
        ty = t.transform.translation.y
        q = t.transform.rotation
        mx, my = _point_laser_to_map(
            px, py,
            tx, ty, q.w, q.x, q.y, q.z,
        )
        return (mx, my)

    def timer_publish(self):
        if self._recommended_path is None or len(self._recommended_path.poses) < 2:
            return

        if not self.use_fgm:
            self.pub_path.publish(self._recommended_path)
            return

        dist = _closest_obstacle_distance(
            self._obstacle_data, forward_cone_rad=self.forward_cone_rad
        )

        if dist > self.avoid_threshold:
            self.pub_path.publish(self._recommended_path)
            return

        current = self._get_current_pose_map()
        fgm_xy = self._get_fgm_target_in_map()
        if current is None or fgm_xy is None:
            if not hasattr(self, "_last_avoid_warn_ns"):
                self._last_avoid_warn_ns = 0
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_avoid_warn_ns > 2_000_000_000:  # 2초마다
                self.get_logger().warn(
                    f"회피 시도했으나 사용 불가: dist={dist:.2f}m<=threshold. "
                    f"current={'None' if current is None else 'ok'}, fgm_target={'None' if fgm_xy is None else 'ok'}. TF 및 /fgm_target 확인."
                )
                self._last_avoid_warn_ns = now_ns
            self.pub_path.publish(self._recommended_path)
            return

        out = Path()
        out.header.frame_id = self.map_frame
        out.header.stamp = self.get_clock().now().to_msg()

        p0 = PoseStamped()
        p0.header = out.header
        p0.pose = current.pose
        out.poses.append(p0)

        p1 = PoseStamped()
        p1.header = out.header
        p1.pose.position.x = fgm_xy[0]
        p1.pose.position.y = fgm_xy[1]
        p1.pose.position.z = 0.0
        p1.pose.orientation.w = 1.0
        out.poses.append(p1)

        cur_x = current.pose.position.x
        cur_y = current.pose.position.y
        fgm_x, fgm_y = fgm_xy[0], fgm_xy[1]
        best_i = 0
        best_d2 = float("inf")
        for i, pose in enumerate(self._recommended_path.poses):
            x = pose.pose.position.x
            y = pose.pose.position.y
            d2 = (x - fgm_x) ** 2 + (y - fgm_y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        for i in range(best_i, len(self._recommended_path.poses)):
            out.poses.append(self._recommended_path.poses[i])
            if len(out.poses) >= 200:
                break

        if len(out.poses) >= 2:
            self.pub_path.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
