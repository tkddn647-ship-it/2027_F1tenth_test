"""
lidar_race_env.py
=================
Roboracer-2026 (실차 스택) 인터페이스에 맞춘 LiDAR 레이싱 시뮬.

실차 파이프라인과의 대응:
  /scan  (LaserScan)     ↔  occupancy 레이캐스트 LiDAR
  /drive (Ackermann)     ↔  action = [steering_norm, speed_norm]
  map yaml+png           ↔  시뮬 occupancy만 (에이전트 입력 아님)

mapless ONLY:
  관측: LiDAR 히스토리(40Hz→5Hz 스택) + v + yaw_rate
  보상: 최장 LiDAR 방향 정렬 × 이동 × 속도  (맵/센터라인 금지)
  스폰: 하단 가로 직선 구간
  속도: 2~7 m/s (fix_speed=False)
"""

from __future__ import annotations
from collections import deque
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False
    gym = None
    spaces = None

_EnvBase = gym.Env if _HAS_GYM else object

ROOT = Path(__file__).resolve().parent
MAPS_DIR = ROOT / "maps"
RACETRACKS_DIR = ROOT / "f1tenth_racetracks"
ROBORACER_MAPS = ROOT / "Roboracer-2026-main" / "maps"

# 실차 stanley CFG와 맞춤 (대략)
MAX_STEER_RAD = 0.3735   # 실측 전륜각 ±21.4°
MIN_SPEED_MPS = 2.0      # 최저 속도 (정지 금지)
MAX_SPEED_MPS = 7.0      # 최고 속도


def resolve_map_yaml(map_name: str) -> Path:
    """맵 이름 또는 yaml 경로 → yaml Path.

    우선순위:
      1) 절대/상대 yaml 경로
      2) Roboracer-2026-main/maps/<name>.yaml  (실차 Cartographer)
      3) maps/<name>.yaml
      4) f1tenth_racetracks/<Name>/<Name>_map.yaml
    """
    p = Path(map_name)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p.resolve()

    # 짧은 별칭
    aliases = {
        "ajou": "cartographer_map_20260817_003202",
        "ajou_latest": "cartographer_map_20260817_003202",
        "ajou_prev": "cartographer_map_20260817_002900",
    }
    key = aliases.get(map_name, map_name)

    for base in (ROBORACER_MAPS, MAPS_DIR):
        cand = base / f"{key}.yaml"
        if cand.exists():
            return cand

    race = RACETRACKS_DIR / key / f"{key}_map.yaml"
    if race.exists():
        return race

    if RACETRACKS_DIR.exists():
        for d in RACETRACKS_DIR.iterdir():
            if d.is_dir() and d.name.replace(" ", "") == key.replace(" ", ""):
                y = next(d.glob("*_map.yaml"), None)
                if y:
                    return y

    # Roboracer maps glob partial
    if ROBORACER_MAPS.exists():
        hits = list(ROBORACER_MAPS.glob(f"*{key}*.yaml"))
        hits = [h for h in hits if "origin" not in h.name and h.suffix == ".yaml"]
        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"맵 yaml 없음: {map_name}\n"
        f"  예: ajou | cartographer_map_20260817_003202 | Spielberg | vegas\n"
        f"  또는 yaml 전체 경로"
    )


def load_occupancy(map_yaml: Path):
    meta = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
    img_path = map_yaml.parent / meta["image"]
    if not img_path.exists():
        stem = Path(meta["image"]).stem
        for ext in (".png", ".pgm", ".jpg"):
            alt = map_yaml.parent / f"{stem}{ext}"
            if alt.exists():
                img_path = alt
                break
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    # ROS map: 밝음=free, 어두움=occupied. trinary 회색(unknown)은 벽.
    free_thresh = float(meta.get("free_thresh", 0.196))
    # occ_prob 낮을수록 free. free_thresh=0.196 → img/255 > 1-0.196=0.804 → img>205
    occupied = (img / 255.0) < (1.0 - free_thresh)
    return occupied, float(meta["resolution"]), meta["origin"], meta, img_path


class LidarRaceEnv(_EnvBase):
    """실차 /scan + /drive 와 맞춘 Gym 환경."""

    N_RAYS = 20          # 실차 LaserScan 다운샘플
    RAY_RANGE = 8.0
    FOV = 4.71238898     # 270 deg (sllidar 계열과 유사)
    DT = 0.025           # 40 Hz — 실차 LiDAR 주기와 맞춤
    MAX_STEPS = 16000    # ≈ 400 s (이전 20Hz×8000과 동일 시간)
    WHEELBASE = 0.33
    # 과거 LiDAR: 40Hz 스트림에서 5Hz로 서브샘플 → 움직임 추정
    LIDAR_HZ = 40
    HIST_FPS = 5
    HIST_LEN = 4         # 0.6 s lookback

    def __init__(
        self,
        map_name: str = "ajou",
        seed: int | None = None,
        min_speed_mps: float = MIN_SPEED_MPS,
        max_speed_mps: float = MAX_SPEED_MPS,
        max_steer_rad: float = MAX_STEER_RAD,
        fix_speed: bool = False,
    ):
        self.map_yaml = resolve_map_yaml(map_name)
        self.map_name = self.map_yaml.stem
        self.occ, self.resolution, self.origin, self.meta, self.map_png = load_occupancy(self.map_yaml)
        self.h, self.w = self.occ.shape
        self.centerline = None  # mapless: 센터라인 미로드·미사용
        self.rng = np.random.default_rng(seed)
        self.MIN_SPEED = float(min_speed_mps)
        self.MAX_SPEED = float(max_speed_mps)
        if self.MAX_SPEED <= self.MIN_SPEED:
            raise ValueError("max_speed_mps must be > min_speed_mps")
        self.MAX_STEER = float(max_steer_rad)
        self.fix_speed = bool(fix_speed)
        self._speed_span = self.MAX_SPEED - self.MIN_SPEED
        self._hist_interval = max(1, int(round(self.LIDAR_HZ / self.HIST_FPS)))
        self._lidar_hist: deque[np.ndarray] = deque(maxlen=self.HIST_LEN)
        self._obs_dim = self.N_RAYS * self.HIST_LEN + 2

        if _HAS_GYM:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.x = self.y = self.theta = self.v = 0.0
        self.yaw_rate = 0.0
        self.steps = 0
        self._dist_acc = 0.0
        self.lap_completed = False
        self.last_steer = 0.0
        self.last_speed_cmd = 0.0
        self._steer_filt = 0.0
        self.spawn_mode = "straight"  # straight: 고정 직선 스폰 / random: 매 에피소드 재샘플
        self._start_pose = self._sample_start_pose()
        print(
            f"[lidar_race] MAPLESS map={self.map_yaml.name} "
            f"{self.w}x{self.h} res={self.resolution} "
            f"obs={self._obs_dim} lidar={self.LIDAR_HZ}Hz hist={self.HIST_LEN}@{self.HIST_FPS}Hz "
            f"fix_speed={self.fix_speed}"
        )

    def set_spawn_mode(self, mode: str) -> None:
        if mode not in ("straight", "random"):
            raise ValueError(f"spawn_mode must be straight|random, got {mode}")
        if mode != self.spawn_mode:
            print(f"[lidar_race] spawn_mode: {self.spawn_mode} -> {mode}")
        self.spawn_mode = mode

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        ox, oy = float(self.origin[0]), float(self.origin[1])
        col = int((x - ox) / self.resolution)
        row = int(self.h - (y - oy) / self.resolution)
        return row, col

    def pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        ox, oy = float(self.origin[0]), float(self.origin[1])
        x = ox + (col + 0.5) * self.resolution
        y = oy + (self.h - row - 0.5) * self.resolution
        return x, y

    def is_occupied(self, x: float, y: float, inflate: float = 0.12) -> bool:
        for dx, dy in ((0.0, 0.0), (inflate, 0), (-inflate, 0), (0, inflate), (0, -inflate)):
            r, c = self.world_to_pixel(x + dx, y + dy)
            if r < 0 or c < 0 or r >= self.h or c >= self.w:
                return True
            if self.occ[r, c]:
                return True
        return False

    def _ray_clear(self, x: float, y: float, th: float, max_dist: float = 6.0) -> float:
        ca, sa = np.cos(th), np.sin(th)
        clear = 0.0
        for dist in np.linspace(0.15, max_dist, 40):
            if self.is_occupied(x + dist * ca, y + dist * sa, 0.10):
                return dist
            clear = dist
        return clear

    def _sample_start_pose(self) -> np.ndarray:
        """하단 긴 가로 직선 구간, 차로 중앙, 직선 방향 헤딩."""
        free = np.argwhere(~self.occ)
        if len(free) == 0:
            raise RuntimeError("free 셀 없음")
        oy = float(self.origin[1])
        map_h_m = self.h * self.resolution
        # 맵 이미지 하단 = 낮은 y (유저가 가리킨 긴 가로 직선)
        y_hi = oy + 0.38 * map_h_m
        # 가로 직선만: 0° / 180°
        headings = (0.0, np.pi)
        best = None
        best_score = -1.0
        self.rng.shuffle(free)
        for row, col in free[:15000]:
            x, y = self.pixel_to_world(int(row), int(col))
            if y > y_hi:
                continue
            if self.is_occupied(x, y, 0.18):
                continue
            for th in headings:
                fwd = self._ray_clear(x, y, th, 8.0)
                if fwd < 4.0:
                    continue
                left = self._ray_clear(x, y, th + np.pi / 2, 1.5)
                right = self._ray_clear(x, y, th - np.pi / 2, 1.5)
                width = left + right
                if width < 0.50 or width > 1.60:
                    continue
                center = 1.0 - abs(left - right) / max(width, 1e-3)
                if center < 0.55:
                    continue
                # 가장 긴 직선 + 중앙
                score = fwd * 3.0 + center * 2.0
                if score > best_score:
                    best_score = score
                    mid_off = 0.5 * (right - left)
                    nx = x + mid_off * np.cos(th - np.pi / 2)
                    ny = y + mid_off * np.sin(th - np.pi / 2)
                    if self.is_occupied(nx, ny, 0.16):
                        nx, ny = x, y
                    best = np.array([nx, ny, float(th)], dtype=np.float64)
            if best_score >= 20.0:
                break
        if best is None:
            raise RuntimeError("하단 가로 직선 스폰을 찾지 못함")
        print(
            f"[lidar_race] bottom-straight spawn "
            f"({best[0]:.2f},{best[1]:.2f}) th={np.degrees(best[2]):.0f}deg "
            f"score={best_score:.1f}"
        )
        return best

    def _find_start_pose(self) -> np.ndarray:
        if self.spawn_mode == "straight" and self._start_pose is not None:
            return self._start_pose.copy()
        return self._sample_start_pose()

    def _cast_lidar(self) -> np.ndarray:
        half = self.FOV / 2.0
        angles = self.theta + np.linspace(-half, half, self.N_RAYS)
        dists = np.full(self.N_RAYS, self.RAY_RANGE, dtype=np.float32)
        for i, a in enumerate(angles):
            ca, sa = np.cos(a), np.sin(a)
            for t in np.linspace(0.05, self.RAY_RANGE, 60):
                if self.is_occupied(self.x + t * ca, self.y + t * sa, inflate=0.0):
                    dists[i] = float(t)
                    break
        return dists

    def _push_lidar_hist(self, d_norm: np.ndarray, force: bool = False) -> None:
        if force or (self.steps % self._hist_interval == 0) or len(self._lidar_hist) == 0:
            self._lidar_hist.append(np.asarray(d_norm, dtype=np.float32).copy())
            while len(self._lidar_hist) < self.HIST_LEN:
                self._lidar_hist.appendleft(self._lidar_hist[0].copy())

    def _get_obs(self) -> np.ndarray:
        d = self._cast_lidar() / self.RAY_RANGE
        self._push_lidar_hist(d)
        hist = np.concatenate(list(self._lidar_hist), axis=0)
        v_norm = (self.v - self.MIN_SPEED) / max(self._speed_span, 1e-6)
        return np.concatenate([
            hist,
            [float(np.clip(v_norm, 0, 1))],
            [np.clip(self.yaw_rate / 3.0, -1, 1)],
        ]).astype(np.float32)

    def action_to_ackermann(self, action) -> tuple[float, float]:
        steer = float(np.clip(action[0], -1, 1)) * self.MAX_STEER
        u = float(np.clip(action[1], 0, 1))
        speed = self.MIN_SPEED + u * self._speed_span
        return steer, speed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "pose" in options:
            pose = np.asarray(options["pose"], dtype=np.float64)
        else:
            self._start_pose = self._find_start_pose()
            pose = self._start_pose.copy()
        self.x, self.y, self.theta = map(float, pose)
        self.v = self.MIN_SPEED
        self.yaw_rate = 0.0
        self.steps = 0
        self._dist_acc = 0.0
        self.last_steer = 0.0
        self.last_speed_cmd = self.MIN_SPEED
        self._steer_filt = 0.0
        self._lidar_hist.clear()
        obs = self._get_obs()
        info = {"map": self.map_name, "map_yaml": str(self.map_yaml), "reward_mode": "mapless"}
        return (obs, info) if _HAS_GYM else obs

    def step(self, action):
        prev_x, prev_y = self.x, self.y
        steer, speed_cmd = self.action_to_ackermann(action)
        if self.fix_speed:
            speed_cmd = self.MIN_SPEED
        self._steer_filt = 0.7 * self._steer_filt + 0.3 * steer
        steer = self._steer_filt
        self.last_steer, self.last_speed_cmd = steer, speed_cmd
        prev_th = self.theta

        self.v = float(np.clip(
            self.v + 0.4 * (speed_cmd - self.v),
            self.MIN_SPEED,
            self.MAX_SPEED,
        ))
        steer_c = float(np.clip(steer, -self.MAX_STEER, self.MAX_STEER))
        self.x += self.v * np.cos(self.theta) * self.DT
        self.y += self.v * np.sin(self.theta) * self.DT
        self.theta += (self.v / self.WHEELBASE) * np.tan(steer_c) * self.DT
        self.yaw_rate = (self.theta - prev_th) / self.DT
        self.steps += 1

        collided = self.is_occupied(self.x, self.y, inflate=0.10)
        moved = float(np.hypot(self.x - prev_x, self.y - prev_y))
        if not collided:
            self._dist_acc += moved

        scans = self._cast_lidar() / self.RAY_RANGE
        # 최장 LiDAR 빔 방향 = 가야 할 쪽
        half = self.FOV / 2.0
        rels = np.linspace(-half, half, self.N_RAYS)
        i_max = int(np.argmax(scans))
        d_max = float(scans[i_max])
        align = float(np.cos(rels[i_max]))  # 1이면 최장빔이 정면

        # ===== MAPLESS: 최장 LiDAR 쪽 이동 + 열리면 가속 / 막히면 감속 =====
        reward = 10.0 * moved * max(align, 0.0) * (0.25 + 0.75 * d_max)
        v_frac = self.v / self.MAX_SPEED
        if d_max > 0.45 and align > 0.5:
            reward += 3.0 * v_frac          # 앞 뚫리면 빠르게
        else:
            reward -= 4.0 * v_frac          # 막히면 고속 페널티 → 감속 학습
        if d_max < 0.12:
            reward -= 2.0
        if collided:
            reward -= 12.0

        terminated = collided
        truncated = self.steps >= self.MAX_STEPS
        obs = self._get_obs()
        info = {
            "collided": collided,
            "lap_completed": False,
            "lap_time": None,
            "track_name": self.map_name,
            "progress": 0.0,
            "dist_m": self._dist_acc,
            "pos": np.array([self.x, self.y], dtype=np.float32),
            "theta": self.theta,
            "steering_angle": self.last_steer,
            "speed": self.last_speed_cmd,
            "reward_mode": "mapless_max_lidar",
        }
        return (obs, reward, terminated, truncated, info) if _HAS_GYM else (obs, reward, terminated or truncated, info)


if __name__ == "__main__":
    env = LidarRaceEnv("ajou", seed=0)
    obs, info = env.reset()
    print(f"map={info['map']} start=({env.x:.2f},{env.y:.2f}) mode={info.get('reward_mode')}")
    for t in range(800):
        off = env.N_RAYS * (env.HIST_LEN - 1)
        rays = obs[off: off + env.N_RAYS]
        left, mid, right = rays[:7].mean(), rays[7:13].mean(), rays[13:].mean()
        steer = float(np.clip((right - left) * 2.0, -1, 1))
        obs, r, term, trunc, info = env.step(np.array([steer, 0.0]))
        if term or trunc:
            print(f"end t={t} crash={info['collided']} dist={info['dist_m']:.1f}m")
            break
    else:
        print(f"ok dist={info['dist_m']:.1f}m")
