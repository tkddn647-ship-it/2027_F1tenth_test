"""
lidar_race_env.py
=================
Roboracer-2026 (실차 스택) 인터페이스에 맞춘 LiDAR 레이싱 시뮬.

실차 파이프라인과의 대응:
  /scan  (LaserScan)     ↔  occupancy 레이캐스트 LiDAR
  /drive (Ackermann)     ↔  action = [steering_norm, speed_norm]
  map yaml+png           ↔  Cartographer / f1tenth_racetracks 맵
  centerline.csv         ↔  채점용만 (관측 X) — Stanley가 쓰는 CSV와 동일 계열

mapless 관측: LiDAR + v + yaw_rate 만 (맵/센터라인 입력 X).
보상: 한 바퀴 목표를 위해 센터라인 진행은 privileged 보상으로만 사용.
센터라인 CSV는 관측이 아니라 랩 학습·채점용.
커넥톰 정책은 실차에서 stanley_waypoint_follow 대신 /drive 드롭인.
"""

from __future__ import annotations
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
ROBORACER_CFG = ROOT / "Roboracer-2026-main" / "src" / "path_following" / "config"

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


def _read_xy_csv(path: Path) -> np.ndarray | None:
    try:
        raw = np.genfromtxt(path, delimiter=",", comments="#")
        if raw.ndim == 1:
            return None
        xy = raw[:, :2].astype(np.float64)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 20:
            return None
        if len(xy) > 500:
            idx = np.linspace(0, len(xy) - 1, 500).astype(int)
            xy = xy[idx]
        return xy
    except Exception:
        return None


def load_centerline_for_scoring(map_yaml: Path) -> np.ndarray | None:
    """채점용 centerline. 맵 stem 전용 CSV → 맵 폴더 → 실차 config."""
    stem = map_yaml.stem
    preferred = [
        map_yaml.parent / f"{stem}_centerline.csv",
        map_yaml.parent / f"{stem}_raceline.csv",
        map_yaml.parent / "centerline.csv",
        ROBORACER_CFG / "centerline.csv",
        ROOT / "Roboracer-2026-main" / "src" / "race_pkg" / "config" / "centerline.csv",
    ]
    for c in preferred:
        if c.exists():
            xy = _read_xy_csv(c)
            if xy is not None:
                print(f"[lidar_race] scoring centerline: {c.name} ({len(xy)} pts)")
                return xy

    for folder in (map_yaml.parent, ROBORACER_CFG):
        if not folder.exists():
            continue
        for pat in ("*centerline*.csv", "*raceline*.csv"):
            for c in sorted(folder.glob(pat)):
                xy = _read_xy_csv(c)
                if xy is not None:
                    print(f"[lidar_race] scoring centerline: {c.name} ({len(xy)} pts)")
                    return xy
    return None


class LidarRaceEnv(_EnvBase):
    """실차 /scan + /drive 와 맞춘 Gym 환경."""

    N_RAYS = 20          # 실차 LaserScan 다운샘플
    RAY_RANGE = 8.0
    FOV = 4.71238898     # 270 deg (sllidar 계열과 유사)
    DT = 0.05            # 20 Hz (실차 stanley 주기와 비슷한 스케일)
    MAX_STEPS = 8000
    WHEELBASE = 0.33

    def __init__(
        self,
        map_name: str = "ajou",
        seed: int | None = None,
        min_speed_mps: float = MIN_SPEED_MPS,
        max_speed_mps: float = MAX_SPEED_MPS,
        max_steer_rad: float = MAX_STEER_RAD,
    ):
        self.map_yaml = resolve_map_yaml(map_name)
        self.map_name = self.map_yaml.stem
        self.occ, self.resolution, self.origin, self.meta, self.map_png = load_occupancy(self.map_yaml)
        self.h, self.w = self.occ.shape
        self.centerline = load_centerline_for_scoring(self.map_yaml)
        self.rng = np.random.default_rng(seed)
        self.MIN_SPEED = float(min_speed_mps)
        self.MAX_SPEED = float(max_speed_mps)
        if self.MAX_SPEED <= self.MIN_SPEED:
            raise ValueError("max_speed_mps must be > min_speed_mps")
        self.MAX_STEER = float(max_steer_rad)
        self._speed_span = self.MAX_SPEED - self.MIN_SPEED

        if _HAS_GYM:
            dim = self.N_RAYS + 2
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)
            # [steering_norm, speed_norm] → 실차 AckermannDrive.steering_angle / .speed
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.x = self.y = self.theta = self.v = 0.0
        self.yaw_rate = 0.0
        self.steps = 0
        self.cl_idx = 0
        self._progress_acc = 0
        self.lap_completed = False
        self.last_steer = 0.0
        self.last_speed_cmd = 0.0
        self._start_pose = self._find_start_pose()
        print(f"[lidar_race] map={self.map_yaml} size={self.w}x{self.h} res={self.resolution}")

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

    def _find_start_pose(self) -> np.ndarray:
        if self.centerline is not None:
            for i in range(0, len(self.centerline), max(1, len(self.centerline) // 40)):
                x, y = self.centerline[i]
                if not self.is_occupied(x, y, inflate=0.2):
                    nxt = self.centerline[(i + 5) % len(self.centerline)]
                    th = float(np.arctan2(nxt[1] - y, nxt[0] - x))
                    return np.array([x, y, th], dtype=np.float64)
        free = np.argwhere(~self.occ)
        self.rng.shuffle(free)
        for row, col in free[:8000]:
            x, y = self.pixel_to_world(int(row), int(col))
            if self.is_occupied(x, y, 0.25):
                continue
            for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                if not self.is_occupied(x + 0.8 * np.cos(th), y + 0.8 * np.sin(th), 0.12):
                    return np.array([x, y, th], dtype=np.float64)
        raise RuntimeError("시작 포즈를 찾지 못함 — 맵 free 공간 확인")

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

    def _get_obs(self) -> np.ndarray:
        d = self._cast_lidar() / self.RAY_RANGE
        # 속도: [MIN, MAX] → [0, 1]
        v_norm = (self.v - self.MIN_SPEED) / self._speed_span
        return np.concatenate([
            d,
            [float(np.clip(v_norm, 0, 1))],
            [np.clip(self.yaw_rate / 3.0, -1, 1)],
        ]).astype(np.float32)

    def _nearest_cl(self, pos: np.ndarray) -> int:
        if self.centerline is None:
            return 0
        n = len(self.centerline)
        cand = (self.cl_idx + np.arange(-5, 25)) % n
        d = np.linalg.norm(self.centerline[cand] - pos, axis=1)
        return int(cand[np.argmin(d)])

    def action_to_ackermann(self, action) -> tuple[float, float]:
        """정규화 행동 → (steering_angle_rad, speed_mps) = 실차 /drive.

        action[1] ∈ [0,1] → speed ∈ [MIN_SPEED, MAX_SPEED] (기본 2~7 m/s)
        """
        steer = float(np.clip(action[0], -1, 1)) * self.MAX_STEER
        u = float(np.clip(action[1], 0, 1))
        speed = self.MIN_SPEED + u * self._speed_span
        return steer, speed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._start_pose = self._find_start_pose()
        pose = self._start_pose.copy()
        if options and "pose" in options:
            pose = np.asarray(options["pose"], dtype=np.float64)
        self.x, self.y, self.theta = map(float, pose)
        self.v = self.MIN_SPEED  # 정지 없음
        self.yaw_rate = 0.0
        self.steps = 0
        self.lap_completed = False
        self._progress_acc = 0
        self.last_steer = 0.0
        self.last_speed_cmd = self.MIN_SPEED
        self.cl_idx = self._nearest_cl(np.array([self.x, self.y])) if self.centerline is not None else 0
        obs = self._get_obs()
        info = {"map": self.map_name, "map_yaml": str(self.map_yaml)}
        return (obs, info) if _HAS_GYM else obs

    def step(self, action):
        steer, speed_cmd = self.action_to_ackermann(action)
        self.last_steer, self.last_speed_cmd = steer, speed_cmd
        prev_th = self.theta

        # bicycle: R = L/tan(δ). (이전 0.5*tan 모델은 회전반경이 약 2배 → 코너 불가)
        self.v = float(np.clip(
            self.v + 0.35 * (speed_cmd - self.v),
            self.MIN_SPEED,
            self.MAX_SPEED,
        ))
        steer_c = float(np.clip(steer, -self.MAX_STEER, self.MAX_STEER))
        self.x += self.v * np.cos(self.theta) * self.DT
        self.y += self.v * np.sin(self.theta) * self.DT
        self.theta += (self.v / self.WHEELBASE) * np.tan(steer_c) * self.DT
        self.yaw_rate = (self.theta - prev_th) / self.DT
        self.steps += 1

        collided = self.is_occupied(self.x, self.y, inflate=0.14)

        # 진행도(센터라인): 관측에는 안 넣음. 한 바퀴 학습용 privileged 보상만.
        delta = 0
        if self.centerline is not None and not collided:
            new_idx = self._nearest_cl(np.array([self.x, self.y]))
            delta = new_idx - self.cl_idx
            n = len(self.centerline)
            if delta < -n / 2:
                delta += n
            elif delta > n / 2:
                delta -= n
            self.cl_idx = new_idx
            self._progress_acc += max(delta, 0)
            if self._progress_acc >= n - 3:
                self.lap_completed = True

        scans = self._cast_lidar() / self.RAY_RANGE
        mid = float(scans[7:13].mean())
        front = float(scans[9:11].mean())
        v_norm = (self.v - self.MIN_SPEED) / self._speed_span
        # 관측 = LiDAR only. 보상 = 진행(랩) + 속도 + 여유
        reward = (
            2.5 * float(max(delta, 0))             # 앞으로 나가라 (한 바퀴 핵심)
            + 1.0 * v_norm                         # 느린 고착(2m/s) 방지
            + 0.6 * mid
            + 0.4 * min(front, 0.5)
        )
        if front > 0.35 and v_norm < 0.15:
            reward -= 0.4                          # 앞 뚫렸는데 최저속이면 감점
        if collided:
            reward -= 30.0
        if front < 0.12:
            reward -= 0.6
        if self.lap_completed:
            reward += max(0.0, 200.0 - self.steps * self.DT)

        terminated = collided or self.lap_completed
        truncated = self.steps >= self.MAX_STEPS
        obs = self._get_obs()
        info = {
            "collided": collided,
            "lap_completed": self.lap_completed,
            "lap_time": self.steps * self.DT if self.lap_completed else None,
            "track_name": self.map_name,
            "progress": (
                self._progress_acc / max(len(self.centerline), 1)
                if self.centerline is not None else 0.0
            ),
            "pos": np.array([self.x, self.y], dtype=np.float32),
            "theta": self.theta,
            "steering_angle": self.last_steer,
            "speed": self.last_speed_cmd,
            "reward_mode": "mapless_obs_lap_reward",
        }
        return (obs, reward, terminated, truncated, info) if _HAS_GYM else (obs, reward, terminated or truncated, info)


if __name__ == "__main__":
    env = LidarRaceEnv("ajou", seed=0)
    obs, info = env.reset()
    print(f"map={info['map']} start=({env.x:.2f},{env.y:.2f}) lidar={obs[:5]}")
    for t in range(500):
        rays = obs[: env.N_RAYS]
        left, mid, right = rays[:7].mean(), rays[7:13].mean(), rays[13:].mean()
        steer = float(np.clip((right - left) * 2.0, -1, 1))
        speed_u = 0.4 if mid > 0.35 else 0.0
        obs, r, term, trunc, info = env.step(np.array([steer, speed_u]))
        if t == 0:
            print(f"speed_cmd={info['speed']:.2f} (must be in [2,7])")
        if term or trunc:
            print(f"end t={t} crash={info['collided']} lap={info['lap_completed']}")
            break
    else:
        print("ok 500 steps")
