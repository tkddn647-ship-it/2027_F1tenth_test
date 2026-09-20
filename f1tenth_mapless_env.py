"""
f1tenth_mapless_env.py
======================
F1TENTH mapless Gymnasium env (관측 mapless / 보상 privileged).

실차 정렬:
  LiDAR range 40 m, 측정 주기 40 Hz → DT=0.025
  제어는 frame_skip=4 → 약 10 Hz (히스토리도 이 간격)

관측 (dim=680): LiDAR 135×5 + yaw×5
행동: [steer_norm, speed_norm]
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
RACETRACKS_DIR = ROOT / "f1tenth_racetracks"

N_BEAMS = 135
HIST_LEN = 5
OBS_DIM = N_BEAMS * HIST_LEN + HIST_LEN  # 680
MIN_SPEED = 2.0
MAX_SPEED = 7.0
MAX_STEER = 0.4189
WHEELBASE = 0.33
DT = 0.025          # 40 Hz (실차 LiDAR 주기)
FRAME_SKIP = 4      # 제어 ~10 Hz; hist span ≈ 0.4 s
RAY_RANGE = 40.0    # 실차 40 m
FOV = 4.71238898    # 270 deg


def _resolve_map_yaml(map_name: str) -> Path:
    p = Path(map_name)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p.resolve()
    race = RACETRACKS_DIR / map_name / f"{map_name}_map.yaml"
    if race.exists():
        return race
    maps = ROOT / "maps" / f"{map_name}.yaml"
    if maps.exists():
        return maps
    raise FileNotFoundError(f"맵 yaml 없음: {map_name}")


def _load_occupancy(map_yaml: Path):
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
    free_thresh = float(meta.get("free_thresh", 0.196))
    occupied = (img / 255.0) < (1.0 - free_thresh)
    return occupied, float(meta["resolution"]), meta["origin"], meta


def _load_centerline(map_yaml: Path) -> np.ndarray | None:
    folder = map_yaml.parent
    for pat in ("*centerline*.csv", "*raceline*.csv"):
        hits = sorted(folder.glob(pat))
        if not hits:
            continue
        raw = np.genfromtxt(hits[0], delimiter=",", comments="#")
        if raw.ndim == 1 or raw.shape[0] < 20:
            continue
        xy = raw[:, :2].astype(np.float64)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) > 600:
            idx = np.linspace(0, len(xy) - 1, 600).astype(int)
            xy = xy[idx]
        return xy
    return None


class F1TenthMaplessEnv(_EnvBase):
    """Mapless obs + privileged progress reward."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str = "Spielberg",
        seed: int | None = None,
        min_speed: float = MIN_SPEED,
        max_speed: float = MAX_SPEED,
        max_steer: float = MAX_STEER,
        max_steps: int = 3000,
        frame_skip: int = FRAME_SKIP,
    ):
        self.map_yaml = _resolve_map_yaml(map_name)
        self.map_name = self.map_yaml.stem.replace("_map", "")
        self.occ, self.resolution, self.origin, self.meta = _load_occupancy(self.map_yaml)
        self.h, self.w = self.occ.shape
        self.centerline = _load_centerline(self.map_yaml)
        self.cl_s = None
        if self.centerline is not None and len(self.centerline) > 1:
            seg = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
            self.cl_s = np.concatenate([[0.0], np.cumsum(seg)])
            self.cl_loop = float(
                self.cl_s[-1]
                + np.linalg.norm(self.centerline[0] - self.centerline[-1])
            )
        else:
            self.cl_loop = 1.0
        self.rng = np.random.default_rng(seed)

        self.MIN_SPEED = float(min_speed)
        self.MAX_SPEED = float(max_speed)
        self.MAX_STEER = float(max_steer)
        self._speed_span = self.MAX_SPEED - self.MIN_SPEED
        self.MAX_STEPS = int(max_steps)
        self.frame_skip = int(max(1, frame_skip))
        self.dt_ctrl = DT * self.frame_skip  # ~0.1 s

        self._lidar_hist: deque[np.ndarray] = deque(maxlen=HIST_LEN)
        self._yaw_hist: deque[float] = deque(maxlen=HIST_LEN)
        # resolution-step ray marching
        self._ray_step = max(float(self.resolution), 0.05)

        if _HAS_GYM:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.x = self.y = self.theta = self.v = 0.0
        self.yaw_rate = 0.0
        self.steps = 0
        self.phys_steps = 0
        self.cl_idx = 0
        self._s_travel = 0.0
        self.lap_completed = False
        self.last_steer = 0.0
        self.last_speed_cmd = self.MIN_SPEED
        self._start_pool = self._build_start_pool()
        self._start_pose = self._find_start_pose()
        print(
            f"[f1tenth_mapless] map={self.map_name} "
            f"obs={OBS_DIM} beams={N_BEAMS}x{HIST_LEN} "
            f"lidar={RAY_RANGE}m@{1.0/DT:.0f}Hz ctrl~{1.0/self.dt_ctrl:.0f}Hz "
            f"skip={self.frame_skip} speed=[{self.MIN_SPEED},{self.MAX_SPEED}] "
            f"cl_loop={self.cl_loop:.1f}m starts={len(self._start_pool)}"
        )

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

    def _ray_clear(self, x: float, y: float, th: float, max_d: float = 12.0, inflate: float = 0.15) -> float:
        ca, sa = np.cos(th), np.sin(th)
        d = 0.2
        while d <= max_d:
            if self.is_occupied(x + d * ca, y + d * sa, inflate):
                return float(d)
            d += self._ray_step
        return float(max_d)

    def _side_clear(self, x: float, y: float, th: float) -> tuple[float, float]:
        left = self._ray_clear(x, y, th + np.pi / 2, max_d=4.0, inflate=0.05)
        right = self._ray_clear(x, y, th - np.pi / 2, max_d=4.0, inflate=0.05)
        return left, right

    def _lane_center(self, x: float, y: float, th: float) -> tuple[float, float]:
        left, right = self._side_clear(x, y, th)
        lateral = 0.5 * (right - left)
        nx, ny = np.cos(th - np.pi / 2), np.sin(th - np.pi / 2)
        return float(x + lateral * nx), float(y + lateral * ny)

    def _build_start_pool(self) -> list[np.ndarray]:
        """CL 전체에 균일 샘플 (직선만 고르지 않음)."""
        pool: list[np.ndarray] = []
        if self.centerline is None or len(self.centerline) < 20:
            return pool
        n = len(self.centerline)
        step = max(1, n // 120)
        for i in range(0, n, step):
            p = self.centerline[i]
            nxt = self.centerline[(i + 5) % n]
            th = float(np.arctan2(nxt[1] - p[1], nxt[0] - p[0]))
            x0, y0 = float(p[0]), float(p[1])
            if self.is_occupied(x0, y0, 0.22):
                continue
            # 최소 전방 여유만 (코너 포함)
            if self._ray_clear(x0, y0, th, max_d=3.0, inflate=0.12) < 1.2:
                continue
            x, y = self._lane_center(x0, y0, th)
            if self.is_occupied(x, y, 0.22):
                continue
            pool.append(np.array([x, y, th], dtype=np.float64))
        return pool

    def _find_start_pose(self) -> np.ndarray:
        if self._start_pool:
            base = self._start_pool[int(self.rng.integers(0, len(self._start_pool)))].copy()
            x, y, th = float(base[0]), float(base[1]), float(base[2])
            # 헤딩·횡방향 노이즈 → 코너/오프셋 데이터
            th = th + float(self.rng.uniform(-0.12, 0.12))
            lat = float(self.rng.uniform(-0.15, 0.15))
            nx, ny = np.cos(th - np.pi / 2), np.sin(th - np.pi / 2)
            x2, y2 = x + lat * nx, y + lat * ny
            if not self.is_occupied(x2, y2, 0.2):
                x, y = x2, y2
            return np.array([x, y, th], dtype=np.float64)

        free = np.argwhere(~self.occ)
        self.rng.shuffle(free)
        for row, col in free[:4000]:
            x, y = self.pixel_to_world(int(row), int(col))
            if self.is_occupied(x, y, 0.2):
                continue
            for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                if self._ray_clear(x, y, float(th), max_d=3.0) >= 2.0:
                    return np.array([x, y, float(th)], dtype=np.float64)
        raise RuntimeError("start pose not found")

    def _cast_lidar(self) -> np.ndarray:
        half = FOV / 2.0
        angles = self.theta + np.linspace(-half, half, N_BEAMS)
        dists = np.full(N_BEAMS, RAY_RANGE, dtype=np.float32)
        step = self._ray_step
        for i, a in enumerate(angles):
            ca, sa = np.cos(a), np.sin(a)
            t = step
            while t <= RAY_RANGE:
                if self.is_occupied(self.x + t * ca, self.y + t * sa, inflate=0.0):
                    dists[i] = float(t)
                    break
                t += step
        return dists

    def _nearest_cl(self, pos: np.ndarray) -> int:
        if self.centerline is None:
            return 0
        n = len(self.centerline)
        cand = (self.cl_idx + np.arange(-3, 50)) % n
        d = np.linalg.norm(self.centerline[cand] - pos, axis=1)
        return int(cand[int(np.argmin(d))])

    def _cl_tangent(self, idx: int) -> np.ndarray:
        n = len(self.centerline)
        a = self.centerline[idx]
        b = self.centerline[(idx + 3) % n]
        t = b - a
        nrm = np.linalg.norm(t)
        if nrm < 1e-6:
            return np.array([1.0, 0.0])
        return t / nrm

    def _arc_ds(self, i0: int, i1: int) -> float:
        if self.cl_s is None:
            return 0.0
        s0 = float(self.cl_s[i0])
        s1 = float(self.cl_s[i1])
        ds = s1 - s0
        if ds < -0.5 * self.cl_loop:
            ds += self.cl_loop
        elif ds > 0.5 * self.cl_loop:
            ds -= self.cl_loop
        return float(ds)

    def _get_obs(self) -> np.ndarray:
        d = self._cast_lidar() / RAY_RANGE
        self._lidar_hist.append(d.astype(np.float32))
        while len(self._lidar_hist) < HIST_LEN:
            self._lidar_hist.appendleft(self._lidar_hist[0].copy())
        yaw_n = float(np.clip(self.yaw_rate / 3.0, -1.0, 1.0))
        self._yaw_hist.append(yaw_n)
        while len(self._yaw_hist) < HIST_LEN:
            self._yaw_hist.appendleft(self._yaw_hist[0])
        lidar = np.concatenate(list(self._lidar_hist), axis=0)
        yaw = np.asarray(list(self._yaw_hist), dtype=np.float32)
        return np.concatenate([lidar, yaw]).astype(np.float32)

    def action_to_controls(self, action) -> tuple[float, float]:
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
            pose = self._find_start_pose()
            self._start_pose = pose.copy()
        self.x, self.y, self.theta = map(float, pose)
        self.v = self.MIN_SPEED
        self.yaw_rate = 0.0
        self.steps = 0
        self.phys_steps = 0
        self._s_travel = 0.0
        self.lap_completed = False
        self.last_steer = 0.0
        self.last_speed_cmd = self.MIN_SPEED
        self.cl_idx = self._nearest_cl(np.array([self.x, self.y]))
        self._lidar_hist.clear()
        self._yaw_hist.clear()
        obs = self._get_obs()
        info = {
            "map": self.map_name,
            "obs_dim": OBS_DIM,
            "dt": DT,
            "frame_skip": self.frame_skip,
            "ray_range": RAY_RANGE,
        }
        return (obs, info) if _HAS_GYM else obs

    def _physics_once(self, steer: float, speed_cmd: float) -> tuple[float, float, float, bool]:
        """One 40 Hz tick. Returns (ds, cte, heading_cos, collided)."""
        prev_th = self.theta
        self.v = float(np.clip(
            self.v + 0.5 * (speed_cmd - self.v),
            self.MIN_SPEED,
            self.MAX_SPEED,
        ))
        steer_c = float(np.clip(steer, -self.MAX_STEER, self.MAX_STEER))
        self.x += self.v * np.cos(self.theta) * DT
        self.y += self.v * np.sin(self.theta) * DT
        self.theta += (self.v / WHEELBASE) * np.tan(steer_c) * DT
        self.yaw_rate = (self.theta - prev_th) / DT
        self.phys_steps += 1

        collided = self.is_occupied(self.x, self.y, inflate=0.12)
        ds = cte = heading_cos = 0.0
        if self.centerline is not None and not collided:
            pos = np.array([self.x, self.y])
            new_idx = self._nearest_cl(pos)
            ds = self._arc_ds(self.cl_idx, new_idx)
            ds = float(np.clip(ds, -1.5, 2.5))
            self.cl_idx = new_idx
            if ds > 0:
                self._s_travel += ds
            cte = float(np.linalg.norm(pos - self.centerline[new_idx]))
            tang = self._cl_tangent(new_idx)
            heading_cos = float(
                np.cos(self.theta) * tang[0] + np.sin(self.theta) * tang[1]
            )
            if self._s_travel >= 0.98 * self.cl_loop:
                self.lap_completed = True
        return ds, cte, heading_cos, collided

    def step(self, action):
        steer, speed_cmd = self.action_to_controls(action)
        self.last_steer, self.last_speed_cmd = steer, speed_cmd

        sum_ds = 0.0
        last_cte = 0.0
        last_hcos = 0.0
        collided = False
        for _ in range(self.frame_skip):
            ds, cte, hcos, collided = self._physics_once(steer, speed_cmd)
            sum_ds += ds
            last_cte, last_hcos = cte, hcos
            if collided or self.lap_completed:
                break

        self.steps += 1
        obs = self._get_obs()
        v_norm = (self.v - self.MIN_SPEED) / max(self._speed_span, 1e-6)

        # 생존+진행이 짧은 충돌보다 항상 유리 (음수 누적 붕괴 방지)
        reward = (
            4.0 * max(sum_ds, 0.0)
            + 0.2  # alive
            + 0.15 * max(last_hcos, 0.0)
            + 0.25 * v_norm * max(last_hcos, 0.0)
            - 0.1 * max(last_cte - 0.7, 0.0)
            - 0.4 * max(-sum_ds, 0.0)
        )
        reward = float(np.clip(reward, -2.0, 10.0))
        if collided:
            reward = -10.0
        if self.lap_completed:
            reward += 30.0

        # 충돌만 terminate. 랩은 truncate (bootstrap 유지)
        terminated = collided
        truncated = self.lap_completed or self.steps >= self.MAX_STEPS
        info = {
            "collided": collided,
            "lap_completed": self.lap_completed,
            "lap_time": self.phys_steps * DT if self.lap_completed else None,
            "progress": float(self._s_travel / max(self.cl_loop, 1e-6)),
            "s_travel": self._s_travel,
            "speed": self.v,
            "steering_angle": self.last_steer,
            "pos": np.array([self.x, self.y], dtype=np.float32),
            "theta": self.theta,
            "track_name": self.map_name,
            "reward_mode": "privileged_progress_v4",
            "ctrl_hz": 1.0 / self.dt_ctrl,
        }
        return (obs, reward, terminated, truncated, info) if _HAS_GYM else (
            obs, reward, terminated or truncated, info
        )


def try_official_f1tenth_gym() -> bool:
    try:
        import f110_gym  # noqa: F401
        return True
    except ImportError:
        try:
            import f1tenth_gym  # noqa: F401
            return True
        except ImportError:
            return False


if __name__ == "__main__":
    print("official_f1tenth_gym=", try_official_f1tenth_gym())
    env = F1TenthMaplessEnv("Spielberg", seed=0)
    obs, info = env.reset()
    print("obs", obs.shape, info)
    assert obs.shape == (OBS_DIM,)
    for t in range(50):
        a = np.array([0.0, 0.3], dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            print("end", t, info)
            break
    else:
        print("ok progress", info.get("progress"), "v", info.get("speed"), "r", r)
