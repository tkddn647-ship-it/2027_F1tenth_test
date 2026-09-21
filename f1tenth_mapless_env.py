"""
f1tenth_mapless_env.py
======================
F1TENTH mapless Gymnasium env (관측 mapless / 보상 privileged).

실차 정렬:
  LiDAR range 40 m, 측정 주기 40 Hz → DT=0.025
  물리 서브스텝 5 ms, 제어 frame_skip=4 → ~10 Hz
  동역학: f1tenth_gym Single-Track (타이어 슬립 / μ)

관측 (dim=680): LiDAR 135×5 + yaw×5
행동: [steer_norm, speed_norm] → 조향각 목표 + 속도 목표 (PID→가속/조향속도)
"""

from __future__ import annotations
from collections import deque
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from vehicle_dynamics import (
    STParams,
    ROBORACER_A_LAT,
    ROBORACER_HALF_WIDTH,
    ROBORACER_STEER_MAX,
    ROBORACER_WHEELBASE,
    integrate_st_rk4,
    lateral_accel,
    pid_speed_steer,
)

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
ROBORACER_MAPS = ROOT / "Roboracer-2026-main" / "maps"
MAPS_DIR = ROOT / "maps"

MAP_ALIASES = {
    "ajou": "cartographer_map_20260817_003202",
    "ajou_latest": "cartographer_map_20260817_003202",
    "ajou_prev": "cartographer_map_20260817_002900",
    "ifac": "ifac_roboracer",
    "ifac_roboracer": "ifac_roboracer",
}

N_BEAMS = 135
HIST_LEN = 5
OBS_DIM = N_BEAMS * HIST_LEN + HIST_LEN  # 680
MIN_SPEED = 2.0   # Physical AI 커리큘럼 하한 (코너 감속 여유는 보상/슬립)
MAX_SPEED = 7.0
MAX_STEER = ROBORACER_STEER_MAX  # 실측 전륜각 ±21.4°
WHEELBASE = ROBORACER_WHEELBASE
COLLISION_INFLATE = ROBORACER_HALF_WIDTH  # 실측 반폭 0.15 m
DT = 0.025          # 40 Hz lidar / control tick aggregation
PHYS_DT = 0.005     # 5 ms dynamics substep
FRAME_SKIP = 4      # 제어 ~10 Hz
RAY_RANGE = 40.0
FOV = 4.71238898
G = 9.81
REVERSE_GRACE_STEPS = 25  # 스폰 직후 역주행 판정 유예 (~2.5 s @10 Hz)
REVERSE_STREAK = 8        # 연속 역주행 프레임 후 terminate


def _resolve_map_yaml(map_name: str) -> Path:
    """맵 이름/별칭/yaml 경로 → yaml Path.

    우선순위: yaml 경로 → Roboracer maps → maps/ → f1tenth_racetracks
    """
    p = Path(map_name)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p.resolve()

    key = MAP_ALIASES.get(map_name, map_name)
    for base in (ROBORACER_MAPS, MAPS_DIR):
        cand = base / f"{key}.yaml"
        if cand.exists():
            return cand.resolve()

    race = RACETRACKS_DIR / key / f"{key}_map.yaml"
    if race.exists():
        return race.resolve()

    if ROBORACER_MAPS.exists():
        hits = [
            h
            for h in ROBORACER_MAPS.glob(f"*{key}*.yaml")
            if "origin" not in h.name
        ]
        if hits:
            return sorted(hits)[0].resolve()

    raise FileNotFoundError(
        f"맵 yaml 없음: {map_name}\n"
        f"  예: ajou | cartographer_map_20260817_003202 | Spielberg\n"
        f"  또는 yaml 전체 경로"
    )


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
    """Prefer CSV matching map stem; fall back to any *centerline*/*raceline*."""
    folder = map_yaml.parent
    stem = map_yaml.stem
    candidates: list[Path] = []
    for pat in (f"{stem}_centerline.csv", f"{stem}*centerline*.csv", "*centerline*.csv", "*raceline*.csv"):
        candidates.extend(sorted(folder.glob(pat)))
    seen: set[Path] = set()
    for hit in candidates:
        if hit in seen:
            continue
        seen.add(hit)
        raw = np.genfromtxt(hit, delimiter=",", comments="#")
        if raw.ndim == 1 or raw.shape[0] < 20:
            continue
        xy = raw[:, :2].astype(np.float64)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 20:
            continue
        if len(xy) > 600:
            idx = np.linspace(0, len(xy) - 1, 600).astype(int)
            xy = xy[idx]
        print(f"[f1tenth_mapless] centerline: {hit.name} ({len(xy)} pts)")
        return xy
    return None


class F1TenthMaplessEnv(_EnvBase):
    """Mapless obs + privileged progress reward + single-track tire dynamics."""

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
        mu: float | None = None,
        physics: str = "st",  # "st" | "kinematic"
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
        self._speed_span = max(self.MAX_SPEED - self.MIN_SPEED, 1e-6)
        self.MAX_STEPS = int(max_steps)
        self.frame_skip = int(max(1, frame_skip))
        self.dt_ctrl = DT * self.frame_skip
        self.physics = physics
        # 기본: Roboracer 실차 제원 ST (슬립/μ/조향한계)
        self.st_params = STParams.roboracer()
        self.st_params.s_min = -self.MAX_STEER
        self.st_params.s_max = self.MAX_STEER
        self.st_params.v_max = max(self.MAX_SPEED * 1.05, self.st_params.v_max)
        if mu is not None:
            self.st_params.mu = float(mu)
        self.a_lat_lim = float(ROBORACER_A_LAT)
        self._lidar_hist: deque[np.ndarray] = deque(maxlen=HIST_LEN)
        self._yaw_hist: deque[float] = deque(maxlen=HIST_LEN)
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
        self.beta = 0.0  # slip angle
        self.delta = 0.0  # front wheel steer
        self.ay = 0.0
        self.steps = 0
        self.phys_steps = 0
        self.cl_idx = 0
        self._s_travel = 0.0
        self.lap_completed = False
        self.last_steer = 0.0
        self.last_speed_cmd = self.MIN_SPEED
        self._prev_steer_cmd = 0.0
        self._start_pool = self._build_start_pool()
        self._start_pose = self._find_start_pose()
        print(
            f"[f1tenth_mapless] map={self.map_name} "
            f"obs={OBS_DIM} beams={N_BEAMS}x{HIST_LEN} "
            f"lidar={RAY_RANGE}m@{1.0/DT:.0f}Hz ctrl~{1.0/self.dt_ctrl:.0f}Hz "
            f"skip={self.frame_skip} speed=[{self.MIN_SPEED},{self.MAX_SPEED}] "
            f"physics={self.physics} mu={self.st_params.mu:.2f} "
            f"steer±{self.MAX_STEER:.3f} a_lat={self.a_lat_lim:.1f} "
            f"cl_loop={self.cl_loop:.1f}m starts={len(self._start_pool)}"
        )
        self._reverse_streak = 0
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

    def is_occupied(self, x: float, y: float, inflate: float | None = None) -> bool:
        if inflate is None:
            inflate = COLLISION_INFLATE
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
            # 횡방향만 소량 노이즈 (헤딩은 CL 진행 방향으로 재정렬)
            lat = float(self.rng.uniform(-0.12, 0.12))
            nx, ny = np.cos(th - np.pi / 2), np.sin(th - np.pi / 2)
            x2, y2 = x + lat * nx, y + lat * ny
            if not self.is_occupied(x2, y2, 0.2):
                x, y = x2, y2
            # 진행 방향 = 센터라인 접선 (역헤딩 스폰 방지)
            idx = self._nearest_cl(np.array([x, y]))
            tang = self._cl_tangent(idx)
            th = float(np.arctan2(tang[1], tang[0]))
            th += float(self.rng.uniform(-0.06, 0.06))
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
        # 전·후방 모두 검색 (한쪽만 보면 스폰 직후 인덱스 점프 → 가짜 역주행)
        cand = (self.cl_idx + np.arange(-25, 40)) % n
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
        self.v = float(np.clip(self.MIN_SPEED + 0.3, self.MIN_SPEED, self.MAX_SPEED))
        self.yaw_rate = 0.0
        self.beta = 0.0
        self.delta = 0.0
        self.ay = 0.0
        self.steps = 0
        self.phys_steps = 0
        self._s_travel = 0.0
        self.lap_completed = False
        self.last_steer = 0.0
        self.last_speed_cmd = self.v
        self._prev_steer_cmd = 0.0
        self._reverse_streak = 0
        self.cl_idx = self._nearest_cl(np.array([self.x, self.y]))
        # 커스텀 pose가 아니면 헤딩을 CL 접선에 한 번 더 맞춤
        if not (options and "pose" in options) and self.centerline is not None:
            tang = self._cl_tangent(self.cl_idx)
            self.theta = float(np.arctan2(tang[1], tang[0]))
        self._lidar_hist.clear()
        self._yaw_hist.clear()
        obs = self._get_obs()
        info = {
            "map": self.map_name,
            "obs_dim": OBS_DIM,
            "dt": DT,
            "frame_skip": self.frame_skip,
            "ray_range": RAY_RANGE,
            "physics": self.physics,
            "mu": self.st_params.mu,
            "a_lat_lim": self.a_lat_lim,
            "max_steer": self.MAX_STEER,
        }
        return (obs, info) if _HAS_GYM else obs

    def _st_state(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.delta, self.v, self.theta, self.yaw_rate, self.beta],
            dtype=np.float64,
        )

    def _apply_st_state(self, s: np.ndarray) -> None:
        self.x, self.y = float(s[0]), float(s[1])
        self.delta = float(np.clip(s[2], self.st_params.s_min, self.st_params.s_max))
        self.v = float(s[3])
        self.theta = float(s[4])
        self.yaw_rate = float(s[5])
        self.beta = float(s[6])
        self.ay = lateral_accel(self.v, self.yaw_rate)

    def _physics_once(self, steer: float, speed_cmd: float) -> tuple[float, float, float, bool]:
        """One lidar-period tick (DT), with PHYS_DT ST substeps."""
        prev_th = self.theta
        steer_c = float(np.clip(steer, -self.MAX_STEER, self.MAX_STEER))
        speed_c = float(np.clip(speed_cmd, self.MIN_SPEED, self.MAX_SPEED))

        if self.physics == "kinematic":
            self.v = float(np.clip(
                self.v + 0.5 * (speed_c - self.v),
                self.MIN_SPEED,
                self.MAX_SPEED,
            ))
            self.delta = steer_c
            self.x += self.v * np.cos(self.theta) * DT
            self.y += self.v * np.sin(self.theta) * DT
            self.theta += (self.v / WHEELBASE) * np.tan(steer_c) * DT
            self.yaw_rate = (self.theta - prev_th) / DT
            self.beta = 0.0
            self.ay = self.v * self.yaw_rate
            self.phys_steps += 1
        else:
            n_sub = max(1, int(round(DT / PHYS_DT)))
            sub_dt = DT / n_sub
            state = self._st_state()
            for _ in range(n_sub):
                accl, sv = pid_speed_steer(
                    speed_c,
                    steer_c,
                    float(state[3]),
                    float(state[2]),
                    self.st_params.sv_max,
                    self.st_params.a_max,
                    self.st_params.v_max,
                    self.st_params.v_min,
                )
                u = np.array([sv, accl], dtype=np.float64)
                state = integrate_st_rk4(state, u, self.st_params, sub_dt)
                self.phys_steps += 1
            self._apply_st_state(state)
            # keep within training speed band softly
            if self.v > self.MAX_SPEED:
                self.v = self.MAX_SPEED

        collided = self.is_occupied(self.x, self.y)
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
        steer_rate = abs(steer - self._prev_steer_cmd) / max(self.dt_ctrl, 1e-6)
        self._prev_steer_cmd = steer
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

        # Physical AI 보상: 진행 + 실차 a_lat/슬립/조향급변 패널티
        slip = abs(self.beta)
        ay_excess = max(abs(self.ay) - 0.85 * self.a_lat_lim, 0.0)
        # 속도 활용 장려 (상한 근처만 살짝) — 슬립/ay 패널티와 균형
        v_norm = (self.v - self.MIN_SPEED) / max(self._speed_span, 1e-6)
        reward = (
            5.0 * max(sum_ds, 0.0)
            - 1.0 * max(-sum_ds, 0.0)
            - 0.04 * steer_rate
            - 0.6 * max(slip - 0.04, 0.0)
            - 0.05 * ay_excess
            - 0.12 * max(last_cte - 0.55, 0.0)
            + 0.15 * max(sum_ds, 0.0) * v_norm
        )
        reward = float(np.clip(reward, -3.0, 12.0))
        if collided:
            reward = -10.0
        if self.lap_completed:
            reward += 30.0

        # 역주행: 스폰 유예 + 연속 프레임 요구 (가짜 reverse 학살 방지)
        reverse_now = last_hcos < -0.35 and sum_ds < -0.02
        if reverse_now and self.steps >= REVERSE_GRACE_STEPS:
            self._reverse_streak += 1
        else:
            self._reverse_streak = 0
        reverse = self._reverse_streak >= REVERSE_STREAK
        if reverse_now and not reverse:
            reward -= 0.5  # soft penalty while streak builds
        terminated = bool(collided or reverse)
        truncated = self.lap_completed or self.steps >= self.MAX_STEPS
        info = {
            "collided": collided,
            "reversed": reverse,
            "lap_completed": self.lap_completed,
            "lap_time": self.phys_steps * PHYS_DT if self.lap_completed else None,
            "progress": float(self._s_travel / max(self.cl_loop, 1e-6)),
            "s_travel": self._s_travel,
            "speed": self.v,
            "slip_angle": self.beta,
            "yaw_rate": self.yaw_rate,
            "ay": self.ay,
            "mu": self.st_params.mu,
            "a_lat_lim": self.a_lat_lim,
            "steering_angle": self.last_steer,
            "pos": np.array([self.x, self.y], dtype=np.float32),
            "theta": self.theta,
            "track_name": self.map_name,
            "reward_mode": "privileged_progress_st_roboracer_v2",
            "physics": self.physics,
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
