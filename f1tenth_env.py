"""
f1tenth_env.py
================
F1TENTH/RoboRacer 스타일 Gymnasium 환경.

- 공식 f1tenth_gym 맵(yaml + png/pgm) 사용
- 관측: LiDAR 거리 (정규화) — 커넥톰 RNN 입력과 동일 인터페이스
- 행동: [조향, 가속] in [-1, 1] — toy_obstacle_env / RoboRacer와 동일
- 보상: 전진 진행도 - 충돌 페널티 (+ 약한 속도 보너스)
- 랩: 출발점 근처를 한 바퀴 돌아오면 lap +1 (간단 규정)

공식 f110_gym은 구버전 gym + Python 제약으로, 여기선 맵/규정을
가져오고 물리·센서를 gymnasium 네이티브로 재구현했다.
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

MAPS_DIR = Path(__file__).resolve().parent / "maps"

# 맵별 알려진 안전한 시작 포즈 (없으면 자동 탐색)
KNOWN_STARTS = {
    "berlin": (0.0, 0.0, 1.4),
    "vegas": (0.0, -0.5, 1.4),
    "skirk": (0.0, 0.0, 0.0),
    "levine": (-15.0, 0.0, 0.0),
    "stata_basement": (0.0, 0.0, 0.0),
}


def _resolve_map(map_name: str) -> Path:
    p = Path(map_name)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    cand = MAPS_DIR / f"{map_name}.yaml"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"맵을 찾을 수 없습니다: {map_name}\n"
        f"maps/ 에 {{name}}.yaml + 이미지 가 있어야 합니다. 현재: {list(MAPS_DIR.glob('*'))}"
    )


def load_occupancy(map_yaml: Path):
    meta = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
    img_name = meta["image"]
    img_path = map_yaml.parent / img_name
    if not img_path.exists():
        stem = Path(img_name).stem
        for ext in (".png", ".pgm", ".jpg"):
            alt = map_yaml.parent / f"{stem}{ext}"
            if alt.exists():
                img_path = alt
                break
    if not img_path.exists():
        raise FileNotFoundError(f"맵 이미지 없음: {img_name} (yaml={map_yaml})")

    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    # ROS: 흰색=자유, 검정=점유
    occ_prob = 1.0 - (img / 255.0)
    occupied = occ_prob >= float(meta.get("occupied_thresh", 0.65))
    resolution = float(meta["resolution"])
    origin = meta["origin"]  # [x, y, yaw]
    return occupied, resolution, origin, meta


class F1TenthEnv(_EnvBase):
    """단일 에이전트 F1TENTH-lite. SB3 / 커넥톰 학습용."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        map_name: str = "berlin",
        n_beams: int = 36,
        lidar_max: float = 10.0,
        fov: float = 4.7,          # ~270 deg, F1TENTH 유사
        dt: float = 0.05,
        max_steps: int = 800,
        wheelbase: float = 0.33,
        max_steer: float = 0.34,
        max_accel: float = 4.0,
        max_speed: float = 6.0,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        self.map_yaml = _resolve_map(map_name)
        self.map_name = self.map_yaml.stem
        self.occ, self.resolution, self.origin, self.meta = load_occupancy(self.map_yaml)
        self.h, self.w = self.occ.shape

        self.n_beams = n_beams
        self.lidar_max = lidar_max
        self.fov = fov
        self.dt = dt
        self.max_steps = max_steps
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_accel = max_accel
        self.max_speed = max_speed
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed)
        if _HAS_GYM:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(n_beams,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.x = self.y = self.theta = self.v = 0.0
        self.steps = 0
        self.lap_count = 0
        self._start = np.zeros(3)
        self._near_start = True
        self._toggle = 0
        self._path_len = 0.0
        self._default_start = self._find_start_pose(KNOWN_STARTS.get(self.map_name))

    # ---- 좌표 변환 -------------------------------------------------
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

    def is_occupied(self, x: float, y: float, inflate: float = 0.15) -> bool:
        # 차량 반경만큼 몇 점 샘플
        for dx, dy in ((0, 0), (inflate, 0), (-inflate, 0), (0, inflate), (0, -inflate)):
            r, c = self.world_to_pixel(x + dx, y + dy)
            if r < 0 or c < 0 or r >= self.h or c >= self.w:
                return True
            if self.occ[r, c]:
                return True
        return False

    def _find_start_pose(self, hint: tuple | None):
        if hint is not None:
            x, y, th = hint
            if not self.is_occupied(x, y, inflate=0.25):
                return np.array([x, y, th], dtype=np.float64)

        free = np.argwhere(~self.occ)
        if len(free) == 0:
            raise RuntimeError("맵에 자유 공간이 없습니다.")
        self.rng.shuffle(free)
        for row, col in free[:5000]:
            x, y = self.pixel_to_world(int(row), int(col))
            if self.is_occupied(x, y, inflate=0.3):
                continue
            # 전방에 여유가 있는 방향 찾기
            for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                nx = x + 1.5 * np.cos(th)
                ny = y + 1.5 * np.sin(th)
                if not self.is_occupied(nx, ny, inflate=0.2):
                    return np.array([x, y, th], dtype=np.float64)
        # fallback
        row, col = free[0]
        x, y = self.pixel_to_world(int(row), int(col))
        return np.array([x, y, 0.0], dtype=np.float64)

    # ---- LiDAR -----------------------------------------------------
    def _cast_lidar(self) -> np.ndarray:
        half = self.fov / 2.0
        angles = self.theta + np.linspace(-half, half, self.n_beams)
        dists = np.full(self.n_beams, self.lidar_max, dtype=np.float32)
        n_samp = 60
        for i, a in enumerate(angles):
            ca, sa = np.cos(a), np.sin(a)
            for t in np.linspace(0.05, self.lidar_max, n_samp):
                if self.is_occupied(self.x + t * ca, self.y + t * sa, inflate=0.0):
                    dists[i] = t
                    break
        return dists

    def _get_obs(self) -> np.ndarray:
        return (self._cast_lidar() / self.lidar_max).astype(np.float32)

    # ---- Gym API ---------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._default_start = self._find_start_pose(KNOWN_STARTS.get(self.map_name))
        pose = self._default_start.copy()
        if options and "pose" in options:
            pose = np.asarray(options["pose"], dtype=np.float64)

        self.x, self.y, self.theta = float(pose[0]), float(pose[1]), float(pose[2])
        self.v = 0.0
        self.steps = 0
        self.lap_count = 0
        self._start = pose.copy()
        self._near_start = True
        self._toggle = 0
        self._path_len = 0.0

        obs = self._get_obs()
        info = {"lap_count": 0, "map": self.map_name}
        if _HAS_GYM:
            return obs, info
        return obs

    def _update_laps(self):
        # 출발점 반경 통과 토글 → 2번이면 1랩 (공식 env와 유사한 단순 규정)
        dx = self.x - self._start[0]
        dy = self.y - self._start[1]
        dist = float(np.hypot(dx, dy))
        close = dist < 1.0
        if close and not self._near_start:
            self._near_start = True
            self._toggle += 1
        elif (not close) and self._near_start:
            self._near_start = False
            self._toggle += 1
        self.lap_count = self._toggle // 2

    def step(self, action):
        steer_n = float(np.clip(action[0], -1, 1))
        accel_n = float(np.clip(action[1], -1, 1))
        steer = steer_n * self.max_steer
        accel = accel_n * self.max_accel

        prev = np.array([self.x, self.y])
        # kinematic bicycle
        self.v = float(np.clip(self.v + accel * self.dt, 0.0, self.max_speed))
        beta = np.arctan(0.5 * np.tan(steer))
        self.x += self.v * np.cos(self.theta + beta) * self.dt
        self.y += self.v * np.sin(self.theta + beta) * self.dt
        self.theta += (self.v / self.wheelbase) * np.sin(beta) * self.dt
        self.steps += 1

        collided = self.is_occupied(self.x, self.y, inflate=0.18)
        progress = float(np.linalg.norm(np.array([self.x, self.y]) - prev))
        if not collided:
            self._path_len += progress

        self._update_laps()

        reward = 2.0 * progress + 0.05 * self.v
        if collided:
            reward -= 5.0
        # 제자리 정지는 약하게만 억제 (과도하면 '안 움직이기'로 수렴함)
        if self.v < 0.05:
            reward -= 0.005

        terminated = collided or self.lap_count >= 1
        truncated = self.steps >= self.max_steps
        if self.lap_count >= 1 and not collided:
            reward += 50.0  # 완주 보너스

        obs = self._get_obs()
        info = {
            "collided": collided,
            "lap_count": int(self.lap_count),
            "speed": self.v,
            "pos": np.array([self.x, self.y], dtype=np.float32),
            "theta": self.theta,
            "path_len": self._path_len,
            "map": self.map_name,
        }
        if _HAS_GYM:
            return obs, reward, terminated, truncated, info
        return obs, reward, (terminated or truncated), info

    def render(self):
        """맵 + 차량 + LiDAR를 RGB 배열로."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyArrow, Rectangle
        import io
        from PIL import Image as PILImage

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(
            self.occ,
            cmap="gray_r",
            origin="upper",
            extent=[
                self.origin[0],
                self.origin[0] + self.w * self.resolution,
                self.origin[1],
                self.origin[1] + self.h * self.resolution,
            ],
        )
        ax.add_patch(Circle((self.x, self.y), 0.2, color="#1a7"))
        ax.add_patch(FancyArrow(
            self.x, self.y,
            0.5 * np.cos(self.theta), 0.5 * np.sin(self.theta),
            width=0.05, head_width=0.2, color="#0a5", length_includes_head=True,
        ))
        scans = self._cast_lidar()
        half = self.fov / 2.0
        angles = self.theta + np.linspace(-half, half, self.n_beams)
        for d, a in zip(scans, angles):
            ax.plot(
                [self.x, self.x + d * np.cos(a)],
                [self.y, self.y + d * np.sin(a)],
                color="#4af", alpha=0.25, lw=0.6,
            )
        ax.set_title(f"{self.map_name}  v={self.v:.1f}  lap={self.lap_count}")
        ax.set_aspect("equal")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return np.asarray(PILImage.open(buf).convert("RGB"))


if __name__ == "__main__":
    env = F1TenthEnv(map_name="berlin", seed=0)
    obs, info = env.reset()
    print(f"map={info['map']}, obs_shape={obs.shape}, start OK")
    total = 0.0
    for t in range(50):
        a = env.action_space.sample() if _HAS_GYM else np.array([0.0, 0.5])
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            print(f"done at {t}: collided={info['collided']} reward={total:.2f}")
            break
    else:
        print(f"50 steps ok, reward={total:.2f}, path={info['path_len']:.2f}")
