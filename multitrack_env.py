"""
multitrack_env.py
===================
다중 트랙 레이싱 환경 (랩타임).

- --race-maps / racetracks_dir: 실제 f1tenth_racetracks (Spielberg, Silverstone…)
- 기본: procedural oval 등 (빠른 디버그용)
관측은 LiDAR + 속도 + 각속도만 (mapless).
"""

from __future__ import annotations
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False
    gym = None
    spaces = None

from procedural_tracks import build_track_pool

_EnvBase = gym.Env if _HAS_GYM else object


class MultiTrackEnv(_EnvBase):
    N_RAYS = 20
    RAY_RANGE = 8.0
    MAX_STEPS = 5000
    DT = 0.05
    MAX_SPEED = 5.0
    DEFAULT_HALF_WIDTH = 1.2

    def __init__(
        self,
        track_names: list[str] | None = None,
        seed: int | None = None,
        track_switch_every_episode: bool = True,
        racetracks_dir: str | None = None,
        fixed_start: bool = False,
    ):
        self.rng = np.random.default_rng(seed)
        self.track_switch_every_episode = track_switch_every_episode
        self.fixed_start = fixed_start
        self.racetracks_dir = racetracks_dir
        self.map_png = None
        self.half_widths = None

        if racetracks_dir:
            from racetrack_loader import build_race_track_pool
            names = track_names or ["Spielberg", "MoscowRaceway"]
            raw = build_race_track_pool(names, racetracks_dir)
            self.track_pool = {k: v["centerline"] for k, v in raw.items()}
            self._meta = raw
        else:
            self.track_pool = build_track_pool(track_names, seed=seed or 0)
            self._meta = {
                k: {"centerline": v, "half_widths": None, "map_png": None}
                for k, v in self.track_pool.items()
            }

        self.track_names = list(self.track_pool.keys())
        self.current_track_name = None
        self.centerline = None
        self.n_cl = 0
        self._episode_count = 0

        if _HAS_GYM:
            obs_dim = self.N_RAYS + 2
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.pos = np.zeros(2)
        self.heading = 0.0
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.cl_idx = 0
        self.steps = 0
        self.lap_completed = False
        self._progress_acc = 0  # 한 바퀴 진행량(인덱스 누적)

    def _load_track(self, name: str):
        self.current_track_name = name
        meta = self._meta[name]
        self.centerline = meta["centerline"]
        self.n_cl = len(self.centerline)
        self.half_widths = meta.get("half_widths")
        self.map_png = meta.get("map_png")
        nxt = np.roll(self.centerline, -1, axis=0)
        tang = nxt - self.centerline
        norms = np.linalg.norm(tang, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        self.tangents = tang / norms

    def _half_w_at(self, idx: int) -> float:
        if self.half_widths is None:
            return self.DEFAULT_HALF_WIDTH
        return float(self.half_widths[idx])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._episode_count += 1

        if self.centerline is None or self.track_switch_every_episode:
            name = str(self.rng.choice(self.track_names))
            self._load_track(name)

        if self.fixed_start or (options and options.get("fixed_start")):
            start_idx = 0
        else:
            start_idx = int(self.rng.integers(0, self.n_cl))
        self.pos = self.centerline[start_idx].copy()
        self.heading = float(np.arctan2(*self.tangents[start_idx][::-1]))
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.cl_idx = start_idx
        self.steps = 0
        self.lap_completed = False
        self._progress_acc = 0
        self._start_idx = start_idx

        obs = self._get_obs()
        info = {"track_name": self.current_track_name}
        if _HAS_GYM:
            return obs, info
        return obs

    def _nearest_centerline_idx(self, pos: np.ndarray, search_window: int = 20) -> int:
        cand = (self.cl_idx + np.arange(-3, search_window)) % self.n_cl
        d = np.linalg.norm(self.centerline[cand] - pos, axis=1)
        return int(cand[np.argmin(d)])

    def _cast_rays(self) -> np.ndarray:
        angles = self.heading + np.linspace(-np.pi / 2, np.pi / 2, self.N_RAYS)
        dists = np.full(self.N_RAYS, self.RAY_RANGE, dtype=np.float32)
        for i, a in enumerate(angles):
            direction = np.array([np.cos(a), np.sin(a)])
            for t in np.linspace(0.05, self.RAY_RANGE, 28):
                p = self.pos + direction * t
                idx = self._nearest_centerline_idx(p, search_window=12)
                lateral = np.linalg.norm(p - self.centerline[idx])
                if lateral > self._half_w_at(idx):
                    dists[i] = t
                    break
        return dists

    def _get_obs(self) -> np.ndarray:
        d = self._cast_rays() / self.RAY_RANGE
        speed_norm = np.array([self.speed / self.MAX_SPEED], dtype=np.float32)
        yaw_rate_norm = np.array([np.clip(self.yaw_rate / 3.0, -1, 1)], dtype=np.float32)
        return np.concatenate([d, speed_norm, yaw_rate_norm]).astype(np.float32)

    def step(self, action):
        steer, accel = float(np.clip(action[0], -1, 1)), float(np.clip(action[1], -1, 1))
        prev_heading = self.heading
        self.heading += steer * 1.8 * self.DT
        self.yaw_rate = (self.heading - prev_heading) / self.DT
        self.speed = float(np.clip(self.speed + accel * self.MAX_SPEED * self.DT, 0.0, self.MAX_SPEED))
        self.pos = self.pos + self.speed * self.DT * np.array(
            [np.cos(self.heading), np.sin(self.heading)]
        )
        self.steps += 1

        new_idx = self._nearest_centerline_idx(self.pos)
        lateral_offset = np.linalg.norm(self.pos - self.centerline[new_idx])
        off_track = lateral_offset > self._half_w_at(new_idx) * 1.05

        delta = new_idx - self.cl_idx
        if delta < -self.n_cl / 2:
            delta += self.n_cl
        elif delta > self.n_cl / 2:
            delta -= self.n_cl
        self.cl_idx = new_idx
        self._progress_acc += delta
        # 출발점 기준 한 바퀴(거의 full loop) 진행하면 완주
        if self._progress_acc >= self.n_cl - 2:
            self.lap_completed = True

        reward = (
            float(delta) * 1.5
            + 0.2 * self.speed
            - (30.0 if off_track else 0.0)
            - (0.1 if self.speed < 0.4 else 0.0)
            - 0.03 * abs(lateral_offset)
        )
        if self.lap_completed:
            lap_time = self.steps * self.DT
            reward += max(0.0, 120.0 - lap_time)

        terminated = off_track or self.lap_completed
        truncated = self.steps >= self.MAX_STEPS

        obs = self._get_obs()
        info = {
            "off_track": off_track,
            "lap_completed": self.lap_completed,
            "lap_time": self.steps * self.DT if self.lap_completed else None,
            "track_name": self.current_track_name,
            "progress": self._progress_acc / max(self.n_cl, 1),
        }
        if _HAS_GYM:
            return obs, reward, terminated, truncated, info
        return obs, reward, (terminated or truncated), info


if __name__ == "__main__":
    print("[multitrack] Spielberg 휴리스틱 완주 테스트...")
    env = MultiTrackEnv(
        track_names=["Spielberg"],
        racetracks_dir="f1tenth_racetracks",
        track_switch_every_episode=False,
        fixed_start=True,
        seed=0,
    )
    obs, _ = env.reset()
    for t in range(env.MAX_STEPS):
        rays = obs[: env.N_RAYS]
        left, right = rays[: env.N_RAYS // 2].mean(), rays[env.N_RAYS // 2 :].mean()
        steer = float(np.clip((right - left) * 3.0, -1, 1))
        accel = 0.6 if obs[env.N_RAYS] < 0.7 else 0.2
        obs, reward, term, trunc, info = env.step(np.array([steer, accel]))
        if term or trunc:
            print(
                f"step={t} lap={info['lap_completed']} time={info['lap_time']} "
                f"off={info['off_track']} progress={info['progress']:.2f}"
            )
            break
    else:
        print("MAX_STEPS")
