"""
lidarfly.env
============
MaplessRaceEnv40 차량·보상·레이싱라인. 관측만 LiDAR 1080빔 × 3프레임.

  scan  (3, 1080) float16  — n-2, n-1, n. 최신이 마지막 행
  state (14,)     float32  — v̄×3, ω̄×3, ω_latest, (δ,v)×2, Δt×3
  priv  (27,)     float32  — critic 전용. actor 에 넣지 않는다

맵 문자열 "ifac:3,roboracer_0817:3,random:2" 의 random 은 randmap 폐곡선.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from gymnasium import spaces

from mapless40.config import PRIV_DIM, EnvConfig
from mapless40.env import MaplessRaceEnv40, load_track
from mapless40.obs_builder import preprocess_scan

from .config import DEFAULT_MAPS, HIST, lidarfly_config, state_dim
from .randmap import make_random_track


def parse_maps(maps) -> tuple[list[str], list[float]]:
    if isinstance(maps, str):
        maps = [m for m in maps.split(",") if m]
    names, weights = [], []
    for m in maps:
        n, _, w = str(m).partition(":")
        names.append(n)
        weights.append(float(w) if w else 1.0)
    return names, weights


class ScanHistory:
    """최근 hist 장 스캔과 구간 평균. state 길이는 hist 에 맞춰 14 (hist=3)."""

    def __init__(self, cfg: EnvConfig, hist: int = HIST):
        self.cfg = cfg
        self.hist = hist
        self.scans: deque = deque(maxlen=hist)
        self.v: deque = deque(maxlen=hist)
        self.w: deque = deque(maxlen=hist)
        self.dt: deque = deque(maxlen=hist)
        self.cmds: deque = deque(maxlen=2)
        self.w_latest = 0.0

    def reset(self, scan, v: float, w: float, cmd=None) -> None:
        for d in (self.scans, self.v, self.w, self.dt, self.cmds):
            d.clear()
        for _ in range(self.hist):
            self.push(scan, v, w, w, self.cfg.norm.dt_nominal)
        c = cmd if cmd is not None else (0.0, v)
        self.cmds.extend([c, c])

    def push(self, scan, v_mean, w_mean, w_latest, dt) -> None:
        self.scans.append(np.asarray(scan, dtype=np.float32))
        self.v.append(float(v_mean))
        self.w.append(float(w_mean))
        self.dt.append(float(dt))
        self.w_latest = float(w_latest)

    def push_cmd(self, steer, v_cmd) -> None:
        self.cmds.append((float(steer), float(v_cmd)))

    def state(self) -> np.ndarray:
        n, a, H = self.cfg.norm, self.cfg.act, self.hist
        s = np.zeros(state_dim(H), np.float32)
        s[0:H] = np.asarray(self.v) / n.v_scale
        s[H:2 * H] = np.clip(np.asarray(self.w) / n.w_scale, -1.0, 1.0)
        s[2 * H] = np.clip(self.w_latest / n.w_scale, -1.0, 1.0)
        c = list(self.cmds)
        s[2 * H + 1:2 * H + 5] = (
            c[0][0] / a.steer_max, c[0][1] / n.v_scale,
            c[1][0] / a.steer_max, c[1][1] / n.v_scale,
        )
        s[2 * H + 5:3 * H + 5] = np.clip(
            (np.asarray(self.dt) - n.dt_nominal) / n.dt_nominal, -2.0, 4.0)
        return s

    def observation(self, priv=None) -> dict:
        obs = {
            "scan": np.stack(list(self.scans), axis=0).astype(np.float16),
            "state": self.state(),
        }
        if priv is not None:
            obs["priv"] = np.asarray(priv, dtype=np.float32)
        return obs


class LidarFlyEnv(MaplessRaceEnv40):
    def __init__(self, maps=DEFAULT_MAPS, cfg: EnvConfig | None = None,
                 seed: int | None = None, sensor_noise: bool = True,
                 randomize: bool = True, tracks=None):
        cfg = cfg or lidarfly_config()
        names, weights = parse_maps(maps)
        own_tracks = tracks is None
        if own_tracks:
            rng = np.random.default_rng(seed)
            tracks = [
                make_random_track(cfg, rng) if n == "random" else load_track(n, cfg)
                for n in names
            ]
        super().__init__(maps=maps, cfg=cfg, seed=seed, sensor_noise=sensor_noise,
                         randomize=randomize, tracks=tracks)
        if own_tracks:
            w = np.asarray(weights, dtype=np.float64)
            self.track_p = w / w.sum()
        self._install_obs()

    def _install_obs(self) -> None:
        c = self.cfg
        h = c.timing.hist
        self.hist = ScanHistory(c, h)
        self.observation_space = spaces.Dict({
            "scan": spaces.Box(0.0, 1.0, (h, c.lidar.n_beams), np.float16),
            "state": spaces.Box(-5.0, 5.0, (state_dim(h),), np.float32),
            "priv": spaces.Box(-10.0, 10.0, (PRIV_DIM,), np.float32),
        })

    def _raw_scan(self) -> np.ndarray:
        """1080빔 · FOV 270° (사용자 지정) 로 정규화한 한 장. 값 ∈ [0, 1]."""
        sp = self.cfg.lidar
        raw = self.lidar.scan(self.state[0], self.state[1], self.state[4],
                              noise=self.sensor_noise, obstacles=self.obstacles)
        return preprocess_scan(raw, -sp.fov / 2.0, sp.angle_inc, sp)


def make_env(maps=DEFAULT_MAPS, seed: int = 0, cfg: EnvConfig | None = None, **kw):
    def _f():
        return LidarFlyEnv(maps=maps, cfg=cfg, seed=seed, **kw)
    return _f
