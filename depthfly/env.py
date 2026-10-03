"""
depthfly.env
============
mapless40 차량·보상·장애물 + depth 파리 눈 관측.

관측: near (3, 16, 64) float16 — 가까움 영상 n-2, n-1, n
      state (14,) — v̄×3, ω̄×3, ω_latest, 직전 명령 2개, Δt×3
      priv (27,)  — critic 전용 (레이싱라인·장애물)
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from camfly.env import CamHistory
from mapless40.config import PRIV_DIM
from mapless40.env import MaplessRaceEnv40

from .config import DepthFlyConfig, state_dim
from .sensor import DepthEyeSim


class NearHistory(CamHistory):
    def observation(self, priv=None) -> dict:
        obs = {"near": np.stack(list(self.frames)).astype(np.float16), "state": self.state()}
        if priv is not None:
            obs["priv"] = np.asarray(priv, np.float32)
        return obs


class DepthFlyEnv(MaplessRaceEnv40):
    def __init__(self, maps=("ifac",), cfg=None, seed=None, sensor_noise=True, randomize=True, tracks=None):
        self.cf = cfg or DepthFlyConfig()
        super().__init__(maps=maps, cfg=self.cf.env, seed=seed, sensor_noise=sensor_noise,
                         randomize=randomize, tracks=tracks)
        c = self.cf.cam
        self.eye = DepthEyeSim(c, self.cf.scene, range_max=self.cf.env.lidar.range_max)
        self.hist = NearHistory(self.cf)
        H = self.cf.hist
        self.observation_space = spaces.Dict({
            "near": spaces.Box(0.0, 1.0, (H, c.n_rows, c.n_cols), np.float16),
            "state": spaces.Box(-5.0, 5.0, (state_dim(H),), np.float32),
            "priv": spaces.Box(-10.0, 10.0, (PRIV_DIM,), np.float32),
        })

    def reset(self, *, seed=None, options=None):
        self.eye.hold.reset()
        return super().reset(seed=seed, options=options)

    def _raw_scan(self):
        x, y, yaw = self.state[0], self.state[1], self.state[4]
        return self.eye.frame(self.track.grid, x, y, yaw, self.obstacles, self.rng, noise=self.sensor_noise)


def make_env(maps, seed: int = 0, cfg=None, **kw):
    def _f():
        return DepthFlyEnv(maps=maps, cfg=cfg, seed=seed, **kw)
    return _f
