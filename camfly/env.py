"""
camfly.env
==========
mapless40.MaplessRaceEnv40 (차량·보상·레이싱라인·장애물) 위에 센서만 카메라로 바꾼 환경.

관측 (Dict):
  eye    (hist, R, C) float16  — 파리 눈 흑백, 광수용체 적응 후 [-1, 1]. 최근 3프레임 n-2, n-1, n
  dscan  (hist, C)    float16  — depth 가짜 스캔 /7 m, 같은 열
  state  (3·hist+5,)  float32  — v̄×3, ω̄×3, ω_latest, (δ,v)×2, Δt×3
  priv   (27,)        float32  — critic 전용 (mapless40 과 동일: 레이싱라인·장애물)
제어 주기 = 카메라 30 Hz.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from gymnasium import spaces

from mapless40.config import PRIV_DIM
from mapless40.env import MaplessRaceEnv40

from .config import CamFlyConfig, state_dim
from .eye import FlyEyeRenderer, adapt


class CamHistory:
    """최근 hist 프레임의 (eye, dscan) 과 구간 평균 v̄, ω̄, Δt, 직전 명령 2개."""

    def __init__(self, cfg: CamFlyConfig):
        self.cfg, self.hist = cfg, cfg.hist
        self.frames: deque = deque(maxlen=self.hist)
        self.v: deque = deque(maxlen=self.hist)
        self.w: deque = deque(maxlen=self.hist)
        self.dt: deque = deque(maxlen=self.hist)
        self.cmds: deque = deque(maxlen=2)
        self.w_latest = 0.0

    def reset(self, frame, v: float, w: float, cmd=None) -> None:
        for d in (self.frames, self.v, self.w, self.dt, self.cmds):
            d.clear()
        for _ in range(self.hist):
            self.push(frame, v, w, w, self.cfg.env.norm.dt_nominal)
        c = cmd if cmd is not None else (0.0, v)
        self.cmds.extend([c, c])

    def push(self, frame, v_mean, w_mean, w_latest, dt) -> None:
        self.frames.append(frame)
        self.v.append(float(v_mean))
        self.w.append(float(w_mean))
        self.dt.append(float(dt))
        self.w_latest = float(w_latest)

    def push_cmd(self, steer, v_cmd) -> None:
        self.cmds.append((float(steer), float(v_cmd)))

    def state(self) -> np.ndarray:
        n, a, H = self.cfg.env.norm, self.cfg.env.act, self.hist
        s = np.zeros(state_dim(H), np.float32)
        s[0:H] = np.asarray(self.v) / n.v_scale
        s[H:2 * H] = np.clip(np.asarray(self.w) / n.w_scale, -1, 1)
        s[2 * H] = np.clip(self.w_latest / n.w_scale, -1, 1)
        c = list(self.cmds)
        s[2 * H + 1:2 * H + 5] = [c[0][0] / a.steer_max, c[0][1] / n.v_scale,
                                  c[1][0] / a.steer_max, c[1][1] / n.v_scale]
        s[2 * H + 5:3 * H + 5] = np.clip((np.asarray(self.dt) - n.dt_nominal) / n.dt_nominal, -2, 4)
        return s

    def observation(self, priv=None) -> dict:
        obs = {"eye": np.stack([f[0] for f in self.frames]).astype(np.float16),
               "dscan": np.stack([f[1] for f in self.frames]).astype(np.float16),
               "state": self.state()}
        if priv is not None:
            obs["priv"] = np.asarray(priv, np.float32)
        return obs


class CamFlyEnv(MaplessRaceEnv40):
    def __init__(self, maps=("ifac",), cfg: CamFlyConfig | None = None, seed: int | None = None,
                 sensor_noise: bool = True, randomize: bool = True, tracks=None):
        self.cf = cfg or CamFlyConfig()
        super().__init__(maps=maps, cfg=self.cf.env, seed=seed, sensor_noise=sensor_noise,
                         randomize=randomize, tracks=tracks)
        c = self.cf.cam
        self.renderer = FlyEyeRenderer(c, self.cf.scene, range_max=self.cf.env.lidar.range_max)
        self.hist = CamHistory(self.cf)
        H = self.cf.hist
        self.observation_space = spaces.Dict({
            "eye": spaces.Box(-1.0, 1.0, (H, c.n_rows, c.n_cols), np.float16),
            "dscan": spaces.Box(0.0, 1.0, (H, c.n_cols), np.float16),
            "state": spaces.Box(-5.0, 5.0, (state_dim(H),), np.float32),
            "priv": spaces.Box(-10.0, 10.0, (PRIV_DIM,), np.float32),
        })
        self.last_raw_eye = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.renderer.randomize(self.rng)                # 장면 밝기·질감 (에피소드마다)
        return super().reset(seed=seed, options=options)

    def _raw_scan(self):
        """mapless40 의 'LiDAR 스캔' 자리에 카메라 프레임 (eye, dscan) 을 넣는다."""
        x, y, yaw = self.state[0], self.state[1], self.state[4]
        img, dw, do, *_ = self.renderer.render(self.track.grid, x, y, yaw, self.obstacles, self.rng,
                                               noise=self.sensor_noise)
        self.last_raw_eye = img
        ds = self.renderer.depth_scan(dw, do, self.rng, noise=self.sensor_noise)
        return (adapt(img).astype(np.float32), ds)


def make_env(maps, seed: int = 0, cfg: CamFlyConfig | None = None, **kw):
    def _f():
        return CamFlyEnv(maps=maps, cfg=cfg, seed=seed, **kw)
    return _f
