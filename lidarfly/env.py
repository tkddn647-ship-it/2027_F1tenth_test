"""
lidarfly.env
============
MaplessRaceEnv40 차량·보상·레이싱라인. 관측만 LiDAR 1080빔 × 3프레임.

  scan  (3, 1080) float16  — n-2, n-1, n. 최신이 마지막 행
  state (14,)     float32  — v̄×3, ω̄×3, ω_latest, (δ,v)×2, Δt×3
  priv  (27,)     float32  — critic 전용. actor 에 넣지 않는다

맵 문자열 "ifac:3,roboracer_0817:3,random:2" 의 random 은 자리표시자.
random 이 뽑힌 에피소드에서만 env 시드로 randmap 폐곡선을 다시 만든다.
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np
from gymnasium import spaces

from mapless40.config import PRIV_DIM, EnvConfig
from mapless40.env import MaplessRaceEnv40, load_track
from mapless40.obs_builder import preprocess_scan

from .config import DEFAULT_MAPS, HIST, lidarfly_config, state_dim
from .randmap import make_random_track

# 폐곡선 자기교차 등으로 생성이 실패하면 이 횟수만큼 다시 뽑고, 그래도 실패하면 이전 트랙을 유지한다.
_RANDOM_TRIES = 8


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
            # 실패 시 남길 자리표시자. 에피소드 트랙은 reset 이 self.rng(환경 시드)로 다시 만든다.
            # 자리표시자 시드를 한 칸 비켜 첫 reset 과 같은 형상이 되지 않게 한다.
            ph = None if seed is None else int(seed) + 1_000_003
            rng = np.random.default_rng(ph)
            tracks = [
                make_random_track(cfg, rng) if n == "random" else load_track(n, cfg)
                for n in names
            ]
        super().__init__(maps=maps, cfg=cfg, seed=seed, sensor_noise=sensor_noise,
                         randomize=randomize, tracks=tracks)
        if own_tracks:
            w = np.asarray(weights, dtype=np.float64)
            self.track_p = w / w.sum()
        # 부모 reset 은 시드가 있으면 self.rng 를 덮어쓴다. 에피소드 번호는 여기 두어
        # 같은 시드로 reset 을 반복해도 트랙이 다시 뽑히게 한다.
        self._base_seed = 0 if seed is None else int(seed)
        self._random_ep = 0
        self._install_obs()

    def _rng_matching_parent(self, seed: int | None) -> np.random.Generator:
        """부모 reset 이 트랙을 고를 때와 같은 난수. self.rng 는 바꾸지 않는다."""
        if seed is not None:
            return np.random.default_rng(seed)
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        return rng

    def _selected_track_index(self, seed: int | None, options: dict) -> int | None:
        """이번 에피소드 트랙. random 이 아니면 None (폐곡선을 만들지 않음)."""
        if "map" in options:
            if options["map"] != "random":
                return None
            for i, tr in enumerate(self.tracks):
                if tr.name == "random":
                    return i
            return None
        idx = int(self._rng_matching_parent(seed).choice(len(self.tracks), p=self.track_p))
        if self.tracks[idx].name != "random":
            return None
        return idx

    def _episode_track_rng(self, seed: int | None) -> np.random.Generator:
        """이번 에피소드의 트랙 난수. self.rng(환경 시드)에서 하나 뽑아 갈라낸다."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._base_seed = int(seed)
        ep = self._random_ep
        self._random_ep += 1
        draw = int(self.rng.integers(0, 2**32))
        return np.random.default_rng(np.random.SeedSequence([self._base_seed, ep, draw]))

    def _rebuild_random_slot(self, idx: int, rng: np.random.Generator) -> None:
        """random 슬롯을 교체. 실패하면 그 칸의 이전 트랙을 남긴다. name 은 'random'."""
        prev = self.tracks[idx]
        name = prev.name
        for _ in range(_RANDOM_TRIES):
            try:
                tr = make_random_track(self.cfg, rng, name=name)
            except Exception:
                continue
            line = tr.line
            if line.length > 1.0 and np.isfinite(line.x).all() and np.isfinite(line.y).all():
                self.tracks[idx] = tr
                return

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # 부모가 고른 칸이 random 일 때만 폐곡선을 만든다. ifac 등 고정 맵은 그대로 둔다.
        options = options or {}
        idx = self._selected_track_index(seed, options)
        if idx is not None:
            # seed 가 없으면 트랙 난수가 self.rng 를 한 번 쓴다. 부모의 맵 추첨은 원래 상태로 되돌린다.
            saved = copy.deepcopy(self.rng.bit_generator.state) if seed is None else None
            self._rebuild_random_slot(idx, self._episode_track_rng(seed))
            if saved is not None:
                self.rng.bit_generator.state = saved
        return super().reset(seed=seed, options=options)

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


def make_env(maps=DEFAULT_MAPS, seed: int = 0, cfg: EnvConfig | None = None,
             rank: int = 0, **kw):
    """병렬 env 팩토리. rank 마다 시드가 seed+rank 라 랜덤 맵이 갈라진다."""
    env_seed = int(seed) + int(rank)

    def _f():
        return LidarFlyEnv(maps=maps, cfg=cfg, seed=env_seed, **kw)
    return _f
