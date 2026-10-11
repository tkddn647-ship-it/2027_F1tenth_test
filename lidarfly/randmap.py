"""
lidarfly.randmap
================
노이즈 기반 폐곡선 센터라인 → occupancy 복도 → mapless40 Raceline.

극좌표 r(θ)>0 이라 자기교차가 없다. 벽은 센터라인에서 half_width 밖.
레이싱라인은 mapless40.raceline.Raceline (학습용 privileged 전용).
"""

from __future__ import annotations

import numpy as np

from mapless40.config import EnvConfig
from mapless40.env import Track
from mapless40.raceline import Raceline
from mapless40.raycast import GridMap

# 복도 반폭 [m]. 차 반폭 0.15, 레이싱라인 wall_margin 0.45 보다 넓게.
HALF_WIDTH_M = 1.35
RESOLUTION_M = 0.05
PAD_M = 2.0


def closed_centerline(rng: np.random.Generator, n: int = 360, r0: float = 6.5) -> np.ndarray:
    """저차 푸리에 노이즈를 반지름에 더한 폐곡선. (n, 2) [m]."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = np.full(n, r0, dtype=np.float64)
    for k, sigma in ((1, 1.1), (2, 0.7), (3, 0.4), (4, 0.25)):
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(rng.normal(0.0, sigma))
        r += amp * np.cos(k * t + phase)
    r = np.maximum(r, 3.5)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _densify(xy: np.ndarray, step: float) -> np.ndarray:
    closed = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.ceil(s[-1] / step)), xy.shape[0])
    su = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(su, s, closed[:, 0]), np.interp(su, s, closed[:, 1])], axis=1)


def occupancy_corridor(xy: np.ndarray, half_width: float = HALF_WIDTH_M,
                       res: float = RESOLUTION_M, pad: float = PAD_M
                       ) -> tuple[np.ndarray, float, tuple[float, float, float]]:
    """센터라인 주변만 빈칸인 occupancy. 반환 (occupied, resolution, origin xyz)."""
    pts = _densify(np.asarray(xy, dtype=np.float64), res)
    minx = float(pts[:, 0].min() - half_width - pad)
    miny = float(pts[:, 1].min() - half_width - pad)
    maxx = float(pts[:, 0].max() + half_width + pad)
    maxy = float(pts[:, 1].max() + half_width + pad)
    w = int(np.ceil((maxx - minx) / res)) + 1
    h = int(np.ceil((maxy - miny) / res)) + 1
    xs = minx + (np.arange(w) + 0.5) * res
    ys = miny + (h - 1 - np.arange(h) + 0.5) * res
    free = np.zeros((h, w), dtype=bool)
    lim2 = half_width * half_width
    for i0 in range(0, h, 24):
        i1 = min(h, i0 + 24)
        yy = ys[i0:i1][:, None, None]
        xx = xs[None, :, None]
        d2 = (xx - pts[None, None, :, 0]) ** 2 + (yy - pts[None, None, :, 1]) ** 2
        free[i0:i1] = d2.min(axis=2) <= lim2
    origin = (minx, miny, 0.0)
    return ~free, res, origin


def _decel(cfg: EnvConfig):
    a = cfg.act
    if a.brake_enabled:
        return lambda v: 0.9 * a.brake_decel_max
    return lambda v: 0.9 * (a.coast_decel_c0 + a.coast_decel_c1 * v)


def make_random_track(cfg: EnvConfig, rng: np.random.Generator, name: str = "random") -> Track:
    """랜덤 폐곡선 트랙. 레이싱라인은 센터라인을 최소곡률 근사로 다시 계산한다."""
    xy = closed_centerline(rng)
    occ, res, origin = occupancy_corridor(xy)
    grid = GridMap(occ, res, origin)
    spec = cfg.raceline
    line = Raceline(xy, spec, grid=grid, decel_fn=_decel(cfg), optimize=spec.optimize_centerline)
    grid.release_grad()
    return Track(name=name, grid=grid, line=line, line_file="random")
