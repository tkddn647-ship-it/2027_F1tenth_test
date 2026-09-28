"""
mapless40.raycast
=================
벡터화 2D LiDAR 시뮬레이터 (1125빔 @ 40 Hz 가 현실적으로 돌 수 있게).

Sphere tracing: 맵 점유칸까지의 거리장(EDT)을 한 번 계산해 두고,
모든 빔을 동시에 "지금 위치에서 가장 가까운 벽까지의 거리"만큼씩 전진시킨다.
scipy 가 있으면 EDT 는 scipy, 없으면 느린 numpy 폴백.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt as _edt
except Exception:  # pragma: no cover
    _edt = None


def _edt_numpy(free: np.ndarray) -> np.ndarray:
    """scipy 없을 때: 점유칸까지 거리 (칸 단위) — brute force 청크 계산 (느림, 맵 로드시 1회)."""
    occ_rc = np.argwhere(~free)
    h, w = free.shape
    out = np.zeros((h, w), dtype=np.float32)
    if len(occ_rc) == 0:
        out[:] = max(h, w)
        return out
    rows, cols = np.nonzero(free)
    for i in range(0, len(rows), 4096):
        r = rows[i:i + 4096, None]
        c = cols[i:i + 4096, None]
        d2 = (r - occ_rc[None, :, 0]) ** 2 + (c - occ_rc[None, :, 1]) ** 2
        out[rows[i:i + 4096], cols[i:i + 4096]] = np.sqrt(d2.min(axis=1))
    return out


class GridMap:
    """occupancy 격자 + 거리장. 좌표계: ROS map (origin = 좌하단)."""

    def __init__(self, occupied: np.ndarray, resolution: float, origin):
        self.occ = occupied.astype(bool)
        self.res = float(resolution)
        self.ox, self.oy = float(origin[0]), float(origin[1])
        self.h, self.w = self.occ.shape
        free = ~self.occ
        dist_cells = _edt(free) if _edt is not None else _edt_numpy(free)
        self.dist = (np.asarray(dist_cells, dtype=np.float32) * self.res)
        g_row, g_col = np.gradient(self.dist)
        self._gx = (g_col / self.res).astype(np.float32)      # ∂d/∂x
        self._gy = (-g_row / self.res).astype(np.float32)     # ∂d/∂y (행은 아래로 증가)

    def release_grad(self) -> None:
        """레이싱라인 생성 후 기울기 배열 해제 (env 여러 개 띄울 때 메모리 절약)."""
        self._gx = self._gy = None

    def grad(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """거리장 기울기 (벽에서 멀어지는 방향)."""
        row, col = self.to_rc(x, y)
        row = np.clip(row, 0, self.h - 1)
        col = np.clip(col, 0, self.w - 1)
        return self._gx[row, col], self._gy[row, col]

    def to_rc(self, x, y):
        col = np.floor((np.asarray(x) - self.ox) / self.res).astype(np.int64)
        row = (self.h - 1 - np.floor((np.asarray(y) - self.oy) / self.res)).astype(np.int64)
        return row, col

    def distance(self, x, y) -> np.ndarray:
        """점 → 가장 가까운 점유칸 거리 [m]. 맵 밖은 0 (벽 취급)."""
        row, col = self.to_rc(x, y)
        inside = (row >= 0) & (row < self.h) & (col >= 0) & (col < self.w)
        out = np.zeros(np.shape(row), dtype=np.float32)
        out[inside] = self.dist[row[inside], col[inside]]
        return out

    def occupied(self, x, y) -> np.ndarray:
        row, col = self.to_rc(x, y)
        inside = (row >= 0) & (row < self.h) & (col >= 0) & (col < self.w)
        out = np.ones(np.shape(row), dtype=bool)
        out[inside] = self.occ[row[inside], col[inside]]
        return out


def cast_rays(grid: GridMap, x: float, y: float, yaw: float,
              angles: np.ndarray, range_max: float, max_iter: int = 96) -> np.ndarray:
    """모든 빔 동시 sphere tracing. 반환 거리 [m] (벽 없으면 range_max)."""
    th = yaw + angles
    c, s = np.cos(th), np.sin(th)
    n = th.size
    t = np.zeros(n, dtype=np.float64)
    hit_eps = 0.75 * grid.res
    min_step = 0.5 * grid.res
    active = np.ones(n, dtype=bool)
    for _ in range(max_iter):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        px = x + t[idx] * c[idx]
        py = y + t[idx] * s[idx]
        d = grid.distance(px, py)
        hit = d <= hit_eps
        step = np.maximum(d - hit_eps, min_step)
        t[idx[~hit]] += step[~hit]
        done = hit | (t[idx] >= range_max)
        active[idx[done]] = False
    t = np.minimum(t, range_max)
    return t


class LidarSim:
    """차량 pose → 노이즈 포함 LaserScan 유사 배열 (실차와 같은 전처리를 거치게 raw 로 반환)."""

    def __init__(self, grid: GridMap, spec, rng: np.random.Generator):
        from .obs_builder import beam_angles
        self.grid, self.spec, self.rng = grid, spec, rng
        self.angles = beam_angles(spec)

    def scan(self, x: float, y: float, yaw: float, noise: bool = True) -> np.ndarray:
        sp = self.spec
        lx = x + sp.mount_x * np.cos(yaw)
        ly = y + sp.mount_x * np.sin(yaw)
        r = cast_rays(self.grid, lx, ly, yaw, self.angles, sp.range_max)
        if noise:
            r = r + self.rng.normal(0.0, sp.noise_std, r.size)
            drop = self.rng.random(r.size) < sp.dropout_prob
            r[drop] = np.inf          # 무반사 → 전처리에서 range_max
        return r
