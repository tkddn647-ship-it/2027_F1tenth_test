"""
depthfly.sensor
===============
depth → 파리 눈 '가까움' 영상 (R×C).  시뮬과 실차가 같은 격자·같은 정의를 쓴다.

  가까움 = NEAR_REF / r_h  (r_h = 수평거리 [m], NEAR_REF = 0.25 m)  → 0.25 m 에서 1, 1 m 0.25, 7 m 0.036
  측정 없음 (하늘·7 m 밖·depth 구멍) = 0
  바닥 제거: 바닥에서 min_obj_height(5 cm) 아래 점은 버린다 → 덕트·장애물만 남김 (바닥이 늘 가장 가까워 다른 걸 가리는 것 방지)
  구멍 메우기: 이번 프레임에 값이 없는 칸은 직전 값(최대 hold 프레임)을 쓴다 → 구멍 깜빡임이 가짜 움직임이 되는 것 방지

시뮬: camfly 렌더러의 레이캐스팅 거리로 각 칸이 장애물 / 덕트 / 바닥 / 없음 중 무엇인지 → 그 수평거리.
      Gemini 2L 오차: σ = k·z² (4 m 에서 2%), 구멍 p0 + p1·(z/7)².
실차: depth 영상(m, color 에 정렬) → 칸마다 주변 픽셀들의 최대 가까움 (가장 가까운 점 = 보수적).
"""

from __future__ import annotations

import numpy as np

from camera.camfly.config import CameraSpec, SceneSpec
from camera.camfly.eye import FlyEyeRenderer, eye_pixel_map

NEAR_REF = 0.25


class HoleHold:
    """값 없는 칸은 직전 유효값을 hold 프레임까지 유지."""

    def __init__(self, hold: int = 2):
        self.hold, self.prev, self.age = hold, None, None

    def reset(self):
        self.prev = self.age = None

    def __call__(self, near: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev, self.age = near.copy(), np.zeros(near.shape, np.int32)
            return near
        miss = near <= 0.0
        use = miss & (self.prev > 0) & (self.age < self.hold)
        out = np.where(use, self.prev, near).astype(np.float32)
        self.age = np.where(use, self.age + 1, 0)
        self.prev = out
        return out


def near_from_range(r: np.ndarray, depth_max: float) -> np.ndarray:
    r = np.asarray(r, np.float64)
    ok = np.isfinite(r) & (r > 0.05) & (r <= depth_max)
    out = np.zeros(r.shape, np.float32)
    out[ok] = np.minimum(NEAR_REF / r[ok], 1.0)
    return out


class DepthEyeSim:
    """시뮬 depth 파리 눈."""

    def __init__(self, cam: CameraSpec, scene: SceneSpec, range_max: float = 15.0):
        self.cam, self.scene = cam, scene
        self.r = FlyEyeRenderer(cam, scene, range_max)
        self.el = cam.row_elevations()[:, None]
        self.hold = HoleHold()

    def ranges(self, grid, x, y, yaw, obstacles):
        """칸별 수평거리 (R, C). 아무것도 안 맞으면 inf."""
        cam, s = self.cam, self.scene
        dw, do, *_ = self.r.distances(grid, x, y, yaw, obstacles)
        tan = np.tan(self.el)
        h = cam.height
        z_w = h + dw[None, :] * tan
        z_o = h + do[None, :] * tan
        rm = self.r.range_max - 1e-3
        hit_o = (do[None, :] < dw[None, :]) & (z_o >= 0) & (z_o <= s.obstacle_height) & (do[None, :] < rm)
        hit_w = ~hit_o & (z_w >= 0) & (z_w <= s.duct_height) & (dw[None, :] < rm)
        R, C = self.el.size, dw.size
        r = np.full((R, C), np.inf)                     # 바닥 = 제거 (값 없음)
        # 높이 min_obj_height 미만으로 보이는 벽·장애물 아랫부분도 제거 (실차 바닥 제거와 같은 기준)
        hit_w = hit_w & (z_w >= cam.min_obj_height)
        hit_o = hit_o & (z_o >= cam.min_obj_height)
        r = np.where(hit_w, np.broadcast_to(dw[None, :], (R, C)), r)
        r = np.where(hit_o, np.broadcast_to(do[None, :], (R, C)), r)
        return r

    def frame(self, grid, x, y, yaw, obstacles, rng: np.random.Generator, noise: bool = True) -> np.ndarray:
        cam = self.cam
        r = self.ranges(grid, x, y, yaw, obstacles)
        if noise:
            fin = np.isfinite(r)
            r = np.where(fin, r + rng.normal(0, 1, r.shape) * cam.depth_k * np.where(fin, r, 0) ** 2, r)
            hole = rng.random(r.shape) < cam.depth_hole_p0 + cam.depth_hole_p1 * np.where(fin, r / cam.depth_max, 1) ** 2
            r = np.where(hole, np.inf, r)
        return self.hold(near_from_range(r, cam.depth_max))


class DepthEyeReal:
    """실차: depth 영상 [m] (color 정렬, 왜곡 보정) → 가까움 (R, C).  칸마다 (2·box+1)² 픽셀 중 최대 가까움."""

    def __init__(self, cam: CameraSpec, K: np.ndarray, width: int, height: int, pitch_deg: float | None = None,
                 box: int = 3):
        self.cam = cam
        self.pitch = np.radians(cam.pitch_deg if pitch_deg is None else pitch_deg)
        u, v, ok = eye_pixel_map(cam, K, width, height, pitch_deg)
        di, dj = np.meshgrid(np.arange(-box, box + 1), np.arange(-box, box + 1), indexing="ij")
        self.ii = np.clip(np.rint(v)[..., None, None].astype(int) + di, 0, height - 1)   # (R,C,b,b)
        self.jj = np.clip(np.rint(u)[..., None, None].astype(int) + dj, 0, width - 1)
        self.ok = ok
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.xn = (self.jj - cx) / fx
        self.yn = (self.ii - cy) / fy
        self.hold = HoleHold()

    def frame(self, depth_m: np.ndarray) -> np.ndarray:
        z = depth_m[self.ii, self.jj].astype(np.float64)
        p = self.pitch
        bx = z * np.cos(p) - (-self.yn * z) * np.sin(p)
        by = -self.xn * z
        bz = z * np.sin(p) + (-self.yn * z) * np.cos(p) + self.cam.height
        r = np.hypot(bx, by)
        r = np.where((z > 0.1) & np.isfinite(z) & (bz >= self.cam.min_obj_height), r, np.inf)   # 바닥 제거
        near = near_from_range(r, self.cam.depth_max).max(axis=(-1, -2))
        return self.hold(np.where(self.ok, near, 0.0).astype(np.float32))
