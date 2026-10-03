"""
depthfly.sensor
===============
depth → 파리 눈 '가까움' 영상 (R×C).  시뮬과 실차가 같은 격자·같은 정의를 쓴다.

  가까움 = NEAR_REF / r_h  (r_h = 수평거리 [m], NEAR_REF = 0.25 m)  → 0.25 m 에서 1, 1 m 0.25, 7 m 0.036
  측정 없음 (하늘·7 m 밖·0.25 m 안·depth 구멍) = 0
  높이 판정: 높이가 5 cm + d·tan(1.5°) 보다 낮거나 0.40 m + d·tan(1.5°) 보다 높은 점은 버린다 → 덕트·장애물만 남김
            (거리 비례 여유 = pitch 오차로 먼 바닥이 떠 보이는 것 방지. 시뮬도 바닥을 쏘고 같은 판정을 거친다)
  구멍 메우기: 이번 프레임에 값이 없는 칸은 직전 값(최대 hold 프레임)을 쓴다 → 구멍 깜빡임이 가짜 움직임이 되는 것 방지

시뮬: camfly 렌더러의 레이캐스팅 거리로 각 칸이 장애물 / 덕트 / 바닥 / 없음 중 무엇인지 → 그 수평거리.
      Gemini 2L 오차: σ = k·z² (2 m 에서 2% = 데이터시트 상한), 구멍 p0 + p1·(z/7)², 0.25 m 안쪽 = 측정 없음.
실차: depth 영상(m) → 칸마다 주변 픽셀들 중 유효한 것의 중앙값.
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


def near_from_range(r: np.ndarray, depth_max: float, depth_min: float = 0.25) -> np.ndarray:
    r = np.asarray(r, np.float64)
    ok = np.isfinite(r) & (r >= depth_min) & (r <= depth_max)
    out = np.zeros(r.shape, np.float32)
    out[ok] = np.minimum(NEAR_REF / r[ok], 1.0)
    return out


def keep_height(z, r, cam: CameraSpec):
    """높이 판정 (시뮬·실차 공통): 바닥 + 여유 위, 덕트 높이 근처까지만. 여유는 거리에 비례 (pitch 오차)."""
    m = r * np.tan(np.radians(cam.floor_margin_deg))
    return (z >= cam.min_obj_height + m) & (z <= cam.max_obj_height + m)


class DepthEyeSim:
    """시뮬 depth 파리 눈.  광선은 실제 pitch (가정 + 오차) 로 쏘고, 맞은 점은 실차처럼 가정 pitch 로 해석한다."""

    def __init__(self, cam: CameraSpec, scene: SceneSpec, range_max: float = 15.0):
        self.cam, self.scene = cam, scene
        self.r = FlyEyeRenderer(cam, scene, range_max)
        self.el = cam.row_elevations()[:, None]
        self.hold = HoleHold()
        self.pitch_err = None                           # 에피소드 pitch 오차 [rad], 첫 프레임에서 뽑음

    def reset(self):
        self.hold.reset()
        self.pitch_err = None

    def ranges(self, grid, x, y, yaw, obstacles, pitch_err: float = 0.0):
        """칸별 수평거리 (R, C). 아무것도 안 맞거나 높이 판정에서 버려지면 inf."""
        cam, s = self.cam, self.scene
        dw, do, *_ = self.r.distances(grid, x, y, yaw, obstacles)
        dw, do = dw[None, :], do[None, :]
        el_t = self.el + pitch_err                      # 실제 광선 고도각
        tan = np.tan(el_t)
        h = cam.height
        z_w, z_o = h + dw * tan, h + do * tan
        rm = self.r.range_max - 1e-3
        hit_o = (do < dw) & (z_o >= 0) & (z_o <= s.obstacle_height) & (do < rm)
        hit_w = ~hit_o & (z_w >= 0) & (z_w <= s.duct_height) & (dw < rm)
        d_f = np.where(tan < 0, h / np.maximum(-tan, 1e-9), np.inf)          # 바닥에 닿는 수평거리 (R,1)
        hit_f = ~hit_o & ~hit_w & (d_f < dw) & (d_f < rm)
        d = np.where(hit_o, do, np.where(hit_w, dw, np.where(hit_f, d_f, np.inf)))
        fin = np.isfinite(d)
        rho = np.where(fin, d, 0.0) / np.cos(el_t)      # 광선 길이 → 가정 고도각으로 해석 (실차와 같은 계산)
        r, z = rho * np.cos(self.el), h + rho * np.sin(self.el)
        return np.where(fin & keep_height(z, r, cam), r, np.inf)

    def frame(self, grid, x, y, yaw, obstacles, rng: np.random.Generator, noise: bool = True) -> np.ndarray:
        cam = self.cam
        e = 0.0
        if noise:
            if self.pitch_err is None:
                self.pitch_err = np.radians(rng.uniform(-cam.pitch_err_deg, cam.pitch_err_deg))
            e = self.pitch_err + np.radians(rng.normal(0.0, cam.pitch_jitter_deg))
        r = self.ranges(grid, x, y, yaw, obstacles, e)
        if noise:
            fin = np.isfinite(r)
            r = np.where(fin, r + rng.normal(0, 1, r.shape) * cam.depth_k * np.where(fin, r, 0) ** 2, r)
            hole = rng.random(r.shape) < cam.depth_hole_p0 + cam.depth_hole_p1 * np.where(fin, r / cam.depth_max, 1) ** 2
            r = np.where(hole, np.inf, r)
        return self.hold(near_from_range(r, cam.depth_max, cam.depth_min))


class DepthEyeReal:
    """실차: depth 영상 [m] (왜곡 보정) → 가까움 (R, C).  칸마다 (2·box+1)² 픽셀 중 유효한 것들의 중앙값.

    (최대 가까움은 노이즈·가장자리 가짜 점을 골라 잡는다. 시뮬은 칸 중심 광선 1개라 중앙값이 더 가깝다.)
    유효 픽셀이 min_valid 비율보다 적으면 값 없음.
    """

    def __init__(self, cam: CameraSpec, K: np.ndarray, width: int, height: int, pitch_deg: float | None = None,
                 box: int = 3, min_valid: float = 0.1):
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
        self.n_min = max(1, int(round(min_valid * (2 * box + 1) ** 2)))

    def frame(self, depth_m: np.ndarray) -> np.ndarray:
        z = depth_m[self.ii, self.jj].astype(np.float64)
        p = self.pitch
        bx = z * np.cos(p) - (-self.yn * z) * np.sin(p)
        by = -self.xn * z
        bz = z * np.sin(p) + (-self.yn * z) * np.cos(p) + self.cam.height
        r = np.hypot(bx, by)
        r = np.where((z > 0.1) & np.isfinite(z) & keep_height(bz, r, self.cam), r, np.inf)   # 바닥·배경 제거
        n = np.sort(near_from_range(r, self.cam.depth_max, self.cam.depth_min).reshape(*r.shape[:2], -1), axis=-1)    # 0 (없음) 이 앞
        k = n.shape[-1]
        nv = (n > 0).sum(axis=-1)
        idx = np.minimum(k - nv + nv // 2, k - 1)
        near = np.where(nv >= self.n_min, np.take_along_axis(n, idx[..., None], axis=-1)[..., 0], 0.0)
        return self.hold(np.where(self.ok, near, 0.0).astype(np.float32))
