"""
camfly.eye
==========
파리 눈(겹눈) 입력 만들기 — 시뮬 렌더러와 실제 카메라 재샘플링이 **같은 격자**를 쓴다.

  격자: 열 n_cols(정면 촘촘), 행 n_rows. 각 칸 = (방위각, 고도각) 한 방향의 밝기.

시뮬 (FlyEyeRenderer):
  2D 맵에서 열마다 벽·장애물까지 레이캐스팅 → 행마다 그 방향 시선이
  장애물(높이 0.30) / 덕트(0.33) / 바닥 / 배경 중 무엇에 먼저 닿는지 계산 → 질감 있는 밝기.
  같은 레이캐스팅 거리로 depth 가짜 스캔(열별 최근접 거리 + Gemini 2L 오차 모델)도 만든다.

실차 (resample_gray / depth_to_scan):
  핀홀 내부 파라미터 K 로 각 칸 방향을 픽셀 좌표로 투영해 흑백 영상을 샘플링,
  depth 영상은 열마다 바닥보다 높은 점의 최소 거리.

광수용체 적응 (adapt): 프레임마다 평균을 빼고 표준편차로 나눠 [-1, 1] 근처로.
"""

from __future__ import annotations

import numpy as np

from mapless40.raycast import cast_rays, ray_circles

from .config import CameraSpec, SceneSpec


# ---------------------------------------------------------------- 공통
def adapt(img: np.ndarray, eps: float = 0.02) -> np.ndarray:
    """광수용체 밝기 적응: (I − 평균) / (표준편차 + eps), ±3 에서 자르고 /3."""
    m = float(img.mean())
    s = float(img.std())
    return np.clip((img - m) / (s + eps), -3.0, 3.0) / 3.0


def _tex(u: np.ndarray, lam: float, phase: float) -> np.ndarray:
    """주기 lam 의 줄무늬 + 조금 긴 주기 섞음 (덕트 주름·바닥 이음매 흉내)."""
    return 0.6 * np.sin(2 * np.pi * u / lam + phase) + 0.4 * np.sin(2 * np.pi * u / (2.7 * lam) + 1.7 * phase)


# ---------------------------------------------------------------- 시뮬
class FlyEyeRenderer:
    def __init__(self, cam: CameraSpec, scene: SceneSpec, range_max: float = 15.0):
        self.cam, self.scene = cam, scene
        self.az = cam.col_azimuths()                     # (C,)
        self.el = cam.row_elevations()                   # (R,)
        self.range_max = range_max
        self.tan_el = np.tan(self.el)[:, None]           # (R,1)
        self.p = None

    def randomize(self, rng: np.random.Generator) -> None:
        s = self.scene
        u = lambda r: float(rng.uniform(*r))  # noqa: E731
        self.p = dict(
            wall=u(s.wall_level), floor=u(s.floor_level), obst=u(s.obstacle_level), bg=u(s.bg_level),
            amp_w=u(s.texture_amp), amp_f=u(s.texture_amp), amp_b=u(s.texture_amp),
            rib=u(s.duct_rib_m), floor_lam=float(rng.uniform(0.2, 0.8)),
            ph=float(rng.uniform(0, 2 * np.pi, )), ph2=float(rng.uniform(0, 2 * np.pi)),
            noise=u(s.pixel_noise),
        )

    def distances(self, grid, x: float, y: float, yaw: float, obstacles: np.ndarray):
        """열별 (벽 거리, 장애물 거리, 렌즈 위치, 세계 방위각)."""
        c = self.cam
        lx, ly = x + c.mount_x * np.cos(yaw), y + c.mount_x * np.sin(yaw)
        dw = cast_rays(grid, lx, ly, yaw, self.az, self.range_max)
        th = yaw + self.az
        if obstacles is not None and len(obstacles):
            do = ray_circles(lx, ly, th, obstacles, self.range_max)
        else:
            do = np.full_like(dw, self.range_max)
        return dw, do, lx, ly, th

    def render(self, grid, x, y, yaw, obstacles, rng: np.random.Generator, noise: bool = True):
        """→ (eye (R,C) 밝기 0~1, dw, do, lx, ly, th)."""
        if self.p is None:
            self.randomize(rng)
        p, s, c = self.p, self.scene, self.cam
        dw, do, lx, ly, th = self.distances(grid, x, y, yaw, obstacles)
        h = c.height
        ct, st = np.cos(th)[None, :], np.sin(th)[None, :]
        img = np.full((self.el.size, self.az.size), p["bg"], dtype=np.float64)
        # 배경: 세계 방위각 기준 저주파 무늬 (방이 돌면 같이 흐름)
        img += p["amp_b"] * np.sin(3.0 * th[None, :] + p["ph2"])

        tan = self.tan_el
        # 바닥: 고도 < 0 인 시선이 바닥에 닿는 수평거리
        with np.errstate(divide="ignore"):
            d_floor = np.where(tan < 0, h / np.maximum(-tan, 1e-6), np.inf)          # (R,1)
        d_floor = np.broadcast_to(d_floor, img.shape)
        # 각 시선이 거리 d 에서의 높이 = h + d·tan(el)
        z_w = h + dw[None, :] * tan                                                  # (R,C)
        z_o = h + do[None, :] * tan
        hit_o = (do[None, :] < dw[None, :]) & (z_o >= 0.0) & (z_o <= s.obstacle_height) \
            & (do[None, :] < self.range_max - 1e-3)
        hit_w = (~hit_o) & (z_w >= 0.0) & (z_w <= s.duct_height) & (dw[None, :] < self.range_max - 1e-3)
        floor = (~hit_o) & (~hit_w) & np.isfinite(d_floor)

        # 덕트: 주름 = 벽 위 점의 세계 좌표를 따라 줄무늬
        hxw, hyw = lx + dw * np.cos(th), ly + dw * np.sin(th)
        rib = _tex(hxw + hyw, p["rib"], p["ph"])[None, :]
        shade = 0.85 + 0.15 * (z_w / s.duct_height)                                  # 위가 조금 밝음
        wall_val = p["wall"] * shade + p["amp_w"] * rib
        hxo, hyo = lx + do * np.cos(th), ly + do * np.sin(th)
        obst_val = p["obst"] + 0.5 * p["amp_w"] * _tex(hxo - hyo, 0.07, p["ph2"])[None, :]
        fx = lx + d_floor * ct
        fy = ly + d_floor * st
        with np.errstate(invalid="ignore"):
            ftex = _tex(fx * 0.7 + fy * 0.3, p["floor_lam"], p["ph"]) + 0.5 * _tex(fy - fx, 0.37, p["ph2"])
            far = np.clip(d_floor / 6.0, 0, 1)                                       # 멀면 무늬가 뭉개짐
            floor_val = p["floor"] + p["amp_f"] * np.nan_to_num(ftex) * (1 - far)

        img = np.where(hit_o, obst_val, img)
        img = np.where(hit_w, wall_val, img)
        img = np.where(floor, floor_val, img)
        if noise:
            img = img * (1.0 + rng.normal(0, self.scene.gain_jitter))
            img = img + rng.normal(0, p["noise"], img.shape)
        return np.clip(img, 0.0, 1.0).astype(np.float32), dw, do, lx, ly, th

    def depth_scan(self, dw: np.ndarray, do: np.ndarray, rng: np.random.Generator,
                   noise: bool = True) -> np.ndarray:
        """열별 최근접 물체 거리 → /depth_max ∈ [0,1].  구멍·범위 밖 = 1 (모름 = 멀다고 처리)."""
        c = self.cam
        d = np.minimum(dw, do)
        if noise:
            d = d + rng.normal(0, 1, d.shape) * c.depth_k * d * d
            hole = rng.random(d.shape) < c.depth_hole_p0 + c.depth_hole_p1 * (d / c.depth_max) ** 2
            d = np.where(hole, np.inf, d)
        d = np.where((d > c.depth_max) | ~np.isfinite(d), c.depth_max, np.maximum(d, 0.0))
        return (d / c.depth_max).astype(np.float32)


# ---------------------------------------------------------------- 실차
def eye_pixel_map(cam: CameraSpec, K: np.ndarray, width: int, height: int,
                  cam_pitch_deg: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """파리 눈 격자 각 칸 방향 → 핀홀 카메라 픽셀 (u, v) 와 유효 마스크.

    K = [[fx,0,cx],[0,fy,cy],[0,0,1]] (왜곡 보정된 영상 기준). 카메라 광축은 차 정면,
    pitch 만큼 숙어 있다고 가정. 좌표: 카메라 x 오른쪽, y 아래, z 앞.
    """
    pitch = np.radians(cam.pitch_deg if cam_pitch_deg is None else cam_pitch_deg)
    az = cam.col_azimuths()[None, :]
    el = cam.row_elevations()[:, None]
    # 차체 좌표 방향 (앞 x, 왼 y, 위 z)
    dx, dy, dz = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
    # 카메라 좌표 (pitch: 차체 y 축으로 아래로 회전)
    zc = dx * np.cos(pitch) + dz * np.sin(pitch)
    yc = -(-dx * np.sin(pitch) + dz * np.cos(pitch))
    xc = -dy
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * xc / zc + cx
        v = fy * yc / zc + cy
    ok = (zc > 1e-3) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    return np.nan_to_num(u).astype(np.float32), np.nan_to_num(v).astype(np.float32), ok


def resample_gray(gray: np.ndarray, u: np.ndarray, v: np.ndarray, ok: np.ndarray,
                  box: int = 2) -> np.ndarray:
    """흑백 영상(0~255 또는 0~1) → 파리 눈 (R,C) 0~1. 각 칸은 주변 (2·box+1)² 평균 (겹눈 한 칸의 넓이)."""
    g = gray.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    H, W = g.shape
    ii = np.clip(np.rint(v).astype(int), 0, H - 1)
    jj = np.clip(np.rint(u).astype(int), 0, W - 1)
    acc = np.zeros(u.shape, np.float32)
    n = 0
    for di in range(-box, box + 1):
        for dj in range(-box, box + 1):
            acc += g[np.clip(ii + di, 0, H - 1), np.clip(jj + dj, 0, W - 1)]
            n += 1
    out = acc / n
    return np.where(ok, out, float(out[ok].mean()) if ok.any() else 0.5).astype(np.float32)


def depth_to_scan(depth_m: np.ndarray, K: np.ndarray, cam: CameraSpec,
                  cam_pitch_deg: float | None = None, stride: int = 4) -> np.ndarray:
    """depth 영상 [m] (RGB 에 정렬) → 열별 최근접 '바닥보다 높은' 거리 /depth_max.

    각 픽셀을 차체 좌표 점으로 올리고, 높이 > min_obj_height 인 점만 남겨
    그 방위각이 속한 파리 눈 열에 최소 수평거리를 넣는다.
    """
    pitch = np.radians(cam.pitch_deg if cam_pitch_deg is None else cam_pitch_deg)
    d = depth_m[::stride, ::stride].astype(np.float32)
    H, W = d.shape
    fx, fy, cx, cy = K[0, 0] / stride, K[1, 1] / stride, K[0, 2] / stride, K[1, 2] / stride
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float32)
    xc = (uu - cx) / fx * d
    yc = (vv - cy) / fy * d
    zc = d
    # 카메라 → 차체 (앞 x, 왼 y, 위 z)
    bx = zc * np.cos(pitch) - (-yc) * np.sin(pitch)
    bz = zc * np.sin(pitch) + (-yc) * np.cos(pitch) + cam.height
    by = -xc
    valid = (d > 0.1) & np.isfinite(d) & (bz > cam.min_obj_height) & (bz < 1.0)
    rng_h = np.hypot(bx, by)
    az = np.arctan2(by, bx)
    edges = cam.col_azimuths()
    mids = np.concatenate([[-np.inf], 0.5 * (edges[1:] + edges[:-1]), [np.inf]])
    col = np.clip(np.searchsorted(mids, az) - 1, 0, cam.n_cols - 1)
    out = np.full(cam.n_cols, cam.depth_max, np.float32)
    np.minimum.at(out, col[valid], np.minimum(rng_h[valid], cam.depth_max))
    return out / cam.depth_max
