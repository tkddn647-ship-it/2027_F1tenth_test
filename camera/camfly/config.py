"""
camfly.config
=============
Orbbec Gemini 2L 한 대 + 초파리 시각엽(커넥톰) 정책 설정.

차량 동역학·보상·레이싱라인(critic 전용)은 mapless40 을 그대로 쓰고,
센서만 LiDAR → 카메라(흑백 파리 눈 + depth 가짜 스캔)로 바꾼다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mapless40.config import EnvConfig


@dataclass
class CameraSpec:
    """Gemini 2L 데이터시트 v1.0: RGB·IR 글로벌 셔터, 베이스라인 100 mm, depth 91°×66°, 0.25~10 m (최적 0.30~7.0 m),
    정확도 ≤ 2 % (1280×800, 2 m, 반사율 > 80 % 평면의 RMS). 지연은 데이터시트에 없음."""

    # 장착 (뒷바퀴축 기준 0.31 m 앞 = CG 기준 0.139 m, mapless40 LiDAR 자리)
    mount_x: float = 0.139
    height: float = 0.18                # 바닥에서 렌즈 높이 [m]
    pitch_deg: float = -10.0            # 아래로 숙임 (가까운 바닥·덕트 아랫부분을 보려고)
    hfov_deg: float = 91.0              # depth 기준 가로 시야 (RGB 94°)
    vfov_deg: float = 66.0
    # 파리 눈 재샘플링: 가로는 정면이 촘촘한 acute zone, 세로는 균일
    n_cols: int = 64
    n_rows: int = 16
    acute_half_deg: float = 15.0        # 정면 ±15° 에 열의 절반을 몰아줌
    acute_frac: float = 0.5
    # 세로 acute zone: 0 이면 균일 (camfly). depthfly 는 지평선 ±row_acute_half 에 행을 몰아줌
    row_acute_half_deg: float = 0.0
    row_acute_frac: float = 0.75
    # depth 가짜 스캔
    depth_max: float = 7.0              # 최적 범위 끝
    depth_min: float = 0.25             # 이보다 가까우면 측정 없음 (데이터시트 최소 거리)
    depth_k: float = 0.01               # σ ≈ k·d² [m]  (데이터시트 상한 2 % @ 2 m = 4 cm 에 맞춤. 스테레오 오차 ∝ d²)
    depth_hole_p0: float = 0.02         # 기본 구멍 확률
    depth_hole_p1: float = 0.15         # + p1·(d/depth_max)²  (멀수록·균일한 면일수록 구멍)
    min_obj_height: float = 0.05        # 바닥보다 이만큼 높은 점만 장애물로
    # depthfly: 높이 판정 (시뮬·실차 공통) 과 장착 pitch 오차 (시뮬)
    floor_margin_deg: float = 1.5       # 높이 기준을 거리에 따라 늘림: min_obj_height + d·tan(이 각) (pitch 오차 여유)
    max_obj_height: float = 0.40        # 이보다 높은 점은 버림 (덕트 너머 배경 벽·사람)
    pitch_err_deg: float = 1.0          # [추정] 에피소드마다 장착·자세 pitch 오차 ±
    pitch_jitter_deg: float = 0.3       # [추정] 프레임마다 흔들림 σ (가감속·진동)

    @property
    def hfov(self) -> float:
        return math.radians(self.hfov_deg)

    @property
    def vfov(self) -> float:
        return math.radians(self.vfov_deg)

    def col_azimuths(self) -> np.ndarray:
        """열별 방위각 [rad], 오른쪽(−) → 왼쪽(+). 정면 ±acute_half 에 acute_frac 만큼의 열."""
        n = self.n_cols
        n_in = int(round(n * self.acute_frac)) // 2 * 2
        n_out = (n - n_in) // 2
        h, a = self.hfov / 2.0, math.radians(self.acute_half_deg)
        inner = np.linspace(-a, a, n_in + 2)[1:-1] if n_in > 0 else np.zeros(0)
        right = np.linspace(-h, -a, n_out + 1)[:-1]
        left = np.linspace(a, h, n_out + 1)[1:]
        az = np.concatenate([right, inner, left])
        assert az.size == n, az.size
        return az.astype(np.float64)

    def row_elevations(self) -> np.ndarray:
        """행별 고도각 [rad] (차체 수평 기준, 위 +). 행 0 = 위쪽."""
        p, v = math.radians(self.pitch_deg), self.vfov / 2.0
        if self.row_acute_half_deg <= 0.0:
            return (p + np.linspace(v, -v, self.n_rows)).astype(np.float64)
        # 지평선 기준 대칭: ±a 안에 row_acute_frac 만큼의 행, 나머지는 시야 끝(±lim)까지
        n = self.n_rows
        n_in = int(round(n * self.row_acute_frac)) // 2 * 2
        n_out = (n - n_in) // 2
        a, lim = math.radians(self.row_acute_half_deg), v - abs(p)
        inner = np.linspace(a, -a, n_in)
        up = np.linspace(lim, a, n_out + 1)[:-1]
        el = np.concatenate([up, inner, -up[::-1]])
        assert el.size == n and lim > a, (el.size, lim, a)
        return el.astype(np.float64)


@dataclass
class SceneSpec:
    """시뮬 장면 밝기·질감 (에피소드마다 랜덤화). 대회 트랙 = 지름 33 cm 에어 덕트."""

    duct_height: float = 0.33
    obstacle_height: float = 0.30
    duct_rib_m: tuple[float, float] = (0.04, 0.12)     # 덕트 주름 간격
    wall_level: tuple[float, float] = (0.45, 0.85)
    floor_level: tuple[float, float] = (0.10, 0.45)
    obstacle_level: tuple[float, float] = (0.05, 0.95)
    bg_level: tuple[float, float] = (0.20, 0.90)
    texture_amp: tuple[float, float] = (0.03, 0.15)
    pixel_noise: tuple[float, float] = (0.005, 0.03)
    gain_jitter: float = 0.05                           # 프레임마다 노출 흔들림


@dataclass
class CamFlyConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    cam: CameraSpec = field(default_factory=CameraSpec)
    scene: SceneSpec = field(default_factory=SceneSpec)
    hist: int = 3                       # 과거 프레임 수 (n-2, n-1, n)
    fps: float = 30.0

    def __post_init__(self):
        t = self.env.timing
        t.dt_scan = 1.0 / self.fps                     # 제어 주기 = 카메라 프레임
        t.phys_dt = t.dt_scan / 6.0
        t.imu_every = 2                                # ≈ 90 Hz
        t.speed_every = 4                              # ≈ 45 Hz
        t.compute_latency = 0.030                      # [측정 필요] 노출·USB·처리 (틱 안에서 근사)
        t.hist = self.hist
        self.env.norm.dt_nominal = t.dt_scan
        self.env.max_episode_s = self.env.max_episode_s


# 관측 크기
def state_dim(hist: int) -> int:
    """v̄×hist, ω̄×hist, ω_latest, (δ,v)×2, Δt×hist."""
    return 3 * hist + 5
