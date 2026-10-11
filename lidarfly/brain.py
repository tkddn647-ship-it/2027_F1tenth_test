"""
lidarfly.brain
==============
LiDAR 1080빔 × 3프레임(거리, 0~1 = 거리/range_max) → 커넥톰만 (CNN·MLP 없음).

  구간 풀링   15빔씩 72열.  열마다 거리의 중앙값·최솟값 (최솟값 = 그 방향에서 가장 가까운 점).
  가까움      0.25 m / 거리  (depthfly 와 같은 1/거리 정의. 라이다 데이터시트 값이 아님).
              0.25 m 보다 가까우면 1 로 포화.  카메라는 그 안쪽이 측정 없음(0)이지만 라이다는 보인다.
  눈          (프레임 3, 행 2, 열 72).  행 0 = 중앙값 가까움, 행 1 = 최근접 가까움.
              열 0 = 오른쪽(−135°), 열이 커질수록 왼쪽.  depthfly 와 같은 좌우 약속.
  lamina      프레임 차 ON/OFF (빠른 n−(n−1), 느린 (n−1)−(n−2)), 대비 적응.
  이후        T4/T5 4방향 → HS/VS, LPLC2(가까움 증가율), LC, L3.  depthfly 회로와 같은 식.
  DN 48       좌우 대칭 배선: 같은 쪽 흥분·반대쪽 억제, 세기만 학습 → tanh.

TODO: torch 없는 numpy 추론(np_actor)은 이번 범위 밖.  forward_numpy 가 그 참조 구현이다.
"""

from __future__ import annotations

import numpy as np

from camera.depthfly.brain import (
    N_BAND,
    N_SECT,
    feature_layout,          # 재수출: 특징 배치 (HS…NEAR)
    feature_meta,            # 재수출: 섹터·HS 여부·좌우 거울 인덱스
    features_numpy as _df_features,
    forward_numpy as _df_forward,
    mirror_dn_index,         # 재수출: DN 좌우 짝
)

from .config import N_BEAMS, lidarfly_config

# 1080 / 72 = 15빔(약 3.75°).  커넥톰 망막 크기 — 센서 분해능으로 역산한 값이 아니다.
N_COLS = 72
BEAM_W = N_BEAMS // N_COLS
assert BEAM_W * N_COLS == N_BEAMS and N_COLS % N_SECT == 0

# depthfly 와 같은 가까움: 0.25 m / 거리.  회로가 쓰는 스케일이지 라이다 교정값이 아니다.
NEAR_REF_M = 0.25
DIST_EPS = 1e-4


def scans_to_near_np(scan, range_max: float) -> np.ndarray:
    """(B, 3, 1080) 정규화 거리 → 가까움 (B, 3, 2, 72).  행 0 중앙값, 행 1 최근접."""
    x = np.asarray(scan, np.float32)
    if x.ndim != 3 or x.shape[1] != 3 or x.shape[2] != N_BEAMS:
        raise ValueError(f"scan shape {x.shape}, expected (B, 3, {N_BEAMS})")
    cols = x.reshape(x.shape[0], 3, N_COLS, BEAM_W)
    d_med = np.median(cols, axis=-1)
    d_min = cols.min(axis=-1)
    d = np.stack([d_med, d_min], axis=-2)
    dist_m = np.maximum(d, DIST_EPS) * float(range_max)
    return np.clip(NEAR_REF_M / dist_m, 0.0, 1.0).astype(np.float32)


def features_numpy(scan, p, range_max: float | None = None, n_sect: int = N_SECT, n_band: int = N_BAND):
    if range_max is None:
        range_max = float(lidarfly_config().lidar.range_max)
    near = scans_to_near_np(scan, range_max)
    return _df_features(near, p, n_sect, n_band)


def forward_numpy(scan, p, mask, sign, range_max: float | None = None,
                  n_sect: int = N_SECT, n_band: int = N_BAND):
    """→ DN (B, n_dn) ∈ (−1, 1)."""
    if range_max is None:
        range_max = float(lidarfly_config().lidar.range_max)
    near = scans_to_near_np(scan, range_max)
    return _df_forward(near, p, mask, sign, n_sect, n_band)


try:
    import torch
    import torch.nn as nn

    from camera.depthfly.brain import DepthFlyBrain
except ImportError:  # pragma: no cover
    torch = None


if torch is not None:

    def scans_to_near_t(scan, range_max):
        x = scan.float()
        if x.ndim != 3 or x.shape[1] != 3 or x.shape[2] != N_BEAMS:
            raise ValueError(f"scan shape {tuple(x.shape)}, expected (B, 3, {N_BEAMS})")
        cols = x.reshape(x.shape[0], 3, N_COLS, BEAM_W)
        d_med = cols.median(dim=-1).values
        d_min = cols.amin(dim=-1)
        d = torch.stack([d_med, d_min], dim=-2)
        dist_m = torch.clamp(d, min=DIST_EPS) * range_max
        return (NEAR_REF_M / dist_m).clamp(0.0, 1.0)


if torch is not None:

    class LidarFlyBrain(nn.Module):
        """scan (B, 3, 1080) → DN (B, n_dn).  배선 부호는 고정, 세기만 파라미터."""

        def __init__(self, n_dn: int = 48, fan_in: int = 12, seed: int = 0,
                     n_sect: int = N_SECT, n_band: int = N_BAND, range_max: float | None = None):
            super().__init__()
            if range_max is None:
                range_max = float(lidarfly_config().lidar.range_max)
            self.core = DepthFlyBrain(n_dn=n_dn, fan_in=fan_in, seed=seed, n_sect=n_sect,
                                      n_band=n_band, wiring="bilateral")
            self.register_buffer("range_max", torch.tensor(float(range_max)))
            self.features_dim = self.core.features_dim

        @property
        def mask(self):
            return self.core.mask

        @property
        def sign(self):
            return self.core.sign

        def numpy_params(self) -> dict:
            return self.core.numpy_params()

        def forward(self, scan):
            near = scans_to_near_t(scan, self.range_max)
            return self.core(near)
