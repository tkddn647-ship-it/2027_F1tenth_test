"""
mapless40.policy
================
Asymmetric SAC 정책 (SB3).

  actor  : scan(4×1125) → LiDAR 인코더 ─┐
           state(17)    → FC 32 ───────┴→ [Z;u] → MLP 256·256 → (μ, logσ) → tanh → (δ, v)
  critic : 위 인코더(별도 가중치) + priv(24) + a(2) → MLP 256·256 → Q      (×2, +target)

LiDAR 인코더 두 종류 (--encoder):
  conv1d : 프레임마다 Conv1d×3 → 48, 4장 concat → 192          (range view, 기준선)
  bev    : IMU·속도로 4프레임을 현재 차 기준에 정렬해 150×150 격자에 찍고
           (프레임별 점유 4ch + 최신 프레임 빈공간 1ch) → Conv2d×4 → 192
  both   : conv1d 192 ‖ bev 192 → 384  (코랩 T4 에선 conv1d 대비 약 10배 느림)

배포되는 건 actor 뿐 → priv 는 actor 에 절대 들어가지 않는다 (make_actor 참고).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy

from .config import PRIV_DIM, LidarSpec, NormSpec


# --------------------------------------------------------------------------- 1D
from .policy_spec import CONV1D_LAYERS  # (out_ch, kernel, stride, pad), viz_encoder 와 공유


class ScanEncoder1D(nn.Module):
    """(B, T, N) → (B, T·out_dim).  프레임마다 같은 가중치.

    1125 → 375 → 125 → 63.  예전 (32·64·64, stride 2·2·2, 1125→563→282→141) 은
    학습 업데이트 1회에 약 130 GFLOP 라 코랩 T4 에서 15 steps/s 밖에 안 나왔다 → 연산 약 1/6.
    첫 층 커널 7(빔 7개 = 1.7°)이 빔을 전부 덮으므로 0.24° 빔을 버리지 않는다.
    """

    def __init__(self, n_beams: int, hist: int, out_dim: int = 48):
        super().__init__()
        self.hist, self.n_beams, self.out_dim = hist, n_beams, out_dim
        layers, c_in = [], 1
        for c_out, k, s, p in CONV1D_LAYERS:
            layers += [nn.Conv1d(c_in, c_out, k, stride=s, padding=p), nn.ReLU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            L = self.conv(torch.zeros(1, 1, n_beams)).shape[-1]     # 1125 → 63
        k = max(1, L // 8)
        self.pool = nn.AvgPool1d(k, stride=k)                      # ONNX 친화 (adaptive 대신)
        with torch.no_grad():
            flat = self.pool(self.conv(torch.zeros(1, 1, n_beams))).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flat, out_dim), nn.ReLU())
        self.features_dim = hist * out_dim

    def forward(self, scan: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        b = scan.shape[0]
        x = scan.reshape(b * self.hist, 1, self.n_beams)
        z = self.fc(self.pool(self.conv(x)))
        return z.reshape(b, self.hist * self.out_dim)


# --------------------------------------------------------------------------- BEV
class BEVRasterizer(nn.Module):
    """scan (B,T,N) + state (B,17) → BEV 이미지 (B, T+1, H, W).  미분 불필요(입력 변환)."""

    def __init__(self, lidar: LidarSpec, norm: NormSpec, hist: int,
                 x_min: float = -1.5, x_max: float = 13.5, y_half: float = 7.5,
                 res: float = 0.1):
        super().__init__()
        half = lidar.fov / 2.0
        ang = torch.linspace(-half, half, lidar.n_beams)
        self.register_buffer("cos_a", torch.cos(ang), persistent=False)
        self.register_buffer("sin_a", torch.sin(ang), persistent=False)
        # 빈공간: 격자칸 → (빔 번호, 거리) 조회표로 한 번에 판정 (gather, 배치 학습에서 빠름)
        from .obs_builder import bev_polar_lut
        lut_b, lut_rho, lut_ok = bev_polar_lut(lidar, x_min, x_max, y_half, res)
        self.register_buffer("lut_beam", torch.from_numpy(lut_b), persistent=False)
        self.register_buffer("lut_rho", torch.from_numpy(lut_rho), persistent=False)
        self.register_buffer("lut_ok", torch.from_numpy(lut_ok), persistent=False)
        self.range_max, self.mount_x = lidar.range_max, lidar.mount_x
        self.v_scale, self.w_scale, self.dt_nom = norm.v_scale, norm.w_scale, norm.dt_nominal
        self.hist, self.x_min, self.y_half, self.res = hist, x_min, y_half, res
        self.H = int(round((x_max - x_min) / res))
        self.W = int(round(2 * y_half / res))
        self.C = hist + 1

    def _cells(self, x: torch.Tensor, y: torch.Tensor, ch: int) -> torch.Tensor:
        """좌표 → 평탄화 인덱스 (범위 밖은 쓰레기칸 C·H·W)."""
        row = torch.floor((x - self.x_min) / self.res).long()          # 앞쪽 = 행 증가
        col = torch.floor((self.y_half - y) / self.res).long()         # 왼쪽 = 열 감소
        ok = (row >= 0) & (row < self.H) & (col >= 0) & (col < self.W)
        idx = ch * self.H * self.W + row * self.W + col
        trash = self.C * self.H * self.W
        return torch.where(ok, idx, torch.full_like(idx, trash))

    def forward(self, scan: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b, T, N = scan.shape
        assert T == 4, "state 레이아웃(obs_builder.ObsHistory)은 hist=4 기준"
        r = scan.float() * self.range_max
        hit = scan < 0.999
        v = state[:, 0:T] * self.v_scale
        w = state[:, 4:4 + T] * self.w_scale
        dt = state[:, 13:13 + T] * self.dt_nom + self.dt_nom

        # 프레임 k 의 pose 를 최신 프레임(T-1) 좌표로: 구간 j(=k+1..T-1) 운동을 역으로 누적
        px = [None] * T; py = [None] * T; pth = [None] * T
        zero = torch.zeros(b, device=scan.device)
        px[T - 1], py[T - 1], pth[T - 1] = zero, zero, zero
        for k in range(T - 2, -1, -1):
            j = k + 1
            dth = w[:, j] * dt[:, j]
            d = v[:, j] * dt[:, j]
            lx = -d * torch.cos(dth / 2)          # frame k 원점을 frame j 에서 본 위치
            ly = d * torch.sin(dth / 2)
            c, s = torch.cos(pth[j]), torch.sin(pth[j])
            px[k] = px[j] + c * lx - s * ly
            py[k] = py[j] + s * lx + c * ly
            pth[k] = pth[j] - dth

        # 점유 채널 (프레임별)
        lx = r * self.cos_a + self.mount_x                     # (B,T,N) 라이다점, 프레임 자체 좌표
        ly = r * self.sin_a
        idx_all = []
        for k in range(T):
            c, s = torch.cos(pth[k])[:, None], torch.sin(pth[k])[:, None]
            gx = px[k][:, None] + c * lx[:, k] - s * ly[:, k]
            gy = py[k][:, None] + s * lx[:, k] + c * ly[:, k]
            idx = self._cells(gx, gy, k)
            idx = torch.where(hit[:, k], idx, torch.full_like(idx, self.C * self.H * self.W))
            idx_all.append(idx)
        idx = torch.cat(idx_all, dim=1)
        img = torch.zeros(b, self.C * self.H * self.W + 1, device=scan.device)
        img.scatter_(1, idx, 1.0)
        # 빈공간 채널 (최신 프레임): 칸 거리 < 그 방향 빔 거리 − res
        r_cell = r[:, T - 1].index_select(1, self.lut_beam)            # (B, H·W)
        free = (self.lut_rho[None] < r_cell - self.res) & self.lut_ok[None]
        img[:, T * self.H * self.W:self.C * self.H * self.W] = free.float()
        return img[:, :-1].reshape(b, self.C, self.H, self.W)


class ScanEncoderBEV(nn.Module):
    def __init__(self, lidar: LidarSpec, norm: NormSpec, hist: int, out_dim: int = 192):
        super().__init__()
        self.raster = BEVRasterizer(lidar, norm, hist)
        C = self.raster.C
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AvgPool2d(2), nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.cnn(torch.zeros(1, C, self.raster.H, self.raster.W)).shape[-1]
        self.fc = nn.Sequential(nn.Linear(flat, out_dim), nn.ReLU())
        self.features_dim = out_dim

    def forward(self, scan: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img = self.raster(scan, state)
        return self.fc(self.cnn(img))


class ScanEncoderBoth(nn.Module):
    """1D CNN(거리 배열) 과 BEV 2D CNN(x,y 격자) 을 나란히 돌려 이어 붙인다 → 192 + 192 = 384.

    1D 는 각도별 거리 패턴(틈 방향), BEV 는 공간 모양(장애물·벽 위치)을 맡는다.
    """

    def __init__(self, lidar: LidarSpec, norm: NormSpec, hist: int):
        super().__init__()
        self.d1 = ScanEncoder1D(lidar.n_beams, hist)
        self.bev = ScanEncoderBEV(lidar, norm, hist)
        self.features_dim = self.d1.features_dim + self.bev.features_dim

    def forward(self, scan: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.d1(scan, state), self.bev(scan, state)], dim=1)


# ----------------------------------------------------------------- extractor
class AsymFeatures(BaseFeaturesExtractor):
    """Dict obs → 특징. use_priv=False 면 priv 를 읽지도 않는다 (actor)."""

    def __init__(self, observation_space: spaces.Dict, use_priv: bool = False,
                 encoder: str = "conv1d", lidar_cfg: dict | None = None,
                 norm_cfg: dict | None = None, state_dim_out: int = 32):
        hist, n_beams = observation_space["scan"].shape
        state_dim = observation_space["state"].shape[0]
        lidar = LidarSpec(**(lidar_cfg or {}))
        norm = NormSpec(**(norm_cfg or {}))
        assert lidar.n_beams == n_beams, f"lidar_cfg n_beams {lidar.n_beams} != obs {n_beams}"
        if encoder == "conv1d":
            enc = ScanEncoder1D(n_beams, hist)
        elif encoder == "bev":
            enc = ScanEncoderBEV(lidar, norm, hist)
        elif encoder == "both":
            enc = ScanEncoderBoth(lidar, norm, hist)
        else:
            raise ValueError(f"unknown encoder {encoder}")
        feat = enc.features_dim + state_dim_out + (PRIV_DIM if use_priv else 0)
        super().__init__(observation_space, features_dim=feat)
        self.use_priv = use_priv
        self.scan_enc = enc
        self.state_enc = nn.Sequential(nn.Linear(state_dim, state_dim_out), nn.ReLU())

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        scan = obs["scan"].float()
        state = obs["state"].float()
        parts = [self.scan_enc(scan, state), self.state_enc(state)]
        if self.use_priv:
            parts.append(obs["priv"].float())
        return torch.cat(parts, dim=1)


class AsymSACPolicy(SACPolicy):
    """actor 는 use_priv=False, critic(+target) 은 use_priv=True 추출기를 따로 만든다."""

    def __init__(self, *args: Any, share_features_extractor: bool = False, **kwargs: Any):
        if share_features_extractor:
            raise ValueError("asymmetric critic 은 추출기를 공유할 수 없다")
        super().__init__(*args, share_features_extractor=False, **kwargs)

    def _make_fe(self, use_priv: bool) -> BaseFeaturesExtractor:
        kw = dict(self.features_extractor_kwargs)
        kw["use_priv"] = use_priv
        return self.features_extractor_class(self.observation_space, **kw)

    def make_actor(self, features_extractor=None):
        if features_extractor is None:
            features_extractor = self._make_fe(False)
        return super().make_actor(features_extractor)

    def make_critic(self, features_extractor=None):
        if features_extractor is None:
            features_extractor = self._make_fe(True)
        return super().make_critic(features_extractor)


class DeterministicActor(nn.Module):
    """배포용: (scan, state) → tanh(μ) ∈ [−1,1]².  priv 입력 없음."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.fe = actor.features_extractor
        self.latent_pi = actor.latent_pi
        self.mu = actor.mu

    def forward(self, scan: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        feats = self.fe({"scan": scan, "state": state})
        return torch.tanh(self.mu(self.latent_pi(feats)))
