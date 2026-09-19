"""
policy_sac_connectome.py
========================
SAC policy features:
  LiDAR Encoder (1D Conv) + IMU Encoder (MLP)
  → ConnectomeRNN as temporal memory over 5-frame window
  → features for SAC actor/critic heads

고속 성능 우선: 인코더는 성능형, 커넥톰은 토폴로지 prior + temporal mixing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy

from connectome_rnn import ConnectomeRNN

N_BEAMS = 135
HIST_LEN = 5
LIDAR_DIM = N_BEAMS * HIST_LEN
YAW_DIM = HIST_LEN
OBS_DIM = LIDAR_DIM + YAW_DIM


class LidarEncoder(nn.Module):
    """Per-frame 1D conv over 135 beams; stack 5 frames as channels when fused later."""

    def __init__(self, n_beams: int = N_BEAMS, out_dim: int = 64):
        super().__init__()
        self.n_beams = n_beams
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 135) or (B, 1, 135)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class IMUEncoder(nn.Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x)


class ConnectomeTemporalFeatures(BaseFeaturesExtractor):
    """
    Obs layout: [lidar_0..lidar_4 (each 135), yaw_0..yaw_4]
    LiDAR/IMU encoders → (skip MLP ‖ ConnectomeRNN) → fused features for SAC.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        A_signed: np.ndarray,
        input_idx: np.ndarray,
        output_idx: np.ndarray,
        lidar_out: int = 64,
        imu_out: int = 16,
        n_inner_steps: int = 2,
        fuse_dim: int = 256,
    ):
        dn_dim = len(output_idx)
        enc_dim = lidar_out + imu_out
        super().__init__(observation_space, features_dim=fuse_dim)

        self.n_beams = N_BEAMS
        self.hist_len = HIST_LEN
        self.lidar_enc = LidarEncoder(N_BEAMS, lidar_out)
        self.imu_enc = IMUEncoder(1, imu_out)

        self.brain = ConnectomeRNN(
            A_signed,
            input_idx,
            output_idx,
            n_obs=enc_dim,
            n_act=dn_dim,
            dt=1.0,
            tau=3.0,
        )
        self.brain.W_out = nn.Identity()
        self.n_inner_steps = n_inner_steps
        # Fixed connectome prior: topology features without unstable scale grads
        for p in self.brain.parameters():
            p.requires_grad = False

        # primary learnable path (must be strong enough to drive)
        self.skip = nn.Sequential(
            nn.Linear(enc_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(128 + dn_dim, fuse_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(inplace=True),
        )
        self._features_dim = fuse_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b = observations.shape[0]
        lidar = observations[:, :LIDAR_DIM].reshape(b, self.hist_len, self.n_beams)
        yaw = observations[:, LIDAR_DIM:]

        z_list = []
        for t in range(self.hist_len):
            z_l = self.lidar_enc(lidar[:, t, :])
            z_i = self.imu_enc(yaw[:, t])
            z_list.append(torch.cat([z_l, z_i], dim=-1))
        feat = torch.stack(z_list, dim=1).mean(dim=1)
        with torch.no_grad():
            y, _h = self.brain(feat.detach(), h=None, n_steps=max(3, self.n_inner_steps))
        return self.fuse(torch.cat([self.skip(feat), y], dim=-1))


def make_sac_policy_kwargs(
    A_signed: np.ndarray,
    input_idx: np.ndarray,
    output_idx: np.ndarray,
    net_arch: list | dict | None = None,
) -> dict:
    if net_arch is None:
        net_arch = dict(pi=[256, 256], qf=[256, 256])

    return dict(
        features_extractor_class=ConnectomeTemporalFeatures,
        features_extractor_kwargs=dict(
            A_signed=A_signed,
            input_idx=input_idx,
            output_idx=output_idx,
            lidar_out=64,
            imu_out=16,
            n_inner_steps=1,
            fuse_dim=256,
        ),
        net_arch=net_arch,
    )


class ConnectomeSACPolicy(SACPolicy):
    """Thin wrapper so zip load can resolve the policy class name."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
