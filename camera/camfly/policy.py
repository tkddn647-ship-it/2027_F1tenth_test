"""
camfly.policy
=============
SB3 SAC 용 특징 추출기.  actor/critic 분리는 mapless40.policy.AsymSACPolicy 를 그대로 쓴다.

  actor  : [FlyBrain DN 48 ‖ state 14]  → (net_arch pi=[] 이면) 선형 읽기 → tanh → (조향, 속도)
           = "CNN·MLP 없이 커넥톰 + 선형 읽기층" (기본값)
  critic : 같은 구조의 별도 가중치 + priv 27 → MLP 256·256 → Q   (critic 은 배포 안 함)

encoder:
  fly : 합성 커넥톰 시각엽 (camfly.flybrain.FlyBrain)
  cnn : 비교용 작은 CNN (같은 입력)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from mapless40.config import PRIV_DIM

from .flybrain import FlyBrain, SmallCNN


class CamFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, use_priv: bool = False, encoder: str = "fly",
                 n_dn: int = 48, fan_in: int = 12, wiring_seed: int = 0):
        _, R, C = observation_space["eye"].shape
        sdim = observation_space["state"].shape[0]
        if encoder == "fly":
            enc = FlyBrain(n_dn=n_dn, fan_in=fan_in, seed=wiring_seed)
        elif encoder == "cnn":
            enc = SmallCNN(out_dim=n_dn, n_rows=R, n_cols=C)
        else:
            raise ValueError(encoder)
        super().__init__(observation_space, features_dim=enc.features_dim + sdim + (PRIV_DIM if use_priv else 0))
        self.brain = enc
        self.use_priv = use_priv

    def forward(self, obs: dict) -> torch.Tensor:
        parts = [self.brain(obs["eye"], obs["dscan"]), obs["state"].float()]
        if self.use_priv:
            parts.append(obs["priv"].float())
        return torch.cat(parts, dim=1)


class CamDeterministicActor(nn.Module):
    """배포용 (eye, dscan, state) → tanh(μ) ∈ [−1,1]²."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.fe = actor.features_extractor
        self.latent_pi = actor.latent_pi
        self.mu = actor.mu

    def forward(self, eye: torch.Tensor, dscan: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        f = self.fe({"eye": eye, "dscan": dscan, "state": state})
        return torch.tanh(self.mu(self.latent_pi(f)))
