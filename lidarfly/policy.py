"""
lidarfly.policy
===============
actor  = LidarFlyBrain DN 48 ‖ state 14 → 선형 읽기 (train 의 pi=[]).  CNN·은닉 MLP 없음.
critic = 같은 커넥톰(별도 가중치) + privileged 27 → MLP.  mapless40.AsymSACPolicy 가 둘을 나눈다.

TODO: torch 없는 numpy 추론(np_actor)은 이번 범위 밖.
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from mapless40.config import PRIV_DIM

from .brain import LidarFlyBrain


class LidarFlyFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, use_priv: bool = False, n_dn: int = 48,
                 fan_in: int = 12, wiring_seed: int = 0, range_max: float | None = None):
        sdim = observation_space["state"].shape[0]
        brain = LidarFlyBrain(n_dn=n_dn, fan_in=fan_in, seed=wiring_seed, range_max=range_max)
        super().__init__(observation_space, features_dim=brain.features_dim + sdim + (PRIV_DIM if use_priv else 0))
        self.brain = brain
        self.use_priv = use_priv

    def forward(self, obs):
        parts = [self.brain(obs["scan"]), obs["state"].float()]
        if self.use_priv:
            parts.append(obs["priv"].float())
        return torch.cat(parts, dim=1)
