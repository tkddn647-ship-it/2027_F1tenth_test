"""
depthfly.policy — actor = DepthFlyBrain DN 48 ‖ state 14 → 선형 읽기 (pi=[]).  critic 은 + priv → MLP (학습 전용).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from mapless40.config import PRIV_DIM

from .brain import DepthFlyBrain


class NearFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, use_priv: bool = False, n_dn: int = 48, fan_in: int = 12,
                 wiring_seed: int = 0, wiring: str = "bilateral"):
        sdim = observation_space["state"].shape[0]
        brain = DepthFlyBrain(n_dn=n_dn, fan_in=fan_in, seed=wiring_seed, wiring=wiring)
        super().__init__(observation_space, features_dim=n_dn + sdim + (PRIV_DIM if use_priv else 0))
        self.brain, self.use_priv = brain, use_priv

    def forward(self, obs):
        parts = [self.brain(obs["near"]), obs["state"].float()]
        if self.use_priv:
            parts.append(obs["priv"].float())
        return torch.cat(parts, dim=1)


class NearDeterministicActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.fe, self.latent_pi, self.mu = actor.features_extractor, actor.latent_pi, actor.mu

    def forward(self, near, state):
        return torch.tanh(self.mu(self.latent_pi(self.fe({"near": near, "state": state}))))
