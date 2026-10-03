"""
camfly.np_actor
===============
학습된 SB3 zip → numpy 로 actor 실행 (torch 없이).  encoder=fly 전용.
Jetson 에서 torch 없이 돌리거나, torch 없는 PC 에서 평가·GIF 를 만들 때.
"""

from __future__ import annotations

import numpy as np

from mapless40.np_actor import load_sb3_zip

from .flybrain import forward_numpy


class CamNumpyActor:
    def __init__(self, path: str):
        sd, self.data = load_sb3_zip(path)
        self.num_timesteps = int(self.data.get("num_timesteps", 0))
        p = "actor.features_extractor.brain."
        if f"{p}w_dn" not in sd:
            raise ValueError("encoder=fly 모델만 지원 (cnn 은 torch 로)")
        g = lambda k: sd[k].astype(np.float32)  # noqa: E731
        self.params = {k: g(p + k) for k in ("g_t4", "g_t5", "g_lp", "w_dn", "b_dn")}
        self.mask, self.sign = g(p + "mask"), g(p + "sign")
        self.mlp = []
        i = 0
        while f"actor.latent_pi.{i}.weight" in sd:
            self.mlp.append((g(f"actor.latent_pi.{i}.weight"), g(f"actor.latent_pi.{i}.bias")))
            i += 2
        self.mu = (g("actor.mu.weight"), g("actor.mu.bias"))

    def dn(self, obs) -> np.ndarray:
        return forward_numpy(np.asarray(obs["eye"], np.float32)[None], np.asarray(obs["dscan"], np.float32)[None],
                             self.params, self.mask, self.sign)[0]

    def __call__(self, obs, env=None) -> np.ndarray:
        h = np.concatenate([self.dn(obs), np.asarray(obs["state"], np.float32)])
        for w, b in self.mlp:
            h = np.maximum(w @ h + b, 0.0)
        return np.tanh(self.mu[0] @ h + self.mu[1]).astype(np.float32)
