"""depthfly.np_actor — 학습 zip → numpy actor (torch 없이, Jetson 용)."""

from __future__ import annotations

import numpy as np

from mapless40.np_actor import load_sb3_zip

from .brain import PARAM_KEYS, forward_numpy


class NearNumpyActor:
    def __init__(self, path: str):
        sd, self.data = load_sb3_zip(path)
        self.num_timesteps = int(self.data.get("num_timesteps", 0))
        p = "actor.features_extractor.brain."
        g = lambda k: sd[k].astype(np.float32)  # noqa: E731
        self.params = {k: g(p + k) for k in PARAM_KEYS}
        self.mask, self.sign = g(p + "mask"), g(p + "sign")
        self.mlp, i = [], 0
        while f"actor.latent_pi.{i}.weight" in sd:
            self.mlp.append((g(f"actor.latent_pi.{i}.weight"), g(f"actor.latent_pi.{i}.bias")))
            i += 2
        self.mu = (g("actor.mu.weight"), g("actor.mu.bias"))

    def __call__(self, obs, env=None) -> np.ndarray:
        dn = forward_numpy(np.asarray(obs["near"], np.float32)[None], self.params, self.mask, self.sign)[0]
        h = np.concatenate([dn, np.asarray(obs["state"], np.float32)])
        for w, b in self.mlp:
            h = np.maximum(w @ h + b, 0.0)
        return np.tanh(self.mu[0] @ h + self.mu[1]).astype(np.float32)
